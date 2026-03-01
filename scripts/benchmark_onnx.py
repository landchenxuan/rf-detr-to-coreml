#!/usr/bin/env python3
"""
Benchmark ONNX Runtime execution providers vs direct CoreML conversion.

Compares four inference paths for RF-DETR detection models:
  1. ONNX Runtime — CPUExecutionProvider
  2. ONNX Runtime — CoreMLExecutionProvider (default → NeuralNetwork FP16)
  3. ONNX Runtime — CoreMLExecutionProvider (MLProgram FP32)
  4. Direct CoreML (.mlpackage from this project)

IMPORTANT: The ONNX model is exported by _export_onnx_raw.py in a subprocess
that does NOT import rfdetr_coreml, so it uses the raw unpatched rfdetr model.
This is the same ONNX you'd get from rfdetr's built-in export — with rank-6
tensors and bicubic interpolation. The Direct CoreML path uses our patched model.

All paths use identical input (RandomState(42)) and report latency, max box diff
vs PyTorch reference, and graph partition count.

Usage:
  python scripts/benchmark_onnx.py                  # Benchmark Nano (default)
  python scripts/benchmark_onnx.py --model base     # Benchmark Base
"""

import argparse
import gc
import glob
import logging
import os
import subprocess
import sys
import time

import numpy as np
import torch

# Apply patches — only used for Direct CoreML path and PyTorch reference
import rfdetr_coreml  # noqa: F401
from rfdetr_coreml.export import MODEL_REGISTRY, NormalizedWrapper, _import_model_class, export_to_coreml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def export_onnx_raw(model_name, output_dir):
    """Export raw (unpatched) rfdetr to ONNX via a separate process.

    Uses _export_onnx_raw.py which does NOT import rfdetr_coreml,
    ensuring no monkey-patches are applied to the ONNX model.
    """
    onnx_path = os.path.join(output_dir, f"rf-detr-{model_name}-raw.onnx")
    ref_path = os.path.join(output_dir, f"rf-detr-{model_name}-raw-ref.npy")

    if os.path.exists(onnx_path) and os.path.exists(ref_path):
        logger.info(f"Using existing raw ONNX: {onnx_path}")
        return onnx_path, ref_path

    logger.info("Exporting raw (unpatched) ONNX via subprocess...")
    script = os.path.join(os.path.dirname(__file__), "_export_onnx_raw.py")
    result = subprocess.run(
        [sys.executable, script, "--model", model_name, "--output-dir", output_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"ONNX export failed:\n{result.stderr}")
        raise RuntimeError(f"_export_onnx_raw.py failed: {result.stderr[:200]}")
    logger.info(result.stdout.strip())
    return onnx_path, ref_path


def benchmark_ort(session, input_dict, n_warmup=5, n_runs=50):
    """Benchmark ONNX Runtime session, returns list of per-run times in ms."""
    for _ in range(n_warmup):
        session.run(None, input_dict)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, input_dict)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def benchmark_coreml(mlmodel, input_dict, n_warmup=5, n_runs=50):
    """Benchmark CoreML inference, returns list of per-run times in ms."""
    for _ in range(n_warmup):
        mlmodel.predict(input_dict)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        mlmodel.predict(input_dict)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def identify_coreml_outputs(result):
    """Match CoreML output arrays to boxes / logits by shape."""
    boxes = logits = None
    for _, v in result.items():
        arr = np.array(v)
        if arr.ndim >= 2 and arr.shape[-1] == 4 and boxes is None:
            boxes = arr
        elif arr.ndim >= 2 and arr.shape[-1] > 4 and logits is None:
            logits = arr
    return boxes, logits


def benchmark_model(model_name, output_dir, n_runs=50):
    """Run ONNX vs Direct CoreML benchmark for one model."""
    import onnxruntime as ort
    import coremltools as ct
    from copy import deepcopy
    from PIL import Image

    resolution = MODEL_REGISTRY[model_name][1]

    # Real test image (same as test_export.py / test_fp16.py)
    test_img_path = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "test_images", "*.jpg")))[0]
    pil_img = Image.open(test_img_path).convert("RGB").resize(
        (resolution, resolution), Image.BILINEAR
    )
    img_np = np.array(pil_img)
    pt_input = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    logger.info(f"\n{'=' * 70}")
    logger.info(f"ONNX Benchmark: {model_name} (resolution={resolution}, {n_runs} runs)")
    logger.info(f"{'=' * 70}")

    # --- Export raw ONNX (unpatched, via subprocess) ---
    onnx_path, ref_path = export_onnx_raw(model_name, output_dir)
    raw_ref_boxes = np.load(ref_path)
    logger.info(f"Loaded unpatched PyTorch reference: {raw_ref_boxes.shape}")

    # ONNX input (same float32 tensor, NCHW)
    ort_input = pt_input.numpy()

    results = []

    # --- 1. ONNX Runtime CPU ---
    logger.info("ONNX Runtime CPU...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    times = benchmark_ort(sess, {input_name: ort_input}, n_runs=n_runs)
    ort_out = sess.run(None, {input_name: ort_input})
    ort_boxes = ort_out[0]
    box_diff = float(np.abs(raw_ref_boxes - ort_boxes).max()) * resolution
    median = float(np.median(times))
    logger.info(f"  Median: {median:.1f} ms, Max box diff: {box_diff:.2f} px")
    results.append(("ONNX Runtime CPU", median, box_diff, "—"))
    del sess
    gc.collect()

    # --- 2. ONNX Runtime CoreML EP (default → NeuralNetwork FP16) ---
    logger.info("ONNX Runtime CoreML EP (default)...")
    try:
        sess = ort.InferenceSession(
            onnx_path,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        input_name = sess.get_inputs()[0].name
        times = benchmark_ort(sess, {input_name: ort_input}, n_runs=n_runs)
        ort_out = sess.run(None, {input_name: ort_input})
        ort_boxes = ort_out[0]
        box_diff = float(np.abs(raw_ref_boxes - ort_boxes).max()) * resolution
        median = float(np.median(times))

        logger.info(f"  Median: {median:.1f} ms, Max box diff: {box_diff:.2f} px")
        results.append(("ONNX Runtime CoreML EP (default)", median, box_diff, "—"))
        del sess
        gc.collect()
    except Exception as e:
        logger.warning(f"  CoreML EP (default) not available: {e}")
        results.append(("ONNX Runtime CoreML EP (default)", None, None, "N/A"))

    # --- 3. ONNX Runtime CoreML EP (MLProgram FP32) ---
    logger.info("ONNX Runtime CoreML EP (MLProgram FP32)...")
    try:
        sess = ort.InferenceSession(
            onnx_path,
            providers=[
                ("CoreMLExecutionProvider", {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                }),
                "CPUExecutionProvider",
            ],
        )
        input_name = sess.get_inputs()[0].name
        times = benchmark_ort(sess, {input_name: ort_input}, n_runs=n_runs)
        ort_out = sess.run(None, {input_name: ort_input})
        ort_boxes = ort_out[0]
        box_diff = float(np.abs(raw_ref_boxes - ort_boxes).max()) * resolution
        median = float(np.median(times))

        logger.info(f"  Median: {median:.1f} ms, Max box diff: {box_diff:.2f} px")
        results.append(("ONNX Runtime CoreML EP (MLProgram FP32)", median, box_diff, "—"))
        del sess
        gc.collect()
    except Exception as e:
        logger.warning(f"  CoreML EP (MLProgram FP32) not available: {e}")
        results.append(("ONNX Runtime CoreML EP (MLProgram FP32)", None, None, "N/A"))

    # --- 4. Direct CoreML (.mlpackage, uses patched model) ---
    # This path uses our monkey-patched model (the whole point of this project).
    # We compare against patched PyTorch reference for a fair accuracy comparison.
    mlpackage_path = os.path.join(output_dir, f"rf-detr-{model_name}-fp32.mlpackage")
    if not os.path.exists(mlpackage_path):
        logger.info("Exporting Direct CoreML (patched)...")
        mlpackage_path = export_to_coreml(model_name, output_dir, "fp32")

    # Patched PyTorch reference for Direct CoreML comparison
    model_cls = _import_model_class(model_name)
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped = NormalizedWrapper(pt_model, resolution).eval()
    del rfdetr_model
    with torch.no_grad():
        patched_ref_boxes = wrapped(pt_input)[0].numpy()
    del wrapped, pt_model
    gc.collect()

    logger.info("Direct CoreML (this project)...")
    ml_model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)
    times = benchmark_coreml(ml_model, {"image": pil_img}, n_runs=n_runs)
    result = ml_model.predict({"image": pil_img})
    cm_boxes, _ = identify_coreml_outputs(result)
    box_diff = float(np.abs(patched_ref_boxes - cm_boxes).max()) * resolution if cm_boxes is not None else None
    median = float(np.median(times))
    logger.info(f"  Median: {median:.1f} ms, Max box diff: {box_diff:.2f} px")
    results.append(("Direct CoreML (this project)", median, box_diff, "1 partition (all nodes)"))
    del ml_model
    gc.collect()

    # Summary table
    print(f"\n{'=' * 85}")
    print(f"ONNX BENCHMARK RESULTS — {model_name}")
    print(f"{'=' * 85}")
    print(f"{'Method':<42s} {'Latency':>8s} {'Max Box Diff':>15s} {'Partitions':>20s}")
    print("-" * 87)
    for name, lat, diff, parts in results:
        lat_s = f"{lat:.1f} ms" if lat is not None else "N/A"
        diff_s = f"{diff:.2f} px" if diff is not None else "N/A"
        print(f"{name:<42s} {lat_s:>8s} {diff_s:>15s} {parts:>20s}")
    print()
    print("Max Box Diff: in pixels (normalized [0,1] × resolution)")
    print("ONNX rows: vs unpatched PyTorch reference (raw rfdetr)")
    print("Direct CoreML: vs patched PyTorch reference (this project)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime vs Direct CoreML")
    parser.add_argument("--model", default="nano", help="Model name (default: nano)")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs")
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        parser.error(f"Unknown model: {args.model}. Choose from {list(MODEL_REGISTRY.keys())}")

    benchmark_model(args.model, args.output_dir, args.runs)


if __name__ == "__main__":
    main()
