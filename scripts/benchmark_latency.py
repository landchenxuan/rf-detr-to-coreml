#!/usr/bin/env python3
"""
Benchmark RF-DETR CoreML model latency across compute unit configurations.

Tests ALL, CPU_AND_GPU, CPU_AND_NE, and CPU_ONLY for the selected export
precision. For FP32 models, GPU is the only accelerator providing speedup
because ANE is unused.
Also benchmarks PyTorch CPU and MPS for comparison.

Usage:
  python scripts/benchmark_latency.py                       # Benchmark Nano FP32
  python scripts/benchmark_latency.py --model medium        # Benchmark Medium FP32
  python scripts/benchmark_latency.py --model detection --precision fp16
"""

import argparse
import gc
import glob
import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch

# Apply patches before any rfdetr imports
import rfdetr_coreml  # noqa: F401
from rfdetr_coreml.export import MODEL_REGISTRY, NormalizedWrapper, _import_model_class, export_to_coreml

DETECTION_MODELS = [k for k in MODEL_REGISTRY if not k.startswith("seg-")]
SEGMENTATION_MODELS = [k for k in MODEL_REGISTRY if k.startswith("seg-")]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def benchmark_pytorch(model, dummy, n_warmup=5, n_runs=50, device="cpu"):
    """Benchmark PyTorch inference, returns list of per-run times in ms."""
    model = model.to(device)
    x = dummy.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
    if device == "mps":
        torch.mps.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == "mps":
                torch.mps.synchronize()
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


def stats(times):
    """Compute timing statistics from a list of latencies."""
    arr = np.array(times)
    return {
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def benchmark_model(model_name, output_dir, precision="fp32", n_runs=50):
    """Run full benchmark for one model variant."""
    import coremltools as ct
    from PIL import Image

    resolution = MODEL_REGISTRY[model_name][1]
    mlpackage_path = os.path.join(output_dir, f"rf-detr-{model_name}-{precision}.mlpackage")

    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"Benchmarking: {model_name} "
        f"(resolution={resolution}, precision={precision}, {n_runs} runs)"
    )
    logger.info(f"{'=' * 60}")

    # Export if needed
    if not os.path.exists(mlpackage_path):
        logger.info(f"Exporting to CoreML {precision.upper()}...")
        mlpackage_path = export_to_coreml(model_name, output_dir, precision)

    # Real test image (same as all other scripts)
    test_img_path = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "test_images", "*.jpg")))[0]
    pil_img = Image.open(test_img_path).convert("RGB").resize(
        (resolution, resolution), Image.BILINEAR
    )
    img_np = np.array(pil_img)
    pt_input = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    coreml_input = {"image": pil_img}

    result = {"model": model_name}

    # PyTorch CPU
    logger.info("PyTorch CPU...")
    model_cls = _import_model_class(model_name)
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped = NormalizedWrapper(pt_model, resolution).eval()
    del rfdetr_model

    s = stats(benchmark_pytorch(wrapped, pt_input, n_runs=n_runs, device="cpu"))
    result["pytorch_cpu"] = s["median"]
    logger.info(f"  Median: {s['median']:.1f} ms")

    # PyTorch MPS
    if torch.backends.mps.is_available():
        logger.info("PyTorch MPS...")
        s = stats(benchmark_pytorch(wrapped, pt_input, n_runs=n_runs, device="mps"))
        result["pytorch_mps"] = s["median"]
        logger.info(f"  Median: {s['median']:.1f} ms")

    del wrapped, pt_model
    gc.collect()

    # CoreML — load one compute unit at a time to avoid EP conflicts.
    for label, cu in [
        ("ALL", ct.ComputeUnit.ALL),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
    ]:
        logger.info(f"CoreML {label}...")
        ml_model = ct.models.MLModel(mlpackage_path, compute_units=cu)
        s = stats(benchmark_coreml(ml_model, coreml_input, n_runs=n_runs))
        result[f"coreml_{label.lower()}"] = s["median"]
        logger.info(f"  Median: {s['median']:.1f} ms")
        del ml_model
        gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark RF-DETR CoreML latency")
    parser.add_argument(
        "--model", default="nano",
        help="Model name, 'detection', 'segmentation', or 'all' (default: nano)",
    )
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs")
    args = parser.parse_args()

    if args.model == "all":
        models = list(MODEL_REGISTRY.keys())
    elif args.model == "detection":
        models = DETECTION_MODELS
    elif args.model == "segmentation":
        models = SEGMENTATION_MODELS
    elif args.model in MODEL_REGISTRY:
        models = [args.model]
    else:
        parser.error(
            f"Unknown model: {args.model}. "
            f"Choose from {list(MODEL_REGISTRY.keys())} or all/detection/segmentation"
        )

    results = []
    for name in models:
        try:
            r = benchmark_model(name, args.output_dir, args.precision, args.runs)
            results.append(r)
        except Exception as e:
            logger.error(f"FAILED: {name} — {e}", exc_info=True)

    # Summary
    print(f"\n{'=' * 90}")
    print(f"LATENCY SUMMARY (median ms, precision={args.precision})")
    print(f"{'=' * 90}")
    print(
        f"{'Model':<14s} {'PT CPU':>7s} {'PT MPS':>7s} "
        f"{'CM ALL':>7s} {'CM GPU':>7s} {'CM NE':>7s} {'CM CPU':>7s} {'Speedup':>8s}"
    )
    print("-" * 90)
    for r in results:
        mps = f"{r['pytorch_mps']:.1f}" if "pytorch_mps" in r else "—"
        speedup_val = r.get("pytorch_mps", r["pytorch_cpu"]) / r["coreml_all"]
        print(
            f"{r['model']:<14s} "
            f"{r['pytorch_cpu']:>6.1f} {mps:>7s} "
            f"{r['coreml_all']:>6.1f} {r['coreml_cpu_and_gpu']:>6.1f} "
            f"{r['coreml_cpu_and_ne']:>6.1f} {r['coreml_cpu_only']:>6.1f} "
            f"{speedup_val:>7.1f}x"
        )
    print()
    print("CM ALL = CPU+GPU+NeuralEngine, CM GPU = CPU+GPU, CM NE = CPU+NeuralEngine, CM CPU = CPU only")
    if args.precision == "fp32":
        print("Note: for FP32 models, CM NE ≈ CM CPU because ANE cannot run FP32 ops.")
    print("Use scripts/test_export.py with the same precision for accuracy validation.")


if __name__ == "__main__":
    main()
