#!/usr/bin/env python3
"""
Test FP16 precision strategies for RF-DETR CoreML conversion.

Exports three mixed-precision CoreML models and compares each against the
FP32 PyTorch reference to demonstrate why FP16 is unusable:
  1. Full FP16 — all ops in FP16
  2. Conv/linear FP16 only — only conv/linear weights in FP16, rest FP32
  3. Resample+softmax FP32 — everything FP16 except resample and softmax

All use identical input (RandomState(42)) and report max box diff in pixels.

Usage:
  python scripts/test_fp16.py                 # Test Nano (default)
  python scripts/test_fp16.py --model base    # Test Base
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
from rfdetr_coreml.export import MODEL_REGISTRY, NormalizedWrapper, _import_model_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_traced_model(model_name):
    """Build and trace a NormalizedWrapper for the given model variant."""
    resolution = MODEL_REGISTRY[model_name][1]
    model_cls = _import_model_class(model_name)
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped = NormalizedWrapper(pt_model, resolution).eval()
    del rfdetr_model

    dummy = torch.rand(1, 3, resolution, resolution)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy)
    return wrapped, traced


def export_with_precision(traced, resolution, strategy_name, output_path):
    """Convert traced model to CoreML with the given precision strategy."""
    import coremltools as ct

    if strategy_name == "full_fp16":
        compute_precision = ct.precision.FLOAT16
    elif strategy_name == "conv_linear_fp16":
        compute_precision = ct.transform.FP16ComputePrecision(
            op_selector=lambda op: op.op_type in ("conv", "linear")
        )
    elif strategy_name == "resample_softmax_fp32":
        compute_precision = ct.transform.FP16ComputePrecision(
            op_selector=lambda op: op.op_type not in ("resample", "softmax")
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, resolution, resolution), scale=1.0 / 255.0)],
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.save(output_path)
    return output_path


def identify_coreml_boxes(result):
    """Extract boxes array from CoreML prediction result."""
    for _, v in result.items():
        arr = np.array(v)
        if arr.ndim >= 2 and arr.shape[-1] == 4:
            return arr
    return None


def test_model(model_name, output_dir, n_runs=50):
    """Run FP16 precision tests for one model."""
    import coremltools as ct
    from PIL import Image

    resolution = MODEL_REGISTRY[model_name][1]

    logger.info(f"\n{'=' * 70}")
    logger.info(f"FP16 Precision Test: {model_name} (resolution={resolution})")
    logger.info(f"{'=' * 70}")

    # Real test image (same as test_export.py / benchmark_onnx.py)
    test_img_path = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "test_images", "*.jpg")))[0]
    pil_img = Image.open(test_img_path).convert("RGB").resize(
        (resolution, resolution), Image.BILINEAR
    )
    img_np = np.array(pil_img)
    pt_input = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # PyTorch FP32 reference
    logger.info("Computing PyTorch FP32 reference...")
    wrapped, traced = build_traced_model(model_name)
    with torch.no_grad():
        pt_out = wrapped(pt_input)
    ref_boxes = pt_out[0].numpy()
    del wrapped
    gc.collect()

    strategies = [
        ("full_fp16", "Full FP16"),
        ("conv_linear_fp16", "Conv/linear weights FP16 only"),
        ("resample_softmax_fp32", "Resample+softmax keep FP32"),
    ]

    results = []
    for strategy_key, strategy_label in strategies:
        output_path = os.path.join(output_dir, f"rf-detr-{model_name}-{strategy_key}.mlpackage")

        logger.info(f"\n--- {strategy_label} ---")

        # Export
        if not os.path.exists(output_path):
            logger.info("Exporting...")
            t0 = time.time()
            try:
                export_with_precision(traced, resolution, strategy_key, output_path)
                logger.info(f"Exported in {time.time() - t0:.1f}s")
            except Exception as e:
                logger.error(f"Export failed: {e}")
                results.append((strategy_label, None, "export failed"))
                continue
        else:
            logger.info(f"Using existing: {output_path}")

        # Load and predict
        try:
            ml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.ALL)
            result = ml_model.predict({"image": pil_img})
            cm_boxes = identify_coreml_boxes(result)

            if cm_boxes is not None:
                box_diff_px = float(np.abs(ref_boxes - cm_boxes).max()) * resolution
                verdict = "OK" if box_diff_px < 2.0 else "Unusable"
                logger.info(f"  Max box diff: {box_diff_px:.1f} px — {verdict}")
                results.append((strategy_label, box_diff_px, verdict))
            else:
                logger.error("  Could not identify boxes in CoreML output")
                results.append((strategy_label, None, "no boxes"))

            del ml_model
            gc.collect()
        except Exception as e:
            logger.error(f"  Inference failed: {e}")
            results.append((strategy_label, None, str(e)[:50]))

    del traced
    gc.collect()

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"FP16 PRECISION RESULTS — {model_name}")
    print(f"{'=' * 60}")
    print(f"{'Strategy':<36s} {'Max Box Diff':>15s} {'Verdict':>10s}")
    print("-" * 62)
    for label, diff, verdict in results:
        diff_s = f"{diff:.1f} px" if diff is not None else "N/A"
        print(f"{label:<36s} {diff_s:>15s} {verdict:>10s}")
    print()
    print("FP32 reference: 0 (by definition)")
    print("Threshold: < 2 px = OK, >= 2 px = Unusable")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test FP16 precision strategies for RF-DETR CoreML")
    parser.add_argument("--model", default="nano", help="Model name (default: nano)")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        parser.error(f"Unknown model: {args.model}. Choose from {list(MODEL_REGISTRY.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)
    test_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()
