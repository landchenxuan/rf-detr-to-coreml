#!/usr/bin/env python3
"""
Automated test for RF-DETR CoreML models.

For each model variant × precision:
  1. Export to CoreML
  2. Run inference on a test image
  3. Compare vs PyTorch output (box diff, class agreement)
  4. Benchmark: ALL vs CPU_ONLY compute units
"""

import argparse
import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch

# Apply patches before any rfdetr imports
import rfdetr_coreml  # noqa: F401

from rfdetr_coreml.export import MODEL_REGISTRY, export_to_coreml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_pytorch_predictions(model_name, test_image):
    """Run PyTorch inference and return (boxes, scores, labels)."""
    from rfdetr_coreml.export import _import_model_class, IMAGENET_MEAN, IMAGENET_STD

    model_cls = _import_model_class(model_name)
    resolution = MODEL_REGISTRY[model_name][1]

    rfdetr = model_cls()
    model = deepcopy(rfdetr.model.model).cpu().eval()
    model.export()

    # Preprocess: resize + normalize
    img = torch.nn.functional.interpolate(
        test_image, size=(resolution, resolution), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    img = (img - mean) / std

    with torch.no_grad():
        outputs = model(img)
    # outputs is (pred_boxes, pred_logits) tuple from forward_export
    boxes, logits = outputs[0], outputs[1]
    return boxes.numpy(), logits.numpy()


def get_coreml_predictions(mlpackage_path, test_image, compute_units="ALL"):
    """Run CoreML inference and return (boxes, logits, latency_ms)."""
    import coremltools as ct

    if compute_units == "ALL":
        cu = ct.ComputeUnit.ALL
    else:
        cu = ct.ComputeUnit.CPU_ONLY

    model = ct.models.MLModel(mlpackage_path, compute_units=cu)

    # Convert test image to PIL for CoreML ImageType input
    from PIL import Image
    img_np = (test_image.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    # Warmup
    for _ in range(3):
        model.predict({"image": pil_img})

    # Benchmark
    n_runs = 20
    t0 = time.time()
    for _ in range(n_runs):
        result = model.predict({"image": pil_img})
    latency_ms = (time.time() - t0) / n_runs * 1000

    # Extract outputs — keys depend on traced model output names
    keys = list(result.keys())
    # forward_export returns (outputs_coord, outputs_class)
    # CoreML names them var_XXX or similar — take the two largest outputs
    arrays = [(k, np.array(result[k])) for k in keys]
    arrays.sort(key=lambda x: x[1].size, reverse=True)

    # The boxes output has shape (..., 4) and logits has shape (..., num_classes)
    boxes = None
    logits = None
    for name, arr in arrays:
        if arr.ndim >= 2:
            if arr.shape[-1] == 4 and boxes is None:
                boxes = arr
            elif arr.shape[-1] > 4 and logits is None:
                logits = arr

    return boxes, logits, latency_ms


def compare_outputs(pt_boxes, pt_logits, cm_boxes, cm_logits, model_name):
    """Compare PyTorch vs CoreML outputs."""
    results = {"model": model_name}

    if cm_boxes is not None and pt_boxes is not None:
        # Reshape to match if needed
        pt_b = pt_boxes.reshape(-1, 4)
        cm_b = cm_boxes.reshape(-1, 4)
        min_len = min(len(pt_b), len(cm_b))
        box_diff = np.abs(pt_b[:min_len] - cm_b[:min_len]).max()
        results["max_box_diff"] = float(box_diff)
        logger.info(f"  Max box diff: {box_diff:.4f}")
    else:
        results["max_box_diff"] = None
        logger.warning("  Could not compare boxes")

    if cm_logits is not None and pt_logits is not None:
        pt_l = pt_logits.reshape(-1)
        cm_l = cm_logits.reshape(-1)
        min_len = min(len(pt_l), len(cm_l))
        logit_diff = np.abs(pt_l[:min_len] - cm_l[:min_len]).max()
        results["max_logit_diff"] = float(logit_diff)
        logger.info(f"  Max logit diff: {logit_diff:.4f}")

        # Check top-K class agreement
        pt_top = np.argsort(pt_logits.reshape(pt_logits.shape[-2], -1).max(axis=0))[-10:]
        cm_top = np.argsort(cm_logits.reshape(cm_logits.shape[-2], -1).max(axis=0))[-10:]
        overlap = len(set(pt_top) & set(cm_top))
        results["top10_class_overlap"] = overlap
        logger.info(f"  Top-10 class overlap: {overlap}/10")
    else:
        results["max_logit_diff"] = None
        logger.warning("  Could not compare logits")

    return results


def test_model(model_name, precision, output_dir, skip_export=False):
    """Full test for a single model variant."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {model_name} ({precision})")
    logger.info(f"{'='*60}")

    resolution = MODEL_REGISTRY[model_name][1]
    mlpackage_path = os.path.join(output_dir, f"rf-detr-{model_name}-{precision}.mlpackage")

    # Export
    if not skip_export or not os.path.exists(mlpackage_path):
        logger.info("Exporting...")
        t0 = time.time()
        mlpackage_path = export_to_coreml(model_name, output_dir, precision)
        logger.info(f"Export took {time.time() - t0:.1f}s")
    else:
        logger.info(f"Using existing model: {mlpackage_path}")

    # Model size
    total_size = 0
    for dirpath, _, filenames in os.walk(mlpackage_path):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    size_mb = total_size / (1024 * 1024)
    logger.info(f"Model size: {size_mb:.1f} MB")

    # Test image at the model's native resolution (CoreML expects fixed input size)
    test_image = torch.rand(1, 3, resolution, resolution)

    # PyTorch baseline
    logger.info("Running PyTorch inference...")
    pt_boxes, pt_logits = get_pytorch_predictions(model_name, test_image)

    # CoreML ALL
    logger.info("Running CoreML inference (ALL compute units)...")
    cm_boxes, cm_logits, latency_all = get_coreml_predictions(mlpackage_path, test_image, "ALL")
    logger.info(f"  Latency (ALL): {latency_all:.1f} ms")

    # CoreML CPU_ONLY
    logger.info("Running CoreML inference (CPU_ONLY)...")
    _, _, latency_cpu = get_coreml_predictions(mlpackage_path, test_image, "CPU_ONLY")
    logger.info(f"  Latency (CPU_ONLY): {latency_cpu:.1f} ms")

    # Compare
    logger.info("Comparing outputs...")
    comparison = compare_outputs(pt_boxes, pt_logits, cm_boxes, cm_logits, model_name)

    # Speedup
    speedup = latency_cpu / latency_all if latency_all > 0 else 0
    logger.info(f"  Speedup (CPU/ALL): {speedup:.2f}x")

    return {
        "model": model_name,
        "precision": precision,
        "size_mb": size_mb,
        "latency_all_ms": latency_all,
        "latency_cpu_ms": latency_cpu,
        "speedup": speedup,
        **comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Test RF-DETR CoreML models")
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32", "both"])
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--skip-export", action="store_true", help="Skip export if .mlpackage exists")
    args = parser.parse_args()

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    precisions = ["fp16", "fp32"] if args.precision == "both" else [args.precision]

    all_results = []
    for model_name in models:
        for precision in precisions:
            try:
                result = test_model(model_name, precision, args.output_dir, args.skip_export)
                all_results.append(result)
            except Exception as e:
                logger.error(f"FAILED: {model_name} {precision} — {e}", exc_info=True)
                all_results.append({"model": model_name, "precision": precision, "error": str(e)})

    # Final summary table
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    header = f"{'Model':>8} {'Prec':>4} {'Size':>7} {'ALL ms':>7} {'CPU ms':>7} {'Speed':>5} {'Box Δ':>7} {'Logit Δ':>8}"
    logger.info(header)
    logger.info("-" * 80)
    for r in all_results:
        if "error" in r:
            logger.info(f"  {r['model']:>8} {r['precision']:>4}   ERROR: {r['error'][:50]}")
        else:
            box_d = f"{r['max_box_diff']:.4f}" if r.get('max_box_diff') is not None else "N/A"
            logit_d = f"{r['max_logit_diff']:.4f}" if r.get('max_logit_diff') is not None else "N/A"
            logger.info(
                f"  {r['model']:>8} {r['precision']:>4} {r['size_mb']:>6.1f}M "
                f"{r['latency_all_ms']:>6.1f} {r['latency_cpu_ms']:>6.1f} "
                f"{r['speedup']:>4.1f}x {box_d:>7} {logit_d:>8}"
            )


if __name__ == "__main__":
    main()
