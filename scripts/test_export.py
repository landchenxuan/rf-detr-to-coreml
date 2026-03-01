#!/usr/bin/env python3
"""
Export RF-DETR models to CoreML and verify accuracy against PyTorch.

For each selected model:
  1. Export to CoreML FP32
  2. Run inference with identical input on both PyTorch and CoreML
  3. Compare outputs (boxes, logits, and masks for segmentation models)
  4. Report accuracy diff and latency

Usage:
  python scripts/test_export.py                    # Test all models
  python scripts/test_export.py --model nano       # Test single model
  python scripts/test_export.py --model detection  # Test all detection models
  python scripts/test_export.py --model segmentation  # Test all segmentation models
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
from rfdetr_coreml.export import MODEL_REGISTRY, NormalizedWrapper, _import_model_class, export_to_coreml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DETECTION_MODELS = [k for k in MODEL_REGISTRY if not k.startswith("seg-")]
SEGMENTATION_MODELS = [k for k in MODEL_REGISTRY if k.startswith("seg-")]


def build_pytorch_model(model_name):
    """Instantiate RF-DETR model in export mode, wrapped with normalization."""
    resolution = MODEL_REGISTRY[model_name][1]
    model_cls = _import_model_class(model_name)
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped = NormalizedWrapper(pt_model, resolution).eval()
    del rfdetr_model
    return wrapped


def identify_coreml_outputs(result):
    """Match CoreML output arrays to boxes / logits / masks by shape."""
    boxes = logits = masks = None
    for _, v in result.items():
        arr = np.array(v)
        if arr.ndim >= 2 and arr.shape[-1] == 4 and boxes is None:
            boxes = arr
        elif arr.ndim == 4 and masks is None:
            masks = arr
        elif arr.ndim >= 2 and arr.shape[-1] > 4 and logits is None:
            logits = arr
    return boxes, logits, masks


def test_model(model_name, output_dir, skip_export=False):
    """Export one model and compare CoreML vs PyTorch outputs."""
    import coremltools as ct
    from PIL import Image

    resolution = MODEL_REGISTRY[model_name][1]
    is_seg = model_name.startswith("seg-")
    mlpackage_path = os.path.join(output_dir, f"rf-detr-{model_name}-fp32.mlpackage")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing: {model_name} (resolution={resolution})")
    logger.info(f"{'=' * 60}")

    # Export
    if not skip_export or not os.path.exists(mlpackage_path):
        logger.info("Exporting to CoreML FP32...")
        t0 = time.time()
        mlpackage_path = export_to_coreml(model_name, output_dir, "fp32")
        logger.info(f"Export took {time.time() - t0:.1f}s")
    else:
        logger.info(f"Using existing: {mlpackage_path}")

    # Model size
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(mlpackage_path)
        for f in fns
    )
    size_mb = total_size / (1024 * 1024)

    # Create test image (uint8 so both paths get identical input)
    rng = np.random.RandomState(42)
    img_uint8 = rng.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pt_input = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # PyTorch inference
    logger.info("Running PyTorch inference...")
    wrapped_pt = build_pytorch_model(model_name)
    with torch.no_grad():
        pt_out = wrapped_pt(pt_input)
    pt_boxes = pt_out[0].numpy()
    pt_logits = pt_out[1].numpy()
    pt_masks = pt_out[2].numpy() if len(pt_out) > 2 else None
    del wrapped_pt

    # CoreML inference (ALL compute units)
    logger.info("Running CoreML inference (ALL)...")
    ml_model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)
    for _ in range(3):
        ml_model.predict({"image": pil_img})
    n_runs = 20
    t0 = time.time()
    for _ in range(n_runs):
        result = ml_model.predict({"image": pil_img})
    latency_all = (time.time() - t0) / n_runs * 1000

    cm_boxes, cm_logits, cm_masks = identify_coreml_outputs(result)
    del ml_model

    # Compare
    box_diff = np.abs(pt_boxes - cm_boxes).max() if cm_boxes is not None else None
    logit_diff = np.abs(pt_logits - cm_logits).max() if cm_logits is not None else None
    mask_diff = None
    if is_seg and cm_masks is not None and pt_masks is not None:
        mask_diff = np.abs(pt_masks - cm_masks).max()

    logger.info(f"  Size: {size_mb:.1f} MB")
    logger.info(f"  Latency (ALL): {latency_all:.1f} ms")
    if box_diff is not None:
        logger.info(f"  Max box diff: {box_diff:.6f}")
    if logit_diff is not None:
        logger.info(f"  Max logit diff: {logit_diff:.6f}")
    if mask_diff is not None:
        logger.info(f"  Max mask diff: {mask_diff:.6f}")

    return {
        "model": model_name,
        "type": "seg" if is_seg else "det",
        "size_mb": size_mb,
        "latency_all_ms": latency_all,
        "box_diff": box_diff,
        "logit_diff": logit_diff,
        "mask_diff": mask_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Test RF-DETR CoreML export accuracy")
    parser.add_argument(
        "--model", default="all",
        help="Model name, 'all', 'detection', or 'segmentation'",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export if .mlpackage already exists")
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
        parser.error(f"Unknown model: {args.model}. "
                     f"Choose from {list(MODEL_REGISTRY.keys())} or all/detection/segmentation")

    results = []
    for name in models:
        try:
            r = test_model(name, args.output_dir, args.skip_export)
            results.append(r)
        except Exception as e:
            logger.error(f"FAILED: {name} — {e}", exc_info=True)
            results.append({"model": name, "error": str(e)})

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Model':<14s} {'Size':>7s} {'ALL ms':>7s} {'Box Diff':>10s} {'Logit Diff':>11s} {'Mask Diff':>10s}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<14s}  ERROR: {r['error'][:50]}")
        else:
            bd = f"{r['box_diff']:.6f}" if r["box_diff"] is not None else "—"
            ld = f"{r['logit_diff']:.6f}" if r["logit_diff"] is not None else "—"
            md = f"{r['mask_diff']:.6f}" if r["mask_diff"] is not None else "—"
            print(f"{r['model']:<14s} {r['size_mb']:>6.1f}M {r['latency_all_ms']:>6.1f} {bd:>10s} {ld:>11s} {md:>10s}")


if __name__ == "__main__":
    main()
