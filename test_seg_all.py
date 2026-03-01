#!/usr/bin/env python3
"""Export and benchmark all RF-DETR Seg models to CoreML FP32."""

import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch

import rfdetr_coreml  # noqa: F401
from rfdetr_coreml.export import MODEL_REGISTRY, export_to_coreml, NormalizedWrapper, _import_model_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEG_MODELS = ["seg-nano", "seg-small", "seg-medium", "seg-large", "seg-xlarge", "seg-2xlarge"]


def test_seg_model(model_name):
    import coremltools as ct
    from PIL import Image

    resolution = MODEL_REGISTRY[model_name][1]
    output_dir = "output"
    precision = "fp32"
    mlpackage_path = os.path.join(output_dir, f"rf-detr-{model_name}-{precision}.mlpackage")

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name} (resolution={resolution})")
    logger.info(f"{'='*60}")

    # Export
    if not os.path.exists(mlpackage_path):
        logger.info("Exporting...")
        t0 = time.time()
        mlpackage_path = export_to_coreml(model_name, output_dir, precision)
        logger.info(f"Export done in {time.time() - t0:.1f}s")
    else:
        logger.info(f"Using existing: {mlpackage_path}")

    # Model size
    total_size = 0
    for dirpath, _, filenames in os.walk(mlpackage_path):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    size_mb = total_size / (1024 * 1024)

    # PyTorch model
    logger.info("Loading PyTorch model...")
    model_cls = _import_model_class(model_name)
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped_pt = NormalizedWrapper(pt_model, resolution).eval()
    del rfdetr_model

    # Test image (uint8, same for both paths)
    rng = np.random.RandomState(42)
    img_uint8 = rng.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pt_input = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # PyTorch inference
    with torch.no_grad():
        pt_out = wrapped_pt(pt_input)
    pt_boxes = pt_out[0].numpy()
    pt_logits = pt_out[1].numpy()
    pt_masks = pt_out[2].numpy() if len(pt_out) > 2 else None

    # CoreML ALL
    logger.info("CoreML ALL...")
    ml_model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)
    for _ in range(3):
        ml_model.predict({"image": pil_img})
    n_runs = 10
    t0 = time.time()
    for _ in range(n_runs):
        result = ml_model.predict({"image": pil_img})
    latency_all = (time.time() - t0) / n_runs * 1000

    # CoreML CPU_ONLY
    logger.info("CoreML CPU_ONLY...")
    ml_cpu = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    for _ in range(3):
        ml_cpu.predict({"image": pil_img})
    t0 = time.time()
    for _ in range(n_runs):
        ml_cpu.predict({"image": pil_img})
    latency_cpu = (time.time() - t0) / n_runs * 1000

    # Match CoreML outputs
    cm_boxes = cm_logits = cm_masks = None
    for k, v in result.items():
        arr = np.array(v)
        if arr.ndim >= 2 and arr.shape[-1] == 4 and cm_boxes is None:
            cm_boxes = arr
        elif arr.ndim == 4 and cm_masks is None:
            cm_masks = arr
        elif arr.ndim >= 2 and arr.shape[-1] > 4 and cm_logits is None:
            cm_logits = arr

    # Compute diffs
    box_diff = np.abs(pt_boxes - cm_boxes).max() if cm_boxes is not None else None
    logit_diff = np.abs(pt_logits - cm_logits).max() if cm_logits is not None else None
    mask_diff = np.abs(pt_masks - cm_masks).max() if (cm_masks is not None and pt_masks is not None) else None
    mask_shape = pt_masks.shape if pt_masks is not None else None

    speedup = latency_cpu / latency_all if latency_all > 0 else 0

    result_dict = {
        "model": model_name,
        "resolution": resolution,
        "size_mb": size_mb,
        "latency_all": latency_all,
        "latency_cpu": latency_cpu,
        "speedup": speedup,
        "box_diff": box_diff,
        "logit_diff": logit_diff,
        "mask_diff": mask_diff,
        "mask_shape": mask_shape,
    }

    logger.info(f"  Size: {size_mb:.1f} MB")
    logger.info(f"  ALL: {latency_all:.1f}ms, CPU: {latency_cpu:.1f}ms, Speedup: {speedup:.1f}x")
    logger.info(f"  Box diff: {box_diff:.6f}, Logit diff: {logit_diff:.4f}, Mask diff: {mask_diff:.4f}")
    logger.info(f"  Mask shape: {mask_shape}")

    # Free memory
    del ml_model, ml_cpu, wrapped_pt
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    return result_dict


def main():
    results = []
    for model_name in SEG_MODELS:
        try:
            r = test_seg_model(model_name)
            results.append(r)
        except Exception as e:
            logger.error(f"FAILED: {model_name} — {e}", exc_info=True)
            results.append({"model": model_name, "error": str(e)})

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY: RF-DETR Seg Models — CoreML FP32")
    print(f"{'='*90}")
    print(f"{'Model':<14s} {'Res':>4s} {'Size':>8s} {'ALL ms':>7s} {'CPU ms':>7s} {'Speed':>6s} "
          f"{'BoxΔ':>10s} {'LogitΔ':>8s} {'MaskΔ':>8s} {'Mask shape'}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<14s}  ERROR: {r['error'][:60]}")
        else:
            print(f"{r['model']:<14s} {r['resolution']:>4d} {r['size_mb']:>7.1f}M "
                  f"{r['latency_all']:>6.1f} {r['latency_cpu']:>6.1f} {r['speedup']:>5.1f}x "
                  f"{r['box_diff']:>10.6f} {r['logit_diff']:>8.4f} {r['mask_diff']:>8.4f} "
                  f"{r['mask_shape']}")


if __name__ == "__main__":
    main()
