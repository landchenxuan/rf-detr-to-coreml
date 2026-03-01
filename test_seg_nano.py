#!/usr/bin/env python3
"""Quick test: export SegNano to CoreML FP32 and compare vs PyTorch."""

import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch

# Apply patches before any rfdetr imports
import rfdetr_coreml  # noqa: F401

from rfdetr_coreml.export import MODEL_REGISTRY, export_to_coreml, NormalizedWrapper, _import_model_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    model_name = "seg-nano"
    precision = "fp32"
    output_dir = "output"
    resolution = MODEL_REGISTRY[model_name][1]
    mlpackage_path = os.path.join(output_dir, f"rf-detr-{model_name}-{precision}.mlpackage")

    logger.info(f"Testing {model_name} (resolution={resolution})")

    # Step 1: Export to CoreML (skip if exists)
    if not os.path.exists(mlpackage_path):
        logger.info("Exporting to CoreML...")
        t0 = time.time()
        mlpackage_path = export_to_coreml(model_name, output_dir, precision)
        logger.info(f"Export done in {time.time() - t0:.1f}s: {mlpackage_path}")
    else:
        logger.info(f"Using existing model: {mlpackage_path}")

    # Step 2: Build PyTorch model
    logger.info("Loading PyTorch model...")
    model_cls = _import_model_class(model_name)
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped_pt = NormalizedWrapper(pt_model, resolution).eval()
    del rfdetr_model

    # Step 3: Create test image — start from uint8 PIL so both paths get
    # identical input (avoids uint8 quantization noise dominating the diff)
    from PIL import Image
    rng = np.random.RandomState(42)
    img_uint8 = rng.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_uint8)

    # PyTorch input: same uint8 → float32 as CoreML ImageType(scale=1/255)
    pt_input = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    logger.info("Running PyTorch inference...")
    with torch.no_grad():
        pt_out = wrapped_pt(pt_input)

    logger.info(f"PyTorch outputs: {len(pt_out)} tensors")
    for i, t in enumerate(pt_out):
        logger.info(f"  [{i}] shape={t.shape}")

    pt_boxes = pt_out[0].numpy()
    pt_logits = pt_out[1].numpy()
    pt_masks = pt_out[2].numpy() if len(pt_out) > 2 else None

    # Step 4: Run CoreML inference
    logger.info("Running CoreML inference...")
    import coremltools as ct

    ml_model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)

    # Warmup
    for _ in range(3):
        ml_model.predict({"image": pil_img})

    # Timed runs
    n_runs = 10
    t0 = time.time()
    for _ in range(n_runs):
        result = ml_model.predict({"image": pil_img})
    latency_ms = (time.time() - t0) / n_runs * 1000

    # Identify outputs
    logger.info(f"CoreML outputs: {list(result.keys())}")
    cm_boxes = cm_logits = cm_masks = None
    for k, v in result.items():
        arr = np.array(v)
        logger.info(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.ndim >= 2:
            if arr.shape[-1] == 4 and cm_boxes is None:
                cm_boxes = arr
            elif arr.ndim == 4 and cm_masks is None:
                cm_masks = arr
            elif arr.ndim >= 2 and arr.shape[-1] > 4 and cm_logits is None:
                cm_logits = arr

    # Step 5: Compare
    logger.info("\n=== Comparison (same uint8 input) ===")

    if cm_boxes is not None and pt_boxes is not None:
        box_diff = np.abs(pt_boxes.reshape(-1, 4) - cm_boxes.reshape(-1, 4))
        logger.info(f"Box diff — max: {box_diff.max():.6f}, mean: {box_diff.mean():.6f}")
    else:
        logger.warning("Could not compare boxes")

    if cm_logits is not None and pt_logits is not None:
        logit_diff = np.abs(pt_logits.reshape(-1) - cm_logits.reshape(-1))
        logger.info(f"Logit diff — max: {logit_diff.max():.4f}, mean: {logit_diff.mean():.6f}")
    else:
        logger.warning("Could not compare logits")

    if cm_masks is not None and pt_masks is not None:
        mask_diff = np.abs(pt_masks.reshape(-1) - cm_masks.reshape(-1))
        logger.info(f"Mask diff — max: {mask_diff.max():.4f}, mean: {mask_diff.mean():.6f}")
        logger.info(f"Mask shape: PT={pt_masks.shape}, CM={cm_masks.shape}")
    elif pt_masks is not None:
        logger.warning("CoreML missing mask output!")
    else:
        logger.warning("No mask output from either model")

    logger.info(f"\nLatency (ALL, {n_runs} runs): {latency_ms:.1f} ms")

    # CPU_AND_NE latency
    ml_model_ne = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    for _ in range(3):
        ml_model_ne.predict({"image": pil_img})
    t0 = time.time()
    for _ in range(n_runs):
        ml_model_ne.predict({"image": pil_img})
    latency_ne = (time.time() - t0) / n_runs * 1000
    logger.info(f"Latency (CPU_AND_NE, {n_runs} runs): {latency_ne:.1f} ms")

    # CPU_ONLY latency
    ml_model_cpu = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    for _ in range(3):
        ml_model_cpu.predict({"image": pil_img})
    t0 = time.time()
    for _ in range(n_runs):
        ml_model_cpu.predict({"image": pil_img})
    latency_cpu = (time.time() - t0) / n_runs * 1000
    logger.info(f"Latency (CPU_ONLY, {n_runs} runs): {latency_cpu:.1f} ms")

    # Model size
    total_size = 0
    for dirpath, _, filenames in os.walk(mlpackage_path):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    size_mb = total_size / (1024 * 1024)

    logger.info(f"\n=== Summary: {model_name} {precision.upper()} ===")
    logger.info(f"Model size: {size_mb:.1f} MB")
    logger.info(f"Latency — ALL: {latency_ms:.1f}ms, CPU_AND_NE: {latency_ne:.1f}ms, CPU_ONLY: {latency_cpu:.1f}ms")
    if cm_boxes is not None:
        logger.info(f"Box diff (max): {box_diff.max():.6f}")
    if cm_logits is not None:
        logger.info(f"Logit diff (max): {logit_diff.max():.4f}")
    if cm_masks is not None:
        logger.info(f"Mask diff (max): {mask_diff.max():.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
