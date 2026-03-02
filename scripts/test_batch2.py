"""Test batch=2 CoreML export and benchmark vs batch=1.

Usage:
    python scripts/test_batch2.py
"""

import glob
import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Apply patches first
from rfdetr_coreml.patches import apply_rfdetr_patches
from rfdetr_coreml.coreml_fixes import apply_coremltools_patches as apply_ct_patches
apply_rfdetr_patches()
apply_ct_patches()

from rfdetr_coreml.export import MODEL_REGISTRY, NormalizedWrapper, _import_model_class

MODEL = "small"
RESOLUTION = MODEL_REGISTRY[MODEL][1]
OUTPUT_DIR = "output"


def export_batch(batch_size: int) -> str:
    """Export model with given batch size, return mlpackage path."""
    import coremltools as ct

    filename = f"rf-detr-{MODEL}-fp32-batch{batch_size}.mlpackage"
    output_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(output_path):
        logger.info(f"Already exists: {output_path}")
        return output_path

    logger.info(f"Exporting batch={batch_size}...")
    model_cls = _import_model_class(MODEL)
    rfdetr_model = model_cls()
    model = deepcopy(rfdetr_model.model.model).cpu().eval()
    model.export()
    wrapped = NormalizedWrapper(model, RESOLUTION)
    wrapped.eval()

    dummy = torch.rand(batch_size, 3, RESOLUTION, RESOLUTION)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="image",
            shape=(batch_size, 3, RESOLUTION, RESOLUTION),
            dtype=np.float32,
        )],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS16,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mlmodel.save(output_path)
    logger.info(f"Saved: {output_path}")
    return output_path


def load_all_images():
    """Load all test images as NCHW float32 [0,1] arrays."""
    test_dir = os.path.join(os.path.dirname(__file__), "test_images")
    img_paths = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
    arrays = []
    for p in img_paths:
        pil = Image.open(p).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.BILINEAR)
        arr = np.array(pil).astype(np.float32) / 255.0
        arrays.append(arr.transpose(2, 0, 1))  # (3, H, W)
    return arrays


def benchmark_sequential(mlpackage_path: str, images: list[np.ndarray], n_rounds: int = 5):
    """Run all images sequentially with batch=1 model, return total time."""
    import coremltools as ct
    model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)

    # Warmup
    inp = images[0][np.newaxis, ...]
    for _ in range(3):
        model.predict({"image": inp})

    times = []
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        for img in images:
            model.predict({"image": img[np.newaxis, ...]})
        times.append(time.perf_counter() - t0)

    total_ms = np.median(times) * 1000
    per_photo = total_ms / len(images)
    return total_ms, per_photo


def benchmark_batch2(mlpackage_path: str, images: list[np.ndarray], n_rounds: int = 5):
    """Run all images in pairs with batch=2 model, return total time."""
    import coremltools as ct
    model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)

    # Build pairs (pad last one if odd)
    pairs = []
    for i in range(0, len(images), 2):
        if i + 1 < len(images):
            pairs.append(np.stack([images[i], images[i + 1]], axis=0))
        else:
            pairs.append(np.stack([images[i], images[i]], axis=0))  # dup last

    # Warmup
    for _ in range(3):
        model.predict({"image": pairs[0]})

    times = []
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        for pair in pairs:
            model.predict({"image": pair})
        times.append(time.perf_counter() - t0)

    total_ms = np.median(times) * 1000
    per_photo = total_ms / len(images)
    return total_ms, per_photo


def main():
    path_b1 = export_batch(1)
    path_b2 = export_batch(2)

    images = load_all_images()
    n = len(images)
    logger.info(f"\nBenchmark: {n} images sequentially ({MODEL}, {RESOLUTION}x{RESOLUTION}, FP32, ALL)")

    total_b1, per_b1 = benchmark_sequential(path_b1, images)
    logger.info(f"  batch=1: {total_b1:.0f}ms total, {per_b1:.1f}ms/photo")

    total_b2, per_b2 = benchmark_batch2(path_b2, images)
    logger.info(f"  batch=2: {total_b2:.0f}ms total, {per_b2:.1f}ms/photo")

    speedup = per_b1 / per_b2
    logger.info(f"\n  Speedup: {speedup:.2f}x per photo")
    if speedup > 1.1:
        logger.info("  → Batch=2 is faster!")
    elif speedup > 0.9:
        logger.info("  → No significant difference")
    else:
        logger.info("  → Batch=2 is slower")


if __name__ == "__main__":
    main()
