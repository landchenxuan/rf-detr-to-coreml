"""Benchmark batch=1 vs batch=2 with GPU utilization monitoring.

Samples GPU Device Utilization % via ioreg during inference.

Usage:
    python scripts/test_batch2_gpu.py
"""

import glob
import logging
import os
import re
import subprocess
import threading
import time

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from rfdetr_coreml.patches import apply_rfdetr_patches
from rfdetr_coreml.coreml_fixes import apply_coremltools_patches
apply_rfdetr_patches()
apply_coremltools_patches()

from rfdetr_coreml.export import MODEL_REGISTRY

MODEL = "small"
RESOLUTION = MODEL_REGISTRY[MODEL][1]
OUTPUT_DIR = "output"


def sample_gpu_utilization() -> int | None:
    """Read GPU Device Utilization % from ioreg (no sudo needed)."""
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            text=True, timeout=1,
        )
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


class GPUMonitor:
    """Background thread that samples GPU utilization at ~50ms intervals."""

    def __init__(self):
        self.samples: list[int] = []
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[int]:
        self._stop.set()
        self._thread.join(timeout=2)
        return self.samples

    def _run(self):
        while not self._stop.is_set():
            v = sample_gpu_utilization()
            if v is not None:
                self.samples.append(v)
            time.sleep(0.05)


def load_all_images():
    test_dir = os.path.join(os.path.dirname(__file__), "test_images")
    img_paths = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
    arrays = []
    for p in img_paths:
        pil = Image.open(p).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.BILINEAR)
        arr = np.array(pil).astype(np.float32) / 255.0
        arrays.append(arr.transpose(2, 0, 1))
    return arrays


def run_benchmark(mlpackage_path: str, images: list[np.ndarray], batch_size: int, n_rounds: int = 10):
    import coremltools as ct
    model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)

    if batch_size == 1:
        inputs = [img[np.newaxis, ...] for img in images]
    else:
        inputs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            if len(batch) < batch_size:
                batch = batch + [batch[-1]] * (batch_size - len(batch))
            inputs.append(np.stack(batch, axis=0))

    # Warmup
    for _ in range(3):
        model.predict({"image": inputs[0]})

    # Timed runs with GPU monitoring
    monitor = GPUMonitor()
    monitor.start()

    times = []
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        for inp in inputs:
            model.predict({"image": inp})
        times.append(time.perf_counter() - t0)

    gpu_samples = monitor.stop()
    del model

    total_ms = np.median(times) * 1000
    per_photo = total_ms / len(images)

    gpu_mean = np.mean(gpu_samples) if gpu_samples else 0
    gpu_max = max(gpu_samples) if gpu_samples else 0
    gpu_p50 = np.median(gpu_samples) if gpu_samples else 0

    return total_ms, per_photo, gpu_mean, gpu_max, gpu_p50, len(gpu_samples)


def main():
    path_b1 = os.path.join(OUTPUT_DIR, f"rf-detr-{MODEL}-fp32-batch1.mlpackage")
    path_b2 = os.path.join(OUTPUT_DIR, f"rf-detr-{MODEL}-fp32-batch2.mlpackage")

    if not os.path.exists(path_b1) or not os.path.exists(path_b2):
        logger.info("Models not found. Run test_batch2.py first to export.")
        return

    images = load_all_images()
    n = len(images)

    logger.info(f"Benchmark: {n} images, {MODEL} {RESOLUTION}x{RESOLUTION}, FP32, ALL")
    logger.info(f"GPU sampled via ioreg every ~50ms\n")

    # Baseline: idle GPU
    idle = sample_gpu_utilization()
    logger.info(f"Idle GPU utilization: {idle}%\n")

    total1, per1, gpu1_mean, gpu1_max, gpu1_p50, n1 = run_benchmark(path_b1, images, batch_size=1)
    logger.info(f"batch=1: {total1:.0f}ms total, {per1:.1f}ms/photo")
    logger.info(f"  GPU: mean={gpu1_mean:.0f}%, median={gpu1_p50:.0f}%, max={gpu1_max}% ({n1} samples)")

    # Small gap to let GPU idle
    time.sleep(1)

    total2, per2, gpu2_mean, gpu2_max, gpu2_p50, n2 = run_benchmark(path_b2, images, batch_size=2)
    logger.info(f"\nbatch=2: {total2:.0f}ms total, {per2:.1f}ms/photo")
    logger.info(f"  GPU: mean={gpu2_mean:.0f}%, median={gpu2_p50:.0f}%, max={gpu2_max}% ({n2} samples)")

    speedup = per1 / per2
    logger.info(f"\nSpeedup: {speedup:.2f}x per photo")


if __name__ == "__main__":
    main()
