#!/usr/bin/env python3
"""
Benchmark RF-DETR CoreML model latency across compute unit configurations.

Tests ALL (CPU+GPU), CPU_AND_NE, and CPU_ONLY to demonstrate that GPU is the
only accelerator providing speedup (ANE is unused due to FP32 precision).
Also benchmarks PyTorch CPU and MPS for comparison.

Usage:
  python scripts/benchmark.py                  # Benchmark Nano (default)
  python scripts/benchmark.py --model base     # Benchmark Base
  python scripts/benchmark.py --model all      # Benchmark all models
"""

import argparse
import gc
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


def benchmark_model(model_name, output_dir, n_runs=50):
    """Run full benchmark for one model variant."""
    import coremltools as ct
    from PIL import Image

    resolution = MODEL_REGISTRY[model_name][1]
    mlpackage_path = os.path.join(output_dir, f"rf-detr-{model_name}-fp32.mlpackage")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Benchmarking: {model_name} (resolution={resolution}, {n_runs} runs)")
    logger.info(f"{'=' * 60}")

    # Export if needed
    if not os.path.exists(mlpackage_path):
        logger.info("Exporting to CoreML FP32...")
        mlpackage_path = export_to_coreml(model_name, output_dir, "fp32")

    # Test image
    rng = np.random.RandomState(42)
    img_uint8 = rng.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pt_input = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
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

    # CoreML — three compute unit modes
    for label, cu in [
        ("ALL", ct.ComputeUnit.ALL),
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
        help="Model name or 'all' (default: nano)",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs")
    args = parser.parse_args()

    if args.model == "all":
        models = list(MODEL_REGISTRY.keys())
    elif args.model in MODEL_REGISTRY:
        models = [args.model]
    else:
        parser.error(f"Unknown model: {args.model}. Choose from {list(MODEL_REGISTRY.keys())} or 'all'")

    results = []
    for name in models:
        try:
            r = benchmark_model(name, args.output_dir, args.runs)
            results.append(r)
        except Exception as e:
            logger.error(f"FAILED: {name} — {e}", exc_info=True)

    # Summary
    print(f"\n{'=' * 90}")
    print("LATENCY SUMMARY (median ms)")
    print(f"{'=' * 90}")
    print(f"{'Model':<14s} {'PT CPU':>7s} {'PT MPS':>7s} {'CM ALL':>7s} {'CM NE':>7s} {'CM CPU':>7s} {'Speedup':>8s}")
    print("-" * 90)
    for r in results:
        mps = f"{r['pytorch_mps']:.1f}" if "pytorch_mps" in r else "—"
        speedup_val = r.get("pytorch_mps", r["pytorch_cpu"]) / r["coreml_all"]
        print(
            f"{r['model']:<14s} "
            f"{r['pytorch_cpu']:>6.1f} {mps:>7s} "
            f"{r['coreml_all']:>6.1f} {r['coreml_cpu_and_ne']:>6.1f} {r['coreml_cpu_only']:>6.1f} "
            f"{speedup_val:>7.1f}x"
        )
    print()
    print("CM ALL = CPU+GPU, CM NE = CPU+NeuralEngine, CM CPU = CPU only")
    print("Note: CM NE ≈ CM CPU because ANE cannot run FP32 ops (see README).")


if __name__ == "__main__":
    main()
