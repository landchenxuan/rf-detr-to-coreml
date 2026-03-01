#!/usr/bin/env python3
"""
Benchmark RF-DETR Base model: our CoreML (FP32/FP16) vs PyTorch.
Compares accuracy, latency, and resource consumption.

Roboflow Swift Package uses:
  - Base model (560×560), Float16
  - computeUnits = .cpuAndNeuralEngine
  - Output: boxes [1,300,4], scores [1,300], labels [1,300]
"""

import gc
import os
import sys
import time

import numpy as np
import torch

# Apply patches before any rfdetr imports
import rfdetr_coreml  # noqa: F401

from copy import deepcopy


def get_model_size_mb(path):
    """Get total size of mlpackage directory in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


def benchmark_pytorch(model, dummy, n_warmup=5, n_runs=50, device="cpu"):
    """Benchmark PyTorch inference."""
    model = model.to(device)
    x = dummy.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    if device == "mps":
        torch.mps.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return times


def benchmark_coreml(mlmodel, img_dict, n_warmup=5, n_runs=50):
    """Benchmark CoreML inference."""
    # Warmup
    for _ in range(n_warmup):
        _ = mlmodel.predict(img_dict)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = mlmodel.predict(img_dict)
        times.append((time.perf_counter() - t0) * 1000)

    return times


def stats(times):
    """Return timing statistics."""
    arr = np.array(times)
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "p5": np.percentile(arr, 5),
        "p95": np.percentile(arr, 95),
        "min": np.min(arr),
        "max": np.max(arr),
        "std": np.std(arr),
    }


def main():
    import coremltools as ct
    from PIL import Image

    from rfdetr_coreml.export import (
        MODEL_REGISTRY,
        NormalizedWrapper,
        _import_model_class,
        IMAGENET_MEAN,
        IMAGENET_STD,
        export_to_coreml,
    )

    resolution = MODEL_REGISTRY["base"][1]  # 560
    print(f"Resolution: {resolution}x{resolution}")
    print()

    # =========================================================================
    # Step 1: Export Base FP32 and FP16
    # =========================================================================
    output_dir = "output"
    fp32_path = os.path.join(output_dir, "rf-detr-base-fp32.mlpackage")
    fp16_path = os.path.join(output_dir, "rf-detr-base-fp16.mlpackage")

    if not os.path.exists(fp32_path):
        print("Exporting Base FP32...")
        fp32_path = export_to_coreml("base", output_dir, "fp32")
    else:
        print(f"FP32 model exists: {fp32_path}")

    if not os.path.exists(fp16_path):
        print("Exporting Base FP16...")
        fp16_path = export_to_coreml("base", output_dir, "fp16")
    else:
        print(f"FP16 model exists: {fp16_path}")

    # =========================================================================
    # Step 2: Model sizes
    # =========================================================================
    fp32_size = get_model_size_mb(fp32_path)
    fp16_size = get_model_size_mb(fp16_path)
    print()
    print("=" * 60)
    print("MODEL SIZE")
    print("=" * 60)
    print(f"  FP32: {fp32_size:.1f} MB")
    print(f"  FP16: {fp16_size:.1f} MB")
    print(f"  Roboflow (Float16, estimated): ~{fp16_size:.0f} MB")

    # =========================================================================
    # Step 3: Prepare PyTorch reference model
    # =========================================================================
    print()
    print("=" * 60)
    print("PREPARING PYTORCH REFERENCE")
    print("=" * 60)
    model_cls = _import_model_class("base")
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped_pt = NormalizedWrapper(pt_model, resolution)
    wrapped_pt.eval()
    print("  PyTorch model ready")

    # =========================================================================
    # Step 4: Create test image
    # =========================================================================
    np.random.seed(42)
    test_np = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_np)

    # PyTorch input: (1, 3, H, W) float32 [0, 1]
    pt_input = torch.from_numpy(test_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # CoreML input
    coreml_input = {"image": test_pil}

    # =========================================================================
    # Step 5: Accuracy comparison — PyTorch vs CoreML FP32 vs CoreML FP16
    # =========================================================================
    print()
    print("=" * 60)
    print("ACCURACY COMPARISON (300 queries)")
    print("=" * 60)

    # PyTorch reference
    with torch.no_grad():
        pt_out = wrapped_pt(pt_input)
    # forward_export returns (boxes, logits) — identify by shape
    if pt_out[0].shape[-1] == 4:
        pt_boxes = pt_out[0].numpy()
        pt_logits = pt_out[1].numpy()
    else:
        pt_logits = pt_out[0].numpy()
        pt_boxes = pt_out[1].numpy()
    print(f"  PyTorch output shapes: logits={pt_logits.shape}, boxes={pt_boxes.shape}")

    def identify_outputs(pred):
        """Identify boxes and logits from CoreML output dict."""
        outputs = {}
        for k in sorted(pred.keys()):
            v = pred[k]
            if len(v.shape) == 3 and v.shape[-1] == 4:
                outputs["boxes"] = v
                print(f"    {k}: shape={v.shape} → boxes")
            elif len(v.shape) == 3 and v.shape[-1] > 4:
                outputs["logits"] = v
                print(f"    {k}: shape={v.shape} → logits")
            else:
                print(f"    {k}: shape={v.shape} → unknown")
        return outputs

    # CoreML FP32
    fp32_model = ct.models.MLModel(fp32_path, compute_units=ct.ComputeUnit.ALL)
    fp32_pred = fp32_model.predict(coreml_input)
    print("  CoreML FP32 outputs:")
    fp32_outputs = identify_outputs(fp32_pred)
    fp32_boxes = fp32_outputs["boxes"]
    fp32_logits = fp32_outputs["logits"]

    # CoreML FP16
    fp16_model = ct.models.MLModel(fp16_path, compute_units=ct.ComputeUnit.ALL)
    fp16_pred = fp16_model.predict(coreml_input)
    print("  CoreML FP16 outputs:")
    fp16_outputs = identify_outputs(fp16_pred)
    fp16_boxes = fp16_outputs["boxes"]
    fp16_logits = fp16_outputs["logits"]

    # Per-query box diff (in pixels)
    def per_query_box_diff_px(boxes_a, boxes_b, res):
        """Max absolute box coordinate diff per query, in pixels."""
        diff = np.abs(boxes_a - boxes_b) * res  # normalized → pixels
        return np.max(diff, axis=-1).squeeze()  # (300,)

    fp32_diff = per_query_box_diff_px(pt_boxes, fp32_boxes, resolution)
    fp16_diff = per_query_box_diff_px(pt_boxes, fp16_boxes, resolution)

    # Logit diff
    fp32_logit_diff = np.abs(pt_logits - fp32_logits).max()
    fp16_logit_diff = np.abs(pt_logits - fp16_logits).max()

    print()
    print("  FP32 CoreML vs PyTorch (box diff in pixels):")
    print(f"    P50:  {np.median(fp32_diff):.4f} px")
    print(f"    P95:  {np.percentile(fp32_diff, 95):.4f} px")
    print(f"    Max:  {np.max(fp32_diff):.4f} px")
    print(f"    < 0.4px: {np.sum(fp32_diff < 0.4)}/300")
    print(f"    > 5px:   {np.sum(fp32_diff > 5)}/300")
    print(f"    Max logit diff: {fp32_logit_diff:.6f}")

    print()
    print("  FP16 CoreML vs PyTorch (box diff in pixels):")
    print(f"    P50:  {np.median(fp16_diff):.4f} px")
    print(f"    P95:  {np.percentile(fp16_diff, 95):.4f} px")
    print(f"    Max:  {np.max(fp16_diff):.4f} px")
    print(f"    < 0.4px: {np.sum(fp16_diff < 0.4)}/300")
    print(f"    > 5px:   {np.sum(fp16_diff > 5)}/300")
    print(f"    Max logit diff: {fp16_logit_diff:.6f}")

    # Top-K detection agreement
    pt_scores = pt_logits.squeeze().max(axis=-1)
    fp32_scores = fp32_logits.squeeze().max(axis=-1)
    fp16_scores = fp16_logits.squeeze().max(axis=-1)

    for topk in [10, 50]:
        pt_topk = set(np.argsort(-pt_scores)[:topk])
        fp32_topk = set(np.argsort(-fp32_scores)[:topk])
        fp16_topk = set(np.argsort(-fp16_scores)[:topk])
        print(f"\n  Top-{topk} query overlap:")
        print(f"    FP32 vs PyTorch: {len(pt_topk & fp32_topk)}/{topk}")
        print(f"    FP16 vs PyTorch: {len(pt_topk & fp16_topk)}/{topk}")

    # =========================================================================
    # Step 6: Latency benchmarks
    # =========================================================================
    print()
    print("=" * 60)
    print("LATENCY BENCHMARK (50 runs each)")
    print("=" * 60)

    n_runs = 50

    # PyTorch CPU
    print("\n  Benchmarking PyTorch CPU...")
    pt_cpu_times = benchmark_pytorch(wrapped_pt, pt_input, n_runs=n_runs, device="cpu")
    pt_cpu = stats(pt_cpu_times)
    print(f"    Median: {pt_cpu['median']:.1f} ms, P5-P95: [{pt_cpu['p5']:.1f}, {pt_cpu['p95']:.1f}]")

    # PyTorch MPS
    if torch.backends.mps.is_available():
        print("\n  Benchmarking PyTorch MPS...")
        pt_mps_times = benchmark_pytorch(wrapped_pt, pt_input, n_runs=n_runs, device="mps")
        pt_mps = stats(pt_mps_times)
        print(f"    Median: {pt_mps['median']:.1f} ms, P5-P95: [{pt_mps['p5']:.1f}, {pt_mps['p95']:.1f}]")
    else:
        pt_mps = None
        print("\n  PyTorch MPS not available")

    # Free PyTorch model memory
    del wrapped_pt, pt_model, rfdetr_model
    gc.collect()

    # CoreML ALL
    print("\n  Benchmarking CoreML FP32 ALL...")
    fp32_all_times = benchmark_coreml(fp32_model, coreml_input, n_runs=n_runs)
    fp32_all = stats(fp32_all_times)
    print(f"    Median: {fp32_all['median']:.1f} ms, P5-P95: [{fp32_all['p5']:.1f}, {fp32_all['p95']:.1f}]")

    # CoreML CPU_AND_NEURAL_ENGINE (Roboflow's config)
    print("\n  Benchmarking CoreML FP32 CPU_AND_NE (Roboflow config)...")
    fp32_cpune = ct.models.MLModel(fp32_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    fp32_cpune_times = benchmark_coreml(fp32_cpune, coreml_input, n_runs=n_runs)
    fp32_cpune_s = stats(fp32_cpune_times)
    print(f"    Median: {fp32_cpune_s['median']:.1f} ms, P5-P95: [{fp32_cpune_s['p5']:.1f}, {fp32_cpune_s['p95']:.1f}]")
    del fp32_cpune

    # CoreML CPU_ONLY
    print("\n  Benchmarking CoreML FP32 CPU_ONLY...")
    fp32_cpu = ct.models.MLModel(fp32_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    fp32_cpu_times = benchmark_coreml(fp32_cpu, coreml_input, n_runs=n_runs)
    fp32_cpu_s = stats(fp32_cpu_times)
    print(f"    Median: {fp32_cpu_s['median']:.1f} ms, P5-P95: [{fp32_cpu_s['p5']:.1f}, {fp32_cpu_s['p95']:.1f}]")
    del fp32_cpu

    # CoreML FP16 ALL
    print("\n  Benchmarking CoreML FP16 ALL...")
    fp16_all_times = benchmark_coreml(fp16_model, coreml_input, n_runs=n_runs)
    fp16_all = stats(fp16_all_times)
    print(f"    Median: {fp16_all['median']:.1f} ms, P5-P95: [{fp16_all['p5']:.1f}, {fp16_all['p95']:.1f}]")

    # CoreML FP16 CPU_AND_NE
    print("\n  Benchmarking CoreML FP16 CPU_AND_NE (Roboflow config)...")
    fp16_cpune = ct.models.MLModel(fp16_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    fp16_cpune_times = benchmark_coreml(fp16_cpune, coreml_input, n_runs=n_runs)
    fp16_cpune_s = stats(fp16_cpune_times)
    print(f"    Median: {fp16_cpune_s['median']:.1f} ms, P5-P95: [{fp16_cpune_s['p5']:.1f}, {fp16_cpune_s['p95']:.1f}]")
    del fp16_cpune

    # =========================================================================
    # Step 7: Memory footprint (rough estimate from model loading)
    # =========================================================================
    print()
    print("=" * 60)
    print("RESOURCE SUMMARY")
    print("=" * 60)

    print(f"""
  {'Metric':<35s} {'Our FP32':<15s} {'Our FP16':<15s} {'Roboflow*'}
  {'—' * 35} {'—' * 15} {'—' * 15} {'—' * 15}
  {'Model size (disk)':<35s} {fp32_size:<15.1f} {fp16_size:<15.1f} ~{fp16_size:.0f} MB (FP16)
  {'Precision':<35s} {'FP32':<15s} {'FP16':<15s} Float16
  {'Input resolution':<35s} {'560×560':<15s} {'560×560':<15s} 560×560
  {'Compute units':<35s} {'ALL':<15s} {'ALL':<15s} cpuAndNE

  LATENCY (median, ms):
  {'PyTorch CPU':<35s} {pt_cpu['median']:<15.1f} {'—':<15s} —""")

    if pt_mps:
        print(f"  {'PyTorch MPS':<35s} {pt_mps['median']:<15.1f} {'—':<15s} —")

    print(f"""  {'CoreML ALL':<35s} {fp32_all['median']:<15.1f} {fp16_all['median']:<15.1f} ~{fp16_all['median']:.0f} ms (est.)
  {'CoreML CPU_AND_NE':<35s} {fp32_cpune_s['median']:<15.1f} {fp16_cpune_s['median']:<15.1f} ~{fp16_cpune_s['median']:.0f} ms (est.)
  {'CoreML CPU_ONLY':<35s} {fp32_cpu_s['median']:<15.1f} {'—':<15s} —

  ACCURACY vs PyTorch (box diff px):
  {'P50':<35s} {np.median(fp32_diff):<15.4f} {np.median(fp16_diff):<15.4f} ~{np.median(fp16_diff):.1f} px (est.)
  {'P95':<35s} {np.percentile(fp32_diff, 95):<15.4f} {np.percentile(fp16_diff, 95):<15.4f} ~{np.percentile(fp16_diff, 95):.1f} px (est.)
  {'Max':<35s} {np.max(fp32_diff):<15.4f} {np.max(fp16_diff):<15.4f} ~{np.max(fp16_diff):.0f} px (est.)
  {'Queries > 5px':<35s} {np.sum(fp32_diff > 5):<15d} {np.sum(fp16_diff > 5):<15d} ~{np.sum(fp16_diff > 5)} (est.)

  * Roboflow 数据为估算值，基于相同精度(FP16)和分辨率。
    Roboflow 可能使用不同的转换管线，实际表现可能不同。
""")


if __name__ == "__main__":
    main()
