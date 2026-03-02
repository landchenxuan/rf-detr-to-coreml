# RF-DETR to CoreML

Export [RF-DETR](https://github.com/roboflow/rf-detr) v1.5.1 **detection + segmentation** models to Apple CoreML format with GPU acceleration.

## Installation

```bash
git clone https://github.com/landchenxuan/rf-detr-to-coreml.git
cd rf-detr-to-coreml
pip install -e .
```

## Quick Start

```bash
# Export pre-trained Nano detection model (FP32, recommended)
rfdetr-coreml --model nano

# Export Seg-Nano segmentation model
rfdetr-coreml --model seg-nano

# Export a fine-tuned model
rfdetr-coreml --model nano --weights path/to/finetuned.pth

# Export with batch size 2 (for batched inference)
rfdetr-coreml --model nano --batch-size 2

# Export all pre-trained models (detection + segmentation)
rfdetr-coreml --model all --output-dir output
```

Or run the script directly:

```bash
python export_coreml.py --model nano
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `nano` | Model variant: detection `nano/small/medium/base/large`, segmentation `seg-nano/seg-small/seg-medium/seg-large/seg-xlarge/seg-2xlarge`, or `all` |
| `--precision` | `fp32` | Compute precision: `fp32` (recommended) or `fp16` (has precision issues) |
| `--output-dir` | `output` | Output directory |
| `--weights` | None | Path to custom weights (fine-tuned model). Uses COCO pre-trained weights if not specified |
| `--batch-size` | `1` | Batch size for exported model. `1` uses ImageType input; `>1` uses TensorType (float32 NCHW `[0,1]`) |

## Performance

All benchmarks on **MacBook Pro M4 Pro (24 GB)**, coremltools 8.1, torch 2.7.0, rfdetr 1.5.1. FP32 only (see [why FP16 doesn't work](#fp16-precision-issues)). Accuracy tested on 17 real images (reproducible via `python scripts/test_export.py`).

### Detection Models

| Model | Size | PyTorch MPS | CoreML (GPU) | Speedup | Max Box Diff |
|-------|------|-------------|--------------|---------|--------------|
| Nano | 103 MB | 21.6 ms | 11.2 ms | **1.9x** | < 0.01 px |
| Small | 109 MB | 32.1 ms | 18.0 ms | **1.8x** | < 0.01 px |
| Medium | 115 MB | 41.2 ms | 23.7 ms | **1.7x** | < 0.01 px |
| Base | 103 MB | 44.0 ms | 24.8 ms | **1.7x** | < 0.01 px |
| Large | 116 MB | 59.3 ms | 34.9 ms | **1.7x** | < 0.01 px |

> Max Box Diff: maximum per-coordinate difference (in pixels) between CoreML and PyTorch outputs, measured only among confident detections (logit > 0) across 17 test images.

### Segmentation Models

| Model | Size | PyTorch MPS | CoreML (GPU) | Speedup | Max Box Diff | Max Mask Diff |
|-------|------|-------------|--------------|---------|--------------|---------------|
| Seg-Nano | 117 MB | 29.4 ms | 16 ms | **1.8x** | < 0.01 px | 0.0003 |
| Seg-Small | 117 MB | 35.4 ms | 21 ms | **1.7x** | < 0.01 px | 0.0001 |
| Seg-Medium | 124 MB | 46.7 ms | 29 ms | **1.6x** | < 0.01 px | 0.0006 |
| Seg-Large | 125 MB | 60.8 ms | 38 ms | **1.6x** | < 0.01 px | 0.0005 |
| Seg-XLarge | 132 MB | 99.8 ms | 67 ms | **1.5x** | 0.02 px | 0.0110 |
| Seg-2XLarge | 134 MB | 169.3 ms | 128 ms | **1.3x** | 0.56 px | 0.1666 |

> Accuracy measured among confident detections across 17 test images. Max Mask Diff: maximum per-pixel difference in mask logits. Outputs: boxes `(1,N,4)` + logits `(1,N,91)` + masks `(1,N,H/4,W/4)`.

### Hardware Acceleration Analysis

We used Apple's `MLComputePlan` API (macOS 14.4+) to inspect per-op device assignment. Results for Nano (representative of all variants):

| | CPU+GPU capable | Neural Engine capable | No device (const/reshape) |
|-|-----------------|----------------------|---------------------------|
| **Ops** | 605 (100%) | 0 (0%) | 893 |

**Zero ops are Neural Engine capable** — not even standard ops like `conv`, `linear`, `matmul`, `softmax` that ANE normally supports. The root cause is **FP32 precision**, not any individual op:

```
grid_sample is extremely sensitive to FP16 precision loss
  → entire model must be exported as FP32
    → ANE hardware only operates in FP16
      → all ops lose ANE eligibility, fall back to CPU
        → only .all mode (adding GPU) provides acceleration
```

Standard transformers (ViT, BERT) don't have this problem — they can safely export to FP16 and run on ANE. RF-DETR's deformable attention uses `grid_sample` which amplifies FP16 errors catastrophically, forcing the entire model to FP32 and locking out ANE entirely.

This is confirmed by timing:

| Compute Units | Nano | Large | Explanation |
|---------------|------|-------|-------------|
| CPU_ONLY | 33.6 ms | 124.1 ms | CPU baseline |
| CPU_AND_NE | 32.9 ms | 124.8 ms | Same as CPU (ANE has nothing to run) |
| **ALL (CPU+GPU)** | **11.1 ms** | **34.5 ms** | **3x faster (GPU active)** |

Use `computeUnits = .all` (or `.cpuAndGPU`) in your app. Setting `.cpuAndNeuralEngine` provides no benefit for RF-DETR models.

### Batch Size

The CLI supports exporting with arbitrary batch sizes (`--batch-size N`), but **batch=1 is optimal** on Apple Silicon. Benchmark on M4 Pro with 17 real images (Small, 512x512, FP32, reproducible via `python scripts/test_batch2_gpu.py`):

| Batch Size | ms/photo | GPU Utilization | Speedup |
|------------|----------|-----------------|---------|
| 1 | 20.5 ms | mean 83% | baseline |
| 2 | 21.0 ms | mean 84% | 0.97x |

GPU is already near-saturated at batch=1. Larger batches cannot improve throughput — the GPU simply has no idle cycles to fill. This is expected for transformer models with ~50 GFLOPs on M4 Pro's GPU (achieving ~13-19% compute efficiency, normal for non-GEMM-dominated workloads like deformable attention).

> **Note:** batch>1 models use `TensorType` input (float32 NCHW in `[0,1]` range) instead of `ImageType`. If you have a use case that benefits from batching on other hardware, the option is available.

### Why Not ONNX → CoreML?

A natural question: RF-DETR has [built-in ONNX export](https://github.com/roboflow/rf-detr/blob/main/src/rfdetr/deploy/export.py) that works with **zero patches** (ONNX supports rank-6 tensors and bicubic natively). Could we skip all the monkey-patching and go PyTorch → ONNX → CoreML instead?

We tested this with ONNX Runtime's CoreML Execution Provider (Nano, M4 Pro). The ONNX model uses raw (unpatched) rfdetr — reproducible via `python scripts/benchmark_onnx.py`:

| Method | Latency | Max Box Diff | Graph Partitions |
|--------|---------|--------------|------------------|
| ONNX Runtime CPU | 58.9 ms | 0.05 px | — |
| ONNX Runtime CoreML EP (default) | 63.4 ms | **370 px** | 97 partitions (665/825 nodes) |
| ONNX Runtime CoreML EP (MLProgram FP32) | 17.9 ms | 0.04 px | 13 partitions (781/825 nodes) |
| **Direct CoreML (this project)** | **11.5 ms** | **0.05 px** | **1 partition (all nodes)** |

> Max Box Diff: maximum per-coordinate difference in pixels vs PyTorch reference. The 370 px diff for default CoreML EP is due to silent FP16 conversion (see [FP16 Precision Issues](#fp16-precision-issues)).

Key findings:

1. **Default CoreML EP silently uses FP16** (NeuralNetwork format) and splits the graph into 97 partitions — only 665 of 825 nodes run on CoreML, the rest fall back to CPU
2. **MLProgram FP32 works** but still splits into 13 partitions — ops CoreML can't handle (rank-6 tensors, bicubic) fall back to CPU with data transfer overhead
3. **Our direct conversion is 1.6x faster** because the monkey-patches let the entire model run as a single CoreML graph on GPU — no partitioning, no CPU fallback
4. **coremltools dropped ONNX support** in v6 — there's no official ONNX → `.mlpackage` converter anymore, only ONNX Runtime's EP route remains

The monkey-patch approach isn't just a workaround — it produces a fundamentally better result than letting the runtime handle incompatibilities through graph partitioning.

### FP16 Precision Issues

**FP16 is not usable.** Deformable attention's `F.grid_sample` is extremely sensitive to coordinate precision. FP16 causes sampling from entirely wrong locations, resulting in catastrophic output degradation (hundreds of pixels off vs < 0.01 px for FP32).

All mixed-precision strategies also fail (reproducible via `python scripts/test_fp16.py`):

| Strategy | Max Box Diff | Verdict |
|----------|-------------|---------|
| Full FP16 | 513 px | Unusable |
| Conv/linear weights FP16 only | 427 px | Unusable |
| Resample+softmax keep FP32 | 451 px | Unusable |

**Always use FP32 in production.** The ~103 MB model size is acceptable for mobile, and FP32 GPU acceleration works well (1.7-1.9x vs PyTorch MPS).

## How It Works

### Why Can't RF-DETR Be Directly Converted to CoreML?

RF-DETR uses **Deformable Attention** which has several CoreML incompatibilities:

1. **Rank-6 tensors**: `MSDeformAttn.forward()` reshapes `sampling_offsets` to 6 dimensions. CoreML supports at most rank-5.
2. **Bicubic interpolation**: DinoV2 backbone uses `F.interpolate(mode="bicubic")`, but CoreML only supports nearest and bilinear.
3. **Dict output**: `forward()` returns a dict, which `torch.jit.trace` cannot trace.
4. **coremltools bugs**: `_cast` can't handle shape-(1,) numpy arrays; `view` can't handle non-scalar shape Vars.

### Solution: Monkey-Patch Overlay

Instead of forking upstream code, we apply runtime monkey-patches:

**Patch A: 6D → 5D (MSDeformAttn.forward)** — merge batch and heads dimensions (`N × n_heads`) to keep all tensors at rank-5 or below.

**Patch B: Core Attention Function** — companion to Patch A, accepts merged batch+heads 5D inputs, uses `F.grid_sample` on merged dimensions.

**Patch C: Bicubic → Bilinear** — replaces `mode="bicubic"` with `mode="bilinear"` in DinoV2 backbone. Minimal impact on accuracy.

**coremltools Fixes** — `_cast`: `.item()` on numpy shape-(1,) arrays; `view`: squeeze non-scalar Vars before concat.

### Export Pipeline

```python
import rfdetr_coreml  # Auto-applies all patches
from rfdetr_coreml.export import export_to_coreml

path = export_to_coreml("nano", output_dir="output", precision="fp32")
```

Internal steps:
1. Instantiate RF-DETR model (auto-downloads weights)
2. `deepcopy` + `.eval()` + `.export()` (switches to `forward_export` tuple output)
3. Wrap with `NormalizedWrapper` (bakes ImageNet normalization into model graph)
4. `torch.jit.trace` to generate TorchScript
5. `coremltools.convert` to mlprogram
6. Save as `.mlpackage`

## Project Structure

```
rf-detr-to-coreml/
├── rfdetr_coreml/              # Python package (monkey-patch overlay)
│   ├── __init__.py             # Auto-applies all patches on import
│   ├── patches.py              # 3 runtime patches (6D→5D, bicubic→bilinear)
│   ├── coreml_fixes.py         # coremltools bug fixes (_cast, view)
│   ├── export.py               # NormalizedWrapper + export logic
│   └── cli.py                  # CLI entry point (rfdetr-coreml command)
├── scripts/                    # Test and benchmark scripts
│   ├── test_images/            # 17 real test images (EXIF stripped)
│   ├── test_export.py          # Export + accuracy test (all models, 17 images)
│   ├── benchmark_latency.py    # Latency benchmark (all compute units)
│   ├── benchmark_onnx.py       # ONNX Runtime vs Direct CoreML comparison
│   ├── _export_onnx_raw.py     # Unpatched ONNX export (called by benchmark_onnx.py)
│   ├── test_fp16.py            # FP16 precision strategy tests
│   ├── test_batch2.py          # Batch=2 export + throughput benchmark
│   ├── test_batch2_gpu.py      # Batch=1 vs batch=2 with GPU utilization monitoring
│   └── validate_coreml.swift   # Native Swift/CoreML + MLComputePlan analysis
├── export_coreml.py            # Convenience script (calls cli.main())
├── pyproject.toml              # pip install config
├── requirements.txt            # Pinned dependency versions (optional)
├── LICENSE                     # Apache 2.0
└── README.md
```

## Known Limitations

1. **FP16 is unusable** — deformable attention is precision-sensitive, must use FP32
2. **FP32 models are large** — Nano ~103 MB, Large ~116 MB, Seg-2XLarge ~134 MB
3. **Fixed resolution only** — each model variant has a fixed input resolution
4. **coremltools compatibility** — tested with coremltools 8.1 + torch 2.7.0

## Dependencies

```
Python >=3.10
torch >=2.4.0
coremltools >=8.0
rfdetr >=1.5.0
```

Tested with: Python 3.12, torch 2.7.0, coremltools 8.1, rfdetr 1.5.1

## Acknowledgments

- [timnielen/rf-detr](https://github.com/timnielen/rf-detr) — first demonstrated that RF-DETR can be converted to CoreML by refactoring deformable attention to stay within CoreML's rank-5 tensor limit. Our implementation takes a different approach (runtime monkey-patches instead of forking), but the core insight came from studying this work. See also [rf-detr#318](https://github.com/roboflow/rf-detr/issues/318).
- [Roboflow](https://github.com/roboflow/rf-detr) — for open-sourcing RF-DETR.

## License

Apache 2.0
