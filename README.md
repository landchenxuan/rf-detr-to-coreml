# RF-DETR to CoreML

Export [RF-DETR](https://github.com/roboflow/rf-detr) v1.5.1 **detection + segmentation** models to Apple CoreML format with GPU acceleration.

## Installation

```bash
pip install rfdetr-coreml
```

Or install from source:

```bash
git clone https://github.com/landchenxuan/rf-dert-to-coreml.git
cd rf-dert-to-coreml
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

## Performance

All benchmarks on **MacBook Pro M4 Pro (24 GB)**, coremltools 8.1, torch 2.7.0, rfdetr 1.5.1. FP32 only (see [why FP16 doesn't work](#fp16-precision-issues)).

### Detection Models

| Model | Size | PyTorch MPS | CoreML (GPU) | Speedup | Max Box Diff |
|-------|------|-------------|--------------|---------|--------------|
| Nano | 103 MB | 21.6 ms | 11.2 ms | **1.9x** | 0.96 |
| Small | 109 MB | 32.1 ms | 17.9 ms | **1.8x** | 0.97 |
| Medium | 115 MB | 41.2 ms | 24.1 ms | **1.7x** | 0.97 |
| Base | 103 MB | 44.0 ms | 25.4 ms | **1.7x** | 0.96 |
| Large | 116 MB | 59.3 ms | 35.4 ms | **1.7x** | 0.95 |

> Max Box Diff: maximum per-coordinate difference (in pixels, normalized to model resolution) between CoreML and PyTorch outputs across all 300 queries. Values < 1.0 indicate sub-pixel accuracy.

### Segmentation Models

| Model | Size | PyTorch MPS | CoreML (GPU) | Speedup | Max Mask Diff |
|-------|------|-------------|--------------|---------|---------------|
| Seg-Nano | 117 MB | 29.4 ms | 16 ms | **1.8x** | 0.0027 |
| Seg-Small | 117 MB | 35.4 ms | 21 ms | **1.7x** | 0.0063 |
| Seg-Medium | 124 MB | 46.7 ms | 29 ms | **1.6x** | 0.0115 |
| Seg-Large | 125 MB | 60.8 ms | 38 ms | **1.6x** | 0.0912 |
| Seg-XLarge | 132 MB | 99.8 ms | 67 ms | **1.5x** | 0.0233 |
| Seg-2XLarge | 134 MB | 169.3 ms | 127 ms | **1.3x** | 0.0508 |

> All segmentation models achieve near-lossless FP32 conversion (box/logit diffs all < 0.005). Outputs: boxes `(1,N,4)` + logits `(1,N,91)` + masks `(1,N,H/4,W/4)`.

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

### FP16 Precision Issues

**FP16 is not usable.** Deformable attention's `F.grid_sample` is extremely sensitive to coordinate precision. FP16 causes sampling from entirely wrong locations, resulting in catastrophic output degradation (491px max box diff vs 0.96px for FP32).

All mixed-precision strategies also fail:

| Strategy | Max Box Diff | Verdict |
|----------|-------------|---------|
| Full FP16 | 491 px | Unusable |
| Conv/linear weights FP16 only | 379 px | Unusable |
| Resample+softmax keep FP32 | 387 px | Unusable |

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
rf-dert-to-coreml/
├── rfdetr_coreml/              # Python package (monkey-patch overlay)
│   ├── __init__.py             # Auto-applies all patches on import
│   ├── patches.py              # 3 runtime patches (6D→5D, bicubic→bilinear)
│   ├── coreml_fixes.py         # coremltools bug fixes (_cast, view)
│   ├── export.py               # NormalizedWrapper + export logic
│   └── cli.py                  # CLI entry point (rfdetr-coreml command)
├── scripts/                    # Test and benchmark scripts
│   ├── test_export.py          # Export + accuracy test (all models)
│   ├── benchmark.py            # Latency benchmark (all compute units)
│   └── validate_coreml.swift   # Native Swift/CoreML validation
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

## License

Apache 2.0
