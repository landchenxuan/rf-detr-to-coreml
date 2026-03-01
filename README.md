# RF-DETR to CoreML

Export [RF-DETR](https://github.com/roboflow/rf-detr) v1.5.1 **detection + segmentation** models to Apple CoreML format with GPU/Neural Engine acceleration.

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

## Model Specs

### Detection Models

| Model | Resolution | Patch | Queries | FP32 Size | Latency (ALL) | Latency (CPU) | Speedup |
|-------|-----------|-------|---------|-----------|---------------|---------------|---------|
| Nano | 384 | 16 | 300 | ~103 MB | ~8 ms | ~20 ms | 2.5x |
| Small | 512 | 16 | 300 | ~103 MB | TBD | TBD | TBD |
| Medium | 576 | 16 | 300 | ~103 MB | TBD | TBD | TBD |
| Base | 560 | 14 | 300 | ~103 MB | TBD | TBD | TBD |
| Large | 704 | 16 | 300 | ~103 MB | TBD | TBD | TBD |

### Segmentation Models

| Model | Resolution | Queries | FP32 Size | Latency (ALL) | Latency (CPU) | Speedup | Mask Shape |
|-------|-----------|---------|-----------|---------------|---------------|---------|------------|
| Seg-Nano | 312 | 100 | 116.9 MB | 16 ms | 52 ms | 3.3x | (1, 100, 78, 78) |
| Seg-Small | 384 | 100 | 117.4 MB | 21 ms | 72 ms | 3.4x | (1, 100, 96, 96) |
| Seg-Medium | 432 | 200 | 123.9 MB | 29 ms | 101 ms | 3.5x | (1, 200, 108, 108) |
| Seg-Large | 504 | 200 | 124.6 MB | 38 ms | 149 ms | 3.9x | (1, 200, 126, 126) |
| Seg-XLarge | 624 | 300 | 132.1 MB | 67 ms | 273 ms | 4.1x | (1, 300, 156, 156) |
| Seg-2XLarge | 768 | 300 | 134.2 MB | 127 ms | 516 ms | 4.1x | (1, 300, 192, 192) |

> Segmentation models output 3 tensors: boxes `(1,N,4)` + logits `(1,N,91)` + masks `(1,N,H/4,W/4)`. All use patch_size=12.

> Benchmarked on Apple M-series, coremltools 8.1, torch 2.7.0

## Project Structure

```
rf-dert-to-coreml/
├── rfdetr_coreml/              # Python package (monkey-patch overlay)
│   ├── __init__.py             # Auto-applies all patches on import
│   ├── patches.py              # 3 runtime patches (6D→5D, bicubic→bilinear)
│   ├── coreml_fixes.py         # coremltools bug fixes (_cast, view)
│   ├── export.py               # NormalizedWrapper + export logic
│   └── cli.py                  # CLI entry point (rfdetr-coreml command)
├── export_coreml.py            # Convenience script (calls cli.main())
├── pyproject.toml              # pip install config
├── requirements.txt            # Pinned dependency versions (optional)
├── LICENSE                     # Apache 2.0
└── README.md
```

## How It Works

### Why Can't RF-DETR Be Directly Converted to CoreML?

RF-DETR uses **Deformable Attention**, an efficient attention mechanism that computes attention at learnable sampling points on feature maps. However, this mechanism has several CoreML incompatibilities:

1. **Rank-6 tensors**: In `MSDeformAttn.forward()`, `sampling_offsets` is reshaped to `(N, Len_q, n_heads, n_levels, n_points, 2)` — 6 dimensions. CoreML supports at most rank-5.

2. **Bicubic interpolation**: The DinoV2 backbone's positional encoding uses `F.interpolate(mode="bicubic")`, but CoreML only supports nearest and bilinear.

3. **Dict output**: `forward()` returns a dictionary, which `torch.jit.trace` cannot trace.

4. **coremltools bugs**: The `_cast` function can't handle `shape=(1,)` numpy arrays; the `view` op converter can't handle non-scalar shape Vars.

### Our Solution: Monkey-Patch Overlay

Instead of forking upstream code, we apply 3 runtime monkey-patches:

**Patch A: 6D → 5D (MSDeformAttn.forward)**
- Core idea: merge batch and heads dimensions (`N × n_heads`) to keep all tensors at rank-5 or below
- Original: `(N, Len_q, n_heads, n_levels, n_points, 2)` — 6D
- Patched: `(N*n_heads, Len_q, n_levels, n_points, 2)` — 5D
- Output is reshaped back to `(N, Len_q, C)`

**Patch B: Core Attention Function**
- Companion to Patch A, accepts merged batch+heads 5D inputs
- Uses `F.grid_sample` on merged dimensions internally

**Patch C: Bicubic → Bilinear**
- Replaces `mode="bicubic"` with `mode="bilinear"` in the DinoV2 backbone
- Affects positional encoding interpolation, minimal impact on detection accuracy

**coremltools Fixes**
- `_cast`: call `.item()` on numpy shape-(1,) arrays before passing to `int()`
- `view`: squeeze non-scalar Var elements before concat

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

## Accuracy Comparison

### FP32 CoreML vs PyTorch (Nano model, 300 queries)

Per-query box coordinate difference between CoreML FP32 and PyTorch outputs:

| Metric | Value |
|--------|-------|
| Box diff P50 | 0.04 px |
| Box diff P95 | 1.56 px |
| Box diff max | 3.2 px |
| Queries < 0.4px | 252/300 (84%) |
| Queries > 5px | 0 |

**Conclusion: FP32 conversion is nearly lossless.** Only 2 outlier queries (diff > 2px), both low-confidence false detections that would be filtered by confidence thresholds — no impact on production use.

### FP32 CoreML vs PyTorch (Segmentation models)

| Model | Box diff (max) | Logit diff (max) | Mask diff (max) |
|-------|----------------|------------------|-----------------|
| Seg-Nano | 0.000078 | 0.0003 | 0.0027 |
| Seg-Small | 0.000397 | 0.0005 | 0.0063 |
| Seg-Medium | 0.000259 | 0.0009 | 0.0115 |
| Seg-Large | 0.000235 | 0.0007 | 0.0912 |
| Seg-XLarge | 0.000443 | 0.0013 | 0.0233 |
| Seg-2XLarge | 0.000372 | 0.0032 | 0.0508 |

**All segmentation models achieve near-lossless FP32 conversion.** Box/logit diffs are all < 0.005. Mask diff increases slightly with model size but remains within acceptable range. The segmentation head uses only standard ops (Conv2d, einsum, bilinear interpolate) and introduces no additional precision loss.

### FP16 Precision Issues

| Metric | FP32 CoreML | FP16 CoreML |
|--------|-------------|-------------|
| vs PyTorch box diff (P50) | 0.04 px | **extremely large** |
| vs PyTorch box diff (max) | 3.2 px | **491 px** (catastrophic) |
| Model size | ~103 MB | ~52 MB |

**FP16 is not suitable for production.** Deformable attention sampling coordinates are extremely precision-sensitive — FP16 lacks sufficient precision to represent sampling offsets correctly, causing attention to sample from entirely wrong locations.

We also tested these mixed-precision strategies, none of which solve the problem:

| Strategy | Max box diff | Conclusion |
|----------|-------------|------------|
| Full FP16 | 491 px | Unusable |
| Conv/linear weights only FP16 | 379 px | Unusable |
| Resample+softmax keep FP32 | 387 px | Unusable |

**Root cause**: `F.grid_sample` is extremely sensitive to input coordinate precision. Even quantizing only weights to FP16 causes errors to be amplified catastrophically through the deformable attention sampling process.

**Recommendation: Always use FP32 in production.** 103MB is acceptable for mobile, and FP32 Neural Engine utilization is equally good.

## Inference Performance (Nano FP32)

| Runtime | Latency |
|---------|---------|
| PyTorch CPU | ~41 ms |
| PyTorch MPS (Apple GPU) | ~20 ms |
| **CoreML ALL (GPU + Neural Engine)** | **~8 ms** |
| CoreML CPU_ONLY | ~20 ms |

| Comparison | Speedup |
|------------|---------|
| CoreML ALL vs PyTorch CPU | **5.1x** |
| CoreML ALL vs PyTorch MPS | **2.5x** |
| CoreML ALL vs CoreML CPU | **2.5x** |

The Neural Engine is indeed utilized. `F.grid_sample` (the core of deformable attention) runs on GPU rather than ANE, but overall acceleration is still significant. CoreML's 2.5x speedup over PyTorch MPS indicates the Neural Engine handles a substantial portion of other computations.

## Known Issues and Limitations

### Production Considerations

1. **FP16 is unusable** — deformable attention is precision-sensitive, must use FP32
2. **FP32 models are large** — Nano ~103MB, Large estimated 400MB+
3. **Fixed resolution only** — each model variant has a fixed input resolution, no dynamic input support
4. **coremltools compatibility** — only tested with coremltools 8.1 + torch 2.7.0
5. **Segmentation models are large** — Seg-Nano ~117MB, Seg-2XLarge even larger

### Potential Improvements

- Explore mixed precision (keep critical ops FP32, rest FP16) to reduce model size
- Add CoreML output name mapping (`boxes`, `scores`, `labels`) to match Roboflow format
- Support batch inference
- Optimize `grid_sample` for better ANE utilization

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
