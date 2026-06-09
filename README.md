# RF-DETR to CoreML

Export [RF-DETR](https://github.com/roboflow/rf-detr) detection and
segmentation models to Apple Core ML.

This project converts RF-DETR directly from PyTorch to Core ML's ML Program format
and applies a small runtime patch overlay for RF-DETR/coremltools conversion
gaps. The intended production path is FP32 Core ML running on Apple GPU.

## Install

```bash
git clone https://github.com/landchenxuan/rf-detr-to-coreml.git
cd rf-detr-to-coreml
pip install -e .
```

Python 3.10 or newer is required. The package depends on `torch`,
`coremltools`, and `rfdetr`.

For a known baseline environment:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
# Detection, pre-trained COCO weights
rfdetr-coreml --model nano

# Segmentation
rfdetr-coreml --model seg-nano

# Fine-tuned weights
rfdetr-coreml --model nano --weights path/to/finetuned.pth

# Export all supported variants
rfdetr-coreml --model all --output-dir output
```

You can also run the script entrypoint:

```bash
python export_coreml.py --model nano
```

## Verify

Compare a generated Core ML model against RF-DETR PyTorch on real images:

```bash
python scripts/test_export.py --model nano --output-dir output --skip-export --torch-device mps --max-box-diff-px 0.05 --max-logit-diff 0.001
python scripts/test_export.py --model seg-nano --output-dir output --skip-export --torch-device mps --max-box-diff-px 0.05 --max-logit-diff 0.001 --max-mask-diff 0.001
```

Use `--torch-device auto` to use MPS when available and CPU otherwise.

Pull requests also run a fast no-download smoke check for dependency imports,
CLI wiring, and supported model registry resolution. Full Core ML vs PyTorch
checks use the commands above and require generated model artifacts.

## CLI

| Option | Default | Notes |
| --- | --- | --- |
| `--model` | `nano` | Detection: `nano`, `small`, `medium`, `large`; segmentation: `seg-nano`, `seg-small`, `seg-medium`, `seg-large`, `seg-xlarge`, `seg-2xlarge`; or `all` |
| `--precision` | `fp32` | Use `fp32` for production. `fp16` is available but not reliable for RF-DETR. |
| `--output-dir` | `output` | Output directory for `.mlpackage` files |
| `--weights` | None | Path to fine-tuned `.pth` weights |
| `--batch-size` | `1` | `1` uses Core ML `ImageType`; larger batches use `TensorType` NCHW float32 input in `[0, 1]` |

## Supported Models

Detection:

```text
nano, small, medium, large
```

Segmentation:

```text
seg-nano, seg-small, seg-medium, seg-large, seg-xlarge, seg-2xlarge
```

The package targets released `coremltools` 9.0 and `rfdetr` 1.7.1. Deprecated
RF-DETR 1.7 variants such as `base` and `seg-preview` are not supported.

## Production Notes

- Use `precision=fp32`. RF-DETR deformable attention is sensitive to FP16
  coordinate precision; FP16 exports can produce boxes hundreds of pixels off.
- Use Core ML `computeUnits = .all` or `.cpuAndGPU`. RF-DETR FP32 models do not
  get useful Neural Engine coverage, so `.cpuAndNeuralEngine` behaves like CPU.
- Batch size 1 is usually fastest on Apple Silicon. On an M4 Pro, batch 2 did
  not improve throughput because GPU utilization was already high at batch 1.
- Output resolution is fixed per model variant.

## Performance Snapshot

Benchmarks below were measured on Apple M5 Pro 18-core, RF-DETR 1.7.1,
coremltools 9.0, FP32, Core ML GPU. The full benchmark scripts are in
`scripts/`.

| Model | PyTorch MPS | Core ML GPU | Notes |
| --- | ---: | ---: | --- |
| Nano | 12.1 ms | 8.2 ms | detection |
| Small | 17.7 ms | 13.2 ms | detection |
| Medium | 23.5 ms | 16.8 ms | detection |
| Large | 38.3 ms | 24.8 ms | detection |
| Seg-Nano | 16.6 ms | 12.0 ms | segmentation |
| Seg-Small | 20.1 ms | 15.9 ms | segmentation |
| Seg-Medium | 27.9 ms | 21.3 ms | segmentation |
| Seg-Large | 39.7 ms | 28.2 ms | segmentation |
| Seg-XLarge | 73.9 ms | 49.9 ms | segmentation |
| Seg-2XLarge | 135.0 ms | 94.9 ms | segmentation |

Use `scripts/test_export.py` to verify Core ML outputs against RF-DETR PyTorch
on real images for your target model.

## How It Works

RF-DETR cannot be converted cleanly to Core ML as-is because the model and
converter hit a few specific boundaries:

- Deformable attention creates rank-6 tensors; Core ML supports rank 5 or less.
- DinoV2 positional interpolation uses bicubic mode; Core ML conversion handles
  bilinear more reliably.
- RF-DETR export returns tuples, while normal model inference returns dicts.
- Current released `coremltools` versions still need fixes for a few Torch
  frontend edge cases used by RF-DETR.
- RF-DETR 1.7.x segmentation uses a custom autograd function that traces as a
  PythonOp.

The package applies runtime monkey patches on import:

- `MSDeformAttn.forward` is rewritten to merge batch and heads before attention,
  keeping tensors at rank 5 or below.
- The deformable-attention core is replaced with a rank-5 equivalent.
- DinoV2 bicubic interpolation is switched to bilinear during export.
- `coremltools` Torch frontend fixes are applied for `_cast`, `view`, and
  `meshgrid`.
- The RF-DETR 1.7.x segmentation depthwise-conv custom autograd function is
  replaced with plain `F.conv2d` for export.

Importing `rfdetr_coreml` applies the patch overlay:

```python
import rfdetr_coreml
from rfdetr_coreml.export import export_to_coreml

path = export_to_coreml("nano", output_dir="output", precision="fp32")
print(path)
```

## Why Direct Core ML

RF-DETR 1.7 has an official export path for ONNX and experimental TFLite. Use
the upstream exporter when your target is ONNX Runtime, TensorRT, OpenVINO,
TFLite, or another cross-platform runtime.

This project is Apple-specific. It converts the patched PyTorch model directly
to a Core ML ML Program package so the RF-DETR/Core ML compatibility fixes are
applied before conversion. The direct path keeps the model in one Core ML graph
and was faster than ONNX Runtime's Core ML Execution Provider in this project's
benchmarks.

The ONNX benchmark script uses RF-DETR 1.7's official ONNX exporter in a
patch-isolated subprocess, then compares ONNX Runtime against this project's
direct Core ML path.

Detection-only ONNX benchmark, same machine and dependency versions as above,
50 timed runs per backend:

| Model | ONNX CPU | ONNX CoreML EP default | ONNX CoreML EP MLProgram FP32 | Direct Core ML | Box diff range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nano | 35.1 ms | 54.5 ms | 16.0 ms | 8.2 ms | 0.00-0.12 px |
| Small | 64.1 ms | 103.8 ms | 23.6 ms | 13.2 ms | 0.00-0.16 px |
| Medium | 79.5 ms | 146.3 ms | 30.7 ms | 16.9 ms | 0.00-0.11 px |
| Large | 138.2 ms | 233.8 ms | 41.8 ms | 24.9 ms | 0.00-0.21 px |

The box diff range is measured in pixels over confident PyTorch reference
queries. ONNX rows compare against RF-DETR's official PyTorch export reference;
Direct Core ML compares against this project's patched PyTorch reference.
Segmentation ONNX benchmarks are not included because mask-output handling is
not implemented in `scripts/benchmark_onnx.py`.

## Repository Layout

```text
rfdetr_coreml/
  __init__.py        # applies patches on import
  patches.py         # RF-DETR runtime patches
  coreml_fixes.py    # coremltools frontend patches
  export.py          # export pipeline
  cli.py             # rfdetr-coreml command
scripts/
  test_export.py
  benchmark_latency.py
  benchmark_onnx.py
  smoke_test.py
  _export_onnx_official.py
  test_fp16.py
  test_batch2.py
  validate_coreml.swift
```

## Limitations

- FP32 models are larger than FP16 models.
- FP16 is not production-safe for RF-DETR deformable attention.
- Benchmarks are hardware-specific; validate latency and accuracy on your target
  device.
- Re-run ONNX/Core ML benchmarks after RF-DETR or ONNX Runtime upgrades; the
  upstream RF-DETR export path changed substantially in the 1.7 line.
- Some patches can be removed when released `coremltools` and `rfdetr` versions
  cover the same behavior upstream.

## Acknowledgments

- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr), the upstream model.
- [timnielen/rf-detr](https://github.com/timnielen/rf-detr), which first showed
  a practical Core ML conversion path by refactoring deformable attention around
  Core ML's rank limit.
- [apple/coremltools#2665](https://github.com/apple/coremltools/pull/2665),
  whose meshgrid fix matches one of the released-version workarounds used here.

## License

Apache 2.0
