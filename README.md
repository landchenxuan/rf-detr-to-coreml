# RF-DETR to CoreML

Export [RF-DETR](https://github.com/roboflow/rf-detr) detection and
segmentation models to Apple Core ML.

This project converts RF-DETR directly from PyTorch to Core ML ML Program format
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

## CLI

| Option | Default | Notes |
| --- | --- | --- |
| `--model` | `nano` | Detection: `nano`, `small`, `medium`, `base`, `large`; segmentation: `seg-preview`, `seg-nano`, `seg-small`, `seg-medium`, `seg-large`, `seg-xlarge`, `seg-2xlarge`; or `all` |
| `--precision` | `fp32` | Use `fp32` for production. `fp16` is available but not reliable for RF-DETR. |
| `--output-dir` | `output` | Output directory for `.mlpackage` files |
| `--weights` | None | Path to fine-tuned `.pth` weights |
| `--batch-size` | `1` | `1` uses Core ML `ImageType`; larger batches use `TensorType` NCHW float32 input in `[0, 1]` |

## Supported Models

Detection:

```text
nano, small, medium, base, large
```

Segmentation:

```text
seg-preview, seg-nano, seg-small, seg-medium, seg-large, seg-xlarge, seg-2xlarge
```

The package targets released RF-DETR versions from the 1.5.x line through the
current 1.7.x line. The newest compatibility patches target released
`coremltools` 9.0 and `rfdetr` 1.7.1.

## Production Notes

- Use `precision=fp32`. RF-DETR deformable attention is sensitive to FP16
  coordinate precision; FP16 exports can produce boxes hundreds of pixels off.
- Use Core ML `computeUnits = .all` or `.cpuAndGPU`. RF-DETR FP32 models do not
  get useful Neural Engine coverage, so `.cpuAndNeuralEngine` behaves like CPU.
- Batch size 1 is usually fastest on Apple Silicon. On an M4 Pro, batch 2 did
  not improve throughput because GPU utilization was already high at batch 1.
- Output resolution is fixed per model variant.

## Performance Snapshot

Benchmarks below were measured on a MacBook Pro M4 Pro, FP32, Core ML GPU. The
full benchmark scripts are in `scripts/`.

| Model | PyTorch MPS | Core ML GPU | Notes |
| --- | ---: | ---: | --- |
| Nano | 21.6 ms | 11.2 ms | detection |
| Small | 32.1 ms | 18.0 ms | detection |
| Large | 59.3 ms | 34.9 ms | detection |
| Seg-Nano | 29.4 ms | 16 ms | segmentation |
| Seg-2XLarge | 169.3 ms | 128 ms | segmentation |

Accuracy was checked on 17 real images. Detection box differences were below
0.01 px for most variants; the largest segmentation model has higher mask/box
drift and should be validated for your deployment.

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

RF-DETR can also export to ONNX, but ONNX Runtime's Core ML Execution Provider
partitions this model and may silently use FP16 in default paths. Direct Core ML
conversion keeps the patched model in one ML Program graph and was faster in the
project benchmarks.

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
  test_fp16.py
  test_batch2.py
  validate_coreml.swift
```

## Limitations

- FP32 models are larger than FP16 models.
- FP16 is not production-safe for RF-DETR deformable attention.
- Benchmarks are hardware-specific; validate latency and accuracy on your target
  device.
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
