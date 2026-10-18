# RF-DETR Core ML Image Adapter

A thin wrapper around [RF-DETR](https://github.com/roboflow/rf-detr)'s official
Core ML exporter. Upstream owns model preparation, graph capture and conversion.
This package adds RGB image input, in-graph ImageNet normalization, and class
metadata. It does not monkey-patch RF-DETR or coremltools on import.

## Install

Python 3.10 or newer is required. For the tested environment:

```bash
pip install -r requirements.txt
pip install -e .
```

The current baseline is RF-DETR 1.11.2, coremltools 9.0, torch 2.7.0 and
torchvision 0.22.0. The RF-DETR dependency is bounded to 1.11.x because the
adapter extends its Core ML exporter internals.

## Export

```bash
rfdetr-coreml --model small
rfdetr-coreml --model seg-nano --precision fp16
rfdetr-coreml --model small --weights checkpoint.pth
rfdetr-coreml --model small --batch-size 2
```

Default: `nano`, FP32, batch 1, upstream COCO weights, `output/` directory.
Use `--help` for model choices and `--model all` to export every registered
variant. FP16 is optional; validate it on your own images and target device.

```python
from rfdetr_coreml import export_to_coreml

path = export_to_coreml("small", output_dir="output", precision="fp16")
```

Names retain the existing convention:
`rf-detr-{model}[-{weights-stem}]-{precision}[-batchN].mlpackage`.

## Input and metadata contract

- Batch 1: RGB `ImageType` named `image`. Pass a resized PIL image in Python or
  an RGB-compatible pixel buffer in an Apple application.
- Larger fixed batches: float32 NCHW `TensorType` named `image`, values in
  `[0, 1]`. Both paths apply ImageNet mean/std normalization inside the graph;
  do not normalize inputs again.
- Input resolution is fixed by the chosen variant. Read it from the package
  spec and resize images before prediction.
- Outputs remain upstream raw boxes, class logits and, for segmentation, masks.
  Output names are converter-generated. Detection selection and mask
  post-processing belong to the caller.
- Class metadata retains `class_names`, `class_ids`, `class_mapping`,
  `class_count`, `class_names_source`, and the dense-index `names` alias.
  Custom checkpoints need embedded `class_names`; absent labels are not guessed.
  `rfdetr_version`, `coremltools_version`, `model_variant` and `precision` record
  export provenance.

## Adapter tests

```bash
python scripts/smoke_test.py
python scripts/test_adapter.py
```

Smoke checks cover CLI/registry wiring, metadata and input validation. The
conversion tests use a tiny synthetic model to check RGB
scaling, channel-wise normalization, FP16, batch tensors and saved metadata
through real Core ML prediction, without downloading RF-DETR weights.

Model conversion support and model-quality testing are maintained
[upstream](https://github.com/roboflow/rf-detr).

## License

Apache 2.0
