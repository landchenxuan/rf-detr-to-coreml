#!/usr/bin/env python3
"""
Export raw (UNPATCHED) RF-DETR model to ONNX and save PyTorch reference output.

This script intentionally does NOT import rfdetr_coreml, so the model has no
monkey-patches applied. This produces the same ONNX model you'd get from
rfdetr's built-in export — with rank-6 tensors and bicubic interpolation intact.

Called by benchmark_onnx.py via subprocess to ensure patch isolation.

Usage:
  python scripts/_export_onnx_raw.py --model nano --output-dir output
"""

import argparse
import inspect
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

# Model registry (duplicated from rfdetr_coreml.export to avoid importing it)
MODEL_REGISTRY = {
    "nano": ("rfdetr.detr.RFDETRNano", 384),
    "small": ("rfdetr.detr.RFDETRSmall", 512),
    "medium": ("rfdetr.detr.RFDETRMedium", 576),
    "base": ("rfdetr.detr.RFDETRBase", 560),
    "large": ("rfdetr.detr.RFDETRLarge", 704),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NormalizedWrapper(nn.Module):
    """Wraps model with ImageNet normalization (same as rfdetr_coreml.export)."""

    def __init__(self, model, resolution):
        super().__init__()
        self.model = model
        self.resolution = resolution
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        x = (x - self.mean) / self.std
        return self.model(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model_name = args.model
    output_dir = args.output_dir
    class_path, resolution = MODEL_REGISTRY[model_name]

    onnx_path = os.path.join(output_dir, f"rf-detr-{model_name}-raw.onnx")
    ref_path = os.path.join(output_dir, f"rf-detr-{model_name}-raw-ref.npy")

    os.makedirs(output_dir, exist_ok=True)

    # Import model class
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    # Build model (no patches!)
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped = NormalizedWrapper(pt_model, resolution).eval()
    del rfdetr_model

    # Real test image (same as all other scripts)
    from PIL import Image
    import glob
    test_img_path = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "test_images", "*.jpg")))[0]
    pil_img = Image.open(test_img_path).convert("RGB").resize(
        (resolution, resolution), Image.BILINEAR
    )
    img_np = np.array(pil_img)
    pt_input = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Save PyTorch reference output (unpatched)
    with torch.no_grad():
        pt_out = wrapped(pt_input)
    ref_boxes = pt_out[0].numpy()
    np.save(ref_path, ref_boxes)
    print(f"Saved reference: {ref_path} (shape={ref_boxes.shape})")

    # Export to ONNX
    export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    dummy = torch.rand(1, 3, resolution, resolution)
    torch.onnx.export(
        wrapped,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["boxes", "logits"],
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        verbose=False,
        opset_version=17,
        **export_kwargs,
    )
    print(f"Saved ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
