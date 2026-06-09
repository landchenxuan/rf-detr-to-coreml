#!/usr/bin/env python3
"""Export RF-DETR ONNX through the official RF-DETR exporter.

This helper intentionally does not import ``rfdetr_coreml``. The benchmark uses
it in a subprocess so the ONNX path exercises upstream RF-DETR's
``model.export(format="onnx")`` behavior, while the direct Core ML path still
uses this package's patches.

Usage:
  python scripts/_export_onnx_official.py --model nano --output-dir output
"""

import argparse
import glob
import importlib
import os
from copy import deepcopy

import numpy as np
import torch

from rfdetr.export.main import make_infer_image

# Detection models only. Segmentation ONNX benchmark support needs mask handling.
MODEL_REGISTRY = {
    "nano": ("rfdetr.detr.RFDETRNano", 384),
    "small": ("rfdetr.detr.RFDETRSmall", 512),
    "medium": ("rfdetr.detr.RFDETRMedium", 576),
    "large": ("rfdetr.detr.RFDETRLarge", 704),
}


def _import_model_class(model_name):
    class_path = MODEL_REGISTRY[model_name][0]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_test_input(resolution):
    test_img_path = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "test_images", "*.jpg")))[0]
    return make_infer_image(test_img_path, (resolution, resolution), batch_size=1, device="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model_name = args.model
    output_dir = args.output_dir
    _, resolution = MODEL_REGISTRY[model_name]

    os.makedirs(output_dir, exist_ok=True)
    model_cls = _import_model_class(model_name)
    rfdetr_model = model_cls(device="cpu")

    onnx_path = rfdetr_model.export(
        output_dir=output_dir,
        shape=(resolution, resolution),
        batch_size=1,
        format="onnx",
        verbose=False,
    )

    ref_path = os.path.join(output_dir, f"rf-detr-{model_name}-official-ref.npz")
    pt_input = _load_test_input(resolution)

    model = deepcopy(rfdetr_model.model.model).cpu().eval()
    model.export()
    with torch.no_grad():
        outputs = model(pt_input)
    if isinstance(outputs, tuple):
        boxes, logits = outputs[:2]
    else:
        boxes = outputs["pred_boxes"]
        logits = outputs["pred_logits"]
    np.savez(
        ref_path,
        input=pt_input.numpy(),
        boxes=boxes.detach().cpu().numpy(),
        logits=logits.detach().cpu().numpy(),
    )

    print(f"Saved official ONNX: {onnx_path}")
    print(f"Saved reference: {ref_path}")


if __name__ == "__main__":
    main()
