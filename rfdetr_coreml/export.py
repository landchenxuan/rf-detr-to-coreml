"""
CoreML export logic for RF-DETR models.

Provides NormalizedWrapper (embeds ImageNet normalization) and
export_to_coreml() which handles the full pipeline:
  model instantiation → export mode → trace → ct.convert → save
"""

import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model registry: name → (class_path, resolution)
MODEL_REGISTRY = {
    # Detection models
    "nano": ("rfdetr.detr.RFDETRNano", 384),
    "small": ("rfdetr.detr.RFDETRSmall", 512),
    "medium": ("rfdetr.detr.RFDETRMedium", 576),
    "base": ("rfdetr.detr.RFDETRBase", 560),
    "large": ("rfdetr.detr.RFDETRLarge", 704),
    # Segmentation models
    "seg-preview": ("rfdetr.detr.RFDETRSegPreview", 432),
    "seg-nano": ("rfdetr.detr.RFDETRSegNano", 312),
    "seg-small": ("rfdetr.detr.RFDETRSegSmall", 384),
    "seg-medium": ("rfdetr.detr.RFDETRSegMedium", 432),
    "seg-large": ("rfdetr.detr.RFDETRSegLarge", 504),
    "seg-xlarge": ("rfdetr.detr.RFDETRSegXLarge", 624),
    "seg-2xlarge": ("rfdetr.detr.RFDETRSeg2XLarge", 768),
}


class NormalizedWrapper(nn.Module):
    """
    Wraps an RF-DETR model to include ImageNet normalization and resizing
    in the model graph. This means the CoreML model accepts raw [0,1] images.
    """

    def __init__(self, model, resolution, mean=None, std=None):
        super().__init__()
        self.model = model
        self.resolution = resolution
        # Register as buffers so they move with .to(device)
        mean = mean or IMAGENET_MEAN
        std = std or IMAGENET_STD
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        # Resize to model resolution
        x = torch.nn.functional.interpolate(
            x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        # Normalize
        x = (x - self.mean) / self.std
        return self.model(x)


def _import_model_class(model_name):
    """Dynamically import the model class."""
    class_path = MODEL_REGISTRY[model_name][0]
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def export_to_coreml(
    model_name: str,
    output_dir: str = "output",
    precision: str = "fp32",
    weights_path: str | None = None,
    batch_size: int = 1,
) -> str:
    """
    Export an RF-DETR model to CoreML format.

    Args:
        model_name: Model variant key from MODEL_REGISTRY (e.g. 'nano', 'base',
                    'seg-nano'). Use ``list(MODEL_REGISTRY)`` to see all options.
        output_dir: Directory to save the .mlpackage.
        precision: 'fp16' or 'fp32' (default fp32). WARNING: fp16 has known
                   catastrophic precision issues with deformable attention.
        weights_path: Path to custom .pth weights (fine-tuned model).
                      If None, downloads pre-trained COCO weights.
        batch_size: Batch size for the exported model (default 1).
                    batch=1 uses ct.ImageType (accepts 0-255 uint8 or 0-1 float).
                    batch>1 uses ct.TensorType (accepts 0-1 float32 NCHW).

    Returns:
        Path to the saved .mlpackage directory.
    """
    import coremltools as ct

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    resolution = MODEL_REGISTRY[model_name][1]
    batch_desc = f", batch={batch_size}" if batch_size > 1 else ""
    logger.info(f"Exporting RF-DETR {model_name} (resolution={resolution}, precision={precision}{batch_desc})")

    # Step 1: Instantiate model
    t0 = time.time()
    model_cls = _import_model_class(model_name)

    if weights_path:
        # For custom weights: load checkpoint first to detect num_classes,
        # then instantiate model with matching dimensions.
        logger.info(f"Loading custom weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Detect num_classes from the classification head weight.
        # RF-DETR internally adds +1 for background class, so
        # class_embed.weight shape = (num_classes + 1, dim).
        num_classes = None
        for key in ("class_embed.0.weight", "class_embed.weight"):
            if key in state_dict:
                num_classes = state_dict[key].shape[0] - 1
                logger.info(f"Detected num_classes={num_classes} from checkpoint key '{key}' "
                            f"(shape {state_dict[key].shape[0]} - 1 background)")
                break

        if num_classes is not None:
            rfdetr_model = model_cls(pretrain_weights=None, num_classes=num_classes)
        else:
            rfdetr_model = model_cls(pretrain_weights=None)
        rfdetr_model.model.model.load_state_dict(state_dict, strict=False)
    else:
        rfdetr_model = model_cls()

    logger.info(f"Model instantiated in {time.time() - t0:.1f}s")

    # Step 2: Deep copy the inner PyTorch model, move to CPU, eval mode
    t0 = time.time()
    model = deepcopy(rfdetr_model.model.model)
    model = model.cpu().eval()

    # Step 3: Switch to export mode (forward → forward_export, cascades to submodules)
    model.export()

    # Step 4: Wrap with normalization
    wrapped = NormalizedWrapper(model, resolution)
    wrapped.eval()
    logger.info(f"Model prepared in {time.time() - t0:.1f}s")

    # Step 5: Trace with dummy input
    t0 = time.time()
    dummy = torch.rand(batch_size, 3, resolution, resolution)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy)
    logger.info(f"Traced in {time.time() - t0:.1f}s")

    # Step 6: Convert to CoreML
    t0 = time.time()
    compute_precision = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32

    if precision == "fp16":
        logger.warning(
            "FP16 precision may cause significant accuracy degradation in deformable "
            "attention. Use FP32 for production. See README for details."
        )

    if batch_size == 1:
        inputs = [ct.ImageType(name="image", shape=(1, 3, resolution, resolution), scale=1.0 / 255.0)]
    else:
        inputs = [ct.TensorType(
            name="image",
            shape=(batch_size, 3, resolution, resolution),
            dtype=np.float32,
        )]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS16,
    )

    # Add metadata
    mlmodel.author = "rfdetr_coreml"
    mlmodel.short_description = f"RF-DETR {model_name} ({precision.upper()}{batch_desc}) — {resolution}x{resolution}"
    mlmodel.version = "1.5.1"

    logger.info(f"Converted in {time.time() - t0:.1f}s")

    # Step 7: Save
    os.makedirs(output_dir, exist_ok=True)
    suffix = ""
    if weights_path:
        stem = os.path.splitext(os.path.basename(weights_path))[0]
        suffix = f"-{stem}"
    batch_tag = f"-batch{batch_size}" if batch_size > 1 else ""
    filename = f"rf-detr-{model_name}{suffix}-{precision}{batch_tag}.mlpackage"
    output_path = os.path.join(output_dir, filename)
    mlmodel.save(output_path)

    # Report size
    total_size = 0
    for dirpath, _, filenames in os.walk(output_path):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    size_mb = total_size / (1024 * 1024)
    logger.info(f"Saved to {output_path} ({size_mb:.1f} MB)")

    return output_path
