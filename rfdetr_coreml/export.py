"""
CoreML export logic for RF-DETR models.

Provides NormalizedWrapper (embeds ImageNet normalization) and
export_to_coreml() which handles the full pipeline:
  model instantiation → export mode → trace → ct.convert → save
"""

import json
import logging
import os
import time
from collections.abc import Mapping
from copy import deepcopy
from importlib import metadata as importlib_metadata

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
    "large": ("rfdetr.detr.RFDETRLarge", 704),
    # Segmentation models
    "seg-nano": ("rfdetr.detr.RFDETRSegNano", 312),
    "seg-small": ("rfdetr.detr.RFDETRSegSmall", 384),
    "seg-medium": ("rfdetr.detr.RFDETRSegMedium", 432),
    "seg-large": ("rfdetr.detr.RFDETRSegLarge", 504),
    "seg-xlarge": ("rfdetr.detr.RFDETRSegXLarge", 624),
    "seg-2xlarge": ("rfdetr.detr.RFDETRSeg2XLarge", 768),
}


def _package_version(package_name: str) -> str:
    """Return installed package version for Core ML provenance metadata."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


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


def _normalize_class_labels(class_names) -> tuple[list[str], list[int] | None] | None:
    """Return class names and optional class IDs, or None when no names exist."""
    if class_names is None:
        return None
    if isinstance(class_names, str):
        return ([class_names], None) if class_names else None

    if isinstance(class_names, Mapping):
        if not class_names:
            return None
        try:
            items = sorted(class_names.items(), key=lambda item: int(item[0]))
            class_ids = [int(class_id) for class_id, _ in items]
        except (TypeError, ValueError):
            return None
        return [str(name) for _, name in items], class_ids

    try:
        names = list(class_names)
    except TypeError:
        return None
    if not names:
        return None
    return [str(name) for name in names], None


def _checkpoint_arg(checkpoint: dict, name: str):
    args = checkpoint.get("args")
    if isinstance(args, dict):
        return args.get(name)
    return getattr(args, name, None)


def _extract_checkpoint_class_labels(checkpoint: dict) -> tuple[list[str], list[int] | None] | None:
    """Extract class names embedded by RF-DETR training checkpoints."""
    for candidate in (
        checkpoint.get("class_names"),
        _checkpoint_arg(checkpoint, "class_names"),
    ):
        class_labels = _normalize_class_labels(candidate)
        if class_labels is not None:
            return class_labels
    return None


def _class_names_to_metadata(class_names: list[str], class_ids, source: str) -> dict[str, str]:
    class_ids = [int(class_id) for class_id in class_ids]
    if len(class_names) != len(class_ids):
        raise ValueError(
            f"class_names and class_ids must have the same length, got "
            f"{len(class_names)} names and {len(class_ids)} ids"
        )

    return {
        "names": str({index: class_name for index, class_name in enumerate(class_names)}),
        "class_names": json.dumps(class_names, ensure_ascii=False, separators=(",", ":")),
        "class_ids": json.dumps(class_ids, separators=(",", ":")),
        "class_mapping": json.dumps(
            {str(class_id): class_name for class_id, class_name in zip(class_ids, class_names)},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "class_count": str(len(class_names)),
        "class_names_source": source,
    }


def _class_names_metadata(
    rfdetr_model,
    *,
    weights_path: str | None,
    checkpoint_class_labels: tuple[list[str], list[int] | None] | None,
    num_classes: int | None,
) -> dict[str, str]:
    """Build Core ML user-defined metadata for RF-DETR class labels."""
    if checkpoint_class_labels is not None:
        checkpoint_class_names, checkpoint_class_ids = checkpoint_class_labels
        return _class_names_to_metadata(
            checkpoint_class_names,
            checkpoint_class_ids or range(len(checkpoint_class_names)),
            "checkpoint",
        )

    class_labels = _normalize_class_labels(getattr(rfdetr_model, "class_names", None))
    if class_labels is None:
        return {"class_names_source": "unavailable"}
    class_names, class_ids = class_labels

    if weights_path is not None:
        metadata = {"class_names_source": "unavailable"}
        if num_classes is not None:
            metadata["class_count"] = str(num_classes)
        return metadata

    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES, COCO_CLASS_NAMES
    except ImportError:
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES
        except ImportError:
            COCO_CLASSES = {}
        if isinstance(COCO_CLASSES, Mapping):
            COCO_CLASS_NAMES = [name for _, name in sorted(COCO_CLASSES.items())]
        else:
            COCO_CLASSES = {}
            COCO_CLASS_NAMES = []

    if class_names == list(COCO_CLASS_NAMES):
        return _class_names_to_metadata(class_names, COCO_CLASSES.keys(), "coco")
    if class_ids is not None:
        return _class_names_to_metadata(class_names, class_ids, "model")

    return _class_names_to_metadata(class_names, range(len(class_names)), "model")


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
        model_name: Model variant key from MODEL_REGISTRY (e.g. 'nano',
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
    checkpoint_class_labels = None
    num_classes = None

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

        if isinstance(checkpoint, dict):
            checkpoint_class_labels = _extract_checkpoint_class_labels(checkpoint)

        # Detect num_classes from the classification head weight.
        # RF-DETR internally adds +1 for background class, so
        # class_embed.weight shape = (num_classes + 1, dim).
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

    if num_classes is None:
        model_args = getattr(rfdetr_model.model, "args", None)
        num_classes = getattr(model_args, "num_classes", None)

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
    rfdetr_version = _package_version("rfdetr")
    coremltools_version = _package_version("coremltools")
    mlmodel.author = "rfdetr_coreml"
    mlmodel.short_description = f"RF-DETR {model_name} ({precision.upper()}{batch_desc}) — {resolution}x{resolution}"
    mlmodel.version = rfdetr_version
    mlmodel.user_defined_metadata["rfdetr_version"] = rfdetr_version
    mlmodel.user_defined_metadata["coremltools_version"] = coremltools_version
    mlmodel.user_defined_metadata["model_variant"] = model_name
    mlmodel.user_defined_metadata["precision"] = precision
    mlmodel.user_defined_metadata.update(_class_names_metadata(
        rfdetr_model,
        weights_path=weights_path,
        checkpoint_class_labels=checkpoint_class_labels,
        num_classes=num_classes,
    ))

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
