"""RF-DETR official Core ML export with image inputs and class metadata."""

from rfdetr_coreml.export import MODEL_REGISTRY, export_to_coreml

__all__ = ["export_to_coreml", "MODEL_REGISTRY"]
