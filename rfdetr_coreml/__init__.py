"""
rfdetr_coreml — CoreML export overlay for upstream RF-DETR.

Importing this package automatically applies all necessary patches:
  1. coremltools _cast bug workaround
  2. coremltools view op bug workaround
  3. coremltools meshgrid op workaround
  4. Deformable attention 6D → 5D tensor fix
  5. Bicubic → bilinear interpolation fix
  6. RF-DETR 1.7 segmentation PythonOp workaround
"""

__all__ = ["export_to_coreml", "MODEL_REGISTRY"]

from rfdetr_coreml.coreml_fixes import apply_coremltools_patches
from rfdetr_coreml.patches import apply_rfdetr_patches

apply_coremltools_patches()
apply_rfdetr_patches()

from rfdetr_coreml.export import MODEL_REGISTRY, export_to_coreml  # noqa: E402
