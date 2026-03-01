"""
rfdetr_coreml — CoreML export overlay for upstream RF-DETR.

Importing this package automatically applies all necessary patches:
  1. coremltools _cast bug workaround
  2. Deformable attention 6D → 5D tensor fix
  3. Bicubic → bilinear interpolation fix
"""

from rfdetr_coreml.coreml_fixes import apply_coremltools_patches
from rfdetr_coreml.patches import apply_rfdetr_patches

apply_coremltools_patches()
apply_rfdetr_patches()
