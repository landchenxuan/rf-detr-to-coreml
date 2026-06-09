"""
Runtime monkey-patches for upstream rfdetr to enable CoreML conversion.

Three patches are applied:
  A) MSDeformAttn.forward — merge batch+heads to keep tensors ≤ rank-5
  B) ms_deform_attn_core_pytorch — accept 5D (batch*heads merged) inputs
  C) Bicubic → bilinear interpolation in DinoV2 backbone pos-encoding
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_applied = False


# ---------------------------------------------------------------------------
# Patch B: replacement for ms_deform_attn_core_pytorch
# Accepts tensors with batch and heads already merged (all ≤ rank-5).
# ---------------------------------------------------------------------------
def _ms_deform_attn_core_5d(value, value_spatial_shapes, sampling_locations, attention_weights,
                            spatial_shapes_hw=None):
    """
    Core deformable attention with merged batch+heads (max rank 5).

    Args:
        value: (B*H, head_dim, Len_in)
        value_spatial_shapes: (L, 2) — [(H_0, W_0), ...]
        sampling_locations: (B*H, Len_q, L, P, 2) — 5D
        attention_weights: (B*H, Len_q, L*P)
        spatial_shapes_hw: optional list of Python-int (H, W) pairs (rfdetr >=1.7).
            When provided, used for split/view so concrete ints reach trace.
    Returns:
        (B*H, Len_q, head_dim)
    """
    BH, head_dim, _ = value.shape
    _, Len_q, L, P, _ = sampling_locations.shape

    # Prefer Python-int (H, W) pairs (trace-friendly) when the caller supplies them.
    if spatial_shapes_hw is not None:
        shapes = [(int(H_), int(W_)) for (H_, W_) in spatial_shapes_hw]
    else:
        shapes = [(int(H_), int(W_)) for (H_, W_) in value_spatial_shapes]

    value_list = value.split([H_ * W_ for H_, W_ in shapes], dim=2)
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(shapes):
        # (BH, head_dim, H_, W_)
        value_l_ = value_list[lid_].reshape(BH, head_dim, H_, W_)
        # (BH, Len_q, P, 2)
        sampling_grid_l_ = sampling_grids[:, :, lid_]
        # (BH, head_dim, Len_q, P)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    # (BH, head_dim, Len_q, L*P)
    output = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    # (BH, 1, Len_q, L*P)
    attention_weights = attention_weights.unsqueeze(1)
    # (BH, head_dim, Len_q) → (BH, Len_q, head_dim)
    output = (output * attention_weights).sum(-1)
    return output.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Patch A: replacement for MSDeformAttn.forward
# Merges N (batch) and n_heads before computing sampling_locations,
# so we never create rank-6 tensors.
# ---------------------------------------------------------------------------
def _msdeformattn_forward_5d(self, query, reference_points, input_flatten,
                             input_spatial_shapes, input_level_start_index,
                             input_padding_mask=None, input_spatial_shapes_hw=None):
    """
    MSDeformAttn.forward with rank ≤ 5 tensors (batch+heads merged).

    ``input_spatial_shapes_hw`` (rfdetr >=1.7) carries Python-int (H, W) pairs
    for trace-friendly split/view; forwarded to the core function.
    """
    N, Len_q, _ = query.shape
    N, Len_in, _ = input_flatten.shape

    n_heads = self.n_heads
    n_levels = self.n_levels
    n_points = self.n_points
    head_dim = self.d_model // n_heads

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))

    # --- sampling offsets: stay ≤ 5D by merging batch+heads early ---
    # Linear output: (N, Len_q, n_heads * n_levels * n_points * 2)
    # → (N, Len_q, n_heads, n_levels * n_points * 2)
    so = self.sampling_offsets(query).view(N, Len_q, n_heads, n_levels * n_points * 2)
    # → (N, n_heads, Len_q, ...) → (N*n_heads, Len_q, n_levels, n_points, 2)  5D
    so = so.permute(0, 2, 1, 3).reshape(N * n_heads, Len_q, n_levels, n_points, 2)

    # --- attention weights: merge similarly ---
    aw = self.attention_weights(query).view(N, Len_q, n_heads, n_levels * n_points)
    aw = F.softmax(aw, -1)
    # (N*n_heads, Len_q, n_levels * n_points)
    aw = aw.permute(0, 2, 1, 3).reshape(N * n_heads, Len_q, n_levels * n_points)

    # --- reference points: expand for n_heads → (N*n_heads, Len_q, n_levels, ...) ---
    ref = reference_points  # (N, Len_q, n_levels, 2 or 4)
    # (N, 1, Len_q, n_levels, D) → expand → (N*n_heads, Len_q, n_levels, D)
    ref = ref.unsqueeze(1).expand(-1, n_heads, -1, -1, -1).reshape(
        N * n_heads, Len_q, n_levels, ref.shape[-1]
    )

    # --- compute sampling_locations (max rank 5) ---
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        # ref[:, :, :, None, :] → (BH, Len_q, n_levels, 1, 2)  5D
        # offset_normalizer[None, None, :, None, :] → (1, 1, n_levels, 1, 2)  5D
        sampling_locations = (
            ref[:, :, :, None, :]
            + so / offset_normalizer[None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            ref[:, :, :, None, :2]
            + so / n_points * ref[:, :, :, None, 2:] * 0.5
        )
    else:
        raise ValueError(
            f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
        )

    # --- value: merge batch+heads ---
    # (N, Len_in, C) → (N, C, Len_in) → (N*n_heads, head_dim, Len_in)
    value = value.transpose(1, 2).contiguous().reshape(N * n_heads, head_dim, Len_in)

    # --- core attention (all ≤ rank 5) ---
    output = _ms_deform_attn_core_5d(value, input_spatial_shapes, sampling_locations, aw,
                                     spatial_shapes_hw=input_spatial_shapes_hw)
    # output: (N*n_heads, Len_q, head_dim)

    # --- un-merge back to (N, Len_q, C) ---
    output = output.view(N, n_heads, Len_q, head_dim).permute(0, 2, 1, 3).reshape(N, Len_q, -1)

    output = self.output_proj(output)
    return output


# ---------------------------------------------------------------------------
# Patch C: replace bicubic interpolation with bilinear
# ---------------------------------------------------------------------------
def _patch_bicubic_to_bilinear():
    """
    Patch F.interpolate calls that use mode='bicubic' in the DinoV2 backbone.

    Two locations:
    1. dinov2_with_windowed_attn.py — interpolate_pos_encoding (runtime)
    2. dinov2.py — DinoV2.export() inner function (one-time at export)

    We patch both by replacing the relevant methods.
    """
    from rfdetr.models.backbone.dinov2_with_windowed_attn import (
        WindowedDinov2WithRegistersBackbone,
    )
    from rfdetr.models.backbone.dinov2 import DinoV2

    # --- Patch the windowed backbone's embeddings class ---
    # The bicubic call is inside WindowedDinov2WithRegistersEmbeddings.interpolate_pos_encoding
    # We'll wrap it to use bilinear instead.
    embeddings_cls = None
    try:
        from rfdetr.models.backbone.dinov2_with_windowed_attn import (
            WindowedDinov2WithRegistersEmbeddings,
        )
        embeddings_cls = WindowedDinov2WithRegistersEmbeddings
    except ImportError:
        pass

    if embeddings_cls is not None:
        original_interpolate = embeddings_cls.interpolate_pos_encoding

        def patched_interpolate_pos_encoding(self, embeddings, height, width):
            # Temporarily monkey-patch F.interpolate to swap bicubic → bilinear
            original_fi = F.interpolate

            def safe_interpolate(*args, **kwargs):
                if kwargs.get("mode") == "bicubic":
                    kwargs["mode"] = "bilinear"
                    kwargs.pop("antialias", None)
                return original_fi(*args, **kwargs)

            F.interpolate = safe_interpolate
            try:
                return original_interpolate(self, embeddings, height, width)
            finally:
                F.interpolate = original_fi

        embeddings_cls.interpolate_pos_encoding = patched_interpolate_pos_encoding
        logger.info("Patched WindowedDinov2 interpolate_pos_encoding: bicubic → bilinear")

    # --- Patch DinoV2.export() to use bilinear ---
    original_export = DinoV2.export

    def patched_dinov2_export(self):
        original_fi = F.interpolate

        def safe_interpolate(*args, **kwargs):
            if kwargs.get("mode") == "bicubic":
                kwargs["mode"] = "bilinear"
                kwargs.pop("antialias", None)
            return original_fi(*args, **kwargs)

        F.interpolate = safe_interpolate
        try:
            return original_export(self)
        finally:
            F.interpolate = original_fi

    DinoV2.export = patched_dinov2_export
    logger.info("Patched DinoV2.export(): bicubic → bilinear")


# ---------------------------------------------------------------------------
# Patch D: replace segmentation-head custom autograd Function with plain conv2d
# ---------------------------------------------------------------------------
def _patch_segmentation_depthwise_conv():
    """
    The segmentation head's depthwise conv uses a custom ``torch.autograd.Function``
    (`_DepthwiseConvWithoutCuDNN`) purely to disable cuDNN in the *backward* pass on
    certain CUDA stacks (rf-detr issue #731). Its forward is an ordinary ``F.conv2d``.

    ``torch.jit.trace`` captures the custom Function as an opaque ``prim::PythonOp``
    ('pythonop'), which coremltools cannot convert. For export (eval / CPU / no
    backward) a plain ``F.conv2d`` is numerically identical, so we swap the module
    global the call-site resolves at runtime.
    """
    try:
        from rfdetr.models.heads import segmentation as seg_mod
    except ImportError:
        logger.warning("segmentation head not found; skipping depthwise-conv patch")
        return

    class _PlainDepthwiseConv:
        @staticmethod
        def apply(x, weight, bias, stride, padding, dilation, groups):
            return F.conv2d(x, weight, bias, stride=stride, padding=padding,
                            dilation=dilation, groups=groups)

    seg_mod._DepthwiseConvWithoutCuDNN = _PlainDepthwiseConv
    logger.info("Patched segmentation depthwise conv: custom Function → F.conv2d")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def apply_rfdetr_patches() -> None:
    """Apply all runtime patches to make rfdetr CoreML-compatible.

    Patches applied:
      A) MSDeformAttn.forward — merge batch+heads to keep tensors ≤ rank-5
      B) ms_deform_attn_core — accept 5D (batch*heads merged) inputs
      C) DinoV2 interpolation — bicubic → bilinear
    """
    global _applied
    if _applied:
        return
    _applied = True

    # Patch A: MSDeformAttn.forward
    from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
    MSDeformAttn.forward = _msdeformattn_forward_5d
    logger.info("Patched MSDeformAttn.forward (6D → 5D)")

    # Patch B is used internally by Patch A (no separate module-level patch needed)
    # The _ms_deform_attn_core_5d function replaces ms_deform_attn_core_pytorch
    # but is called directly from the patched forward, not imported by name.

    # Patch C: bicubic → bilinear
    _patch_bicubic_to_bilinear()

    # Patch D: segmentation-head custom autograd Function → plain conv2d
    _patch_segmentation_depthwise_conv()

    logger.info("All rfdetr CoreML patches applied")
