"""
Monkey-patches for coremltools bugs that block RF-DETR conversion.

Bug 1: _cast() does `dtype(x.val)` where x.val is a shape-(1,) numpy array.
Bug 2: view() can't handle shape list with non-scalar Var elements.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_applied = False


def apply_coremltools_patches():
    global _applied
    if _applied:
        return
    _applied = True

    try:
        import coremltools.converters.mil.frontend.torch.ops as ct_ops
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs, Var
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError:
        logger.warning("coremltools not installed, skipping patches")
        return

    # --- Patch 1: _cast (numpy scalar bug) ---
    def patched_cast(context, node, dtype, dtype_name):
        inputs = _get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")

        if x.can_be_folded_to_const():
            val = x.val
            if isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()
            if not isinstance(val, dtype):
                res = mb.const(val=dtype(val), name=node.name)
            else:
                res = x
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    ct_ops._cast = patched_cast
    logger.info("Patched coremltools _cast")

    # --- Patch 2: view (shape list with non-scalar Vars) ---
    from coremltools.converters.mil.frontend.torch.ops import ListVar

    original_view = ct_ops.view

    def patched_view(context, node):
        inputs = _get_inputs(context, node, expected=2)
        x = inputs[0]
        shape = inputs[1]

        if isinstance(shape, Var) and np.prod(shape.shape) == 0:
            assert np.prod(x.shape) <= 1, (
                "Reshape to empty shape works only for scalar and single-element tensor"
            )
            context.add(mb.identity(x=x, name=node.name))
            return

        if isinstance(shape, ListVar):
            length = mb.list_length(ls=shape)
            indices = mb.range_1d(start=0, end=length, step=1)
            shape = mb.list_gather(ls=shape, indices=indices)

        # Handle list of Vars — squeeze any 1D single-element Vars to scalars
        if isinstance(shape, list) and all(isinstance(dim, Var) for dim in shape):
            int_shape = []
            for i, size in enumerate(shape):
                s = size
                # Squeeze 1D shape-(1,) Vars to scalar
                if len(s.shape) > 0:
                    s = mb.squeeze(x=s, name=node.name + f"_dim{i}_squeeze")
                if s.dtype != types.int32:
                    s = mb.cast(x=s, dtype="int32", name=node.name + f"_dim{i}_cast")
                int_shape.append(s)
            shape = mb.concat(values=int_shape, axis=0, name=node.name + "_shape")

        if isinstance(shape, Var):
            if shape.dtype != types.int32:
                shape = mb.cast(x=shape, dtype="int32", name=node.name + "_shape_cast")

        view = mb.reshape(x=x, shape=shape, name=node.name)
        context.add(view)

    # Replace the registered op handler
    ct_ops.view = patched_view
    # Also need to update the op registry
    try:
        from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY
        for alias in ["view", "view_copy", "_unsafe_view", "reshape"]:
            if alias in _TORCH_OPS_REGISTRY:
                _TORCH_OPS_REGISTRY[alias] = patched_view
    except ImportError:
        pass

    logger.info("Patched coremltools view op")
