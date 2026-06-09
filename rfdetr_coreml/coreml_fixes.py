"""
Monkey-patches for coremltools bugs that block RF-DETR conversion.

Bug 1: _cast() does `dtype(x.val)` where x.val is a shape-(1,) numpy array.
Bug 2: view() can't handle shape list with non-scalar Var elements.
Bug 3: meshgrid() rejects rank>1 tensors produced by coremltools 9.0 shape
       inference even when the source tensors are semantically 1-D.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_applied = False


def apply_coremltools_patches() -> None:
    """Apply monkey-patches for coremltools bugs that block RF-DETR conversion."""
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

    # --- Patch 3: meshgrid (coremltools 9.0 shape-inference regression) ---
    # In ct 8.1 the meshgrid inputs (torch.linspace / torch.arange, rank-1 in
    # source) trace as rank-1. In ct 9.0 they can arrive as rank>1, tripping the
    # handler's `_check_args` ("meshgrid received non-1d tensor."). RF-DETR's two
    # meshgrid sites (transformer.gen_encoder_output_proposals, box_ops) are both
    # semantically 1-D, so flattening each input to rank-1 is safe and correct.
    from coremltools.converters.mil.frontend.torch.ops import _get_kwinputs
    from coremltools.converters.mil.frontend.torch import ops as _ct_ops_mod

    def patched_meshgrid(context, node):
        inputs = _get_inputs(context, node, expected=[1, 2])
        nargs = len(inputs)
        tensor_inputs = inputs[0]
        indexing = inputs[1].val if nargs > 1 else "ij"
        indexing = _get_kwinputs(context, node, "indexing", default=[indexing])[0]
        if isinstance(indexing, Var):
            indexing = indexing.val

        if not isinstance(tensor_inputs, (list, tuple)):
            raise ValueError("meshgrid requires a list/tuple of tensors.")
        if len(tensor_inputs) < 2:
            raise ValueError("Requires >= 2 tensor inputs.")
        if indexing not in ("ij", "xy"):
            raise ValueError(f"indexing mode {indexing} not supported")

        # Flatten any rank>1 input to rank-1 (inputs are semantically 1-D).
        flat_inputs = []
        for i, ti in enumerate(tensor_inputs):
            if ti.rank > 1:
                ti = mb.reshape(x=ti, shape=[-1], name=node.name + f"_flatten{i}")
            flat_inputs.append(ti)
        tensor_inputs = flat_inputs

        result_symbolic_shape = [ti.shape[0] for ti in tensor_inputs]
        result_shape = _ct_ops_mod._utils.maybe_replace_symbols_with_source_tensor_shape_variables(
            result_symbolic_shape, tensor_inputs
        )

        grids = []
        size = len(tensor_inputs)
        for i in range(size):
            view_shape = [1] * size
            view_shape[i] = -1
            view = mb.reshape(
                x=tensor_inputs[i], shape=tuple(view_shape), name=node.name + "_view_" + str(i)
            )
            reps = result_shape.copy()
            reps[i] = 1
            if any(isinstance(rep, Var) for rep in reps):
                reps = mb.concat(values=reps, axis=0)
            res = mb.tile(x=view, reps=reps, name=node.name + "_expand_" + str(i))
            if indexing == "xy":
                perm = [1, 0] + list(range(2, size))
                res = mb.transpose(x=res, perm=perm, name=node.name + "_transpose_" + str(i))
            grids.append(res)

        context.add(tuple(grids), node.name)

    ct_ops.meshgrid = patched_meshgrid
    try:
        from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY
        for alias in ["meshgrid", "meshgrid.indexing"]:
            if alias in _TORCH_OPS_REGISTRY:
                _TORCH_OPS_REGISTRY[alias] = patched_meshgrid
    except ImportError:
        pass

    logger.info("Patched coremltools meshgrid op")
