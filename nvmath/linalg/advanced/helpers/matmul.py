# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

import math
import warnings

import numpy as np

from nvmath.internal.tensor_ifc import TensorHolder
from nvmath.internal.tensor_wrapper import wrap_operand
from nvmath.internal.utils import create_empty_tensor, get_or_create_stream, infer_object_package

__all__ = [
    "BlockScalingFormat",
    "create_mxfp8_scale",
    "invert_mxfp8_scale",
    "get_mxfp8_scale_offset",
    "apply_mxfp8_scale",
    "unpack_fp4",
    "quantize_to_fp4",
    "get_block_scale_offset",
    "to_block_scale",
    "expand_block_scale",
]

# ====================================================
# helper functions for FP8 and MXFP8
# ====================================================


def _c_strides(operand_shape: tuple[int, ...]) -> tuple[int, ...]:
    ndim = len(operand_shape)
    strides = [0] * ndim
    stride = 1
    for i in range(ndim - 1, -1, -1):
        strides[i] = stride
        stride *= operand_shape[i]
    return tuple(strides)


def _validate_operand_ndim_for_block_scaling(operand_shape: tuple[int, ...]):
    ndim = len(operand_shape)
    if ndim < 2:
        raise ValueError(f"Operand shape must be at least 2D, got {ndim}D with shape {operand_shape}")
    return ndim


def _infer_blocked_axis(operand: TensorHolder):
    # In general, strides are not reliable if corresponding extent is 1,
    # but we expect (and check) operand last two extents to be divisible by
    # 128 or 64.
    return -1 if operand.strides[-2] > operand.strides[-1] else -2


class BlockScalingFormat(str, Enum):
    """
    Block scaling format for microscaling data types.

    Passed as the ``block_scaling_format`` argument to
    :func:`get_block_scale_offset`, :func:`to_block_scale`, and
    :func:`expand_block_scale`.
    """

    NVFP4 = "NVFP4"
    MXFP8 = "MXFP8"


# Per-format properties for microscaling (block-scaled) data types.
# Used by validation helpers and expand_block_scale to dispatch on
# block size, allowed dtypes, and scale interpretation without
# per-format if/elif branches. To add a new format, add an entry here.
_MICROSCALING_FORMAT_PROPERTIES: dict[BlockScalingFormat, dict] = {
    BlockScalingFormat.NVFP4: {
        "num_scalars_in_block": 16,
        "operand_dtypes": ("float4_e2m1fn_x2",),
        # NVFP4 scales are float8_e4m3fn values. uint8 is also accepted
        # because the two dtypes have the same bit-width, and internally
        # the scale bytes are reinterpreted as uint8 for the strided
        # expansion anyway (see _expand_block_scale).
        "scale_dtypes": ("float8_e4m3fn", "uint8"),
        # Raw uint8 scale bytes are reinterpreted as float8_e4m3fn values.
        "scale_interpretation": "float8_e4m3fn",
    },
    BlockScalingFormat.MXFP8: {
        "num_scalars_in_block": 32,
        "operand_dtypes": ("float8_e4m3fn", "float8_e5m2"),
        "scale_dtypes": ("uint8",),
        # Raw uint8 scale bytes are unsigned 8-bit biased exponents: 2^(value - 127).
        "scale_interpretation": "ue8m0",
    },
}

_COMPATIBLE_FORMATS_FOR_OPERAND_DTYPE: dict[str, set[BlockScalingFormat]] = {}
for _fmt, _props in _MICROSCALING_FORMAT_PROPERTIES.items():
    for _dt in _props["operand_dtypes"]:
        _COMPATIBLE_FORMATS_FOR_OPERAND_DTYPE.setdefault(_dt, set()).add(_fmt)


def _validate_operand_dtype_block_scaling_format_compatibility(
    operand_dtype: str,
    block_scaling_format: BlockScalingFormat,
) -> None:
    """Check that *block_scaling_format* is consistent with *operand_dtype*."""
    compatible = _COMPATIBLE_FORMATS_FOR_OPERAND_DTYPE.get(operand_dtype)
    if compatible is not None and block_scaling_format not in compatible:
        compatible_str = {str(f) for f in compatible}
        raise ValueError(
            f"Operand dtype {operand_dtype} requires block_scaling_format in {compatible_str}, got '{block_scaling_format}'."
        )


def _validate_scale_dtype_block_scaling_format_compatibility(
    scale_dtype: str,
    block_scaling_format: BlockScalingFormat,
    param_name: str = "scales",
) -> None:
    """Raise :exc:`TypeError` if *scale_dtype* is not valid for *block_scaling_format*."""
    props = _MICROSCALING_FORMAT_PROPERTIES[block_scaling_format]
    if scale_dtype not in props["scale_dtypes"]:
        allowed = " or ".join(f"torch.{d}" for d in props["scale_dtypes"])
        raise TypeError(f"For {block_scaling_format}, {param_name} must have dtype {allowed}, got {scale_dtype}")


def _infer_operand_shape_axis(
    operand_or_shape: torch.Tensor | tuple[int, ...],
    axis: Literal[-1, -2] | None,
) -> tuple[int, tuple[int, ...], Literal[-1, -2], str | None]:
    """Resolve operand shape, ndim, axis, and (optionally) operand dtype.

    Returns ``(ndim, operand_shape, axis, operand_dtype)`` where
    *operand_dtype* is ``None`` when a plain shape tuple is passed.
    """
    import torch

    if isinstance(operand_or_shape, tuple):
        ndim = _validate_operand_ndim_for_block_scaling(operand_or_shape)
        if axis is None:
            raise ValueError(f"Axis must be specified when operand shape is provided, got {axis}")
        return ndim, operand_or_shape, axis, None
    elif isinstance(operand_or_shape, torch.Tensor):
        operand = wrap_operand(operand_or_shape)
        operand_shape: tuple[int, ...] = operand.shape  # type: ignore
        ndim = _validate_operand_ndim_for_block_scaling(operand_shape)

        # Infer blocked axis
        inferred_axis = _infer_blocked_axis(operand)
        if axis is None:
            axis = inferred_axis
        else:
            if axis >= 0:
                axis -= ndim
            if axis != inferred_axis:
                layout = "row-major" if inferred_axis == -1 else "col-major"
                raise ValueError(f"Incorrect axis: {axis} for {layout} operand, expected {inferred_axis}.")

        if operand.dtype == "float4_e2m1fn_x2":
            operand_shape_list = list(operand_shape)
            operand_shape_list[axis] *= 2
            operand_shape = tuple(operand_shape_list)
        return ndim, operand_shape, axis, operand.dtype
    else:
        raise ValueError(
            f"Expected operand_shape to be a tuple or a torch.Tensor to infer the shape from, got {type(operand_or_shape)}."
        )


def _validate_shape_axes_block_scaling_format(
    operand_or_shape: torch.Tensor | tuple[int, ...],
    axis: Literal[-1, -2] | None,
    block_scaling_format: BlockScalingFormat,
) -> tuple[tuple[int, ...], Literal[-1, -2], Literal[-1, -2], BlockScalingFormat, int]:
    """
    Infers operand_shape and blocked/unblocked axes from ``operand_or_shape`` when a tensor
    is passed; validates ``block_scaling_format`` and related ``num_scalars_in_block``.

    The accepted operand_shape, axis, and block_scaling_format are validated, so that axis
    is one of the last two dimensions of the operand, and operand's shape extents are
    multiples of cuBLAS tile size and num_scalars_in_block.
    """

    ndim, operand_shape, axis, operand_dtype = _infer_operand_shape_axis(operand_or_shape, axis)

    if operand_dtype is not None:
        _validate_operand_dtype_block_scaling_format_compatibility(operand_dtype, block_scaling_format)

    neg_axis = axis if axis < 0 else axis - ndim
    if neg_axis == -1:
        unblocked_axis, blocked_axis = -2, -1  # type: tuple[Literal[-1, -2], Literal[-1, -2]]
    elif neg_axis == -2:
        unblocked_axis, blocked_axis = -1, -2
    else:
        raise ValueError(f"Axis must point to one of the last two dimensions of the operand, got {axis}")

    num_scalars_in_block = _MICROSCALING_FORMAT_PROPERTIES[block_scaling_format]["num_scalars_in_block"]

    if any(extent <= 0 for extent in operand_shape):
        raise ValueError(f"Operand shape must have positive extents, got {operand_shape}")

    tile_height = 128
    tile_width = 4
    scalars_per_tile_row = tile_width * num_scalars_in_block
    unblocked_extent = operand_shape[unblocked_axis]
    blocked_extent = operand_shape[blocked_axis]

    # Validate dimension requirements for NVFP4/MXFP8 tiling
    if unblocked_extent % tile_height != 0:
        raise ValueError(
            f"The extent at axis {unblocked_axis} must be a positive multiple of {tile_height} "
            f"(cuBLASLt uses {tile_height}x{tile_width} scale tiles), "
            f"got operand_shape[{unblocked_axis}] = {unblocked_extent}"
        )

    if blocked_extent % scalars_per_tile_row != 0:
        raise ValueError(
            f"The extent at axis {blocked_axis} must be a positive multiple of {scalars_per_tile_row} "
            f"(cuBLASLt uses {tile_height}x{tile_width} "
            f"scale tiles where each tile row covers {num_scalars_in_block} elements along blocked "
            f"dimension: {tile_width} groups * {num_scalars_in_block} elements in a group), "
            f"got operand_shape[{blocked_axis}] = {blocked_extent}"
        )
    return operand_shape, unblocked_axis, blocked_axis, block_scaling_format, num_scalars_in_block


def _scales_2d_matrix_tiled_layout(
    operand_shape: tuple[int, ...],
    unblocked_axis: int,
    blocked_axis: int,
    num_scalars_in_block: int,
    broadcast_block: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """
    Returns logical 2D shape for a matrix of scales and corresponding 5D (or 6D when
    broadcasting) tensor shape and strides that account for the tiling and interleaved
    layout of the tile.
    """

    tile_width = 4
    tile_col_shape = (4, 32)
    tile_height = 128  # prod(tile_y)

    unblocked_extent = operand_shape[unblocked_axis]
    blocked_extent = operand_shape[blocked_axis]

    num_blocks_in_a_row = blocked_extent // num_scalars_in_block
    num_tiles_in_a_row = num_blocks_in_a_row // tile_width
    num_tiles_in_col = unblocked_extent // tile_height

    # For operand of shape (H, W), the scale tensor is logically
    # a 2D matrix of shape (H, W // num_blocks_in_a_row) (if axis = -1)
    # or (H // num_blocks_in_a_row, W) (if axis = -2).
    # The scale matrix is tiled with tile of shape 128x4.
    # The tile has interleaved layout that splits column into
    # 4 groups of 32 elements, so actual tile shape is (4, 32, 4).
    # To account for the interleaved layout,
    # we need to split the 2D matrix into 5D tensor.
    shape_unblocked = num_tiles_in_col, tile_col_shape[0], tile_col_shape[1]
    squeezed_unblocked = unblocked_extent
    assert squeezed_unblocked == math.prod(shape_unblocked)
    shape_blocked = num_tiles_in_a_row, tile_width
    squeezed_blocked = num_blocks_in_a_row
    assert squeezed_blocked == math.prod(shape_blocked)
    # In total, we get 5D tensor with following dense strides:
    strides_unblocked = (
        512 * num_tiles_in_a_row,  # 1 * tile_size * num_tiles_in_a_row
        4,  # 1 * tile_width
        16,  # 1 * tile_width * tile_col_shape[0]
    )
    strides_blocked = (
        512,  # 1 * tile_width * tile_col_shape[0] * tile_col_shape[1] = tile_size
        1,  # fastest changing dim
    )
    # Broadcast every scale to the whole block if needed.
    if broadcast_block:
        shape_blocked = shape_blocked + (num_scalars_in_block,)  # type: ignore
        strides_blocked = strides_blocked + (0,)  # type: ignore
        squeezed_blocked = blocked_extent
        assert math.prod(shape_blocked) == blocked_extent
    # Finally, we combine the two parts into a 5D (or 6D when broadcasting) tensor.
    if blocked_axis == -1:
        shape = shape_unblocked + shape_blocked
        strides = strides_unblocked + strides_blocked  # type: ignore
        logical_shape = (squeezed_unblocked, squeezed_blocked)
    else:
        assert blocked_axis == -2
        shape = shape_blocked + shape_unblocked
        strides = strides_blocked + strides_unblocked
        logical_shape = (squeezed_blocked, squeezed_unblocked)
    return shape, strides, logical_shape


def _scales_nd_matrix_tiled_layout(
    operand_shape: tuple[int, ...],
    unblocked_axis: Literal[-1, -2],
    blocked_axis: Literal[-1, -2],
    num_scalars_in_block: int,
    broadcast_block: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """
    Returns logical 2D shape for a matrix of scales (or 3D to account for batching) and
    corresponding (1 if batching) + 5D (+ 1 if broadcasting) tensor shape and strides that
    account for the tiling and interleaved layout of the tile.
    """

    matrix_shape, matrix_strides, matrix_logical_shape = _scales_2d_matrix_tiled_layout(
        operand_shape, unblocked_axis, blocked_axis, num_scalars_in_block, broadcast_block
    )

    operand_ndim = len(operand_shape)
    if operand_ndim >= 2:
        batch_shape = operand_shape[:-2]
        batch_size = math.prod(batch_shape)
        batch_stride = math.prod(operand_shape[-2:]) // num_scalars_in_block
        matrix_shape = (batch_size, *matrix_shape)
        matrix_strides = (batch_stride, *matrix_strides)
        matrix_logical_shape = batch_shape + matrix_logical_shape

    return matrix_shape, matrix_strides, matrix_logical_shape


def _validate_tensor(x, where, tensor_name="tensor", dtype=None):
    """
    Validate the tensor package and dtype.

    Args:
        x: wrapped tensor object

        where: name of the function that is performing the validation

        tensor_name: name of the tensor to use in the error messages

        dtype: if not None, check that the object dtype matches the specified dtype

    """
    package = infer_object_package(x)
    if package != "torch":
        raise ValueError(
            f"Only torch.Tensor is currently supported by function '{where}'; the "
            f"specified {tensor_name} belongs to '{package}' package."
        )
    x = wrap_operand(x)
    if dtype is not None and x.dtype != dtype:
        raise ValueError(
            f"The function '{where}' requires the specified {tensor_name} to have dtype "
            f"'{dtype}', whereas it has dtype '{x.dtype}'."
        )
    return x


def _validate_mxfp8_scale(mx_scales, where, x=None):
    if x is not None:
        x = _validate_tensor(x, where)
    mx_scales = _validate_tensor(mx_scales, where, tensor_name="mx_scales tensor", dtype="uint8")

    if x is None:
        return mx_scales

    if mx_scales.shape != (x.size // 32,):
        raise ValueError(
            f"The shape of mx_scales {mx_scales.shape} is not compatible with a tensor of shape {x.shape}. "
            f"The expected mx_scales shape is {(x.size // 32,)}."
        )
    return mx_scales, x


def create_mxfp8_scale(x, exponent, stream=None):
    """
    .. experimental:: function

    Create MXFP8 block scale with the same value for the whole tensor ``x``.

    Args:
        x: The tensor to create the block scale for

        exponent: An integer from [-127, 128] range. Effective scale will be ``2^exponent``.

        stream: Optional stream to create the block scale on.
            Defaults to the stream of ``x``.

    Returns:
        An MXFP8 block scale factors tensor (1D, cuBLAS-compatible interleaved layout) to be
        used with MXFP8 computations.
    """
    x = _validate_tensor(x, "create_mxfp8_scale")

    if not -127 <= exponent <= 128:
        raise ValueError("The exponent should be an integer from [-127, 128] range.")

    stream_holder = None if x.device_id == "cpu" else get_or_create_stream(x.device_id, stream, x.name)
    mx_scales = create_empty_tensor(
        x.__class__, (x.size // 32,), "uint8", device_id=x.device_id, stream_holder=stream_holder, verify_strides=False
    )
    mx_scales.tensor[:] = exponent + 127
    return mx_scales.tensor


def invert_mxfp8_scale(mx_scales):
    """
    .. experimental:: function

    Compute a reciprocal of MXFP8 block scale.

    Args:
        mx_scales: MXFP8 block scale tensor (UE8M0 format).

    Returns:
        Tensor of the same shape as ``mx_scales`` with exponents replaced by the
        reciprocals.
    """
    _validate_mxfp8_scale(mx_scales, "invert_mxfp8_scale")

    ret = mx_scales.clone()
    ret[ret == 255] = 254  # Prevent the overflow
    # remove the bias (127), negate, add the bias back: -(scale - 127) + 127
    ret[:] = 254 - ret
    return ret


def _idx_batch_offset(
    operand_shape: tuple[int, ...], index: tuple[int, ...] | tuple[torch.Tensor, ...], num_scalars_in_block: int
) -> int | torch.Tensor:
    ndim = len(operand_shape)
    if ndim == 2:
        return 0

    assert ndim > 2
    batch_strides = _c_strides(operand_shape)[:-2]
    batch_index = index[:-2]
    batch_offset = sum(i * stride for i, stride in zip(batch_index, batch_strides, strict=True))  # type: ignore
    batch_offset //= num_scalars_in_block
    return batch_offset


def _get_block_scale_offset(
    operand_or_shape: torch.Tensor | tuple[int, ...],
    index: tuple[int, ...] | tuple[torch.Tensor, ...],
    axis: Literal[-1, -2] | None,
    block_scaling_format: BlockScalingFormat,
    is_block_idx: bool,
) -> int | torch.Tensor:
    # infer operand_shape, unblocked_axis, blocked_axis, num_scalars_in_block
    # validate operand_shape's dim and extents for divisibility by cuBLAS tile
    # and num_scalars_in_block
    operand_shape, unblocked_axis, blocked_axis, _, num_scalars_in_block = _validate_shape_axes_block_scaling_format(
        operand_or_shape, axis, block_scaling_format
    )

    # We need to validate index against the operand_shape
    ndim = len(operand_shape)
    if len(index) != ndim:
        raise ValueError("Index length must match the number of dimensions of the operand.")

    blocked_dim = operand_shape[blocked_axis]
    unblocked_idx = index[unblocked_axis]
    if is_block_idx:
        blocked_group_idx = index[blocked_axis]
    else:
        blocked_group_idx = index[blocked_axis] // num_scalars_in_block

    # Tile dimensions
    TILE_OUTER = 128
    TILE_INNER_GROUPS = 4

    # Determine tile coordinates
    sf_outer = unblocked_idx // TILE_OUTER
    sf_inner = (blocked_group_idx // TILE_INNER_GROUPS) * TILE_INNER_GROUPS
    sf_inner_dim = blocked_dim // num_scalars_in_block

    # Compute tile offset (global layout)
    # see https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    tile_offset = (sf_inner + sf_outer * sf_inner_dim) * 128

    # Compute intra-tile offset
    intra_outer = unblocked_idx % TILE_OUTER
    intra_inner = blocked_group_idx % TILE_INNER_GROUPS
    intra_offset = (intra_outer % 32) * 16 + (intra_outer // 32) * 4 + intra_inner

    batch_offset = _idx_batch_offset(operand_shape, index, num_scalars_in_block)

    return batch_offset + tile_offset + intra_offset


def _expand_block_scale(
    scales_1d: torch.Tensor,
    operand_or_shape: torch.Tensor | tuple[int, ...],
    block_scaling_format: BlockScalingFormat,
    axis: Literal[-1, -2] | None,
    device: Literal["cuda", "cpu"] | None = None,
) -> torch.Tensor:
    """
    For a documentation, see the public function :func:`expand_block_scale`.
    This one does most of the work but does not handle conversion of the expanded
    scales to output dtype, instead it returns scales as uint8.
    """
    import torch

    if not isinstance(scales_1d, torch.Tensor):
        raise TypeError(f"scales_1d must be a torch.Tensor, got {type(scales_1d)}")

    scales_wrapped = wrap_operand(scales_1d)
    scales_ndim = len(scales_wrapped.shape)

    if scales_ndim != 1:
        raise ValueError(f"scales_1d must be a 1D tensor, got {scales_ndim}D tensor with shape {scales_wrapped.shape}")

    if scales_wrapped.strides[0] != 1:
        raise ValueError(f"scales_1d must be 1D contiguous tensor, got non-unit stride: {scales_wrapped.strides}")

    operand_shape, unblocked_axis, blocked_axis, block_scaling_format, num_scalars_in_block = (
        _validate_shape_axes_block_scaling_format(operand_or_shape, axis, block_scaling_format)
    )

    _validate_scale_dtype_block_scaling_format_compatibility(scales_wrapped.dtype, block_scaling_format, "scales_1d")

    expected_num_scales = math.prod(operand_shape) // num_scalars_in_block
    num_scales = scales_wrapped.shape[0]
    if num_scales != expected_num_scales:
        raise ValueError(
            f"The scale tensor must have {expected_num_scales} elements: "
            f"{expected_num_scales} = math.prod({operand_shape}) // {num_scalars_in_block},"
            f" got: {num_scales}."
        )

    matrix_shape, matrix_strides, matrix_logical_shape = _scales_nd_matrix_tiled_layout(
        operand_shape, unblocked_axis, blocked_axis, num_scalars_in_block, True
    )

    if device is not None and device not in ("cuda", "cpu"):
        raise ValueError(f"device must be 'cuda', 'cpu', or None, got '{device}'")
    target_device = torch.device(device) if device is not None else scales_wrapped.tensor.device

    scales_on_device = scales_wrapped.tensor.to(target_device)
    scales_on_device = scales_on_device.view(torch.uint8)

    expanded = torch.as_strided(
        scales_on_device,
        size=matrix_shape,
        stride=matrix_strides,
    ).reshape(matrix_logical_shape)
    return expanded


def _convert_to_output_dtype(
    tensor: torch.Tensor,
    output_dtype: Literal["smallest"] | torch.dtype,
) -> torch.Tensor:
    import torch

    def _float_rank(dtype):
        if dtype == torch.float16:
            return 0
        elif dtype == torch.float32:
            return 1
        else:
            return 2

    if output_dtype == tensor.dtype:
        return tensor

    smallest_dtype = _smallest_dtype_that_fits(tensor)

    if output_dtype == "smallest":
        return tensor.type(smallest_dtype)
    else:
        if _float_rank(output_dtype) < _float_rank(smallest_dtype):
            raise ValueError(
                f"Result requires at least {smallest_dtype}; requested {output_dtype} would overflow or underflow."
            )
        return tensor.type(output_dtype)


def _convert_uint8_ue8m0_scale_to_float64(mx_scales: torch.Tensor) -> torch.Tensor:
    import torch

    assert mx_scales.dtype == torch.uint8
    # Use float64 for scale computation to be safe always,
    # this way we avoid overflow for corner case
    # when exponent is 128 (2^128 overflows float32)
    return 2 ** (mx_scales.type(torch.float64) - 127)


def get_mxfp8_scale_offset(
    operand_or_shape: torch.Tensor | tuple[int, ...],
    index: tuple[int, ...] | tuple[torch.Tensor, ...],
    axis: Literal[-1, -2] | None = None,
) -> int | torch.Tensor:
    """
    Computes offset of a scale in the 1D interleaved scales tensor,
    applied to element ``operand[index]``.

    .. deprecated:: 0.9.0
        Please use :func:`get_block_scale_offset` instead.
    """
    warnings.warn("get_mxfp8_scale_offset is deprecated. Please use get_block_scale_offset instead.", DeprecationWarning)
    return _get_block_scale_offset(operand_or_shape, index, axis, BlockScalingFormat.MXFP8, False)


def _smallest_dtype_that_fits(out_64):
    """
    (Private) Return the smallest torch dtype that can
    hold the values in out_64 without overflow or underflow.
    """
    import torch

    nonzero = out_64[out_64 != 0]
    if nonzero.numel() == 0:
        min_abs = 0
        max_abs = 0
    else:
        abs_vals = torch.abs(nonzero)
        min_abs = torch.min(abs_vals).item()
        max_abs = torch.max(abs_vals).item()

    finfo16 = torch.finfo(torch.float16)
    finfo32 = torch.finfo(torch.float32)

    if max_abs <= finfo16.max and (min_abs >= finfo16.tiny or min_abs == 0):
        return torch.float16
    if max_abs <= finfo32.max and (min_abs >= finfo32.tiny or min_abs == 0):
        return torch.float32
    return torch.float64


def apply_mxfp8_scale(
    x: torch.Tensor,
    scales_1d: torch.Tensor,
    output_dtype: Literal["smallest"] | torch.dtype = "smallest",
) -> torch.Tensor:
    """
    .. experimental:: function

    Apply MXFP8 block scale factors to a tensor ``x``.

    Args:
        x: The tensor to which the scaling should be applied.
            Currently it must be a ``torch.Tensor``.

        scales_1d: The block scale factors (stored in cuBLAS-compatible interleaved layout)
            to apply. Its shape must be compatible with ``x``, and currently it must also be
            a ``torch.Tensor``.

        output_dtype: Output dtype. If provided, must be a floating-point
            ``torch.dtype`` (float16, float32, or float64) and must be at least as wide as
            the smallest dtype that can represent the result, or :exc:`ValueError` is
            raised. If 'smallest' (default), the smallest dtype that can represent the
            result is automatically chosen.

    Returns:
        A tensor with values of ``x`` with scales applied, in the chosen or provided dtype.

    Raises:
        ValueError: When the result will over/underflow the requested dtype.

    Behavior:
        The operation is computed in float64. Then, the function determines the smallest
        dtype (float16, float32, or float64) that can represent the result without overflow
        or underflow. If ``output_dtype`` was passed, it must be at least as wide as that
        minimum otherwise :exc:`ValueError` is raised; if ``output_dtype='smallest'``, that
        minimum is used. The result is finally cast to the chosen dtype and returned.

    Note:
        This function is not intended for production usage due to its relatively low
        performance and high memory consumption. Prefer
        :attr:`~nvmath.linalg.advanced.MatmulOptions.result_type` to request non-FP8 output.
    """
    import torch

    if output_dtype != "smallest" and output_dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError("output_dtype must be 'smallest' or one of torch.float16, torch.float32, torch.float64.")

    # Use float64 for scale computation to be safe always,
    # this way we avoid overflow for corner case
    # when exponent is 128 (2^128 overflows float32)
    expanded_scales = _expand_block_scale(scales_1d, x, BlockScalingFormat.MXFP8, axis=None)
    expanded_scales = _convert_uint8_ue8m0_scale_to_float64(expanded_scales)
    # Explicitly cast x to float64, promotion is not guaranteed, see:
    # https://docs.pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc,
    # "Promotion for shell dtypes is not defined".
    out_64 = x.type(torch.float64) * expanded_scales
    return _convert_to_output_dtype(out_64, output_dtype)


# ====================================================
# helper functions for FP4 E2M1 encoding/decoding
# ====================================================


_FP4_DECODE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_FP4_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_FP4_MAG_CODES = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]


def _get_fp4_lookup_table(device):
    """
    (Private) Create FP4 lookup table as
    a torch tensor on the specified device.

    Args:
        device: The device to create the lookup table on.

    Returns:
        A torch tensor with dtype torch.float32 and shape (16,)
        on the specified device.
    """
    import torch

    return torch.tensor(
        _FP4_DECODE_VALUES,
        dtype=torch.float32,
        device=device,
    )


def _quantize_to_fp4_codes_bucketize(x: torch.Tensor, device) -> torch.Tensor:
    """
    Quantize float32 tensor to nearest FP4 value using torch.bucketize on the 8
    distinct FP4 magnitudes.
    """
    import torch

    if x.dtype != torch.float32:
        raise ValueError(f"x must be float32, got {x.dtype}")

    boundaries = torch.tensor(_FP4_BOUNDARIES, dtype=torch.float32, device=device)
    mag_codes = torch.tensor(_FP4_MAG_CODES, dtype=torch.uint8, device=device)

    # Example: x = [1.2, -2.3, 0.1]

    # sign: True where negative (including -0.0).  [False, True, False]
    sign = torch.signbit(x)

    # mag: absolute value.  [1.2, 2.3, 0.1]
    mag = x.abs()

    # Binary-search each magnitude into the boundary bins:
    #   https://docs.pytorch.org/docs/stable/generated/torch.bucketize.html
    #   boundaries = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    #   Default (right=False): boundary[i-1] < value <= boundary[i]
    #   1.2: 0.75 < 1.2 <= 1.25  -> bucket 2
    #   2.3: 1.75 < 2.3 <= 2.5   -> bucket 4
    #   0.1: 0.1  <= 0.25        -> bucket 0
    #   Result: [2, 4, 0]
    bucket = torch.bucketize(mag, boundaries)

    # Map bucket index to the 3-bit magnitude code (lower 3 bits of FP4).
    #   mag_codes = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]
    #   bucket [2, 4, 0] -> code [0x2, 0x4, 0x0]
    #   These represent magnitudes [1.0, 2.0, 0.0].
    code = mag_codes[bucket]

    # Set bit 3 (the FP4 sign bit) for negative values.
    #   sign.to(torch.uint8): [F, T, F] -> [0, 1, 0]
    #   << 3: shift into bit 3: [0x0, 0x8, 0x0]
    #   OR with magnitude code:
    #   code [0x2, 0x4, 0x0] | [0x0, 0x8, 0x0] = [0x2, 0xC, 0x0]
    #   Final FP4 values: [1.0, -2.0, 0.0]
    code = code | (sign.to(torch.uint8) << 3)
    return code


def quantize_to_fp4(
    x: torch.Tensor,
    axis: Literal[-1, -2],
) -> torch.Tensor:
    """
    .. experimental:: function

    Quantize a torch tensor to ``torch.float4_e2m1fn_x2`` dtype.

    The function supports 1D, 2D, and higher-dimensional input torch tensors with dtype
    float32. It quantizes each float32 value to the nearest representable FP4 value and
    packs two 4-bit codes per byte, halving the packed dimension. The packing direction is
    controlled by ``axis``:

    - ``axis=-1``: Packs consecutive elements along the last dimension. Input shape ``(...,
      Q)`` produces output ``(..., Q//2)`` with row-major layout (last stride = 1).

    - ``axis=-2``: Packs consecutive elements along the second-to-last dimension. Input
      shape ``(..., P, Q)`` produces output ``(..., P//2, Q)`` with column-major layout
      (second-to-last stride = 1).

    Args:
        x: Torch tensor with dtype float32 (1D, 2D, or higher-dimensional).

        axis: The axis along which to pack. Must be ``-1`` (last dimension)
            or ``-2`` (second-to-last dimension).

    Returns:
        Torch tensor with dtype ``torch.float4_e2m1fn_x2`` on the same device as the input.

    .. important::
        The packed dimension must have even size.

    Note:
        This helper quantizes a single tensor and is suitable for understanding how packing
        for ``torch.float4_e2m1fn_x2`` works in practice and/or for experimenting with FP4
        GEMMs outside of typical deep-learning workflows. It is not fully optimized for
        performance but should be adequate for most common use cases. For production
        whole-model quantization, consider tools such as `torchao
        <https://github.com/pytorch/ao>`_ or `bitsandbytes
        <https://github.com/bitsandbytes-foundation/bitsandbytes>`_.

    .. seealso::
        :func:`unpack_fp4` — decode packed FP4 values back to float32.
    """
    import torch

    # preconditions
    # -------------
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
    if x.dtype != torch.float32:
        raise ValueError(f"x must be float32, got {x.dtype}")
    if axis not in (-1, -2):
        raise ValueError(f"axis must be -1 or -2, got {axis}")
    if x.ndim < 1:
        raise ValueError(f"x must be at least 1D, got {x.ndim}D")
    if x.ndim == 1 and axis != -1:
        raise ValueError(f"axis must be -1 for 1D tensors, got {axis}")

    if (packed_dim := x.shape[axis]) % 2 != 0:
        raise ValueError(f"Packed dimension must be even, got {packed_dim}")

    device = x.device
    codes = _quantize_to_fp4_codes_bucketize(x, device)

    if axis == -1:
        low = codes[..., 0::2]
        high = codes[..., 1::2]
        packed = low | (high << 4)
        return packed.view(torch.float4_e2m1fn_x2)

    # axis == -2: column-wise packing
    low = codes[..., 0::2, :]
    high = codes[..., 1::2, :]
    packed_codes = low | (high << 4)
    # Transpose so columns become contiguous rows, force a physical copy,
    # reinterpret as fp4x2 (requires stride[-1]==1), then transpose back.
    # The result has the original (…, rows, cols) shape with column-major
    # memory layout, as requested by the caller.
    result = packed_codes.mT.contiguous().view(torch.float4_e2m1fn_x2).mT
    return result


def _decode_fp4_1d_tensor_to_float32(fp4_tensor: torch.Tensor) -> torch.Tensor:
    """
    (Private) Decode a 1D FP4 tensor to float32.

    Args:
        fp4_tensor: 1D FP4 tensor with shape (K//2,)

    Returns:
        Float32 tensor with unpacked shape (K,) and contiguous layout.
    """
    import torch

    assert fp4_tensor.ndim == 1, f"fp4_tensor must be 1D, got {fp4_tensor.ndim}D"
    assert fp4_tensor.dtype == torch.float4_e2m1fn_x2, f"fp4_tensor must be float4_e2m1fn_x2, got {fp4_tensor.dtype}"

    # Validate stride (must be contiguous)
    stride = fp4_tensor.stride()
    assert stride[0] == 1, f"1D FP4 tensor must be contiguous (stride=1), but got stride={stride}"

    # Create FP4 lookup table
    fp4_lookup = _get_fp4_lookup_table(fp4_tensor.device)

    # View as uint8 and extract 4-bit FP4 codes [0..15] from each byte.
    # low needs & 0xF to mask off bits [4..7]; high doesn't because
    # >> 4 on uint8 already zeros bits [4..7].
    fp4_as_uint8 = fp4_tensor.view(torch.uint8)
    low_code = (fp4_as_uint8 & 0xF).int()
    high_code = (fp4_as_uint8 >> 4).int()

    # Map codes to float values via lookup table
    vals_low = fp4_lookup[low_code]
    vals_high = fp4_lookup[high_code]

    # Interleave: (K//2,) -> (K,)
    # stack along last dim pairs [low_i, high_i], reshape flattens to
    # [low0, high0, low1, high1, ...].
    return torch.stack([vals_low, vals_high], dim=-1).reshape(-1)


def _decode_fp4_2d_plus_tensor_to_float32(fp4_tensor: torch.Tensor, row_wise_packing: bool) -> torch.Tensor:
    """
    (Private) Decode a 2-D or higher FP4 tensor to float32.

    Args:
        fp4_tensor: FP4 tensor with shape (..., M, K//2) for row-wise
            packing or (..., K//2, N) for column-wise packing, where ... represents zero or
            more batch dimensions.

        row_wise_packing: If True, uses row-wise packing (trailing
            dimension is packed). If False, uses column-wise packing (second-to-last
            dimension is packed). Determined by the caller from tensor strides.

    Returns:
        Float32 tensor with unpacked shape (..., M, K) for row-wise or (..., K, N) for
        column-wise packing.
    """
    import torch

    assert fp4_tensor.ndim >= 2, f"fp4_tensor must be at least 2D, got {fp4_tensor.ndim}D"
    assert fp4_tensor.dtype == torch.float4_e2m1fn_x2, f"fp4_tensor must be float4_e2m1fn_x2, got {fp4_tensor.dtype}"

    batch_shape = fp4_tensor.shape[:-2]
    device = fp4_tensor.device

    # Create FP4 lookup table
    fp4_lookup = _get_fp4_lookup_table(device)

    # View as uint8 and extract 4-bit values (vectorized across all dims).
    # low needs & 0xF to mask off bits [4..7]; high doesn't because
    # >> 4 on uint8 already zeros bits [4..7].
    fp4_as_uint8 = fp4_tensor.view(torch.uint8)
    low_code = (fp4_as_uint8 & 0xF).int()
    high_code = (fp4_as_uint8 >> 4).int()

    # Lookup decoded values (vectorized)
    vals_low = fp4_lookup[low_code]
    vals_high = fp4_lookup[high_code]

    if row_wise_packing:
        # Row-wise: (..., M, K//2) -> (..., M, K) with row-major layout
        # Expand last dimension by 2
        expanded_shape = batch_shape + (fp4_tensor.shape[-2], fp4_tensor.shape[-1] * 2)
        result = torch.zeros(expanded_shape, dtype=torch.float32, device=device)
        result[..., 0::2] = vals_low
        result[..., 1::2] = vals_high
    else:
        # Column-wise: (..., K//2, N) -> (..., K, N) with column-major layout
        # Expand second-to-last dimension by 2
        rows, cols = fp4_tensor.shape[-2:]
        expanded_shape = batch_shape + (rows * 2, cols)

        # Create column-major layout for batched tensors
        matrix_size = (rows * 2) * cols
        batch_size = int(np.prod(batch_shape)) if batch_shape else 1
        storage = torch.zeros(batch_size * matrix_size, dtype=torch.float32, device=device)
        # Stride: batch elements contiguous, then column-major per matrix
        output_stride = (matrix_size,) + (1, rows * 2)
        result = torch.as_strided(storage, size=(batch_size,) + (rows * 2, cols), stride=output_stride)
        # Flatten batch dims in vals_low/vals_high to match result shape
        vals_low_flat = vals_low.reshape(batch_size, rows, cols)
        vals_high_flat = vals_high.reshape(batch_size, rows, cols)
        result[:, 0::2, :] = vals_low_flat
        result[:, 1::2, :] = vals_high_flat
        result = result.reshape(expanded_shape)

    return result


def unpack_fp4(fp4_tensor: torch.Tensor, axis: Literal[-1, -2]) -> torch.Tensor:
    """
    .. experimental:: function

    Unpack an N-D torch tensor with dtype ``torch.float4_e2m1fn_x2`` to float32.

    Since each byte stores two FP4 values, the output tensor has one dimension doubled along
    ``axis``.

    - ``axis=-1``: The last dimension is the packed axis. Input shape ``(..., Q)`` with
      row-major layout produces output ``(..., 2*Q)``.
    - ``axis=-2``: The second-to-last dimension is the packed axis. Input shape ``(..., P,
      Q)`` with column-major layout produces output ``(..., 2*P, Q)``.

    Args:
        fp4_tensor: FP4 tensor with dtype ``torch.float4_e2m1fn_x2``.

        axis: The axis along which the tensor was packed. Must be ``-1``
            (last dimension) or ``-2`` (second-to-last dimension).

    Returns:
        A torch tensor with dtype float32 with the unpacked shape on the same device as the
        input.

    .. seealso::
        :func:`quantize_to_fp4` — quantize and pack float32 values to FP4.
    """
    import torch

    if not isinstance(fp4_tensor, torch.Tensor):
        raise TypeError(f"fp4_tensor must be a torch.Tensor, got {type(fp4_tensor)}")
    if fp4_tensor.ndim < 1:
        raise ValueError(f"fp4_tensor must be at least 1D, got {fp4_tensor.ndim}D")
    if fp4_tensor.dtype != torch.float4_e2m1fn_x2:
        raise ValueError(f"fp4_tensor must be float4_e2m1fn_x2, got {fp4_tensor.dtype}")
    if axis not in (-1, -2):
        raise ValueError(f"axis must be -1 or -2, got {axis}")
    if fp4_tensor.ndim == 1 and axis != -1:
        raise ValueError(f"axis must be -1 for 1D tensors, got {axis}")

    row_wise_packing = axis == -1

    if fp4_tensor.ndim == 1:
        return _decode_fp4_1d_tensor_to_float32(fp4_tensor)
    return _decode_fp4_2d_plus_tensor_to_float32(fp4_tensor, row_wise_packing)


def get_block_scale_offset(
    index: tuple[int, ...] | tuple[torch.Tensor, ...],
    operand_or_shape: torch.Tensor | tuple[int, ...],
    block_scaling_format: BlockScalingFormat,
    *,
    axis: Literal[-1, -2] | None = None,
) -> int | torch.Tensor:
    """
    .. experimental:: function

    Computes offset of a block scale factor in the 1D interleaved scales tensor.

    Matmul (cuBLAS) expects scale factors in specific `interleaved layout
    <https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout>`_.

    This function aims to abstract away the interleaved layout details, offering indexing
    that more directly corresponds to the operand's shape.

    Example:
        Suppose that you are doing an NVFP4 matmul ``a @ b`` with ``a`` of shape (M=128,
        K=128). For matrix ``a``, a single scale is applied to consecutive 16 elements
        blocks in a row (axis=-1). Therefore, to find the scale applied to ``a[y, x]``, we
        first need to adjust the x index to the index of the 16-element block it belongs to,
        which is ``block_idx = x // 16``. Then, calling: ``get_block_scale_offset((y,
        block_idx), a, BlockScalingFormat.NVFP4)`` will return the offset of the scale
        applied to ``a[y, x]`` (and all other elements in the same 16-element block).

        The schematic below shows matrix ``a`` with the 16-element blocks annotated.
        Asterisks mark two target blocks:

        - elements in ``a`` at indices from (5, 32) to (5, 47), correspond to the same block
          (K-group 2) and map to the same offset ``get_block_scale_offset((5, 2), a,
          BlockScalingFormat.NVFP4) == 82``
        - elements in ``a`` at indices from (5, 80) to (5, 95), correspond to the same block
          (K-group 5) and map to the same offset ``get_block_scale_offset((5, 5), a,
          BlockScalingFormat.NVFP4) == 593``

        .. code-block:: text

                  | K-grp 0  | K-grp 1  | K-grp 2  | K-grp 3  | K-grp 4  | K-grp 5  | ...
                  | [0..15]  | [16..31] | [32..47] | [48..63] | [64..79] | [80..95] | ...
                  +----------+----------+----------+----------+----------+----------+---
            row 0 |          |          |          |          |          |          |
             ...  |          |          |          |          |          |          |
            row 5 |          |          |    *     |          |          |    *     |
             ...  |          |          |          |          |          |          |
            row127|          |          |          |          |          |          |
                  +----------+----------+----------+----------+----------+----------+---
                                          (5,2)                            (5,5)

    Note:
        As far as computing the block scale offset, the only difference between MXFP8 and
        NVFP4 is the number of elements in a block (32 for MXFP8, 16 for NVFP4).

    Args:
        index: A tuple of indices with length equal to ``len(operand_shape)``. Can be:

            - A tuple of integers for single-element query, e.g., ``(10, 20)``
            - A tuple of tensors for batch query, e.g., ``(xs, ys)`` where ``xs`` and ``ys``
              are tensors of the same shape

        operand_or_shape: Operand tensor (that the scales apply to) or
            the operand's logical (non-packed, non-blocked) shape.

        block_scaling_format: The block scaling format of the operand:
            :attr:`BlockScalingFormat.NVFP4` or :attr:`BlockScalingFormat.MXFP8`.
            Internally, it is validated to be consistent with the operand dtype, and a
            :exc:`ValueError` is raised if not.

        axis: The blocked dimension of the operand tensor.
            For example, for NVFP4/MXFP8 matmul, A is blocked in rows (``axis = -1``), and B
            is blocked in columns (``axis = -2``). Depending on ``operand_or_shape``:

            - if a *shape* is passed to ``operand_or_shape``, then ``axis`` is required
            - if an *operand* is passed to ``operand_or_shape``, then ``axis`` can be
              omitted and the blocked dimension is inferred from the operand's layout.

    Returns:
        An integer (if ``index`` contains integers) or a tensor of integers (if ``index``
        contains tensors), indicating the offset(s) to the MXFP8/NVFP4 block scale
        factor(s). The returned offset points to a block scale factor that is applied to:

        - for axis == -2: ``operand[*index[-2:],
          block_size*index[-2]:block_size*(index[-2]+1), index[-1]]``.
        - for axis == -1: ``operand[*index[-2:], index[-1],
          block_size*index[-1]:block_size*(index[-1]+1)]``.

        where the block size is 32 for MXFP8 and 16 for NVFP4.

    Note:
        In typical use-cases, there should be no need to manually modify MXFP8 scales. The
        scales returned as ``"d_out_scale"`` by one matmul, can be directly reused as input
        scales for another matmul.

    Hint:
        - To apply the interleaved scales (e.g. as returned by matmul's ``d_out_scale``) to
          the operand, use :func:`apply_mxfp8_scale` instead.
        - To specify scales as ND tensor and copy them to cuBLAS-compatible interleaved
          layout, use :func:`to_block_scale` instead.

    """
    return _get_block_scale_offset(operand_or_shape, index, axis, block_scaling_format, True)


def to_block_scale(
    scale_tensor: torch.Tensor,
    operand_or_shape: torch.Tensor | tuple[int, ...],
    block_scaling_format: BlockScalingFormat,
    *,
    axis: Literal[-1, -2] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    .. experimental:: function

    Copy ND scale tensor to flat tensor accounting for the tiled layout required by
    cuBLASLt.

    Matmul (cuBLAS) expects scale factors in specific `interleaved layout
    <https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout>`_.

    This function aims to abstract away the interleaved layout details, offering a way to
    specify scales as ND tensor with shape corresponding to the operand's shape and copy
    them to cuBLAS-compatible interleaved layout.

    Example:
        Suppose that you are doing an NVFP4 matmul ``a @ b`` with ``a`` of shape (M=128,
        K=128). For matrix ``a``, a single scale is applied to consecutive 16 elements
        blocks in a row (axis=-1). You can specify the block scales as a ND tensor with
        shape ``(M, K // 16)`` such that scale from ``scale_tensor[i, j]`` will be applied
        to the block of elements ``a[i, j*16:j*16+16]`` and then call
        ``to_block_scale(scale_tensor, a, BlockScalingFormat.NVFP4)``, which will return a
        1D interleaved scale tensor that can be passed as quantization scales for the
        matmul.

    Note:
        As far as computing the block scale offset, the only difference between MXFP8 and
        NVFP4 is the number of elements in a block (32 for MXFP8, 16 for NVFP4).

    Args:
        scale_tensor: ND scale tensor with dtype:

            - for NVFP4: ``torch.float8_e4m3fn`` or ``torch.uint8`` (interpreted as
              ``torch.float8_e4m3fn``)
            - for MXFP8: ``torch.uint8`` (interpreted as ``UE8M0``)

        operand_or_shape: Operand tensor (that the scales apply to) or
            the operand's logical (non-packed, non-blocked) shape.

        block_scaling_format: The block scaling format of the operand:
            :attr:`BlockScalingFormat.NVFP4` or :attr:`BlockScalingFormat.MXFP8`.
            Internally, it is validated to be consistent with the operand dtype, and a
            :exc:`ValueError` is raised if not.

        axis: The blocked dimension of the operand tensor.
            For example, for NVFP4/MXFP8 matmul, A is blocked in rows (``axis = -1``), and B
            is blocked in columns (``axis = -2``). Depending on ``operand_or_shape``:

            - if a *shape* is passed to ``operand_or_shape``, then ``axis`` is required
            - if an *operand* is passed to ``operand_or_shape``, then ``axis`` can be
              omitted and the blocked dimension is inferred from the operand's layout.

        out: Output tensor to copy the scales to. If ``None``, a new tensor is created.

    Returns:
        Flat ``out`` tensor containing the scales copied to match cuBLAS-compatible
        interleaved layout. The `out` dtype is the same as the `scale_tensor` dtype.
    """
    import torch

    if not isinstance(scale_tensor, torch.Tensor):
        raise TypeError(f"scale_tensor must be a torch.Tensor, got {type(scale_tensor)}")

    scale_wrapped = wrap_operand(scale_tensor)
    num_scales = scale_wrapped.size

    operand_shape, unblocked_axis, blocked_axis, block_scaling_format, num_scalars_in_block = (
        _validate_shape_axes_block_scaling_format(operand_or_shape, axis, block_scaling_format)
    )

    _validate_scale_dtype_block_scaling_format_compatibility(scale_wrapped.dtype, block_scaling_format, "scale_tensor")

    num_scales = scale_wrapped.size
    expected_num_scales = math.prod(operand_shape) // num_scalars_in_block
    if num_scales != expected_num_scales:
        raise ValueError(
            f"For operand of shape {operand_shape}, and block_scaling_format {block_scaling_format}, "
            f"the scale_tensor must have shape {expected_num_scales}, got {num_scales} (shape:{scale_wrapped.shape})."
        )

    if out is None:
        out = torch.empty(
            num_scales,
            device=scale_wrapped.tensor.device,
            dtype=scale_wrapped.tensor.dtype,
        )
    else:
        if out.ndim != 1:
            raise ValueError(f"out must be a 1D tensor, got {out.ndim}D tensor with shape {out.shape}")
        if out.shape[0] != num_scales:
            raise ValueError(
                f"Flat scale tensor (out) and ND scale_tensor must have the same number "
                f"of elements, got {out.shape[0]} and {num_scales}"
            )
        if out.dtype != scale_wrapped.tensor.dtype:
            raise ValueError(
                f"Flat scale tensor (out) and ND scale_tensor must "
                f"have the same dtype, got {out.dtype} and {scale_wrapped.dtype}"
            )

    matrix_shape, matrix_strides, matrix_logical_shape = _scales_nd_matrix_tiled_layout(
        operand_shape, unblocked_axis, blocked_axis, num_scalars_in_block, False
    )

    if scale_wrapped.shape[-2:] != matrix_logical_shape[-2:]:
        expected_shape = scale_wrapped.shape[:-2] + matrix_logical_shape[-2:]  # type: ignore
        raise ValueError(
            f"For operand of shape {operand_shape}, block_scaling_format {block_scaling_format}, "
            f"blocked along axis {axis}, "
            f"the scale_tensor must have shape {expected_shape}, got {scale_wrapped.shape}."
        )

    torch.as_strided(
        out,
        size=matrix_shape,
        stride=matrix_strides,
    ).copy_(scale_wrapped.tensor.view(matrix_shape))
    return out


def expand_block_scale(
    scales_1d: torch.Tensor,
    operand_or_shape: torch.Tensor | tuple[int, ...],
    block_scaling_format: BlockScalingFormat,
    *,
    axis: Literal[-1, -2] | None = None,
    output_dtype: Literal["smallest"] | torch.dtype = "smallest",
    device: Literal["cuda", "cpu"] | None = None,
) -> torch.Tensor:
    """
    .. experimental:: function

    Expand NVFP4/MXFP8 block scales from 1D cuBLAS-compatible interleaved array to the full
    operand shape.

    Matmul (cuBLAS) expects and returns the block scale factors in specific `interleaved
    layout
    <https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout>`_.

    This function takes that 1D interleaved scale array (either provided as input or
    returned by cuBLASLt for NVFP4/MXFP8 output) and expands it to a full ND tensor with
    shape ``operand_or_shape`` where each element gets its corresponding scale value. This
    can be useful, for example, to manually dequantize the result of a matmul, by
    elementwise multiplication of the expanded scales with the result.

    Args:
        scales_1d: 1D tensor of scale values with dtype:

            - for NVFP4: ``torch.float8_e4m3fn``, or ``torch.uint8`` (interpreted as
              ``torch.float8_e4m3fn``)
            - for MXFP8: ``torch.uint8``, interpreted as exponent (``UE8M0``)

            The scales are expected to be stored in cuBLAS-compatible interleaved layout
            (e.g. as returned by matmul's ``d_out_scale``). The number of elements in the
            tensor must be equal to the number of elements in the operand tensor, divided by
            the number of elements in a block (for NVFP4: 16, for MXFP8: 32).

        operand_or_shape: Operand tensor or its logical (non-packed, non-blocked) shape.
            The scales are expanded to match this shape.

        block_scaling_format: The block scaling format of the operand:
            :attr:`BlockScalingFormat.NVFP4` or :attr:`BlockScalingFormat.MXFP8`.
            Internally, it is validated to be consistent with the operand dtype, and a
            :exc:`ValueError` is raised if not.

        axis: The blocked dimension of the operand tensor.
            For example, for NVFP4/MXFP8 matmul, A is blocked in rows (``axis = -1``), and B
            is blocked in columns (``axis = -2``). Depending on ``operand_or_shape``:

            - if a *shape* is passed to ``operand_or_shape``, then ``axis`` is required
            - if an *operand* is passed to ``operand_or_shape``, then ``axis`` can be
              omitted and the blocked dimension is inferred from the operand's layout.

        output_dtype: Output dtype.
            If provided, must be a torch's dtype:

            - for NVFP4: ``float8_e4m3fn``, ``float16``, ``float32``, or ``float64``
            - for MXFP8: ``uint8`` (exponent ``UE8M0``), ``float16``, ``float32``, or
              ``float64``

            It must be wide enough to represent the result, or :exc:`ValueError` is raised.
            If 'smallest' (default), the smallest of accepted dtypes that can represent the
            result is automatically chosen (for MXFP8: ``uint8`` interpreted as exponent
            (``UE8M0``), for NVFP4: ``float8_e4m3fn``).

        device: Device for the output tensor. When ``None`` (default), the
            device is inferred from ``scales_1d``. When specified, must be ``"cuda"`` or
            ``"cpu"``.

    Returns:
        Tensor with shape ``operand_or_shape`` (and dtype as specified by ``output_dtype``)
        containing expanded scales, on the target device. Each element contains the scale
        value that applies to the corresponding position in the FP4/FP8 matrix.

    Note:
        For computing a single scale index rather than expanding all scales, use
        :func:`get_block_scale_offset` instead.
    """
    import torch

    _COMMON_EXPAND_OUTPUT_DTYPES = (torch.float16, torch.float32, torch.float64)

    expanded = _expand_block_scale(scales_1d, operand_or_shape, block_scaling_format, axis, device)
    assert expanded.dtype == torch.uint8

    scale_interpretation = _MICROSCALING_FORMAT_PROPERTIES[block_scaling_format]["scale_interpretation"]

    if scale_interpretation == "float8_e4m3fn":
        expanded = expanded.view(torch.float8_e4m3fn)
        if output_dtype == "smallest" or output_dtype == torch.float8_e4m3fn:
            return expanded
        elif output_dtype in _COMMON_EXPAND_OUTPUT_DTYPES:
            return expanded.type(output_dtype)
        else:
            supported_dtypes_str = ", ".join([str(dt) for dt in (torch.float8_e4m3fn,) + _COMMON_EXPAND_OUTPUT_DTYPES])
            raise TypeError(f"output_dtype must be 'smallest' or one of {supported_dtypes_str}. Got {output_dtype}")
    elif scale_interpretation == "ue8m0":
        if output_dtype == "smallest" or output_dtype == torch.uint8:
            return expanded
        elif output_dtype in _COMMON_EXPAND_OUTPUT_DTYPES:
            # TODO: This could be optimized - we can assess the exponents
            # keeping uint8 type and directly convert to output dtype.
            expanded = _convert_uint8_ue8m0_scale_to_float64(expanded)
            return _convert_to_output_dtype(expanded, output_dtype)
        else:
            supported_dtypes_str = ", ".join([str(dt) for dt in (torch.uint8,) + _COMMON_EXPAND_OUTPUT_DTYPES])
            raise TypeError(f"output_dtype must be 'smallest' or one of {supported_dtypes_str}. Got {output_dtype}")
    else:
        raise AssertionError(
            f"Unknown scale_interpretation '{scale_interpretation}' for block_scaling_format '{block_scaling_format}'"
        )
