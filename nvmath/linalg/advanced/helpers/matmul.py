# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath.internal.tensor_wrapper import wrap_operand
from nvmath.internal.utils import create_empty_tensor, infer_object_package, get_or_create_stream

__all__ = ["create_mxfp8_scale", "invert_mxfp8_scale", "get_mxfp8_scale_offset", "apply_mxfp8_scale"]


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


def _validate_mxfp8_scale(scale, where, x=None):
    if x is not None:
        _validate_tensor(x, where)
        x = wrap_operand(x)
    _validate_tensor(scale, where, tensor_name="scale tensor", dtype="uint8")
    scale = wrap_operand(scale)

    if x is not None and scale.shape != (x.size // 32,):
        raise ValueError(
            f"The shape of scale {scale.shape} is not compatible with a tensor of shape {x.shape}. "
            f"The expected scale shape is {(x.size // 32,)}."
        )
    return scale


def create_mxfp8_scale(x, exponent, stream=None):
    """
    Create MXFP8 block scale with the same value for the whole tensor ``x``.

    Args:
        x: The tensor to create the block scale for

        exponent: An integer from [-127, 128] range. Effective scale will be ``2^exponent``.

        stream: Optional stream to create the block scale on.
            Defaults to the stream of ``x``.

    Returns:
        An MXFP8 block scale factors tensor to be used with MXFP8 computations.
    """
    _validate_tensor(x, "create_mxfp8_scale")
    x = wrap_operand(x)

    if not -127 <= exponent <= 128:
        raise ValueError("The exponent should be an integer from [-127, 128] range.")

    stream_holder = None if x.device_id == "cpu" else get_or_create_stream(x.device_id, stream, x.name)
    scale = create_empty_tensor(
        x.__class__, (x.size // 32,), "uint8", device_id=x.device_id, stream_holder=stream_holder, verify_strides=False
    )
    scale.tensor[:] = exponent + 127
    return scale.tensor


def invert_mxfp8_scale(scale):
    """
    Compute a reciprocal of MXFP8 block scale.

    Args:
        scale: MXFP8 block scale tensor.

    Returns:
        An MXFP8 block scale factors tensor with reciprocals of the values in ``scale``.
    """
    _validate_mxfp8_scale(scale, "invert_mxfp8_scale")

    scale[scale == 255] = 254  # Prevent the overflow
    return (127 + 127) - scale


def get_mxfp8_scale_offset(x, index):
    """
    Computes the offset of MXFP8 scale used for element ``x[index]``.

    Args:
        x: The tensor to which ``index`` refers.

        index: A tuple of tensor indices. This function supports broadcasting,
            so the `index` can be a tuple of integers or a tuple of tensors.

    Returns:
        A single integer indicating an offset to the MXFP8 block scale factor which
        is applied to ``x[index]`` during scaling.

    Note:
        In typical use-cases, there should be no need to manually modify MXFP8 scales.
        The scales returned as ``"d_out_scale"`` by one multiplication can be directly
        reused as input scales for another multiplication.
    """

    _validate_tensor(x, where="get_mxfp8_scale_offset")
    x = wrap_operand(x)
    ndim = len(x.shape)
    if len(index) != ndim:
        raise ValueError("Index length should match the number of dimensions of x.")

    if ndim == 2:
        batch_offset = 0
    elif ndim > 2:
        # Compute batch offset
        batch_strides = x.strides[:-2]
        batch_index = index[:-2]
        batch_offset = sum(i * stride for i, stride in zip(batch_index, batch_strides, strict=True)) // min(batch_strides)
    else:
        raise ValueError(f"Got {ndim}-D tensor in `get_mxfp8_scale_offset`, but expected at least 2-D.")
    major_d, minor_d = (-2, -1) if x.strides[-2] > x.strides[-1] else (-1, -2)
    major, minor, minor_length = index[major_d], index[minor_d], x.shape[minor_d]

    # Compute tile offset
    tile_minor = minor // 128
    tile_major = major // 128
    tile_offset = (minor_length // 128) * tile_major + tile_minor
    minor = minor % 128
    major = major % 128

    # Compute offset in the tile
    minor = minor // 32
    offset = (major % 32) * 16 + (major // 32) * 4 + minor

    # Add the offsets together
    tile_size = 128 * 128 // 32
    matrix_size = x.shape[-1] * x.shape[-2] // 32
    return batch_offset * matrix_size + tile_offset * tile_size + offset


def apply_mxfp8_scale(x, scale):
    """
    Apply MXFP8 block scale factors to tensor ``x``.

    Args:
        x: The tensor to which the scaling should be applied.

        scale: The block scale factors to apply.

    Returns:
        A ``float32`` tensor with values of ``x`` with scales applied.

    Note:
        This function is not intended for production usage due to its relatively low
        performance and high memory consumption. Instead of applying the scales
        manually using this function, use
        :attr:`~nvmath.linalg.advanced.MatmulOptions.result_type` to request non-FP8 output.
    """
    scale = _validate_mxfp8_scale(scale, "apply_mxfp8_scale", x=x)
    import torch

    idx = get_mxfp8_scale_offset(x, torch.meshgrid(*(torch.arange(d) for d in x.shape), indexing="ij")).to(x.device)

    actual_scale = 2 ** (scale.tensor.type(torch.float32)[idx] - 127)
    return x.type(torch.float32) * actual_scale
