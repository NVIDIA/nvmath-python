# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

__all__ = ["allocate_symmetric_memory", "free_symmetric_memory"]

from logging import Logger
from collections.abc import Iterable, Sequence
from types import ModuleType
from typing import Literal

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import numpy as np
from numpy.typing import DTypeLike

import nvmath.distributed
from nvmath.internal.utils import device_ctx
from nvmath.distributed._internal import tensor_wrapper


# Supported packages for tensors backed by symmetric memory.
_SUPPORTED_PACKAGES = ("cupy", "torch")


# TODO: do a general refactor of this function in nvmath
def calculate_strides(shape: Sequence[int], axis_order: Iterable[int]):
    """
    Calculate the strides for the provided shape and axis order.
    """
    strides = [0] * len(shape)

    stride = 1
    for axis in axis_order:
        strides[axis] = stride
        stride *= shape[axis]

    return strides


def allocate_symmetric_memory(
    shape: int | Sequence[int],
    package: ModuleType,
    *,
    dtype: DTypeLike | torch.dtype | None = None,
    axis_order: Literal["C", "F"] | Sequence[int] = "C",
    make_symmetric: bool = False,
    skip_symmetric_check: bool = False,
    logger: Logger | None = None,
):
    """Return uninitialized tensor of given shape and type, allocated from the symmetric
    heap, on the device on which nvmath.distributed was initialized. The tensor type is
    determined by the provided package (e.g. cupy, torch).
    **This is a collective operation and must be called by all processes**.
    Note that the buffer size must be the same on all processes, or you can
    use ``make_symmetric=True`` to pad all buffers to the same max size.

    Args:
        shape: Shape of the tensor to allocate.

        package: Python package determining the tensor type (e.g. cupy, torch).

        dtype: Tensor dtype in a form recognized by the package. If None, will use the
            package's default dtype.

        axis_order: Axis order. The default is 'C' (row-major ordering).

        make_symmetric: If buffer sizes do not match across processes, will allocate
            the maximum size on every process to ensure the allocation is symmetric.
            The default is False.

        skip_symmetric_check: Skip checking that the allocation is symmetric (which
            requires inter-process communication). The default is False.

        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.
    """

    if package.__name__ not in _SUPPORTED_PACKAGES:
        raise ValueError(f"The package must be one of {_SUPPORTED_PACKAGES}. Got {package}.")

    distributed_ctx = nvmath.distributed.get_context()
    if distributed_ctx is None:
        raise RuntimeError(
            "nvmath.distributed has not been initialized. Refer to "
            "https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html#initializing-the-distributed-runtime"
            " for more information."
        )

    device_id = distributed_ctx.device_id

    if isinstance(shape, int):
        shape = (shape,)

    strides = None
    if axis_order is not None:
        if axis_order == "C":
            strides = calculate_strides(shape, reversed(range(len(shape))))
        elif axis_order == "F":
            strides = calculate_strides(shape, range(len(shape)))
        else:
            strides = calculate_strides(shape, axis_order)

    if package.__name__ == "cupy":
        from ._internal.tensor_ifc_cupy import CupyDistributedTensor

        if dtype is None:
            # This mimics numpy and cupy
            dtype = np.float64

        dtype = np.dtype(dtype).name  # type: ignore
        return CupyDistributedTensor.empty(
            shape,
            dtype=dtype,
            device_id=device_id,
            strides=strides,
            symmetric_memory=True,
            make_symmetric=make_symmetric,
            skip_symmetric_check=skip_symmetric_check,
            logger=logger,
        ).tensor
    elif package.__name__ == "torch":
        from ._internal.tensor_ifc_torch import TorchDistributedTensor

        if dtype is None:
            import torch

            dtype = torch.get_default_dtype()

        dtype = str(dtype).split(".")[1]
        return TorchDistributedTensor.empty(
            shape,
            dtype=dtype,
            device_id=device_id,
            strides=strides,
            symmetric_memory=True,
            make_symmetric=make_symmetric,
            skip_symmetric_check=skip_symmetric_check,
            logger=logger,
        ).tensor


def free_symmetric_memory(*tensors) -> None:
    """Frees tensors' data buffer where the buffer was allocated on the symmetric heap.
    Note that this is only meant to be called on tensors returned by
    ``allocate_symmetric_memory()``.

    **This is a collective operation and must be called by all processes, with tensors
    in the same order**."""

    device_id = tensor_wrapper.wrap_operand(tensors[0]).device_id
    if device_id == "cpu":
        raise TypeError("free_symmetric_memory called on CPU array/tensor")

    with device_ctx(device_id):
        for tensor in tensors:
            wrapped_tensor = tensor_wrapper.wrap_operand(tensor)
            if wrapped_tensor.device_id == "cpu":
                raise TypeError("free_symmetric_memory called on CPU array/tensor")

            assert wrapped_tensor.device_id == device_id, "Internal error: symmetric memory tensors are not on the same device"

            wrapped_tensor.free_symmetric()
