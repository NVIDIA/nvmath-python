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

from ._internal.nvshmem import free as nvshmem_free_wrapper

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
    axis_order: Literal["C", "F"] | Sequence[int] | None = None,
    make_symmetric: bool = False,
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

        dtype: Tensor dtype.

        axis_order: Axis order.

        make_symmetric: If buffer sizes do not match across processes, will allocate
            the maximum size on every process to ensure the allocation is symmetric.

        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.
    """

    if package.__name__ not in _SUPPORTED_PACKAGES:
        raise ValueError(f"The package must be one of {_SUPPORTED_PACKAGES}. Got {package}.")

    distributed_ctx = nvmath.distributed.get_context()
    if distributed_ctx is None:
        raise RuntimeError("nvmath.distributed has not been initialized")

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
            shape, dtype=dtype, device_id=device_id, strides=strides, make_symmetric=make_symmetric, logger=logger
        ).tensor
    elif package.__name__ == "torch":
        from ._internal.tensor_ifc_torch import TorchDistributedTensor

        if dtype is None:
            import torch

            dtype = torch.get_default_dtype()

        dtype = str(dtype).split(".")[1]
        return TorchDistributedTensor.empty(
            shape, dtype=dtype, device_id=device_id, strides=strides, make_symmetric=make_symmetric, logger=logger
        ).tensor


def free_symmetric_memory(*tensors) -> None:
    """Frees tensors' data buffer where the buffer was allocated on the symmetric heap.
    Note that this is only meant to be called on tensors returned by
    ``allocate_symmetric_memory()``.

    **This is a collective operation and must be called by all processes, with tensors
    in the same order**."""
    for tensor in tensors:
        package = _get_tensor_package(tensor)
        if package not in _SUPPORTED_PACKAGES:
            raise ValueError(
                f"The tensor package must be one of {_SUPPORTED_PACKAGES}. Got {type(tensor)} from package {package}."
            )

    for tensor in tensors:
        wrapped_tensor = tensor_wrapper.wrap_operand(tensor)
        if not isinstance(wrapped_tensor.device_id, int):
            raise ValueError("Tensor must be on GPU symmetric memory")
        with device_ctx(wrapped_tensor.device_id):
            nvshmem_free_wrapper(wrapped_tensor.data_ptr)


def _get_tensor_package(tensor):
    if issubclass(tensor.__class__, np.ndarray):
        return "numpy"
    module = tensor.__class__.__module__
    package = module.split(".")[0]
    return package
