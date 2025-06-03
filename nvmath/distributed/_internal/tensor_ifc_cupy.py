# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use distributed Cupy ndarray objects.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["CupyDistributedTensor"]

import math

try:
    import cupy
except ImportError:
    cupy = None

from collections.abc import Sequence

import nvmath.distributed
from nvmath.internal.tensor_ifc_cupy import CupyTensor
from nvmath.internal.utils import device_ctx
from nvmath.bindings import nvshmem  # type: ignore
from nvmath.distributed._internal.nvshmem import nvshmem_empty_dlpack

from .tensor_ifc_numpy import NumpyDistributedTensor


# Most methods aren't redefined, because they simply act on the local array
class CupyDistributedTensor(CupyTensor):
    """
    Tensor wrapper for distributed cupy ndarrays.
    """

    def __init__(self, tensor):
        super().__init__(tensor)
        if nvshmem.ptr(tensor.data.ptr, nvshmem.my_pe()) == 0:
            raise TypeError(
                "Operand must be on the symmetric heap. Consider allocating it "
                "with nvmath.distributed.allocate_symmetric_memory()."
            )

    @classmethod
    def empty(cls, shape, device_id="cpu", *, dtype="float32", strides=None, **context) -> CupyDistributedTensor:
        """
        Create an empty tensor of the specified shape and data type.

        Note, that the strides, if specified, MUST correspond to a dense (possibly permuted)
        tensor, otherwise the created tensor may be corrupted.
        """
        dtype = CupyTensor.name_to_dtype[dtype]

        from nvmath.distributed._utils import calculate_strides

        ctx = nvmath.distributed.get_context()
        assert ctx is not None, "nvmath.distributed has not been initialized"

        make_symmetric = context.get("make_symmetric", False)
        logger = context.get("logger")

        order = "C"
        if strides is not None:
            if list(strides) == calculate_strides(shape, reversed(range(len(shape)))):
                order = "C"
            elif list(strides) == calculate_strides(shape, range(len(shape))):
                order = "F"
            else:
                raise ValueError("CupyDistributedTensor.empty() only supports 'C' or 'F' order")

        with device_ctx(device_id):
            # TODO: ideally strides should be set in DLPack, but cuda.core doesn't support
            # ndarray yet and instead returns a flat buffer.
            size = math.prod(shape, start=dtype.itemsize)
            dlpack_buf = nvshmem_empty_dlpack(size, device_id, ctx.communicator, make_symmetric=make_symmetric, logger=logger)
            tensor = cupy.from_dlpack(dlpack_buf)
            # Buffer may be padded if make_symmetric=True.
            tensor = tensor[:size].view(dtype).reshape(shape, order=order)
            # assert tensor is not a copy
            assert tensor.base is not None

        return cls(tensor)

    def to(self, device_id, stream_holder) -> NumpyDistributedTensor | CupyDistributedTensor:
        """
        In addition to the base class semantics:
          - Source or target device must be the one used to initialize NVSHMEM on this
            process. This implies that copy from one CUDA device to another is not allowed.
          - Memory layout is preserved.
          - Strides must be dense non-overlapping.
        """
        if not (device_id == "cpu" or isinstance(device_id, int)):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")

        if device_id == "cpu":
            # NOTE: not using self.numpy() because it doesn't preserve memory layout.
            np_tensor = NumpyDistributedTensor.empty(self.shape, dtype=self.dtype, strides=self.strides)
            np_tensor.copy_(self, stream_holder)
            return np_tensor

        if device_id != self.device_id:
            raise ValueError("Cannot copy distributed tensor to a different CUDA device")

        with device_ctx(device_id), stream_holder.ctx:
            return CupyDistributedTensor(cupy.asarray(self.tensor))

    def reshape(self, shape: Sequence[int], *, copy: bool | None = None) -> CupyDistributedTensor:
        if copy:
            raise NotImplementedError("reshape with copy=True is not supported for CupyDistributedTensor")
        return super().reshape(shape, copy=copy)
