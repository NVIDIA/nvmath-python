# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use distributed Torch tensor objects.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["TorchDistributedTensor"]

import math

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from collections.abc import Sequence

import nvmath.distributed
from nvmath.internal.tensor_ifc_torch import TorchTensor
from nvmath.internal.utils import device_ctx
from nvmath.bindings import nvshmem  # type: ignore
from nvmath.distributed._internal.nvshmem import nvshmem_empty_dlpack


# Most methods aren't redefined, because they simply act on the local array
class TorchDistributedTensor(TorchTensor):
    """
    TensorHolder for distributed torch tensors.
    """

    def __init__(self, tensor):
        super().__init__(tensor)
        if tensor.device.index is not None and nvshmem.ptr(tensor.data_ptr(), nvshmem.my_pe()) == 0:
            raise TypeError(
                "Operand must be on the symmetric heap. Consider allocating it "
                "with nvmath.distributed.allocate_symmetric_memory()."
            )

    @classmethod
    def empty(cls, shape, device_id="cpu", *, dtype="float32", strides=None, **context) -> TorchDistributedTensor:
        """
        Create an empty tensor of the specified shape and data type on the specified device
        (None, 'cpu', or device id).

        Note, that the strides, if specified, should correspond to a dense
        (possibly permuted) tensor and MUST NOT overlap.
        Otherwise, the behaviour is not defined.
        """
        if device_id == "cpu":
            return super().empty(shape, device_id, dtype=dtype, strides=strides)

        dtype = TorchTensor.name_to_dtype[dtype]

        ctx = nvmath.distributed.get_context()
        assert ctx is not None, "nvmath.distributed has not been initialized"

        make_symmetric = context.get("make_symmetric", False)
        logger = context.get("logger")

        with device_ctx(device_id):
            size = math.prod(shape, start=dtype.itemsize)
            # TODO: ideally strides should be set in DLPack, but cuda.core doesn't support
            # ndarray yet and instead returns a flat buffer.
            dlpack_buf = nvshmem_empty_dlpack(size, device_id, ctx.communicator, make_symmetric=make_symmetric, logger=logger)
            tensor = torch.from_dlpack(dlpack_buf)
            # Buffer may be padded if make_symmetric=True.
            tensor = tensor[:size]
            if strides is None:
                tensor = tensor.view(dtype).view(shape)
            else:
                tensor = torch.as_strided(tensor.view(dtype), shape, strides)

        return cls(tensor)

    def to(self, device_id, stream_holder) -> TorchDistributedTensor:
        """
        In addition to the base class semantics:
          - Source or target device must be the one used to initialize NVSHMEM on this
            process. This implies that copy from one CUDA device to another is not allowed.
          - Memory layout is preserved.
          - Strides must be non-overlapping.
        """
        if not (device_id == "cpu" or isinstance(device_id, int)):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")

        if device_id == "cpu" or self.device_id == device_id:
            with stream_holder.ctx:
                tensor = self.tensor.to(device=device_id, non_blocking=(device_id != "cpu"))
            return TorchDistributedTensor(tensor)

        if self.device_id != "cpu" and self.device_id != device_id:
            raise ValueError("Cannot copy distributed tensor to a different CUDA device")

        with stream_holder.ctx:
            tensor_device = TorchDistributedTensor.empty(
                self.shape, device_id=device_id, dtype=self.dtype, strides=self.strides, make_symmetric=True
            )
            tensor_device.tensor.copy_(self.tensor, non_blocking=True)
            return tensor_device

    def reshape(self, shape: Sequence[int], *, copy: bool | None = None) -> TorchDistributedTensor:
        if copy and self.device_id != "cpu":
            raise NotImplementedError("reshape with copy=True is not supported for TorchDistributedTensor on GPU")
        return super().reshape(shape, copy=copy)
