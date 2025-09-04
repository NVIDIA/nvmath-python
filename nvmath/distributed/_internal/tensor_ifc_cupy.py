# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use distributed Cupy ndarray objects.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["CupyDistributedTensor", "HostDistributedTensor"]


try:
    import cupy
except ImportError:
    cupy = None


from nvmath.internal.tensor_ifc_cupy import HostTensor, CupyTensor
from nvmath.internal.ndbuffer import ndbuffer

from .tensor_ifc import DistributedTensor
from .tensor_ifc_host_device import HostDistributedTensorMixIn, CudaDistributedTensorMixIn


class HostDistributedTensor(HostDistributedTensorMixIn, HostTensor, DistributedTensor):
    device_tensor_class: type[CupyDistributedTensor]  # set once CupyDistributedTensor is defined


# Most methods aren't redefined, because they simply act on the local array
class CupyDistributedTensor(CudaDistributedTensorMixIn, CupyTensor, DistributedTensor):
    """
    Tensor wrapper for distributed cupy ndarrays.
    """

    host_tensor_class = HostDistributedTensor

    @classmethod
    def wrap_ndbuffer(cls, ndbuffer: ndbuffer.NDBuffer) -> CupyDistributedTensor:
        """
        Wraps NDBuffer into a cupy.ndarray, the method assumes the
        NDBuffer is backed by CUDA device memory.
        """
        mem = cupy.cuda.UnownedMemory(
            ndbuffer.data_ptr,
            ndbuffer.size_in_bytes,
            owner=ndbuffer.data,
            device_id=ndbuffer.device_id,
        )
        memptr = cupy.cuda.MemoryPointer(mem, offset=0)
        dtype = cls.name_to_dtype[ndbuffer.dtype_name]
        tensor = cupy.ndarray(
            ndbuffer.shape,
            dtype=dtype,
            strides=ndbuffer.strides_in_bytes,
            memptr=memptr,
        )
        return cls(tensor)


HostDistributedTensor.device_tensor_class = CupyDistributedTensor
