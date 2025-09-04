# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use distributed Numpy ndarray objects.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["NumpyDistributedTensor", "CudaDistributedTensor"]

from nvmath.internal.tensor_ifc_numpy import CudaTensor, NumpyTensor

from nvmath.internal.ndbuffer import ndbuffer

from .tensor_ifc import DistributedTensor
from .tensor_ifc_host_device import CudaDistributedTensorMixIn, HostDistributedTensorMixIn


class CudaDistributedTensor(CudaDistributedTensorMixIn, CudaTensor, DistributedTensor):
    """
    Tensor wrapper for distributed cuda ndarrays.
    """

    host_tensor_class: type[NumpyDistributedTensor]  # set once NumpyDistributedTensor is defined

    @classmethod
    def wrap_ndbuffer(cls, ndbuffer: ndbuffer.NDBuffer) -> CudaDistributedTensor:
        return cls(ndbuffer)


# Most methods aren't redefined, because they simply act on the local array
class NumpyDistributedTensor(HostDistributedTensorMixIn, NumpyTensor, DistributedTensor):
    """
    Tensor wrapper for distributed numpy ndarrays.
    """

    device_tensor_class = CudaDistributedTensor


CudaDistributedTensor.host_tensor_class = NumpyDistributedTensor
