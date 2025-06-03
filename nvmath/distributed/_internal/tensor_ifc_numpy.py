# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use distributed Numpy ndarray objects.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["NumpyDistributedTensor"]

from nvmath.internal.tensor_ifc_numpy import NumpyTensor
from nvmath.internal.utils import device_ctx

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Can't import CupyDistributedTensor at runtime here due to circular import, but mypy
    # needs it for type checking.
    from .tensor_ifc_cupy import CupyDistributedTensor


# Most methods aren't redefined, because they simply act on the local array
class NumpyDistributedTensor(NumpyTensor):
    """
    Tensor wrapper for distributed numpy ndarrays.
    """

    def __init__(self, tensor):
        super().__init__(tensor)

    def to(self, device_id, stream_holder) -> NumpyDistributedTensor | CupyDistributedTensor:
        """
        In addition to the base class semantics:
          - Target device must be the one used to initialize NVSHMEM on this process.
          - Memory layout is preserved.
          - Strides must be dense non-overlapping.
        """
        if not (device_id == "cpu" or isinstance(device_id, int)):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")

        if device_id == "cpu":
            return NumpyDistributedTensor(self.tensor)

        from .tensor_ifc_cupy import CupyDistributedTensor

        with device_ctx(device_id), stream_holder.ctx:
            tensor_device = CupyDistributedTensor.empty(
                self.shape, dtype=self.dtype, device_id=device_id, strides=self.strides, make_symmetric=True
            )
            tensor_device.copy_(self, stream_holder)
            return tensor_device
