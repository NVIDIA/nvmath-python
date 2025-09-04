# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Common distributed tensor interface.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["DistributedTensor"]

from abc import abstractmethod
from typing import Literal
from collections.abc import Sequence

from nvmath.internal.package_ifc import StreamHolder
from nvmath.internal.tensor_ifc import Tensor, TensorHolder
from nvmath.bindings import nvshmem  # type: ignore
from nvmath.distributed._internal.nvshmem import free


class DistributedTensor(TensorHolder[Tensor]):
    """Base class for distributed tensors.

    Sets flag during construction to indicate if the tensor is on symmetric memory or not.
    """

    def __init__(self, tensor):
        super().__init__(tensor)
        self._is_symmetric_memory = False
        if self.device == "cuda":
            self._is_symmetric_memory = nvshmem.ptr(self.data_ptr, nvshmem.my_pe()) != 0

    @property
    def is_symmetric_memory(self):
        return self._is_symmetric_memory

    @abstractmethod
    def to(
        self, device_id: int | Literal["cpu"], stream_holder: StreamHolder | None, symmetric_memory: bool = False
    ) -> DistributedTensor:
        """Copy the TensorHolder to a different device.

        No copy is performed if the TensorHolder is already on the requested device.
        """
        raise NotImplementedError

    @abstractmethod
    def reshape(self, shape: Sequence[int], *, copy: bool | None = None) -> DistributedTensor:
        raise NotImplementedError

    def free_symmetric(self) -> None:
        """
        Release this tensor's allocation on NVSHMEM symmetric memory heap.
        """
        if not self._is_symmetric_memory:
            raise TypeError("tensor is not on symmetric memory")

        free(self.data_ptr)
