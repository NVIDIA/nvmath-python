# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use sparse tensors from different libraries.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["SparseTensorHolder"]


from abc import ABC, abstractmethod
from typing import Literal

from nvmath.internal.package_ifc import StreamHolder


class SparseTensorHolder(ABC):
    """
    A simple base type for attributes common to *all* sparse formats.
    """

    @classmethod
    @abstractmethod
    def create_from_tensor(cls, attr_name_map, tensor):
        raise NotImplementedError

    @property
    @abstractmethod
    def attr_name_map(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def device_id(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def index_type(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def format_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_dimensions(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

    @abstractmethod
    def to(self, device_id: int | Literal["cpu"], stream_holder: StreamHolder | None):
        """Copy the SparseTensor representation to a different device.

        No copy is performed if the SparseTensor is already on the requested device.
        """
        raise NotImplementedError

    def copy_(self, src: SparseTensorHolder, stream_holder: StreamHolder | None) -> None:
        """Overwrite the sparse tensor (in-place) with a copy of src."""
        raise NotImplementedError
