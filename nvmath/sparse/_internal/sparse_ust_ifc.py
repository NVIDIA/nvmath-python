# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to wrap an UST that encapsulates an uncommon or novel format.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["USTensorHolder"]

from typing import Literal, TypeVar

from nvmath.internal.package_ifc import StreamHolder
from nvmath.internal.tensor_ifc import TensorHolder

from .sparse_tensor_ifc import SparseTensorHolder

"""
A generic wrapper type for a UST.
"""
USTensor = TypeVar("USTensor")


class USTensorHolder(SparseTensorHolder):
    """
    A simple wrapper type for UST objects to enable them to be used in the same way as
    the other abstractions like CSRTensorHolder etc.
    """

    _format_name: str = "UST"

    def __init__(self, tensor):
        # tensor is the native tensor.
        self.tensor = tensor

        # The tensor holder type for the dense constituent tensors.
        self._dense_tensorholder_type = tensor._dense_tensorholder_type

    @classmethod
    def create_from_tensor(cls, tensor, *, attr_name_map=None):
        # TODO: check that tensor is UST.
        return USTensorHolder(tensor=tensor)

    @property
    def attr_name_map(self):
        return {}

    @property
    def device(self):
        return self.tensor.device

    @property
    def device_id(self):
        return self.tensor.device_id

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def index_type(self):
        return self.tensor.index_type

    @property
    def format_name(self) -> str:
        return USTensorHolder._format_name

    @property
    def num_dimensions(self):
        return self.tensor.num_dimensions

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def pos(self):
        """The LevelMap object wrapping the pos array."""
        return self.tensor._pos

    @property
    def crd(self):
        """The LevelMap object wrapping the crd array."""
        return self.tensor._crd

    @property
    def val(self) -> TensorHolder:
        """The TensorHolder object wrapping the val array."""
        return self.tensor._val

    @property
    def values(self) -> TensorHolder:
        """The TensorHolder object wrapping the val array. We use the values
        attribute name to get the NNZ etc."""
        return self.tensor._val

    def to(self, device_id: int | Literal["cpu"], stream_holder: StreamHolder | None):
        """Copy the UST representation to a different device.

        No copy is performed if the UST is already on the requested device.
        """

        # TODO: create an internal function that directly uses the stream holder.
        stream = stream_holder.ptr if stream_holder is not None else None
        target = self.tensor.to(device_id, stream)

        return USTensorHolder(target)

    def copy_(self, src, stream_holder: StreamHolder | None) -> None:
        """Overwrite self.tensor (in-place) with a copy of src (USTensorHolder)."""

        # TODO: create an internal function that directly uses the stream holder.
        stream = stream_holder.ptr if stream_holder is not None else None
        self.tensor.copy_(src.tensor, stream)

    def to_ust(self, *, stream):
        return self.tensor

    def to_package(self):
        """
        This will create a sparse tensor for the original package from which this
        interface was created.
        """
        raise NotImplementedError

    def release(self):
        """
        This method will release the wrapped tensor and any format-specific data
        by setting them to None.
        """
        # The tensor reference.
        self.tensor = None

    def reset_unchecked(self, tensor):
        """
        This method will reset the wrapped tensor to the specified one, and update
        any format-specific data accordingly. It assumes that all attributes
        like the device, shape, etc are consistent between the existing and new tensor.

        Args:
            tensor: The native tensor to wrap.
        """
        # Update the tensor reference.
        self.tensor = tensor
