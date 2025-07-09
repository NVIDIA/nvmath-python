# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use CSR tensors from different libraries.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["CSRTensorHolder"]

from typing import Literal, TypeVar

from .sparse_tensor_ifc import SparseTensorHolder
from nvmath.internal import tensor_wrapper
from nvmath.internal.package_ifc import StreamHolder
from nvmath.internal.tensor_ifc import TensorHolder


"""
A generic type for the third-party CSRTensor which a CSRTensorHolder implementation wraps
around.
"""
CSRTensor = TypeVar("CSRTensor")


class CSRTensorHolder(SparseTensorHolder):
    """
    A simple wrapper type for CSR sparse tensors to make the API package-agnostic. The
    naming convention is sparse format name with the suffix "TensorHolder", which convention
    is used in the sparse format helper.

    Since the implementation is in terms of our dense tensor abstraction, package-specific
    implementation is not needed.
    """

    _format_name: str = "CSR"

    def __init__(self, crow_indices, col_indices, values, *, attr_name_map=None, shape=None, tensor=None):
        # tensor is the native tensor.
        self.tensor = tensor

        # attr_name_map is the canonical attribute name to package attribute name map.
        # It is set when constructing from a tensor, otherwise it is None.
        self._attr_name_map = attr_name_map

        assert (shape is not None) ^ (tensor is not None), "Internal error."

        if shape is not None:
            self._shape = shape
        else:
            self._shape = tuple(tensor.shape)

        # Wrap constituent dense tensors, if required.
        if not isinstance(crow_indices, TensorHolder):
            crow_indices = tensor_wrapper.wrap_operand(crow_indices)

        if not isinstance(col_indices, TensorHolder):
            col_indices = tensor_wrapper.wrap_operand(col_indices)

        if not isinstance(values, TensorHolder):
            values = tensor_wrapper.wrap_operand(values)

        # The wrapped constituent dense tensors.
        self._crow_indices = crow_indices
        self._col_indices = col_indices
        self._values = values

        # Ensure constituent dense tensors are all on the same device.
        device_ids = {d.device_id for d in (self._crow_indices, self._col_indices, self._values)}
        message = f"Internal error: the arrays defining the sparse format aren't on the same device {device_ids}."
        assert len(device_ids) == 1, message
        self._device_id = device_ids.pop()

        self._device = self._values.device

    @classmethod
    def create_from_tensor(cls, tensor, *, attr_name_map=None):
        assert attr_name_map is not None, "Internal error."

        # tensor is the native tensor.
        crow_indices = attr_name_map["crow_indices"](tensor)
        col_indices = attr_name_map["col_indices"](tensor)
        values = attr_name_map["values"](tensor)
        return CSRTensorHolder(crow_indices, col_indices, values, attr_name_map=attr_name_map, tensor=tensor)

    @property
    def attr_name_map(self):
        return self._attr_name_map

    @property
    def device(self):
        return self._device

    @property
    def device_id(self):
        return self._device_id

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def index_type(self):
        return self._crow_indices.dtype

    @property
    def format_name(self) -> str:
        return CSRTensorHolder._format_name

    @property
    def num_dimensions(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def crow_indices(self) -> TensorHolder:
        """The TensorHolder object wrapping the crow_indices array."""
        return self._crow_indices

    @property
    def col_indices(self) -> TensorHolder:
        """The TensorHolder object wrapping the col_indices array."""
        return self._col_indices

    @property
    def values(self) -> TensorHolder:
        """The TensorHolder object wrapping the values array."""
        return self._values

    def to(self, device_id: int | Literal["cpu"], stream_holder: StreamHolder | None):
        """Copy the CSRTensor representation to a different device.

        No copy is performed if the CSRTensor is already on the requested device.
        """

        target_crow_indices = self.crow_indices.to(device_id=device_id, stream_holder=stream_holder)
        target_col_indices = self.col_indices.to(device_id=device_id, stream_holder=stream_holder)
        target_values = self.values.to(device_id=device_id, stream_holder=stream_holder)

        return CSRTensorHolder(
            target_crow_indices, target_col_indices, target_values, attr_name_map=self.attr_name_map, shape=self.shape
        )

    def copy_(self, src, stream_holder: StreamHolder | None) -> None:
        """Overwrite self.tensor (in-place) with a copy of src (CSRTensor)."""

        self.crow_indices.copy_(src=src.crow_indices, stream_holder=stream_holder)
        self.col_indices.copy_(src=src.col_indices, stream_holder=stream_holder)
        self.values.copy_(src=src.values, stream_holder=stream_holder)
