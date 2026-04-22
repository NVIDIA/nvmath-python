# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use CSR tensors from different libraries.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["CSRTensorHolder"]

from typing import Literal, TypeVar

from nvmath._internal.layout import check_monotonic_strides, is_contiguous_and_dense
from nvmath.internal import tensor_wrapper
from nvmath.internal.package_ifc import StreamHolder
from nvmath.internal.tensor_ifc import TensorHolder
from nvmath.internal.utils import infer_object_package

from .sparse_tensor_ifc import SparseTensorHolder

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

        # Capture the native tensor package for use in reset_unchecked().
        self.tensor_package = infer_object_package(self.tensor)

        # The tensor holder type for the dense constituent tensors (take from values, since
        # all constituent tensors have the same type).
        self._dense_tensorholder_type = values.__class__

    @classmethod
    def create_from_tensor(cls, tensor, *, attr_name_map=None):
        assert attr_name_map is not None, "Internal error."

        # tensor is the native tensor.
        # For UST, the buffers will be CuPy ndarrays.
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

    def to_ust(self, *, stream):
        from nvmath.sparse.ust.tensor import Tensor
        from nvmath.sparse.ust.tensor_format import NamedFormats

        wrapped = self
        tensor = wrapped.tensor
        if not (wrapped.attr_name_map["is_coalesced"](tensor) and wrapped.attr_name_map["has_sorted_indices"](tensor)):
            raise ValueError(
                "The CSR tensor is not coalesced (metadata sorted with duplicates removed). \
To coalesce, use the operation provided by your sparse tensor library: `coalesce()`, `sum_duplicates()`, ..."
            )

        if not (
            is_contiguous_and_dense(wrapped.crow_indices.shape, wrapped.crow_indices.strides)
            and is_contiguous_and_dense(wrapped.col_indices.shape, wrapped.col_indices.strides)
            and is_contiguous_and_dense(wrapped.values.shape, wrapped.values.strides)
        ):
            raise ValueError(
                "The sparse tensor representation (crow_indices(), col_indices(), and values() tensors) must be \
dense and contiguous."
            )

        values = wrapped.values.memory_buffer()  # Typed memory viewed as a 1-D tensor.

        num_dim = self.num_dimensions
        num_sparse_dim = self.attr_name_map["num_sparse_dim"](tensor)
        num_dense_dim = self.attr_name_map["num_dense_dim"](tensor)
        num_batch_dim = num_dim - num_sparse_dim - num_dense_dim
        tensor_format = NamedFormats.CSRd(num_batch_dim, num_dense_dim)

        # Check if the batch axes are in C-order for all component dense arrays.
        if not (
            check_monotonic_strides(wrapped.crow_indices.strides[:num_batch_dim], reverse=True)
            and check_monotonic_strides(wrapped.col_indices.strides[:num_batch_dim], reverse=True)
            and check_monotonic_strides(wrapped.values.strides[:num_batch_dim], reverse=True)
        ):
            raise ValueError(
                "The batch dimensions in the sparse tensor representation (crow_indices(), col_indices(), \
and values() tensors) must use the C-layout."
            )

        # Create the UST.
        ust = Tensor(tensor.shape, tensor_format=tensor_format, index_type=wrapped.index_type, dtype=wrapped.dtype)

        ust._pos[num_batch_dim + 1] = wrapped.crow_indices
        ust._crd[num_batch_dim + 1] = wrapped.col_indices
        ust._val = values

        return ust

    def to_package(self):
        """
        This will create a sparse tensor for the original package from which this
        interface was created.
        """
        create_csr = self.attr_name_map["sparse_format_helper"].create_csr
        return create_csr(self.shape, self.crow_indices, self.col_indices, self.values)

    def release(self):
        """
        This method will release the wrapped tensor and any format-specific data
        by setting them to None.
        """
        # The tensor reference.
        self.tensor = None

        # Format-specific data.
        self._crow_indices = None
        self._col_indices = None
        self._values = None

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

        # Format-specific data.

        if self.tensor_package == "nvmath":
            # The data arrays are already wrapped.
            self._crow_indices = self._attr_name_map["crow_indices"](tensor)
            self._col_indices = self._attr_name_map["col_indices"](tensor)
            self._values = self._attr_name_map["values"](tensor)
        else:
            wrapper = self._dense_tensorholder_type

            crow_indices = self._attr_name_map["crow_indices"](tensor)
            self._crow_indices = wrapper(crow_indices)

            col_indices = self._attr_name_map["col_indices"](tensor)
            self._col_indices = wrapper(col_indices)

            values = self._attr_name_map["values"](tensor)
            self._values = wrapper(values)
