# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use COO tensors from different libraries.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

__all__ = ["COOTensorHolder"]

from typing import Literal, TypeVar

from nvmath._internal.layout import is_contiguous_and_dense
from nvmath.internal import tensor_wrapper
from nvmath.internal.package_ifc import StreamHolder
from nvmath.internal.tensor_ifc import TensorHolder
from nvmath.internal.utils import infer_object_package

from .sparse_tensor_ifc import SparseTensorHolder

"""
A generic type for the third-party COOTensor which a COOTensorHolder implementation wraps
around.
"""
COOTensor = TypeVar("COOTensor")


class COOTensorHolder(SparseTensorHolder):
    """
    A simple wrapper type for COO sparse tensors to make the API package-agnostic. The
    naming convention is sparse format name with the suffix "TensorHolder", which convention
    is used in the sparse format helper.

    Since the implementation is in terms of our dense tensor abstraction, package-specific
    implementation is not needed.
    """

    _format_name: str = "COO"

    def __init__(self, indices, values, *, attr_name_map=None, shape=None, tensor=None):
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

        # Wrap constituent dense tensors, if required. If it's a sequence, it will be
        # a sequence of TensorHolder.
        if not isinstance(indices, TensorHolder):
            indices = [tensor_wrapper.wrap_operand(i) for i in indices]

        if not isinstance(values, TensorHolder):
            values = tensor_wrapper.wrap_operand(values)

        # The wrapped constituent dense tensors.
        self._indices = indices
        self._values = values

        # Ensure constituent dense tensors are all on the same device.
        device_ids = {d.device_id for d in (*self._indices, self._values)}
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
        indices = attr_name_map["indices"](tensor)
        values = attr_name_map["values"](tensor)
        return COOTensorHolder(indices, values, attr_name_map=attr_name_map, tensor=tensor)

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
        # TODO: check that all index arrays in the sequence have the same dtype.
        return self._indices[0].dtype

    @property
    def format_name(self) -> str:
        return COOTensorHolder._format_name

    @property
    def num_dimensions(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def indices(self):
        """The TensorHolder object wrapping the indices array."""
        return self._indices

    @property
    def values(self) -> TensorHolder:
        """The TensorHolder object wrapping the values array."""
        return self._values

    def to(self, device_id: int | Literal["cpu"], stream_holder: StreamHolder | None):
        """Copy the COOTensor representation to a different device.

        No copy is performed if the COOTensor is already on the requested device.
        """

        target_indices = [i.to(device_id=device_id, stream_holder=stream_holder) for i in self.indices]
        target_values = self.values.to(device_id=device_id, stream_holder=stream_holder)

        return COOTensorHolder(target_indices, target_values, attr_name_map=self.attr_name_map, shape=self.shape)

    def copy_(self, src, stream_holder: StreamHolder | None) -> None:
        """Overwrite self.tensor (in-place) with a copy of src (COOTensor)."""

        for c, component in enumerate(self.indices):
            component.copy_(src=src.indices[c], stream_holder=stream_holder)
        self.values.copy_(src=src.values, stream_holder=stream_holder)

    def to_ust(self, *, stream):
        from nvmath.sparse.ust.tensor import Tensor, _top_array
        from nvmath.sparse.ust.tensor_format import NamedFormats

        wrapped = self
        tensor = wrapped.tensor
        if not wrapped.attr_name_map["is_coalesced"](tensor):
            raise ValueError(
                "The COO tensor is not coalesced (metadata sorted with duplicates removed). \
To coalesce, use the operation provided by your sparse tensor library: `coalesce()`, `sum_duplicates()`, ..."
            )

        if not (
            all(is_contiguous_and_dense(i.shape, i.strides) for i in self.indices)
            and is_contiguous_and_dense(wrapped.values.shape, wrapped.values.strides)
        ):
            raise ValueError(
                "The sparse tensor representation (indices() and values() tensors) must be \
dense and contiguous."
            )

        values = wrapped.values.memory_buffer()  # Typed memory viewed as a 1-D tensor.

        num_sparse_dim = self.attr_name_map["num_sparse_dim"](tensor)
        num_dense_dim = self.attr_name_map["num_dense_dim"](tensor)
        tensor_format = NamedFormats.COOd(num_sparse_dim, num_dense_dim)

        # Check if the batch axes are in C-order for all component dense arrays.
        if not all(i.strides == (1,) for i in wrapped.indices):
            raise ValueError("The batch dimensions in the sparse tensor representation indices() must use the C-layout.")

        # Create the UST.
        ust = Tensor(tensor.shape, tensor_format=tensor_format, index_type=wrapped.index_type, dtype=wrapped.dtype)

        dense_tensorholder_type = wrapped._dense_tensorholder_type
        if wrapped.device_id == "cpu":
            stream_holder = None
        else:
            from nvmath.internal import utils

            stream_holder = utils.get_or_create_stream(wrapped.device_id, stream, dense_tensorholder_type.name)
        ust._pos[0] = _top_array(
            wrapped.indices[0].size,
            dtype=wrapped.index_type,
            device_id=wrapped.device_id,
            dense_tensorholder_type=dense_tensorholder_type,
            stream_holder=stream_holder,
        )
        for d in range(num_sparse_dim):
            ust._crd[d] = wrapped.indices[d]
        ust._val = values

        return ust

    def to_package(self):
        """
        This will create a sparse tensor for the original package from which this
        interface was created.
        """
        create_coo = self.attr_name_map["sparse_format_helper"].create_coo
        return create_coo(self.shape, self.indices, self.values)

    def release(self):
        """
        This method will release the wrapped tensor and any format-specific data
        by setting them to None.
        """
        # The tensor reference.
        self.tensor = None

        # Format-specific data.
        self._indices = None
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
            self._indices = self._attr_name_map["indices"](tensor)
            self._values = self._attr_name_map["values"](tensor)
        else:
            wrapper = self._dense_tensorholder_type

            indices = self._attr_name_map["indices"](tensor)
            self._indices = [wrapper(i) for i in indices]

            values = self._attr_name_map["values"](tensor)
            self._values = wrapper(values)
