# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
TensorHolder implementation that adapts ndbuffer for use with
the tensor_ifc interface. The class is meant for internal use only
(e.g. for internal operands of host APIs), but the NDBufferTensor
or underlying ndbuffer should never be returned to the user, because
1. the ndbuffer is opaque, i.e. it missies most of typical tensor functionality, and
2. we make no guarantees about the ndbuffer's API stability.
"""

__all__ = ["NDBufferTensor"]

from collections.abc import Sequence

from .ndbuffer import ndbuffer
from . import typemaps
from . import utils
from .tensor_ifc import TensorHolder
from .package_ifc import StreamHolder


class NDBufferTensor(TensorHolder[ndbuffer.NDBuffer]):
    """
    TensorHolder for ndbuffer ndarrays.
    """

    name = "ndbuffer"
    module = ndbuffer
    name_to_dtype = TensorHolder.create_name_dtype_map(conversion_function=lambda name: name, exception_type=TypeError)

    def __init__(self, tensor):
        super().__init__(tensor)

    @property
    def data_ptr(self):
        return self.tensor.data_ptr

    @property
    def device(self):
        return self.tensor.device

    @property
    def device_id(self):
        return self.tensor.device_id

    @property
    def dtype(self):
        """Name of the data type"""
        return self.tensor.dtype_name

    @property
    def itemsize(self):
        return self.tensor.itemsize

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def size(self):
        return self.tensor.size

    @property
    def strides(self):
        return self.tensor.strides

    @classmethod
    def empty(
        cls, shape, device_id="cpu", *, dtype="float32", strides=None, stream_holder: StreamHolder | None = None, **context
    ):
        """
        Create an empty tensor of the specified shape and data type.

        Note, that the strides, if specified, MUST correspond to a dense (possibly permuted)
        tensor, otherwise the created tensor may be corrupted.
        """
        itemsize = typemaps.NAME_TO_ITEM_SIZE[dtype]
        if device_id == "cpu":
            tensor = ndbuffer.empty(
                shape, ndbuffer.CPU_DEVICE_ID, dtype, itemsize, strides=strides, stream=stream_holder, **context
            )
        else:
            assert isinstance(device_id, int), "Internal Error: Cuda tensors must be allocated with an integer device_id."
            with utils.device_ctx(device_id):
                tensor = ndbuffer.empty(shape, device_id, dtype, itemsize, strides=strides, stream=stream_holder, **context)
        return cls(tensor)

    def asndbuffer(self):
        return self.tensor

    def to(self, device_id, stream_holder):
        src_device_id = self.tensor.device_id
        if src_device_id == device_id:
            return self

        if stream_holder is None:
            raise ValueError("Stream holder is required for h2d/d2h transfers.")

        if device_id == "cpu":
            with utils.device_ctx(src_device_id):
                tensor = ndbuffer.empty_like(self.tensor, device_id=ndbuffer.CPU_DEVICE_ID, stream=stream_holder)
                ndbuffer.copy_into(tensor, self.tensor, stream_holder)
        else:
            with utils.device_ctx(device_id):
                tensor = ndbuffer.empty_like(self.tensor, device_id=device_id, stream=stream_holder)
                ndbuffer.copy_into(tensor, self.tensor, stream_holder)

        return NDBufferTensor(tensor)

    def copy_(self, src, stream_holder):
        """
        Inplace copy of src (copy the data from src into self).
        """
        device_id = self.tensor.device_id
        src_nd = src.asndbuffer()
        if device_id == "cpu":
            device_id = src_nd.device_id
        if device_id == "cpu":
            ndbuffer.copy_into(self.tensor, src_nd, stream_holder)
        else:
            with utils.device_ctx(device_id):
                ndbuffer.copy_into(self.tensor, src_nd, stream_holder)

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, ndbuffer.NDBuffer)

    def reshape(self, shape: Sequence[int], *, copy: bool | None = None):
        if copy:
            raise NotImplementedError("Reshape with copy is not supported for ndbuffer")
        return self.__class__(ndbuffer.reshaped_view(self.tensor, shape))
