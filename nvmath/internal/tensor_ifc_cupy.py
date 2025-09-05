# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use Cupy ndarray objects.
"""

__all__ = ["CupyTensor", "HostTensor"]

from collections.abc import Sequence

import cupy
import numpy as np

from . import utils
from .tensor_ifc import TensorHolder
from .package_ifc import StreamHolder
from .ndbuffer import ndbuffer, package_utils
from .tensor_ifc_ndbuffer import NDBufferTensor


class HostTensor(NDBufferTensor):
    """
    Wraps ndbuffer with data residing on the host.
    It serves as a host counterpart for CupyTensor.
    """

    name = "cupy_host"
    device_tensor_class: type["CupyTensor"]  # set once CupyTensor is defined

    def __init__(self, tensor):
        super().__init__(tensor)

    @classmethod
    def create_host_from(cls, tensor: TensorHolder, stream_holder: StreamHolder):
        src_nd = tensor.asndbuffer()
        # empty_like (and not empty_numpy_like) is used as we don't need
        # full-fledged numpy array (with proper layout)
        dst_nd = ndbuffer.empty_like(src_nd, device_id=ndbuffer.CPU_DEVICE_ID)
        ndbuffer.copy_into(dst_nd, src_nd, stream_holder)
        return cls(dst_nd)

    def to(self, device_id, stream_holder):
        if device_id == "cpu":
            return self
        elif isinstance(device_id, int):
            return self.device_tensor_class.create_from_host(self, device_id, stream_holder)
        else:
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")


class _CupyAllocatorAdapter:
    def allocate(self, size, stream, logger=None):
        # we accept the stream and logger, because the ndbuffer.empty_like
        # passes them to the allocator, but we don't use them:
        # 1. cupy.cuda.alloc does not accept the stream, we make sure to set
        # the correct current stream when calling ndbuffer.empty_like
        # 2. we don't log cupy tensor allocations.
        return cupy.cuda.alloc(size)


_cupy_allocator = _CupyAllocatorAdapter()


class CupyTensor(TensorHolder[cupy.ndarray]):
    """
    TensorHolder for cupy ndarrays.
    """

    name = "cupy"
    module = cupy
    name_to_dtype = TensorHolder.create_name_dtype_map(
        conversion_function=lambda name: np.dtype(name), exception_type=TypeError
    )
    host_tensor_class = HostTensor

    def __init__(self, tensor):
        super().__init__(tensor)

    @property
    def data_ptr(self):
        return self.tensor.data.ptr

    @property
    def device(self):
        return "cuda"

    @property
    def device_id(self):
        return self.tensor.device.id

    @property
    def dtype(self):
        """Name of the data type"""
        return self.tensor.dtype.name

    @property
    def itemsize(self):
        return self.tensor.itemsize

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def size(self):
        return self.tensor.size

    @property
    def strides(self):
        return tuple(stride_in_bytes // self.tensor.itemsize for stride_in_bytes in self.tensor.strides)

    @classmethod
    def empty(
        cls, shape, device_id="cpu", *, dtype="float32", strides=None, stream_holder: StreamHolder | None = None, **context
    ):
        """
        Create an empty tensor of the specified shape and data type.

        Note, that the strides, if specified, MUST correspond to a dense (possibly permuted)
        tensor, otherwise the created tensor may be corrupted.
        """
        assert isinstance(device_id, int), "Internal Error: Cupy tensors must be allocated with an integer device_id."
        dtype = CupyTensor.name_to_dtype[dtype]

        # When using the strides, we need an explicit allocation (see below).
        # If the strides are simple enough, we can still avoid this overhead.
        order = "C"
        if strides is not None:
            if len(strides) == 1 and strides[0] == 1:
                strides = None
            elif len(strides) == 2:
                if strides[0] == 1 and strides[1] == shape[0]:
                    strides = None
                    order = "F"
                elif strides[0] == shape[1] and strides[1] == 1:
                    strides = None

        assert isinstance(stream_holder, StreamHolder), "Internal Error: CupyTensors must be allocated on a stream."
        with utils.device_ctx(device_id), stream_holder.ctx:
            if strides:
                # need an explicit allocation due to cupy/cupy#7818
                size = dtype.itemsize
                for s in shape:
                    size = size * s
                ptr = cupy.cuda.alloc(size)
                # when strides is not None, it should be of unit counts not bytes
                strides = tuple(s * dtype.itemsize for s in strides)
                tensor = cupy.ndarray(shape, dtype=dtype, strides=strides, memptr=ptr)
            else:
                tensor = cupy.ndarray(shape, dtype=dtype, order=order)

        return cls(tensor)

    @classmethod
    def create_from_host(cls, tensor: TensorHolder, device_id: int, stream_holder: StreamHolder):
        with utils.device_ctx(device_id), stream_holder.ctx:
            src_nd = tensor.asndbuffer()
            dst_nd = ndbuffer.empty_like(
                src_nd,
                device_id=device_id,
                stream=stream_holder,
                device_memory_pool=_cupy_allocator,
            )
            ndbuffer.copy_into(dst_nd, src_nd, stream_holder)
            dst = cupy.ndarray(dst_nd.shape, dtype=dst_nd.dtype_name, strides=dst_nd.strides_in_bytes, memptr=dst_nd.data)
            return cls(dst)

    def asndbuffer(self):
        return package_utils.wrap_cupy_array(self.tensor)

    def to(self, device_id, stream_holder):
        if device_id == "cpu":
            with utils.device_ctx(self.device_id):
                return self.host_tensor_class.create_host_from(self, stream_holder)
        elif device_id == self.device_id:
            return self
        elif isinstance(device_id, int):
            raise ValueError(f"Unsupported copy between different devices {self.device_id} and {device_id}.")
        raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")

    def copy_(self, src, stream_holder):
        """
        Inplace copy of src (copy the data from src into self).
        """
        with utils.device_ctx(self.device_id):
            ndbuffer.copy_into(self.asndbuffer(), src.asndbuffer(), stream_holder)

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, cupy.ndarray)

    def reshape(self, shape: Sequence[int], *, copy: bool | None = None):
        if copy:
            raise NotImplementedError("reshape with copy=True not implemented")
        if copy is False:
            try:
                reshaped_tensor = self.tensor.view()
                reshaped_tensor.shape = shape
            except AttributeError:
                raise ValueError(f"Could not reshape cupy array without copy: current shape={self.shape}, new shape={shape}")
        else:
            reshaped_tensor = self.tensor.reshape(shape)
        return self.__class__(reshaped_tensor)


HostTensor.device_tensor_class = CupyTensor
