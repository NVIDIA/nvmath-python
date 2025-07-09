# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use Cupy ndarray objects.
"""

__all__ = ["CupyTensor"]

from collections.abc import Sequence

import cupy
import numpy as np

from . import utils
from .tensor_ifc import TensorHolder
from .tensor_ifc_numpy import NumpyTensor
from .package_ifc import StreamHolder


class CupyTensor(TensorHolder[cupy.ndarray]):
    """
    TensorHolder for cupy ndarrays.
    """

    name = "cupy"
    module = cupy
    name_to_dtype = TensorHolder.create_name_dtype_map(
        conversion_function=lambda name: np.dtype(name), exception_type=TypeError
    )

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
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def size(self):
        return self.tensor.size

    @property
    def strides(self):
        return tuple(stride_in_bytes // self.tensor.itemsize for stride_in_bytes in self.tensor.strides)

    def numpy(self, stream_holder: StreamHolder):
        stream = stream_holder.external
        with stream:
            out = self.tensor.get(stream=stream)
        # cupy/cupy#7820
        if stream is not None:
            stream.synchronize()
        return NumpyTensor(out)

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

    def to(self, device_id, stream_holder):
        if not (device_id == "cpu" or isinstance(device_id, int)):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")

        if device_id == "cpu":
            return self.numpy(stream_holder=stream_holder)

        with utils.device_ctx(device_id), stream_holder.ctx:
            return CupyTensor(cupy.asarray(self.tensor))

    def _c2c_copy_(self, src: cupy.ndarray, stream_holder: StreamHolder):
        """
        Inplace copy of src (copy the data from src into self).
        The src must by cupy ndarray
        """
        with stream_holder.ctx:
            cupy.copyto(self.tensor, src)

    def _n2c_copy_(self, src: np.ndarray, stream_holder: StreamHolder):
        """
        Inplace copy of src (copy the data from src into self).
        The src must by numpy ndarray
        """
        stream = stream_holder.external
        try:
            self.tensor.set(src, stream=stream)
        except RuntimeError as e:
            # If self is a strided tensor (neither c nor f layout)
            # cupy refuses to copy from numpy array
            if "set to non-contiguous array" not in str(e):
                raise
            else:
                with stream_holder.ctx:
                    src_gpu = cupy.asarray(src)
                    cupy.copyto(self.tensor, src_gpu)
        # cupy/cupy#7820
        if stream is not None:
            stream.synchronize()

    def copy_(self, src, stream_holder):
        """
        Inplace copy of src (copy the data from src into self).
        """
        match src.name:
            case "cupy":
                self._c2c_copy_(src.tensor, stream_holder)
            case "numpy":
                self._n2c_copy_(src.tensor, stream_holder)
            case _:
                msg = f"CupyTensor does not convert from {src.name}."
                raise NotImplementedError(msg)

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
