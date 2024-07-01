# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use Numpy ndarray objects.
"""

__all__ = ['NumpyTensor']

import cupy
import numpy

from . import utils
from .package_ifc import StreamHolder
from .tensor_ifc import Tensor


class NumpyTensor(Tensor):
    """
    Tensor wrapper for numpy ndarrays.
    """
    name = 'numpy'
    module = numpy
    name_to_dtype = Tensor.create_name_dtype_map(conversion_function=lambda name: numpy.dtype(name), exception_type=TypeError)

    def __init__(self, tensor):
        super().__init__(tensor)

    @property
    def data_ptr(self):
        return self.tensor.ctypes.data

    @property
    def device(self):
        return 'cpu'

    @property
    def device_id(self):
        return None

    @property
    def dtype(self):
        """Name of the data type"""
        return self.tensor.dtype.name

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def strides(self):
        return tuple(stride_in_bytes // self.tensor.itemsize for stride_in_bytes in self.tensor.strides)

    def numpy(self, stream_holder=StreamHolder()):
        return self.tensor

    @classmethod
    def empty(cls, shape, **context):
        """
        Create an empty tensor of the specified shape and data type.
        """
        name = context.get('dtype', 'float32')
        dtype = NumpyTensor.name_to_dtype[name]
        strides = context.get('strides', None)
        # when strides is not None, it should be of unit counts not bytes
        return cls(module.ndarray(shape, dtype=dtype, strides=(tuple(s * dtype.itemsize for s in strides) if strides else None)))

    def to(self, device='cpu', stream_holder=StreamHolder()):
        """
        Create a copy of the tensor on the specified device (integer or
          'cpu'). Copy to  Cupy ndarray on the specified device if it
          is not CPU. Otherwise, return self.
        """
        if device == 'cpu':
            return self.tensor

        if not isinstance(device, int):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device}'.")

        with utils.device_ctx(device), stream_holder.ctx:
            tensor_device = cupy.asarray(self.tensor)

        return tensor_device

    def copy_(self, src, stream_holder=StreamHolder()):
        package = utils.infer_object_package(src)
        # Handle NumPy <=> CuPy CPU-GPU ndarray asymmetry.
        if package == 'cupy':
            stream = stream_holder.obj
            with stream:
                out = src.get(stream=stream, out=self.tensor)
            # cupy/cupy#7820
            if stream is not None:
                stream.synchronize()

            return out
        else:
            raise NotImplementedError

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, numpy.ndarray)

    def reshape_to_match_tensor_descriptor(self, handle, desc_tensor):
        #NOTE: this method is only called for CupyTensor and TorchTensor
        raise NotImplementedError
