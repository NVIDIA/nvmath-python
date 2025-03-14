# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use Numpy ndarray objects.
"""

__all__ = ["NumpyTensor"]

try:
    import cupy
except ImportError:
    cupy = None
import numpy

from . import utils
from .package_ifc import StreamHolder
from .tensor_ifc import Tensor


class NumpyTensor(Tensor):
    """
    Tensor wrapper for numpy ndarrays.
    """

    name = "numpy"
    module = numpy
    name_to_dtype = Tensor.create_name_dtype_map(conversion_function=lambda name: numpy.dtype(name), exception_type=TypeError)

    def __init__(self, tensor):
        super().__init__(tensor)

    @property
    def data_ptr(self):
        return self.tensor.ctypes.data

    @property
    def device(self):
        return "cpu"

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
    def size(self):
        return self.tensor.size

    @property
    def strides(self):
        return tuple(stride_in_bytes // self.tensor.itemsize for stride_in_bytes in self.tensor.strides)

    def numpy(self, stream_holder=StreamHolder()):
        return self.tensor

    @classmethod
    def empty(cls, shape, *, dtype="float32", strides=None, device_id=None):
        """
        Create an empty tensor of the specified shape and data type.
        """
        assert device_id is None
        dtype = NumpyTensor.name_to_dtype[dtype]
        # when strides is not None, it should be of unit counts not bytes
        return cls(
            cls.module.ndarray(shape, dtype=dtype, strides=(tuple(s * dtype.itemsize for s in strides) if strides else None))
        )

    def to(self, device="cpu", stream_holder=StreamHolder()):
        """
        Create a copy of the tensor on the specified device (integer or
          'cpu'). Copy to  Cupy ndarray on the specified device if it
          is not CPU. Otherwise, return self.
        """
        if device == "cpu":
            return self.tensor

        if not isinstance(device, int):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device}'.")

        with utils.device_ctx(device), stream_holder.ctx:
            tensor_device = cupy.asarray(self.tensor)

        return tensor_device

    def n2n_copy_(self, src):
        """
        Inplace copy of src (copy the data from src into self).
        The src must by numpy ndarray
        """
        numpy.copyto(self.tensor, src)

    def c2n_copy_(self, src, stream_holder):
        """
        Inplace copy of src (copy the data from src into self).
        The src must by cupy ndarray
        """
        stream = stream_holder.obj
        try:
            with stream:
                src.get(stream=stream, out=self.tensor)
        except RuntimeError as e:
            # If self is a strided tensor (neither c nor f layout)
            # cupy refuses to copy to numpy array
            if "copying to non-contiguous ndarray" not in str(e):
                raise
            else:
                # we cannot simply use blocking=True, as it is
                # not supported by older cupy releases (<13)
                src_cpu = cupy.asnumpy(src, stream=stream)
                self.n2n_copy_(src_cpu)
        # cupy/cupy#7820
        if stream is not None:
            stream.synchronize()

    def copy_(self, src, stream_holder=StreamHolder()):
        package = utils.infer_object_package(src)
        # Handle NumPy <=> CuPy CPU-GPU ndarray asymmetry.
        if package == "cupy":
            self.c2n_copy_(src, stream_holder)
        elif package == "numpy":
            self.n2n_copy_(src)
        else:
            raise NotImplementedError

        return self.tensor

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, numpy.ndarray)

    def reshape_to_match_tensor_descriptor(self, handle, desc_tensor):
        # NOTE: this method is only called for CupyTensor and TorchTensor
        raise NotImplementedError
