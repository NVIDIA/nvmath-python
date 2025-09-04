# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use Numpy ndarray objects.
"""

__all__ = ["NumpyTensor", "CudaTensor"]

from collections.abc import Sequence

import numpy
import numpy.typing as npt
from nvmath.internal.tensor_ifc_ndbuffer import NDBufferTensor

from .ndbuffer import ndbuffer, package_utils
from . import utils
from .tensor_ifc import TensorHolder
from .package_ifc import StreamHolder


class CudaTensor(NDBufferTensor):
    """
    Wraps ndbuffer with data residing on the GPU.
    It serves as a CUDA counterpart for NumpyTensor.
    """

    name = "cuda"
    host_tensor_class: type["NumpyTensor"]  # set once NumpyTensor is defined

    def __init__(self, tensor):
        super().__init__(tensor)

    @classmethod
    def create_from_host(cls, tensor: TensorHolder, device_id: int, stream_holder: StreamHolder):
        with utils.device_ctx(device_id):
            src_ndbuffer = tensor.asndbuffer()
            dst_ndbuffer = ndbuffer.empty_like(src_ndbuffer, device_id=device_id, stream=stream_holder)
            ndbuffer.copy_into(dst_ndbuffer, src_ndbuffer, stream_holder)
            return cls(dst_ndbuffer)

    def to(self, device_id, stream_holder):
        if device_id == "cpu":
            with utils.device_ctx(self.device_id):
                dst = self.host_tensor_class.create_host_from(self, stream_holder)
                return dst
        elif device_id == self.device_id:
            return self
        elif isinstance(device_id, int):
            raise ValueError(f"Unsupported copy between different devices {self.device_id} and {device_id}.")
        else:
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")


class NumpyTensor(TensorHolder[npt.NDArray]):
    """
    TensorHolder for numpy ndarrays.
    """

    name = "numpy"
    module = numpy
    name_to_dtype = TensorHolder.create_name_dtype_map(
        conversion_function=lambda name: numpy.dtype(name), exception_type=TypeError
    )

    device_tensor_class = CudaTensor

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
        return "cpu"

    @property
    def dtype(self):
        """Name of the data type"""
        return self.tensor.dtype.name

    @property
    def itemsize(self):
        return self.tensor.itemsize

    @classmethod
    def empty(cls, shape, device_id="cpu", *, dtype="float32", strides=None, **context):
        """
        Create an empty tensor of the specified shape and data type.
        """
        assert device_id == "cpu", "Internal Error: Numpy tensors must be allocated with device_id='cpu'"
        dtype = NumpyTensor.name_to_dtype[dtype]
        # when strides is not None, it should be of unit counts not bytes
        return cls(
            cls.module.ndarray(shape, dtype=dtype, strides=(tuple(s * dtype.itemsize for s in strides) if strides else None))
        )

    @classmethod
    def create_host_from(cls, tensor: TensorHolder, stream_holder: StreamHolder):
        src_nd = tensor.asndbuffer()
        wrapped_np = package_utils.empty_numpy_like(src_nd)
        ndbuffer.copy_into(wrapped_np, src_nd, stream_holder)
        return cls(wrapped_np.data)

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def size(self):
        return self.tensor.size

    @property
    def strides(self):
        return tuple(stride_in_bytes // self.tensor.itemsize for stride_in_bytes in self.tensor.strides)

    def asndbuffer(self):
        return package_utils.wrap_numpy_array(self.tensor)

    def to(self, device_id, stream_holder):
        if device_id == "cpu":
            return self
        elif isinstance(device_id, int):
            dst = self.device_tensor_class.create_from_host(self, device_id, stream_holder)
            return dst
        else:
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")

    def copy_(self, src, stream_holder):
        match src.name:
            case "numpy":
                numpy.copyto(self.tensor, src.tensor)
            case _:
                with utils.device_ctx(src.device_id):
                    ndbuffer.copy_into(self.asndbuffer(), src.asndbuffer(), stream_holder)

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, numpy.ndarray)

    def reshape(self, shape: Sequence[int], *, copy: bool | None = None):
        if int(numpy.__version__.split(".")[0]) < 2:
            return self.__class__(numpy.reshape(self.tensor, shape))
        else:
            return self.__class__(numpy.reshape(self.tensor, shape, copy=copy))


CudaTensor.host_tensor_class = NumpyTensor
