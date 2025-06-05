# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use Numpy ndarray objects.
"""

__all__ = ["NumpyTensor"]

try:
    import cupy  # type: ignore
except ImportError:

    class cupy:  # type: ignore
        """A placeholder for the cupy module when it is unavailable."""

        @classmethod
        def asnumpy(cls, *args, **kwargs):
            raise ImportError("Cannot convert cupy to numpy array when cupy is not installed!")

        @classmethod
        def asarray(cls, *args, **kwargs):
            raise ImportError("Cannot convert numpy to cupy array when cupy is not installed!")


from collections.abc import Sequence

import numpy
import numpy.typing as npt

from . import utils
from .package_ifc import StreamHolder
from .tensor_ifc import TensorHolder


class NumpyTensor(TensorHolder[npt.NDArray]):
    """
    TensorHolder for numpy ndarrays.
    """

    name = "numpy"
    module = numpy
    name_to_dtype = TensorHolder.create_name_dtype_map(
        conversion_function=lambda name: numpy.dtype(name), exception_type=TypeError
    )

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

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def size(self):
        return self.tensor.size

    @property
    def strides(self):
        return tuple(stride_in_bytes // self.tensor.itemsize for stride_in_bytes in self.tensor.strides)

    def to(self, device_id, stream_holder):
        if not (device_id == "cpu" or isinstance(device_id, int)):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")

        if device_id == "cpu":
            return NumpyTensor(self.tensor)

        # FIXME: Replace with native tensor implementation to avoid required dep on CuPy
        from .tensor_ifc_cupy import CupyTensor

        with utils.device_ctx(device_id), stream_holder.ctx:
            return CupyTensor(cupy.asarray(self.tensor))

    def _n2n_copy_(self, src: npt.NDArray) -> None:
        """
        Inplace copy of src (copy the data from src into self).
        The src must by numpy ndarray
        """
        numpy.copyto(self.tensor, src)

    def _c2n_copy_(self, src, stream_holder: StreamHolder) -> None:
        """
        Inplace copy of src (copy the data from src into self).
        The src must by cupy ndarray
        """
        stream = stream_holder.external
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
                self._n2n_copy_(src_cpu)
        # cupy/cupy#7820
        if stream is not None:
            stream.synchronize()

    def copy_(self, src, stream_holder):
        # Handle NumPy <=> CuPy CPU-GPU ndarray asymmetry.
        match src.name:
            case "cupy":
                assert stream_holder is not None
                self._c2n_copy_(src.tensor, stream_holder)
            case "numpy":
                self._n2n_copy_(src.tensor)
            case _:
                msg = f"NumpyTensor does not convert from {src.name}."
                raise NotImplementedError(msg)

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
