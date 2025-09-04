# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes

import nvmath.internal.ndbuffer.package_utils as package_utils
from nvmath.internal.memory import free_reserved_memory
from nvmath.internal.tensor_wrapper import maybe_register_package
from nvmath.internal.utils import get_or_create_stream
import nvmath.internal.tensor_wrapper as tw

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


class Param:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __bool__(self):
        return bool(self.value)

    def pretty_name(self):
        if isinstance(self.value, tuple):
            return "x".join(str(arg) for arg in self.value)
        elif hasattr(self.value, "name"):
            value_str = self.value.name
        else:
            value_str = str(self.value)
        return f"{self.name}.{value_str}"


class DummySlice:
    def __getitem__(self, value):
        return value


_SL = DummySlice()


def idfn(val):
    """
    Pytest does not pretty print (repr/str) parameters of custom types.
    """
    if hasattr(val, "pretty_name"):
        return val.pretty_name()
    # use default pytest pretty printing
    return None


def arange(device_id, stream_holder, volume, dtype):
    if device_id == "cpu":
        a = np.arange(1, volume + 1, dtype=dtype)
        if dtype in (np.complex64, np.complex128):
            a = (a + 1j * np.arange(volume, 0, -1, dtype=dtype)).astype(dtype)
        return a
    elif isinstance(device_id, int):
        if cp is None:
            raise ValueError("cupy is not installed")
        with cp.cuda.Device(device_id), stream_holder.ctx:
            a = cp.arange(1, volume + 1, dtype=dtype)
            if dtype in (cp.complex64, cp.complex128):
                a = (a + 1j * cp.arange(volume, 0, -1, dtype=dtype)).astype(dtype)
            return a
    else:
        raise ValueError(f"Invalid device_id: {device_id}")


def zeros(device_id, stream_holder, shape, dtype):
    if device_id == "cpu":
        return np.zeros(shape, dtype=dtype)
    elif isinstance(device_id, int):
        if cp is None:
            raise ValueError("cupy is not installed")
        with cp.cuda.Device(device_id), stream_holder.ctx:
            return cp.zeros(shape, dtype=dtype)
    else:
        raise ValueError(f"Invalid device_id: {device_id}")


def create_stream(device_id):
    if device_id == "cpu":
        return None
    elif isinstance(device_id, int):
        if cp is None:
            raise ValueError("cupy is not installed")
        maybe_register_package("cupy")
        with cp.cuda.Device(device_id):
            stream = cp.cuda.Stream(non_blocking=True)
            return get_or_create_stream(device_id, stream, "cupy")
    else:
        raise ValueError(f"Invalid device_id: {device_id}")


def free_memory():
    free_reserved_memory()
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()


def package(a):
    if isinstance(a, np.ndarray):
        return np
    if isinstance(a, cp.ndarray):
        return cp
    raise ValueError(f"Invalid array: {type(a)}")


def as_ndbuffer(a):
    if isinstance(a, np.ndarray):
        return package_utils.wrap_numpy_array(a)
    if isinstance(a, cp.ndarray):
        return package_utils.wrap_cupy_array(a)
    raise ValueError(f"Invalid array: {type(a)}")


def wrap_operand(a):
    if isinstance(a, np.ndarray):
        wrapped = tw.wrap_operand(a)
        assert isinstance(wrapped, tw.NumpyTensor)
        return wrapped
    if isinstance(a, cp.ndarray):
        wrapped = tw.wrap_operand(a)
        import nvmath.internal.tensor_ifc_cupy as tcupy

        assert isinstance(wrapped, tcupy.CupyTensor)
        return wrapped
    raise ValueError(f"Invalid array: {type(a)}")


def stride_tricks(a, shape, stride, itemsize):
    p = package(a)
    stride_in_bytes = tuple(s * itemsize for s in stride)
    return p.lib.stride_tricks.as_strided(a, shape=shape, strides=stride_in_bytes)


def assert_equal(a, b):
    ap = package(a)
    bp = package(b)
    if ap is bp:
        ap.testing.assert_array_equal(a, b)
    else:
        anp = cp.asnumpy(a)
        bnp = cp.asnumpy(b)
        np.testing.assert_array_equal(anp, bnp)


def sliced_or_broadcast_1d(device_id, stream_holder, volume, stride, dtype):
    if stride == 0:
        a_base = arange(device_id, stream_holder, 1, dtype)
        return stride_tricks(a_base, (volume,), (stride,), np.dtype(dtype).itemsize)
    else:
        a_base = arange(device_id, stream_holder, volume, dtype)
        if stride != 1:
            return a_base[::stride]
        else:
            return a_base


def random_non_empty_slice(rng, a):
    shape = a.shape
    ndim = len(shape)
    slicable_indicies = [i for i in range(ndim) if shape[i] > 1]
    sliced_ndim = rng.randint(1, len(slicable_indicies))
    sliced_indicies = rng.sample(slicable_indicies, sliced_ndim)
    slices = [slice(None)] * ndim
    for i in sliced_indicies:
        slice_size = rng.randint(1, shape[i] - 1)
        slice_start = rng.randint(0, shape[i] - slice_size)
        slice_end = slice_start + slice_size
        slices[i] = slice(slice_start, slice_end)
    return a[tuple(slices)]


def random_negated_strides(rng, a):
    ndim = len(a.shape)
    negated_ndim = rng.randint(1, ndim)
    negated_indicies = rng.sample(range(ndim), negated_ndim)
    slices = [slice(None)] * ndim
    for i in negated_indicies:
        slices[i] = slice(None, None, -1)
    return a[tuple(slices)]


def inv(p):
    inv_p = [0] * len(p)
    for i, d in enumerate(p):
        inv_p[d] = i
    return tuple(inv_p)


def permuted(strides, permutation):
    return tuple(strides[i] for i in permutation)


def dense_c_strides(shape, itemsize):
    strides = [0] * len(shape)
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride * itemsize
        stride *= shape[i]
    return tuple(strides)


def abs_strides(strides):
    return tuple(abs(s) for s in strides)


def as_array(ndbuffer):
    if ndbuffer.device_id == "cpu":
        buffer = (ctypes.c_char * ndbuffer.size_in_bytes).from_address(ndbuffer.data_ptr)
        return np.ndarray(
            shape=ndbuffer.shape,
            strides=ndbuffer.strides_in_bytes,
            dtype=ndbuffer.dtype_name,
            buffer=buffer,
        )
    else:
        mem = cp.cuda.UnownedMemory(
            ndbuffer.data_ptr,
            ndbuffer.size_in_bytes,
            owner=ndbuffer.data,
            device_id=ndbuffer.device_id,
        )
        memptr = cp.cuda.MemoryPointer(mem, offset=0)
        return cp.ndarray(
            shape=ndbuffer.shape,
            strides=ndbuffer.strides_in_bytes,
            dtype=ndbuffer.dtype_name,
            memptr=memptr,
        )
