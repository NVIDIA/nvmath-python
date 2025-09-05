# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import threading
import cython
from .data_layout cimport (
    axis_order_t, parse_py_axis_order, OrderFlag,
    create_layout,
    is_c_or_f as _is_c_or_f,
    tuple2vec, shape_t, strides_t,
)
from .ndbuffer cimport NDBuffer, _no_data_dense_like, _set_flags

import numpy as _numpy


cdef extern from "nd_consts.h":
    cdef const int NDBUFFER_CPU_DEVICE_ID


thread_local = threading.local()


def _name_to_dtype(dtype_name):
    if not hasattr(thread_local, "name_to_dtype_cache"):
        thread_local.name_to_dtype_cache = {}
    dtype = thread_local.name_to_dtype_cache.get(dtype_name)
    if dtype is None:
        dtype = _numpy.dtype(dtype_name)
        thread_local.name_to_dtype_cache[dtype_name] = dtype
    return dtype


def _dtype_to_name(dtype):
    # note, we're relying on the fact that
    # np.dtype is cp.dtype
    if not hasattr(thread_local, "dtype_to_name_cache"):
        thread_local.dtype_to_name_cache = {}
    dtype_name = thread_local.dtype_to_name_cache.get(dtype)
    if dtype_name is None:
        dtype_name = dtype.name
        thread_local.dtype_to_name_cache[dtype] = dtype_name
    return dtype_name


cpdef NDBuffer empty_numpy_like(NDBuffer other, object axis_order='K'):
    cdef axis_order_t axis_order_vec
    cdef OrderFlag order_flag = OrderFlag.C_ORDER
    parse_py_axis_order(order_flag, axis_order_vec, other.layout, axis_order)
    cdef NDBuffer out = _no_data_dense_like(other, &axis_order_vec, order_flag)
    _set_flags(out, is_wrapping_tensor=True)
    out.data_device_id = NDBUFFER_CPU_DEVICE_ID
    out.data = _numpy.ndarray(shape=out.shape, dtype=_name_to_dtype(out.dtype_name), strides=out.strides_in_bytes)
    out.data_ptr = out.data.ctypes.data
    return out


cpdef NDBuffer wrap_numpy_array(object array):
    cdef NDBuffer out = NDBuffer()
    _set_flags(out, is_wrapping_tensor=True)
    out.data = array
    out.data_ptr = array.ctypes.data
    out.data_device_id = NDBUFFER_CPU_DEVICE_ID
    out.dtype_name = _dtype_to_name(array.dtype)
    # accessing array.shape and array.strides is slow, using
    # numpy's C API here could be solution, but that comes with the
    # build-time dependency and compatibility constraints.
    out.layout = create_layout(array.shape, array.strides, array.itemsize, strides_in_bytes=True)
    return out


cpdef NDBuffer wrap_cupy_array(object array):
    cdef NDBuffer out = NDBuffer()
    _set_flags(out, is_wrapping_tensor=True)
    out.data = array
    out.data_ptr = array.data.ptr
    out.data_device_id = array.device.id
    out.dtype_name = _dtype_to_name(array.dtype)
    out.layout = create_layout(array.shape, array.strides, array.itemsize, strides_in_bytes=True)
    return out


cpdef str is_c_or_f(object shape, object strides):
    if strides is None:
        return "C"
    cdef OrderFlag order_flag = OrderFlag.C_ORDER
    cdef shape_t shape_vec
    cdef strides_t strides_vec
    tuple2vec(shape_vec, shape)
    tuple2vec(strides_vec, strides)
    cdef int ndim = shape_vec.size()
    if <size_t>ndim != strides_vec.size():
        raise ValueError("Shape and strides must have the same length")
    if _is_c_or_f(order_flag, shape_vec, strides_vec, ndim):
        return "C" if order_flag == OrderFlag.C_ORDER else "F"
    return "K"
