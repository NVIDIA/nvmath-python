# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from libc.stdint cimport intptr_t
from .data_layout cimport Layout, axis_order_t, OrderFlag


@cython.final
cdef class NDBuffer:
    cdef Layout layout
    cdef readonly object data
    cdef int data_device_id
    cdef int flags
    cdef readonly intptr_t data_ptr
    cdef readonly str dtype_name

    # possibly lazy evaluated properties
    # accessible publicly in python
    cdef prop_strides
    cdef prop_shape
    cdef prop_device
    cdef prop_device_id
    cdef prop_strides_in_bytes


cdef NDBuffer _no_data_dense_like(NDBuffer other, axis_order_t* axis_order_vec, OrderFlag order_flag)
cdef NDBuffer _no_data_like(NDBuffer other, bint copy_data)
cdef int _set_flags(NDBuffer ndbuffer, bint is_wrapping_tensor=*) except -1 nogil

cpdef NDBuffer wrap_external(data, intptr_t ptr, str dtype_name, object shape, object strides, int device_id, int itemsize, bint strides_in_bytes=*)
cpdef NDBuffer empty(object shape, int device_id, str dtype_name, int itemsize, object axis_order=*, object strides=*, object host_memory_pool=*, object device_memory_pool=*, object stream=*, bint strides_in_bytes=*, object logger=*)
cpdef NDBuffer empty_like(NDBuffer other, object axis_order=*, object device_id=*, object stream=*, object host_memory_pool=*, object device_memory_pool=*, object logger=*)
cpdef int copy_into(NDBuffer dst, NDBuffer src, object stream, object host_memory_pool=*, object device_memory_pool=*, object logger=*) except -1
cpdef NDBuffer reshaped_view(NDBuffer other, object shape, object logger=*)
