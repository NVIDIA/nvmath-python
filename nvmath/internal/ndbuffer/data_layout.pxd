# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from libc.stdint cimport int64_t, uint32_t, intptr_t
from libcpp cimport vector

ctypedef int64_t extent_t
ctypedef int64_t stride_t
ctypedef int axis_t

ctypedef uint32_t axes_mask_t

ctypedef vector.vector[extent_t] shape_t
ctypedef vector.vector[stride_t] strides_t
ctypedef vector.vector[axis_t] axis_order_t


ctypedef fused vector_type:
    shape_t
    strides_t
    axis_order_t


cdef enum OrderFlag:
    C_ORDER = 0
    F_ORDER = 1
    CUSTOM_PERMUTATION = 2


@cython.final
cdef class Layout:
    cdef readonly shape_t shape
    cdef readonly strides_t strides
    cdef readonly int64_t volume
    cdef readonly int ndim
    cdef readonly int itemsize


@cython.overflowcheck(True)
cdef inline int64_t size_in_bytes(Layout layout) except? -1 nogil:
    return layout.volume * layout.itemsize


cdef int tuple2vec(vector_type &vec, object t) except -1
cdef int set_strides_tuple(Layout layout, object strides, bint strides_in_bytes) except -1
cdef int64_t set_strides_in_order(strides_t& strides, shape_t &shape, OrderFlag order_flag, axis_order_t *axis_order=*) except -1 nogil
cdef int64_t overflow_checked_volume(shape_t& shape) except? -1 nogil
cdef int zero_strides(strides_t& strides, int ndim) except -1 nogil
cdef int divide_strides(strides_t &strides, int ndim, int itemsize) except -1 nogil
cdef tuple get_strides_in_bytes_tuple(Layout layout)
cdef Layout create_layout_without_strides(object shape, int itemsize)
cdef Layout create_layout(object shape, object strides, int itemsize, bint strides_in_bytes)
cdef Layout copy_layout(Layout src)
cdef Layout empty_layout_with_dtype_like(Layout src)
cdef bint is_overlapping_layout(Layout sorted_layout) except -1 nogil
cdef bint is_overlapping_layout_in_order(Layout layout, axis_order_t& axis_order) except -1 nogil
cdef int64_t transpose_squeeze_zeros_ones_layout(Layout out_layout, Layout in_layout, axis_order_t& axis_order) except -1 nogil
cdef int transpose_layout(Layout layout, axis_order_t& axis_order) except -1 nogil
cdef bint is_c_contiguous_layout(Layout sorted_layout) except -1 nogil
cdef bint is_f_contiguous_layout(Layout sorted_layout) except -1 nogil
cdef bint is_contiguous_layout_in_order(Layout layout, axis_order_t& axis_order) except -1 nogil
cdef int squeeze_layouts_together(Layout layout_a, Layout layout_b, int ndim) except -1 nogil
cdef int squeeze_layout(shape_t& out_shape, strides_t& out_strides, Layout in_layout) except? -1 nogil
cdef bint split_strides(Layout new_layout, shape_t& old_shape, strides_t& old_strides) except -1 nogil
cdef int vectorize_together(Layout layout_a, intptr_t ptr_a, Layout layout_b, intptr_t ptr_b, int max_vec_size=*, int max_itemsize=*) except -1 nogil
cdef int get_axis_order(axis_order_t& axis_order, Layout layout) except -1 nogil
cdef axes_mask_t get_contiguous_axes_up_to_vol(int64_t &suffix_vol, axes_mask_t forbidden_axes, int64_t max_volume, Layout layout, int* axis_order=*) except? -1 nogil
cdef int parse_py_axis_order(OrderFlag& order_flag, axis_order_t& axis_order_vec, Layout other, object axis_order_arg) except -1
cdef bint is_c_or_f(OrderFlag& order_flag, shape_t& shape, strides_t& strides, int ndim) except -1 nogil
cdef bint is_c_or_f_layout(OrderFlag& order_flag, Layout layout) except -1 nogil
