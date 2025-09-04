# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from libcpp.vector cimport vector
from libcpp.algorithm cimport swap
from libc.stdint cimport intptr_t

cdef extern from "nd_consts.h":
    cdef int NDBUFFER_MAX_NDIM


@cython.final
cdef class Layout:
    def __cinit__(Layout self):
        self.shape = shape_t()
        self.strides = strides_t()
        self.volume = 0
        self.ndim = 0
        self.itemsize = 0

    def __repr__(Layout self):
        return (
            f"Layout(shape={self.shape}, strides={self.strides}, itemsize={self.itemsize})"
        )


cdef extern from *:
    """
    #include <cmath>
    int64_t c_abs(int64_t x){
        return std::abs(x);
    }
    """
    int64_t c_abs(int64_t x) nogil


cdef extern from *:
    """
    #include <algorithm>
    #include <vector>
    #include <numeric>
    void _get_axis_order(int ndim, std::vector<int>& indices, const std::vector<int64_t>& strides, const std::vector<int64_t>& shape){
        indices.resize(ndim);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&strides, &shape](int i, int j) {
                    int64_t stride_i = c_abs(strides[i]);
                    int64_t stride_j = c_abs(strides[j]);
                    if (stride_i != stride_j) {
                        return stride_i > stride_j;
                    }
                    int64_t shape_i = shape[i];
                    int64_t shape_j = shape[j];
                    if (shape_i != shape_j) {
                        return shape_i > shape_j;
                    }
                    return i < j;
                  }
                 );
    }
    """
    void _get_axis_order(int ndim, axis_order_t& indices, strides_t& strides, shape_t& shape) nogil


cdef int tuple2vec(vector_type &vec, object t) except -1:
    cdef int ndim = len(t)
    vec.clear()
    vec.reserve(ndim)
    for i in range(ndim):
        vec.push_back(t[i])
    return 0


cdef int64_t _set_c_strides(strides_t& strides, shape_t& shape) except -1 nogil:
    cdef int ndim = shape.size()
    strides.resize(ndim)
    cdef int64_t stride = 1
    cdef int i = ndim - 1
    while i >= 0:
        strides[i] = stride
        stride *= shape[i]
        i -= 1
    return stride


cdef int64_t _set_f_strides(strides_t& strides, shape_t& shape) except -1 nogil:
    cdef int ndim = shape.size()
    strides.clear()
    strides.reserve(ndim)
    cdef int64_t stride = 1
    cdef int i = 0
    while i < ndim:
        strides.push_back(stride)
        stride *= shape[i]
        i += 1
    return stride


cdef int64_t _set_strides(strides_t& strides, shape_t& shape, axis_order_t& axis_order) except -1 nogil:
    cdef int ndim = shape.size()
    if ndim > NDBUFFER_MAX_NDIM:
        raise ValueError(f"Unsupported number of dimensions: {ndim}. Max supported ndim is {NDBUFFER_MAX_NDIM}")
    strides.resize(ndim)
    cdef int64_t stride = 1
    cdef int i = ndim - 1
    cdef axes_mask_t axis_order_mask = 0
    cdef axes_mask_t axis_mask
    cdef axis_t axis
    while i >= 0:
        axis = axis_order[i]
        if axis < 0 or axis >= ndim:
            raise ValueError(f"Invalid axis order: axis {axis} out of range for {ndim}D tensor")
        axis_mask = 1 << axis
        if axis_order_mask & axis_mask:
            raise ValueError(f"The axis order must be a permutation. Axis {axis} appears multiple times.")
        axis_order_mask |= axis_mask
        strides[axis] = stride
        stride *= shape[axis]
        i -= 1
    return stride


cdef int zero_strides(strides_t& strides, int ndim) except -1 nogil:
    strides.clear()
    strides.resize(ndim, 0)
    return 0


cdef int64_t set_strides_in_order(strides_t& strides, shape_t &shape, OrderFlag order_flag, axis_order_t *axis_order=NULL) except -1 nogil:
    cdef int ndim = shape.size()
    cdef int64_t volume
    if order_flag == C_ORDER:
        volume = _set_c_strides(strides, shape)
    elif order_flag == F_ORDER:
        volume = _set_f_strides(strides, shape)
    elif order_flag == CUSTOM_PERMUTATION:
        if not axis_order:
            raise ValueError("axis_order is required for CUSTOM_PERMUTATION")
        volume = _set_strides(strides, shape, axis_order[0])
    else:
        raise ValueError("Invalid axis order flag")
    if volume == 0:
        zero_strides(strides, ndim)
    return volume


@cython.overflowcheck(True)
cdef int64_t overflow_checked_volume(shape_t& shape) except? -1 nogil:
    cdef int64_t vol = 1
    for i in range(shape.size()):
        vol *= shape[i]
    return vol


cdef int divide_strides(strides_t &strides, int ndim, int itemsize) except -1 nogil:
    for i in range(ndim):
        strides[i] //= itemsize
    return 0


cdef tuple get_strides_in_bytes_tuple(Layout layout):
    cdef int ndim = layout.ndim
    cdef strides_t strides_in_bytes
    strides_in_bytes.reserve(ndim)
    cdef int itemsize = layout.itemsize
    for i in range(ndim):
        strides_in_bytes.push_back(layout.strides[i] * itemsize)
    return tuple(strides_in_bytes)


cdef int set_shape_ndim_volume_tuple(Layout layout, object shape) except -1:
    tuple2vec(layout.shape, shape)
    cdef int ndim = layout.shape.size()
    if ndim > NDBUFFER_MAX_NDIM:
        raise ValueError(f"Unsupported number of dimensions: {ndim}. Max supported ndim is {NDBUFFER_MAX_NDIM}")
    layout.ndim = ndim
    for i in range(ndim):
        if layout.shape[i] < 0:
            raise ValueError("extents must be non-negative")
    layout.volume = overflow_checked_volume(layout.shape)
    return 0


cdef int set_strides_tuple(Layout layout, object strides, bint strides_in_bytes) except -1:
    if len(strides) != layout.ndim:
        raise ValueError("strides must have the same length as shape")
    tuple2vec(layout.strides, strides)
    if strides_in_bytes:
        divide_strides(layout.strides, layout.ndim, layout.itemsize)
    return 0


cdef int set_itemsize(Layout layout, int itemsize) except -1 nogil:
    if itemsize <= 0:
        raise ValueError("itemsize must be positive")
    if itemsize & (itemsize - 1):
        raise ValueError("itemsize must be a power of two")
    layout.itemsize = itemsize
    return 0


cdef Layout create_layout_without_strides(object shape, int itemsize):
    cdef Layout layout = Layout()
    set_shape_ndim_volume_tuple(layout, shape)
    set_itemsize(layout, itemsize)
    return layout


cdef Layout create_layout(object shape, object strides, int itemsize, bint strides_in_bytes):
    cdef Layout layout = Layout()
    set_shape_ndim_volume_tuple(layout, shape)
    set_itemsize(layout, itemsize)
    set_strides_tuple(layout, strides, strides_in_bytes)
    return layout


cdef Layout copy_layout(Layout other):
    cdef Layout layout = Layout()
    layout.ndim = other.ndim
    layout.shape = other.shape
    layout.strides = other.strides
    layout.itemsize = other.itemsize
    layout.volume = other.volume
    return layout


cdef Layout empty_layout_with_dtype_like(Layout other):
    cdef Layout layout = Layout()
    layout.ndim = 0
    layout.volume = 1
    layout.shape = shape_t()
    layout.strides = strides_t()
    layout.itemsize = other.itemsize
    return layout


cdef bint is_overlapping_layout(Layout sorted_layout) except -1 nogil:
    """
    Assumes the layout is sorted in C order, i.e. strides increase from right to left.
    Checks for each stride, if it is bigger than maximal offset that can be reached
    with the extents of smaller strides. If so, any two elements cannot map to the same
    offset. While the inverse is not necessarily true, the check is cheap and enough
    to mark as non-overlapping layouts that arise from permuting and slicing of a dense
    tensor.
    """
    cdef int64_t cur_max_offset = 0
    cdef int i = sorted_layout.ndim - 1
    cdef int64_t stride
    while i >= 0:
        stride = c_abs(sorted_layout.strides[i])
        if cur_max_offset >= stride:
            return True
        cur_max_offset += stride * (sorted_layout.shape[i] - 1)
        i -= 1
    return False


cdef bint is_overlapping_layout_in_order(Layout layout, axis_order_t& axis_order) except -1 nogil:
    """
    Same as is_overlapping_layout, but requires passing a permutation of axes so that
    stride[axis_order[i - 1]] >= stride[axis_order[i]] for all i.
    """
    cdef int64_t cur_max_offset = 0
    cdef int i = layout.ndim - 1
    cdef int64_t stride
    cdef axis_t axis
    cdef extent_t extent
    while i >= 0:
        axis = axis_order[i]
        extent = layout.shape[axis]
        if extent != 1:
            stride = c_abs(layout.strides[axis])
            if cur_max_offset >= stride:
                return True
            cur_max_offset += stride * (extent - 1)
        i -= 1
    return False


cdef int64_t transpose_squeeze_zeros_ones_layout(Layout out_layout, Layout in_layout, axis_order_t& axis_order) except -1 nogil:
    cdef int ndim = in_layout.ndim
    out_layout.shape.clear()
    out_layout.shape.reserve(ndim)
    out_layout.strides.clear()
    out_layout.strides.reserve(ndim)
    cdef int out_ndim = 0
    cdef extent_t extent
    cdef int64_t volume = 1
    cdef axis_t axis
    for i in range(ndim):
        axis = axis_order[i]
        extent = in_layout.shape[axis]
        if extent == 0:
            out_layout.shape.clear()
            out_layout.strides.clear()
            out_layout.shape.push_back(0)
            out_layout.strides.push_back(0)
            out_layout.ndim = 1
            out_layout.volume = 0
            return 0
        if extent != 1:
            out_layout.shape.push_back(extent)
            out_layout.strides.push_back(in_layout.strides[axis])
            out_ndim += 1
            volume *= extent
    out_layout.ndim = out_ndim
    out_layout.volume = volume
    return volume


cdef int transpose_layout(Layout layout, axis_order_t& axis_order) except -1 nogil:
    cdef int ndim = layout.ndim
    cdef shape_t new_shape
    cdef strides_t new_strides
    new_shape.reserve(ndim)
    new_strides.reserve(ndim)
    cdef axis_t axis
    for i in range(ndim):
        axis = axis_order[i]
        new_shape.push_back(layout.shape[axis])
        new_strides.push_back(layout.strides[axis])
    swap(layout.shape, new_shape)
    swap(layout.strides, new_strides)
    return 0


cdef bint _is_c_contiguous_layout(shape_t& shape, strides_t& strides, int ndim) except -1 nogil:
    cdef int64_t stride = 1
    cdef int64_t j = ndim - 1
    cdef extent_t extent
    while j >= 0:
        extent = shape[j]
        if extent != 1:
            if strides[j] != stride:
                return False
            stride *= shape[j]
        j -= 1
    return True


cdef bint is_c_contiguous_layout(Layout sorted_layout) except -1 nogil:
    return _is_c_contiguous_layout(sorted_layout.shape, sorted_layout.strides, sorted_layout.ndim)


cdef bint _is_f_contiguous_layout(shape_t& shape, strides_t& strides, int ndim) except -1 nogil:
    cdef int64_t stride = 1
    cdef int64_t j = 0
    cdef extent_t extent
    while j < ndim:
        extent = shape[j]
        if extent != 1:
            if strides[j] != stride:
                return False
            stride *= shape[j]
        j += 1
    return True


cdef bint is_f_contiguous_layout(Layout sorted_layout) except -1 nogil:
    return _is_f_contiguous_layout(sorted_layout.shape, sorted_layout.strides, sorted_layout.ndim)


cdef bint is_contiguous_layout_in_order(Layout layout, axis_order_t& axis_order) except -1 nogil:
    cdef int64_t stride = 1
    cdef int64_t j = layout.ndim - 1
    cdef axis_t axis
    cdef extent_t extent
    while j >= 0:
        axis = axis_order[j]
        extent = layout.shape[axis]
        if extent != 1:
            if layout.strides[axis] != stride:
                return False
            stride *= extent
        j -= 1
    return True


cdef int _squeeze_extents(shape_t& out_shape, strides_t& out_strides, int ndim, shape_t& shape, strides_t& strides) except -1 nogil:
    cdef int group_start = 0
    cdef int group_end = 0
    cdef int64_t group_vol
    cdef int64_t group_stride
    cdef int out_i = 0
    while group_start < ndim:
        group_end = group_start + 1
        group_vol = shape[group_start]
        if group_vol != 1:
            group_stride = strides[group_start]
            while group_end < ndim:
                # whatever the stride for extent one, we can ignore it
                if shape[group_end] == 1:
                    group_end += 1
                elif group_stride == strides[group_end] * shape[group_end]:
                    group_vol *= shape[group_end]
                    group_stride = strides[group_end]
                    group_end += 1
                else:
                    break
            out_shape[out_i] = group_vol
            out_strides[out_i] = group_stride
            out_i += 1
        group_start = group_end
    return out_i


cdef int squeeze_layout(shape_t& out_shape, strides_t& out_strides, Layout in_layout) except? -1 nogil:
    cdef int ndim = in_layout.ndim
    out_shape.resize(ndim)
    out_strides.resize(ndim)
    cdef int out_ndim = _squeeze_extents(out_shape, out_strides, ndim, in_layout.shape, in_layout.strides)
    if out_ndim != ndim:
        out_shape.resize(out_ndim)
        out_strides.resize(out_ndim)
    return out_ndim


cdef int _squeeze_extents_together(shape_t& shape_a, strides_t& strides_a, shape_t& shape_b, strides_t& strides_b, int ndim) except -1 nogil:
    cdef int group_start = 0
    cdef int group_end = 0
    cdef int64_t group_vol
    cdef int64_t group_stride_a
    cdef int64_t group_stride_b
    cdef int out_i = 0
    cdef extent_t extent
    while group_start < ndim:
        # find group start, i.e. an extent where respective
        # extent size is equal in both layouts, otherwise
        # just copy the respective extents and strides
        extent = shape_a[group_start]
        if extent != shape_b[group_start]:
            shape_a[out_i] = extent
            strides_a[out_i] = strides_a[group_start]
            shape_b[out_i] = shape_b[group_start]
            strides_b[out_i] = strides_b[group_start]
            out_i += 1
            group_start += 1
            continue
        # extend the group as long as both layouts are dense
        group_end = group_start + 1
        group_vol = extent
        group_stride_a = strides_a[group_start]
        group_stride_b = strides_b[group_start]
        while group_end < ndim:
            extent = shape_a[group_end]
            if extent == shape_b[group_end] and group_stride_a == strides_a[group_end] * extent and group_stride_b == strides_b[group_end] * extent:
                group_vol *= extent
                group_stride_a = strides_a[group_end]
                group_stride_b = strides_b[group_end]
                group_end += 1
            else:
                break
        # append the volume of the group and the smallest stride from the group
        shape_a[out_i] = group_vol
        strides_a[out_i] = group_stride_a
        shape_b[out_i] = group_vol
        strides_b[out_i] = group_stride_b
        out_i += 1
        group_start = group_end
    return out_i


cdef int squeeze_layouts_together(Layout layout_a, Layout layout_b, int ndim) except -1 nogil:
    cdef int out_ndim = _squeeze_extents_together(layout_a.shape, layout_a.strides, layout_b.shape, layout_b.strides, ndim)
    if out_ndim != ndim:
        layout_a.shape.resize(out_ndim)
        layout_a.strides.resize(out_ndim)
        layout_a.ndim = out_ndim
        layout_b.shape.resize(out_ndim)
        layout_b.strides.resize(out_ndim)
        layout_b.ndim = out_ndim
    return 0


cdef bint split_strides(Layout new_layout, shape_t& old_shape, strides_t& old_strides) except -1 nogil:
    cdef int old_ndim = old_shape.size()
    cdef int new_ndim = new_layout.ndim
    new_layout.strides.resize(new_ndim)
    cdef int old_i = old_ndim - 1
    cdef int new_i = new_ndim - 1
    cdef extent_t old_extent
    cdef extent_t new_extent
    cdef extent_t group_vol
    cdef stride_t group_stride
    while old_i >= 0:
        old_extent = old_shape[old_i]
        group_vol = 1
        group_stride = old_strides[old_i]
        while new_i >= 0 and group_vol < old_extent:
            new_extent = new_layout.shape[new_i]
            if new_extent == 0:
                return False
            group_vol *= new_extent
            new_layout.strides[new_i] = group_stride
            group_stride *= new_extent
            new_i -= 1
        if group_vol != old_extent:
            return False
        old_i -= 1
    return True


cdef int64_t _gcd(int64_t a, int64_t b) except -1 nogil:
    while b != 0:
        a, b = b, a % b
    return a


cdef int _max_compatible_vec_size(Layout layout, intptr_t ptr, int max_vec_size, int max_itemsize) except -1 nogil:
    cdef int one_less_ndim = layout.ndim - 1
    if one_less_ndim < 0 or layout.strides[one_less_ndim] != 1:
        return 1
    cdef int itemsize = layout.itemsize
    cdef int max_compatible = min(max_vec_size, max(max_itemsize // itemsize, 1))
    if max_compatible <= 1:
        return 1
    cdef int64_t n_element_offset = ptr // itemsize
    # make sure the pointer is aligned
    if n_element_offset * itemsize != ptr:
        return 1
    max_compatible = _gcd(max_compatible, c_abs(n_element_offset))
    cdef extent_t last_extent = layout.shape[one_less_ndim]
    max_compatible = _gcd(max_compatible, last_extent)
    if max_compatible == 1:
        return 1
    for i in range(one_less_ndim):
        max_compatible = _gcd(max_compatible, c_abs(layout.strides[i]))
    return max_compatible


cdef int _vectorize_unsafe(Layout layout, int vec_size) except -1 nogil:
    """
    Vectorizes the layout: i.e. multiplies the itemsize by vec_size
    and divides the strides and last extent by the vector size.
    The function does not perform checks assuring that the vec_size is compatible
    with the layout. You should call the _max_compatible_vec_size function first.
    """

    if vec_size == 1 or layout.ndim <= 0:
        return 1
    cdef int one_less_ndim = layout.ndim - 1
    cdef extent_t last_extent = layout.shape[one_less_ndim] // vec_size
    if last_extent != 1:
        layout.shape[one_less_ndim] = last_extent
    else:
        layout.ndim = one_less_ndim
        layout.shape.resize(one_less_ndim)
        layout.strides.resize(one_less_ndim)
    cdef stride_t* strides_data = layout.strides.data()
    for i in range(one_less_ndim):
        strides_data[i] //= vec_size
    layout.itemsize *= vec_size
    layout.volume //= vec_size
    return vec_size


cdef int vectorize_together(Layout layout_a, intptr_t ptr_a, Layout layout_b, intptr_t ptr_b, int max_vec_size=8, int max_itemsize=8) except -1 nogil:
    """
    Find the maximal itemsize that can be used to access elements of both tensors.
    Given vec_size=new_itemsize/itemsize:
        * last extent must be divisible by vec_size
        * last stride must be 1
        * all other strides must be divisible by vec_size
        * the base pointers must be aligned to new_itemsize
    While the copy kernel supports itemsizes up to 16 bytes, we limit the default max itemsize to 8,
    as the itemsize 16 brings the least performance boost on average, and can even degrade it in some cases.
    """
    cdef int vec_size = _max_compatible_vec_size(layout_a, ptr_a, max_vec_size, max_itemsize)
    vec_size = _max_compatible_vec_size(layout_b, ptr_b, vec_size, max_itemsize)
    _vectorize_unsafe(layout_a, vec_size)
    _vectorize_unsafe(layout_b, vec_size)
    return vec_size


cdef int get_axis_order(axis_order_t& axis_order, Layout layout) except -1 nogil:
    _get_axis_order(layout.ndim, axis_order, layout.strides, layout.shape)
    return 0


cdef axes_mask_t get_contiguous_axes_up_to_vol(int64_t &suffix_vol, axes_mask_t forbidden_axes, int64_t max_volume, Layout layout, int* axis_order=NULL) except? -1 nogil:
    cdef int i = layout.ndim - 1
    suffix_vol = 1
    cdef axes_mask_t axes_mask = 0
    cdef axes_mask_t axis_flag
    cdef int axis
    while i >= 0 and suffix_vol < max_volume:
        if axis_order:
            axis = axis_order[i]
        else:
            axis = i
        axis_flag = 1 << axis
        if forbidden_axes & axis_flag:
            break
        if c_abs(layout.strides[axis]) > suffix_vol:
            break
        axes_mask |= axis_flag
        suffix_vol *= layout.shape[axis]
        i -= 1
    return axes_mask


cdef int parse_py_axis_order(OrderFlag& order_flag, axis_order_t& axis_order_vec, Layout other, object axis_order_arg) except -1:
    if axis_order_arg == 'C':
        order_flag = OrderFlag.C_ORDER
        return 0
    elif axis_order_arg == 'F':
        order_flag = OrderFlag.F_ORDER
        return 0
    elif axis_order_arg == 'K':
        get_axis_order(axis_order_vec, other)
        if is_overlapping_layout_in_order(other, axis_order_vec):
            # for overlapping layouts, e.g. broadcast extents (with strides 0),
            # the order is quite arbitrary, default to C order
            order_flag = OrderFlag.C_ORDER
        else:
            order_flag = OrderFlag.CUSTOM_PERMUTATION
        return 0
    elif isinstance(axis_order_arg, tuple):
        tuple2vec(axis_order_vec, axis_order_arg)
        order_flag = OrderFlag.CUSTOM_PERMUTATION
        return 0
    raise ValueError(f"Invalid axis order: {axis_order_arg}")


cdef bint is_c_or_f(OrderFlag& order_flag, shape_t& shape, strides_t& strides, int ndim) except -1 nogil:
    if _is_c_contiguous_layout(shape, strides, ndim):
        order_flag = OrderFlag.C_ORDER
        return True
    if _is_f_contiguous_layout(shape, strides, ndim):
        order_flag = OrderFlag.F_ORDER
        return True
    return False


cdef bint is_c_or_f_layout(OrderFlag& order_flag, Layout layout) except -1 nogil:
    return is_c_or_f(order_flag, layout.shape, layout.strides, layout.ndim)
