# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import threading
cimport cython
from libc.stdint cimport int64_t, intptr_t
from libcpp.vector cimport vector
from libcpp.algorithm cimport swap
from libcpp.memory cimport unique_ptr
from libcpp.functional cimport function
from .data_layout cimport (
    Layout, strides_t, shape_t, stride_t, extent_t, axis_order_t, axes_mask_t, axis_t,
    squeeze_layout, transpose_layout, get_axis_order,
    get_contiguous_axes_up_to_vol
)
from .jit cimport get_kernel, discover_includes, register_includes
from ..bindings cimport launch_kernel, Dim3

ctypedef unique_ptr[void, function[void(void*)]] args_t


cdef extern from "limits.h":
    cdef int INT_MAX
    cdef int INT_MIN


cdef extern from "nd_consts.h":
    cdef int NDBUFFER_MAX_NDIM


cdef extern from *:
    """
    #include <cmath>
    #include <memory>
    #include <type_traits>
    #include "copy_kernel/args.h"
    template <int N>
    void _get_kernel_args_ndim(std::unique_ptr<void, std::function<void(void*)>>& args, void *dst_ptr, const void *src_ptr, int dst_ndim, int src_ndim, int64_t* dst_shape, int64_t* src_shape, int64_t* dst_strides, int64_t* src_strides, int64_t grid_arg){
        auto deleter = [](void *p) {
            delete (static_cast<nvmath::KernelArgs<N>*>(p));
        };
        std::unique_ptr<nvmath::KernelArgs<N>, std::function<void(void*)>> ptr{new nvmath::KernelArgs<N>, std::move(deleter)};
        ptr->dst_ptr = dst_ptr;
        ptr->src_ptr = src_ptr;
        for (int i = 0; i < dst_ndim; i++) {
            ptr->dst_shape[i] = dst_shape[i];
            ptr->dst_strides[i] = dst_strides[i];
        }
        for (int i = 0; i < src_ndim; i++) {
            ptr->src_shape[i] = src_shape[i];
            ptr->src_strides[i] = src_strides[i];
        }
        ptr->grid_arg = grid_arg;
        args = std::move(ptr);
    }
    template <typename F, int i = 1, int max_ndim = NDBUFFER_MAX_NDIM>
    void with_ndim(int ndim, F&& f) {
        if constexpr (i <= max_ndim) {
            if (i == ndim) {
                f(std::integral_constant<int, i>());
            } else {
                with_ndim<F, i + 1, max_ndim>(ndim, std::forward<F>(f));
            }
        } else if constexpr (i > max_ndim) {
            throw std::runtime_error("unsupported ndim");
        }
    }
    void _get_kernel_args(std::unique_ptr<void, std::function<void(void*)>>& args, void *dst_ptr, const void *src_ptr, int dst_ndim, int src_ndim, int64_t* dst_shape, int64_t* src_shape, int64_t* dst_strides, int64_t* src_strides, int64_t grid_arg) {
        int ndim = dst_ndim > src_ndim ? dst_ndim : src_ndim;
        with_ndim(ndim, [&](auto static_ndim_holder) {
            constexpr int static_ndim = decltype(static_ndim_holder)::value;
            _get_kernel_args_ndim<static_ndim>(args, dst_ptr, src_ptr, dst_ndim, src_ndim, dst_shape, src_shape, dst_strides, src_strides, grid_arg);
        });
    }
    """
    void _get_kernel_args(args_t& args, void *dst_ptr, const void *src_ptr, int dst_ndim, int src_ndim, int64_t* dst_shape, int64_t* src_shape, int64_t* dst_strides, int64_t* src_strides, int64_t grid_arg) except + nogil


thread_local = threading.local()

cdef _register_copy_kernel_includes(object logger):
    cdef str copy_kernel_includes_key = "copy_kernel"
    if not hasattr(thread_local, "registered_header_names"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        copy_kernel_dir = os.path.join(current_dir, "copy_kernel")
        copy_kernel_impl_dir = os.path.join(copy_kernel_dir, "copy_kernel_impl")
        include_dirs = [(copy_kernel_dir, copy_kernel_dir), (copy_kernel_dir, copy_kernel_impl_dir)]
        header_names, headers = discover_includes(include_dirs)
        if len(header_names) == 0:
            raise RuntimeError(f"No headers found for copy kernel at {copy_kernel_dir}")
        register_includes(copy_kernel_includes_key, header_names, headers)
        thread_local.registered_header_names = header_names
        if logger is not None:
            logger.debug(f"Registered copy kernel includes: {header_names}")
    return copy_kernel_includes_key


cdef int get_kernel_args(args_t& args, Layout dst, Layout src, intptr_t dst_ptr, intptr_t src_ptr, int64_t grid_arg) except-1 nogil:
    _get_kernel_args(args, <void*>dst_ptr, <const void*>src_ptr, dst.ndim, src.ndim, dst.shape.data(), src.shape.data(), dst.strides.data(), src.strides.data(), grid_arg)
    return 0


cdef inline int _logging_helper(object logger, str msg, fst=None, snd=None, third=None) except -1 nogil:
    with cython.gil:
        logger.debug(msg.format(fst=fst, snd=snd, third=third))
    return 0


cdef inline int _logging_log_axis_order(object logger, str msg, axis_order_t& fst) except -1 nogil:
    with cython.gil:
        logger.debug(msg.format(fst=fst))
    return 0


cdef inline int _logging_log_int(object logger, str msg, int fst=0, int snd=0, int third=0) except -1 nogil:
    with cython.gil:
        logger.debug(msg.format(fst=fst, snd=snd, third=third))
    return 0


cdef inline int64_t _div_ceil(int64_t a, int64_t b) except?-1 nogil:
    return (a + (b - 1)) // b


cdef int _stride_limits(int64_t &min_offset, int64_t &max_offset, Layout layout) except?-1 nogil:
    cdef stride_t local_min_offset = 0
    cdef stride_t local_max_offset = 0
    cdef stride_t stride
    cdef extent_t extent
    for i in range(layout.ndim):
        stride = layout.strides[i]
        extent = layout.shape[i]
        # note, the extent must be positive:
        # 1. negative extent is not allowed
        # 2. if any extent is 0, the volume is 0 and we should
        # have early exited before trying to launch the kernel
        if stride <= 0:
            local_min_offset += (extent - 1) * stride
        else:
            local_max_offset += (extent - 1) * stride
    min_offset = min(min_offset, local_min_offset)
    max_offset = max(max_offset, local_max_offset)
    return 0


cdef bint _needs_wide_strides(int64_t grid_volume, Layout dst, Layout src) except?-1 nogil:
    # grid_volume, i.e the block_size * num_blocks
    if grid_volume > INT_MAX:
        return True
    cdef int64_t min_offset = 0
    cdef int64_t max_offset = 0
    _stride_limits(min_offset, max_offset, dst)
    _stride_limits(min_offset, max_offset, src)
    # forbid INT_MIN too for:
    # 1. abs() to be safe
    # 2. it us used as out_of_bounds_sentinel in the transpose copy kernel
    if min_offset <= INT_MIN or max_offset > INT_MAX:
        return True
    return False


cdef bint _needs_grid_stride_loop(int64_t &cuda_num_blocks, int64_t num_blocks) except?-1 nogil:
    if num_blocks <= INT_MAX:
        cuda_num_blocks = num_blocks
        return False
    else:
        cuda_num_blocks = INT_MAX
        return True


cdef bint _get_transpose_num_blocks(int64_t &num_blocks, int64_t &cuda_num_blocks, int64_t block_size, int block_height, int transposed_dim, Layout layout) except?-1 nogil:
    cdef int ndim = layout.ndim
    cdef int64_t volume = 1
    for i in range(transposed_dim + 1):
        volume *= layout.shape[i]
    volume = _div_ceil(volume, block_height) * block_height
    for i in range(transposed_dim + 1, ndim):
        volume *= layout.shape[i]
    num_blocks = _div_ceil(volume, block_size)
    if num_blocks <= INT_MAX:
        cuda_num_blocks = num_blocks
        return False
    else:
        cuda_num_blocks = INT_MAX
        return True


cdef str _emit_transpose_kernel_code(Layout dst, Layout src, bint needs_wide_strides, bint needs_grid_stride_loop, int block_height, int block_width, char reading_order, int transposed_dim):
    if dst.ndim != src.ndim:
        raise ValueError("dst_ndim and src_ndim must be equal")
    cdef str stride_t_str = "int64_t" if needs_wide_strides else "int32_t"
    cdef str needs_grid_stride_loop_str = "true" if needs_grid_stride_loop else "false"
    kernel_code = f"""
    #include <transposed.h>
    TRANSPOSE_KERNEL({stride_t_str}, {dst.ndim}, {dst.itemsize}, {needs_grid_stride_loop_str}, {transposed_dim}, {block_height}, {block_width}, '{chr(reading_order)}')
    """
    return kernel_code


cdef intptr_t _get_transpose_copy_kernel(Layout dst, Layout src, bint needs_wide_strides, bint needs_grid_stride_loop, int block_height, int block_width, char reading_order, int transposed_dim, int device_id, object logger) except? 0:
    cdef str kernel_code = _emit_transpose_kernel_code(dst, src, needs_wide_strides, needs_grid_stride_loop, block_height, block_width, reading_order, transposed_dim)
    cdef str include_key = _register_copy_kernel_includes(logger)
    return get_kernel(kernel_code, "transpose_copy", device_id, include_key, logger)


cdef str _emit_elementwise_kernel_code(Layout dst, Layout src, bint needs_wide_strides, bint needs_grid_stride_loop):
    cdef str stride_t_str = "int64_t" if needs_wide_strides else "int32_t"
    cdef str needs_grid_stride_loop_str = "true" if needs_grid_stride_loop else "false"
    kernel_code = f"""
    #include <elementwise.h>
    ELEMENTWISE_KERNEL({stride_t_str}, {dst.ndim}, {src.ndim}, {dst.itemsize}, {needs_grid_stride_loop_str})
    """
    return kernel_code


cdef intptr_t _get_elementwise_copy_kernel(Layout dst, Layout src, bint needs_wide_strides, bint needs_grid_stride_loop, int device_id, object logger) except? 0:
    cdef str kernel_code = _emit_elementwise_kernel_code(dst, src, needs_wide_strides, needs_grid_stride_loop)
    cdef str include_key = _register_copy_kernel_includes(logger)
    return get_kernel(kernel_code, "elementwise_copy", device_id, include_key, logger)


cdef int _launch_transpose_copy(Layout dst, Layout src, intptr_t dst_ptr, intptr_t src_ptr, int block_height, int block_width, char reading_order, int transposed_dim, int device_id, intptr_t stream, object logger) except -1 nogil:
    cdef int64_t block_size = block_height * block_width
    cdef int64_t num_blocks = 0
    cdef int64_t cuda_num_blocks = 0
    cdef bint needs_grid_stride_loop = _get_transpose_num_blocks(num_blocks, cuda_num_blocks, block_size, block_height, transposed_dim, dst)
    cdef bint needs_wide_strides = _needs_wide_strides(num_blocks * block_size, dst, src)
    cdef args_t args
    get_kernel_args(args, dst, src, dst_ptr, src_ptr, num_blocks)
    cdef Dim3 grid_dim, block_dim
    grid_dim.x = cuda_num_blocks
    grid_dim.y = 1
    grid_dim.z = 1
    block_dim.x = block_size
    block_dim.y = 1
    block_dim.z = 1
    cdef void* args_ptr = args.get()
    cdef intptr_t kernel_fn_ptr
    with cython.gil:
        kernel_fn_ptr = _get_transpose_copy_kernel(dst, src, needs_wide_strides, needs_grid_stride_loop, block_height, block_width, reading_order, transposed_dim, device_id, logger)
        if logger is not None:
            logger.debug(f"Launching transpose copy kernel {kernel_fn_ptr} with grid {grid_dim} and block {block_dim}.")
    launch_kernel(kernel_fn_ptr, <intptr_t>&args_ptr, grid_dim, block_dim, 0, stream)
    return 0


cdef int _launch_elementwise_copy(Layout dst, Layout src, intptr_t dst_ptr, intptr_t src_ptr, int block_size, int device_id, intptr_t stream, object logger) except -1 nogil:
    cdef int64_t volume = dst.volume
    cdef int64_t num_blocks = _div_ceil(volume, block_size)
    cdef int64_t cuda_num_blocks = 0
    cdef bint needs_grid_stride_loop = _needs_grid_stride_loop(cuda_num_blocks, num_blocks)
    cdef bint needs_wide_strides = _needs_wide_strides(num_blocks * block_size, dst, src)
    cdef args_t args
    get_kernel_args(args, dst, src, dst_ptr, src_ptr, volume)
    cdef Dim3 grid_dim, block_dim
    grid_dim.x = cuda_num_blocks
    grid_dim.y = 1
    grid_dim.z = 1
    block_dim.x = block_size
    block_dim.y = 1
    block_dim.z = 1
    cdef void* args_ptr = args.get()
    cdef intptr_t kernel_fn_ptr
    with cython.gil:
        kernel_fn_ptr = _get_elementwise_copy_kernel(dst, src, needs_wide_strides, needs_grid_stride_loop, device_id, logger)
        if logger is not None:
            logger.debug(f"Launching elementwise copy kernel {kernel_fn_ptr} with grid {grid_dim} and block {block_dim}.")
    launch_kernel(kernel_fn_ptr, <intptr_t>&args_ptr, grid_dim, block_dim, 0, stream)
    return 0


cdef int _get_transpose_copy_order(axis_order_t& copy_order, int ndim, int transposed_dim, axes_mask_t src_axes_mask, axis_order_t &src_order) except -1 nogil:
    copy_order.clear()
    copy_order.reserve(ndim)
    cdef axis_t axis
    cdef axes_mask_t axis_flag
    cdef int i = 0
    # the dims that come before the reading dim (and are not part of reading tile)
    # remain in their original order
    while i < transposed_dim:
        axis_flag = 1 << i
        if not (src_axes_mask & axis_flag):
            copy_order.push_back(i)
        i += 1
    # put the reading dims together, in the order of src_order
    i = 0
    while i < ndim:
        axis = src_order[i]
        axis_flag = 1 << axis
        if src_axes_mask & axis_flag:
            copy_order.push_back(axis)
        i += 1
    # resume putting remaining dims in their original order
    i = transposed_dim
    while i < ndim:
        axis_flag = 1 << i
        if not (src_axes_mask & axis_flag):
            copy_order.push_back(i)
        i += 1
    return 0


cdef int _permute_layouts_for_transpose_copy(Layout dst, Layout src, int ndim, int transposed_dim, axes_mask_t src_axes_mask, axis_order_t &src_order, object logger) except -1 nogil:
    cdef axis_order_t copy_order
    _get_transpose_copy_order(copy_order, ndim, transposed_dim, src_axes_mask, src_order)
    transpose_layout(dst, copy_order)
    transpose_layout(src, copy_order)
    if logger is not None:
        _logging_log_axis_order(logger, "The layouts are permuted to place the read dims together: {fst}", copy_order)
        _logging_helper(logger, "Permuted dst: {fst}, src: {snd}", dst, src)
    return 0


cdef int _permute_layouts_with_src_dims_last(Layout dst, Layout src, int ndim, axis_order_t &src_order, axes_mask_t src_axes_mask, object logger) except -1 nogil:
    cdef axis_order_t copy_order
    cdef int n_read_dims = 0
    cdef axes_mask_t axis_flag
    cdef int i = 0
    while i < ndim:
        axis_flag = 1 << i
        if not (src_axes_mask & axis_flag):
            copy_order.push_back(i)
        i += 1
    i = 0
    cdef axis_t axis
    while i < ndim:
        axis = src_order[i]
        axis_flag = 1 << axis
        if src_axes_mask & axis_flag:
            copy_order.push_back(axis)
            n_read_dims += 1
        i += 1
    if n_read_dims < 1:
        return 0
    transpose_layout(dst, copy_order)
    transpose_layout(src, copy_order)
    if logger is not None:
        _logging_log_axis_order(logger, "Use transposed copy with small src dims last. Copy order: {fst}", copy_order)
        _logging_helper(logger, "Permuted dst: {fst}, src: {snd}", dst, src)
    return n_read_dims


cdef int _adjust_layouts_for_transpose_copy(char &reading_order, int &transposed_dim, int &block_height, int &block_width, Layout dst, Layout src, object logger) except -1 nogil:
    # logical tile extents: 16 threads in the column read together
    # and 32 threads in the row write together
    reading_order = b'F'
    block_height = 16
    block_width = 32
    cdef int ndim = dst.ndim
    cdef int n_read_dims = 0
    if ndim < 2 or dst.volume < block_height * block_width:
        return 0
    cdef int64_t suffix_dst_vol = 1, suffix_src_vol = 1
    # we assume the dst strides are already sorted (increasing right-to-left)
    cdef axes_mask_t dst_axes_mask = get_contiguous_axes_up_to_vol(suffix_dst_vol, 0, block_width, dst)
    # not enough contiguous dims in dst
    if suffix_dst_vol < block_width:
        return 0
    # for src extents, we need to find the axes order to check if there are
    # enough contiguous dims
    cdef axis_order_t src_order
    get_axis_order(src_order, src)
    # get first contiguous dims in the src order, stopping as soon as we
    # have at least block_height elements or we encounter extent
    # that is needed for contiogus writes to the dst.
    cdef axes_mask_t src_axes_mask = get_contiguous_axes_up_to_vol(suffix_src_vol, dst_axes_mask, block_height, src, src_order.data())
    # not enough contiguous dims in src
    if suffix_src_vol < 2:
        return 0
    if logger is not None:
        _logging_log_axis_order(logger, "Src order: {fst}", src_order)
        _logging_log_int(logger, "Dst axes mask: {fst}, src axes mask: {snd}", dst_axes_mask, src_axes_mask)
    # Special case: try to use 2D tile even if there are
    # few contiguous elements to read from. This is particularly important for
    # cases like dst = (SOMETHING_SMALL, SOMETHING_BIG) : (SOMETHING_BIG, 1),
    # and the src having reverse stride order. Here, elementwise copy will
    # suffer from little use of L2 cache.
    if suffix_src_vol < block_height:
        # here, we swap the 2d tile reading/writing order: the threads
        # organized in the same logical row should read together
        # and the same logical column should write together
        reading_order = b'C'
        block_height = 32
        block_width = 16
        n_read_dims = _permute_layouts_with_src_dims_last(dst, src, ndim, src_order, src_axes_mask, logger)
        transposed_dim = ndim - n_read_dims - 1
        return n_read_dims
    # we have enough contiguous elements for tiled reading and writing
    # to simplify the kernel (and recompile less) we want to place extents
    # from src_axes_mask together
    cdef int i = 0
    # Find the max of all axes in src_axes_mask.
    # As dst layout is sorted, this will be the innermost/rightmost
    # extent of all src_axes_mask in the dst layout
    while i < ndim:
        if src_axes_mask & (1 << i):
            transposed_dim = i
            n_read_dims += 1
        i += 1
    if logger is not None:
        _logging_log_int(logger, "There are {fst} dims needed for big enough coalesced reads, transposed dim: {snd}", n_read_dims, transposed_dim)
    # if there is one, large enough extent to read from, there's no need
    # to permute the layouts
    if n_read_dims <= 1:
        return n_read_dims
    # otherwise, we permute the layouts to place the read dims together
    _permute_layouts_for_transpose_copy(dst, src, ndim, transposed_dim, src_axes_mask, src_order, logger)
    return n_read_dims


cdef bint _adjust_layouts_for_elementwise_copy(Layout dst, Layout src, object logger) except -1 nogil:
    cdef shape_t sq_dst_shape, sq_src_shape
    cdef strides_t sq_dst_strides, sq_src_strides
    cdef int sq_dst_ndim = squeeze_layout(sq_dst_shape, sq_dst_strides, dst)
    cdef int sq_src_ndim = squeeze_layout(sq_src_shape, sq_src_strides, src)
    # There is a faster kernel specialized if either of the layouts squeezed to 1D.
    # Note, if either layout was squeezed "a bit", but neither of them down to 1D,
    # we prefer keeping the original layouts, as we know the original shapes are equal
    # so the kernel can unravel flat element index once for both layouts.
    if sq_dst_ndim == 1 or sq_src_ndim == 1:
        swap(dst.shape, sq_dst_shape)
        swap(dst.strides, sq_dst_strides)
        dst.ndim = sq_dst_ndim
        swap(src.shape, sq_src_shape)
        swap(src.strides, sq_src_strides)
        src.ndim = sq_src_ndim
        if logger is not None:
            _logging_helper(logger, "At least one of the layouts was squeezed to 1D: dst {fst}, src {snd}", dst, src)
    return True


cdef bint _use_tranpose_copy_maybe(Layout dst, Layout src, intptr_t dst_ptr, intptr_t src_ptr, int device_id, intptr_t stream_ptr, object logger=None) except -1 nogil:
    # Dimension of the tile
    cdef int block_height = 0
    cdef int block_width = 0
    # Dimension in the src/dst tensor that splits the shape in two parts
    # [0, transposed_dim] and [transposed_dim + 1, ndim - 1]
    # for the purpose of traversing it with the 2D tile.
    cdef int transposed_dim = 0
    cdef char reading_order = b'F'
    cdef int n_read_dims = _adjust_layouts_for_transpose_copy(reading_order, transposed_dim, block_height, block_width, dst, src, logger)
    if n_read_dims <= 0:
        return False
    _launch_transpose_copy(dst, src, dst_ptr, src_ptr, block_height, block_width, reading_order, transposed_dim, device_id, stream_ptr, logger)
    return True


cdef int _use_elementwise_copy(Layout dst, Layout src, intptr_t dst_ptr, intptr_t src_ptr, int device_id, intptr_t stream_ptr, object logger=None) except -1 nogil:
    cdef int block_size = 128
    _adjust_layouts_for_elementwise_copy(dst, src, logger)
    _launch_elementwise_copy(dst, src, dst_ptr, src_ptr, block_size, device_id, stream_ptr, logger)
    return 0


cdef int launch_copy_kernel(Layout dst, Layout src, intptr_t dst_ptr, intptr_t src_ptr, int device_id, intptr_t stream_ptr, object logger=None) except -1 nogil:
    """
    Launches transposed or elementwise copy kernel. Assumes that both src and dst layouts are permuted
    so that the dst strides are increasing righ-to-left (C-like, but with gaps allowed).
    """
    if _use_tranpose_copy_maybe(dst, src, dst_ptr, src_ptr, device_id, stream_ptr, logger):
        return 0
    _use_elementwise_copy(dst, src, dst_ptr, src_ptr, device_id, stream_ptr, logger)
    return 0
