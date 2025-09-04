# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
cimport cpython
from cpython.memoryview cimport PyMemoryView_FromMemory
from libc.stdint cimport int64_t, intptr_t
from libcpp.vector cimport vector
from .data_layout cimport (
    Layout, strides_t, axis_order_t, shape_t,
    OrderFlag, set_strides_in_order, set_strides_tuple,
    tuple2vec, zero_strides,
    create_layout_without_strides, create_layout,
    copy_layout, empty_layout_with_dtype_like,
    is_c_contiguous_layout, is_overlapping_layout,
    transpose_squeeze_zeros_ones_layout,
    squeeze_layout, squeeze_layouts_together,
    vectorize_together, get_axis_order, get_strides_in_bytes_tuple,
    is_c_or_f_layout,
    parse_py_axis_order, split_strides,
    size_in_bytes as _size_in_bytes,
)
from ..memory cimport get_device_current_memory_pool
from ..bindings cimport memcpy_async, stream_sync
from .copy_kernel cimport launch_copy_kernel
import numpy as _numpy


cdef extern from "nd_consts.h":
    cdef const int NDBUFFER_CPU_DEVICE_ID


CPU_DEVICE_ID = NDBUFFER_CPU_DEVICE_ID  # make it accessible from python


@cython.final
cdef class NDBuffer:

    def __repr__(NDBuffer self):
        return (
            f"NDBuffer(ptr={self.data_ptr}, dtype={self.dtype_name}, device_id={self.data_device_id}, layout={self.layout})"
        )

    @property
    def itemsize(self):
        return self.layout.itemsize

    @property
    def ndim(self):
        return self.layout.ndim

    @property
    def device(self):
        if self.prop_device is None:
            self.prop_device = "cpu" if self.data_device_id == NDBUFFER_CPU_DEVICE_ID else "cuda"
        return self.prop_device

    @property
    def device_id(self):
        if self.prop_device_id is None:
            self.prop_device_id = "cpu" if self.data_device_id == NDBUFFER_CPU_DEVICE_ID else self.data_device_id
        return self.prop_device_id

    @property
    def strides(self):
        if self.prop_strides is None:
            self.prop_strides = tuple(self.layout.strides)
        return self.prop_strides

    @property
    def strides_in_bytes(self):
        if self.prop_strides_in_bytes is None:
            self.prop_strides_in_bytes = get_strides_in_bytes_tuple(self.layout)
        return self.prop_strides_in_bytes

    @property
    def shape(self):
        if self.prop_shape is None:
            self.prop_shape = tuple(self.layout.shape)
        return self.prop_shape

    @property
    def size(self):
        return self.layout.volume

    @property
    def size_in_bytes(self):
        return _size_in_bytes(self.layout)

    def cf_order(self):
        cdef OrderFlag order_flag = OrderFlag.CUSTOM_PERMUTATION
        if is_c_or_f_layout(order_flag, self.layout):
            return "C" if order_flag == OrderFlag.C_ORDER else "F"
        return "K"


cdef enum NDBufferFlags:
    # if set, the `data` is a fully-fledged tensor object whose
    # layout matches the NDBuffer's layout.
    NDBUFFER_FLAG_IS_WRAPPING_TENSOR = 1


cdef int _set_flags(NDBuffer ndbuffer, bint is_wrapping_tensor=False) except -1 nogil:
    ndbuffer.flags = 0
    if is_wrapping_tensor:
        ndbuffer.flags |= NDBUFFER_FLAG_IS_WRAPPING_TENSOR
    return 0


cdef bint is_wrapping_tensor(NDBuffer ndbuffer) except -1 nogil:
    return ndbuffer.flags & NDBUFFER_FLAG_IS_WRAPPING_TENSOR


cdef _empty_numpy_data(int64_t size):
    """
    Uses numpy empty array to allocate host buffer, this way we can rely on numpy's
    platform independent memory management and memory initialization.
    Using raw allocation, e.g. malloc with no initialization or no pooling
    degrades performance of pagable D2H copies.
    """
    cdef int64_t num_elements = (size + 15) // 16
    cdef object out = _numpy.empty(num_elements, dtype=_numpy.complex128)
    return out


cdef _allocate_data(NDBuffer buffer, int64_t size, object host_memory_pool=None, object device_memory_pool=None, object stream=None, object logger=None):
    if size == 0:
        buffer.data = None
        buffer.data_ptr = 0
        return
    if buffer.data_device_id == NDBUFFER_CPU_DEVICE_ID:
        if host_memory_pool is None:
            buffer.data = _empty_numpy_data(size)
            buffer.data_ptr = buffer.data.ctypes.data
        else:
            buffer.data = host_memory_pool.allocate(size, stream, logger)
            buffer.data_ptr = buffer.data.ptr
    else:
        if device_memory_pool is None:
            device_memory_pool = get_device_current_memory_pool(buffer.data_device_id)
        buffer.data = device_memory_pool.allocate(size, stream, logger)
        buffer.data_ptr = buffer.data.ptr


cdef NDBuffer _no_data_like(NDBuffer other, bint share_layout):
    """
    Copy the layout and other meta-data from other, but do not allocate data.
    The layout is shared or copied as-is, depending on the share_layout flag,
    without any attempts to make it dense.
    """
    cdef NDBuffer out = NDBuffer()
    _set_flags(out)
    if share_layout:
        out.layout = other.layout
    else:
        out.layout = copy_layout(other.layout)
    out.data_device_id = other.data_device_id
    out.dtype_name = other.dtype_name
    return out


cdef NDBuffer _no_data_dense_like(NDBuffer other, axis_order_t* axis_order, OrderFlag order_flag):
    cdef NDBuffer out = _no_data_like(other, False)
    set_strides_in_order(out.layout.strides, out.layout.shape, order_flag, axis_order)
    return out


cdef NDBuffer _empty_dense_like(NDBuffer other, axis_order_t* axis_order, OrderFlag order_flag, object host_memory_pool=None, object device_memory_pool=None, object stream=None):
    cdef NDBuffer out = _no_data_dense_like(other, axis_order, order_flag)
    _allocate_data(out, _size_in_bytes(out.layout), host_memory_pool, device_memory_pool, stream)
    return out


cdef _as_nonowning_numpy_array(NDBuffer ndbuf, bint readonly=True):
    """
    Note the returned array is non-owning, it's caller responsibility to keep ndbuf alive.
    The function does not perform checks on the ndbuf, e.g. if memory is really on the CPU.
    """
    if is_wrapping_tensor(ndbuf):
        return ndbuf.data
    cdef object buf = PyMemoryView_FromMemory(
        <char*>ndbuf.data_ptr, _size_in_bytes(ndbuf.layout),
        cpython.PyBUF_READ if readonly else cpython.PyBUF_WRITE)
    cdef np_1d = _numpy.frombuffer(buf, dtype=ndbuf.dtype_name)
    return _numpy.lib.stride_tricks.as_strided(np_1d, shape=ndbuf.shape, strides=ndbuf.strides_in_bytes)


cdef NDBuffer _numpy_copy(NDBuffer other, OrderFlag order_flag):
    """
    Copies NDBuffer host array so that it is contiguous in the given order:
    C, F, or K (i.e. keeping the strides order).
    The copy is skipped if the layout is already contiguous in the requested axis order.
    The actual data copy is performed by numpy.
    """
    cdef NDBuffer out
    if order_flag == OrderFlag.C_ORDER:
        src_array = _as_nonowning_numpy_array(other)
        array = _numpy.ascontiguousarray(src_array)
        if array is src_array:
            return other
        else:
            out = _no_data_like(other, False)
            out.data = array
            out.data_ptr = out.data.ctypes.data
            # accessing array.shape and array.strides is slow,
            # so we compute the strides on our own
            set_strides_in_order(out.layout.strides, out.layout.shape, order_flag)
            return out
    elif order_flag == OrderFlag.F_ORDER:
        src_array = _as_nonowning_numpy_array(other)
        array = _numpy.asfortranarray(src_array)
        if array is src_array:
            return other
        else:
            out = _no_data_like(other, False)
            out.data = array
            out.data_ptr = out.data.ctypes.data
            # accessing array.shape and array.strides is slow,
            # so we compute the strides on our own
            set_strides_in_order(out.layout.strides, out.layout.shape, order_flag)
            return out
    elif order_flag == OrderFlag.CUSTOM_PERMUTATION:
        src_array = _as_nonowning_numpy_array(other)
        array = src_array.copy(order='K')
        if array is src_array:
            return other
        else:
            out = _no_data_like(other, False)
            out.data = array
            out.data_ptr = out.data.ctypes.data
            set_strides_tuple(out.layout, array.strides, strides_in_bytes=True)
            return out
    else:
        raise ValueError(f"Unsupported order flag: {order_flag}")


cdef _check_shape_and_dtype(NDBuffer dst, NDBuffer src):
    if dst.dtype_name != src.dtype_name:
        raise ValueError(
            f"The data types of the source and destination buffers must match. "
            f"Got dst dtype:{dst.dtype_name} and src dtype:{src.dtype_name}"
        )
    if dst.layout.itemsize != src.layout.itemsize:
        raise ValueError(
            f"The itemsize of the source and destination buffers must match. "
            f"Got dst itemsize:{dst.layout.itemsize} and src itemsize:{src.layout.itemsize}"
        )
    if dst.layout.shape != src.layout.shape:
        raise ValueError(
            f"The shapes of the source and destination buffers must match. "
            f"Got dst shape:{dst.layout.shape} and src shape:{src.layout.shape}"
        )


cdef inline int _logging_log_axis_order(object logger, str msg, axis_order_t& fst) except -1 nogil:
    with cython.gil:
        logger.debug(msg.format(fst=fst))
    return 0


cdef inline int _logging_helper(object logger, str msg, fst=None, snd=None, third=None) except -1 nogil:
    with cython.gil:
        logger.debug(msg.format(fst=fst, snd=snd, third=third))
    return 0


cdef inline bint _d2d_mem_copy_maybe(Layout dst_normalized, Layout src_normalized, NDBuffer dst, NDBuffer src, intptr_t stream_ptr, object logger) except -1 nogil:
    """
    Returns true iff a copy can be performed disregarding actual strides, i.e.
    both layouts are dense, possibly permuted with the same permutation.
    If a copy is needed (i.e. the volume > 0), launches a memcpy.
    Otherwise, returns false, does not perform any copy and returns pre-processed
    strides in dst_normalized and src_normalized. The returned layouts are permuted
    by the same permutation, so that dst strides increase from right to left (as in C order tensors).
    The returned layouts are squeezed together, i.e. the each fragment of the layouts
    that is elementwise C-contigious in both src and dst is replaced with a single extent.
    """
    if dst.layout.volume == 0:
        return True
    cdef axis_order_t dst_axis_order
    get_axis_order(dst_axis_order, dst.layout)
    # permute dst layout to C-like order of strides and remove all extents equal to 1,
    # as their corresponding strides never contribute to offset of any elements in the tensor.
    transpose_squeeze_zeros_ones_layout(dst_normalized, dst.layout, dst_axis_order)
    transpose_squeeze_zeros_ones_layout(src_normalized, src.layout, dst_axis_order)
    # if dst, src shapes were equal, so are the dst_normalized and src_normalized
    # try to merge extents that are C-contigious in both src and dst
    cdef int ndim = dst_normalized.ndim
    squeeze_layouts_together(dst_normalized, src_normalized, ndim)
    if logger is not None:
        _logging_log_axis_order(logger, "The dst_order is {fst}", dst_axis_order)
        _logging_helper(logger, "Permuted and squeezed strides: dst {fst}, src {snd}", dst_normalized, src_normalized)
    # NB. is_c_contiguous_layout(dst_normalized) <==> dst_normalized.ndim == 0 or dst_normalized.ndim == 1 and dst_normalized.strides[0] == 1
    if is_c_contiguous_layout(dst_normalized) and dst_normalized.strides == src_normalized.strides:
        if logger is not None:
            _logging_helper(logger, "The layouts are dense and have same strides order, we can memcpy")
        memcpy_async(dst.data_ptr, src.data_ptr, _size_in_bytes(dst_normalized), stream_ptr)
        return True
    return False


cdef int _copy_into_d2d(NDBuffer dst, NDBuffer src, object stream, bint sync=False, object logger=None) except -1:
    cdef intptr_t stream_ptr = int(stream.obj.handle)
    cdef intptr_t dst_ptr = dst.data_ptr
    cdef intptr_t src_ptr = src.data_ptr
    # layouts normalized (permuted/squeezed) to be used by the copy kernel
    cdef Layout dst_normalized = empty_layout_with_dtype_like(dst.layout)
    cdef Layout src_normalized = empty_layout_with_dtype_like(src.layout)
    with cython.nogil:
        if _d2d_mem_copy_maybe(dst_normalized, src_normalized, dst, src, stream_ptr, logger):
            if sync:
                stream_sync(stream_ptr)
            return 0
        if is_overlapping_layout(dst_normalized):
            raise ValueError(f"The destination layout could overlap in memory: {dst.layout}")
        vectorize_together(dst_normalized, dst_ptr, src_normalized, src_ptr)
        if logger is not None:
            if dst_normalized.itemsize == dst.layout.itemsize:
                _logging_helper(logger, "Could not vectorize the copy, the itemsize remains unchanged")
            else:
                _logging_helper(logger, "Copy will use bigger/vectorized itemsize: vectorized_dst: {fst}, vectorized_src: {snd}", dst_normalized, src_normalized)
        launch_copy_kernel(dst_normalized, src_normalized, dst_ptr, src_ptr, dst.data_device_id, stream_ptr, logger)
        if sync:
            stream_sync(stream_ptr)
        return 0


cdef int _copy_into_d2h(NDBuffer dst, NDBuffer src, object stream, object host_memory_pool=None, object device_memory_pool=None, object logger=None) except -1:
    """
    Copies tensor from device to host.
    Depending on the src and dst layouts, the function may need to use up two temporary buffers.
    * If the src layout is not contiguous or has different (from the dst) strides order,
      we create a device temporary buffer and perform a d2d copy, transposing the strides order.
    * If the dst layout has gaps, we d2h memcopy into temporary host buffer and use
      numpy to copy-scatter the data.
    Usually, the transposed copy is faster on the GPU, that's why the transposition, if needed,
    is performed while data is still on the device.
    """
    cdef intptr_t stream_ptr = int(stream.obj.handle)
    cdef int64_t size = _size_in_bytes(dst.layout)
    if size == 0:
        return 0
    cdef axis_order_t dst_axis_order
    get_axis_order(dst_axis_order, dst.layout)
    cdef Layout dst_normalized = empty_layout_with_dtype_like(dst.layout)
    cdef Layout src_normalized = empty_layout_with_dtype_like(src.layout)
    transpose_squeeze_zeros_ones_layout(dst_normalized, dst.layout, dst_axis_order)
    transpose_squeeze_zeros_ones_layout(src_normalized, src.layout, dst_axis_order)
    cdef NDBuffer dev_tmp, host_tmp
    if is_overlapping_layout(dst_normalized):
        raise ValueError("The destination layout could overlap in memory")
    # if source layout order matches the dst layout and is dense we can just memcpy
    # it to host. Otherwise if we need to coalesce or transpose - we do it on the device.
    if is_c_contiguous_layout(src_normalized):
        dev_tmp = src
    else:
        dev_tmp = _empty_dense_like(src, &dst_axis_order, OrderFlag.CUSTOM_PERMUTATION, None, device_memory_pool, stream)
        if logger is not None:
            logger.debug(
                f"Src is not contiguous or has a different strides order, "
                f"performing a coalescing/transposing copy into temporary buffer.\n"
                f"dev_tmp: {dev_tmp} <- Src: {src}"
            )
        _copy_into_d2d(dev_tmp, src, stream, False, logger)
        transpose_squeeze_zeros_ones_layout(src_normalized, dev_tmp.layout, dst_axis_order)
    if dst_normalized.strides == src_normalized.strides:
        if logger is not None:
            logger.debug(
                f"The dst and src layouts match, launching direct D2H memcpy.\n"
                f"Dst: {dst} <- dev: {dev_tmp}"
            )
        with cython.nogil:
            memcpy_async(dst.data_ptr, dev_tmp.data_ptr, size, stream_ptr)
            stream_sync(stream_ptr)
            return 0
    else:
        host_tmp = _no_data_like(dev_tmp, True)
        host_tmp.data_device_id = NDBUFFER_CPU_DEVICE_ID
        _allocate_data(host_tmp, size, host_memory_pool, None, stream, logger)
        if logger is not None:
            logger.debug(
                f"The dst and src layouts differ, we D2H memcopy into a temporary host buffer\n"
                f"memcpy: host_tmp: {host_tmp} <- dev: {dev_tmp}, followed by\n"
                f"h2h copy: dst: {dst.layout} <- host: {host_tmp}"
            )
        with cython.nogil:
            memcpy_async(host_tmp.data_ptr, dev_tmp.data_ptr, size, stream_ptr)
            stream_sync(stream_ptr)
        _numpy.copyto(_as_nonowning_numpy_array(dst, readonly=False), _as_nonowning_numpy_array(host_tmp))
        return 0


cdef int _copy_into_h2d(NDBuffer dst, NDBuffer src, object stream, object host_memory_pool=None, object device_memory_pool=None, object logger=None) except -1:
    """
    Copies data from host to device.
    Depending on the src and dst layouts, the function may need to use up two temporary buffers.
    * If the src layout is not contiguous (in any permutation of strides), we need to coalesce it
      before memcpy. We use numpy to do that.
    * If the dst layout has gaps or different strides order than the src, we memcpy into a temporary
      device buffer and perform a d2d copy.
    Usually, the transposed copy is faster on the GPU, that's why the transposition, if needed,
    is performed after the data is copied to the device.
    """
    cdef int64_t size = _size_in_bytes(src.layout)
    if size == 0:
        return 0
    cdef intptr_t stream_ptr = int(stream.obj.handle)
    cdef axis_order_t src_axis_order
    get_axis_order(src_axis_order, src.layout)
    cdef Layout dst_normalized = empty_layout_with_dtype_like(dst.layout)
    cdef Layout src_normalized = empty_layout_with_dtype_like(src.layout)
    transpose_squeeze_zeros_ones_layout(dst_normalized, dst.layout, src_axis_order)
    transpose_squeeze_zeros_ones_layout(src_normalized, src.layout, src_axis_order)
    cdef NDBuffer host_tmp, dev_tmp
    if is_c_contiguous_layout(src_normalized):
        host_tmp = src
    else:
        # For non-overlapping layouts, try to keep the original stride order,
        # this should make the h2h copy faster. Otherwise, the perf of the copy
        # is difficult to predict.
        if is_overlapping_layout(src_normalized):
            host_tmp = _numpy_copy(src, OrderFlag.C_ORDER)
        else:
            host_tmp = _numpy_copy(src, OrderFlag.CUSTOM_PERMUTATION)
        if logger is not None:
            logger.debug(
                f"Src is not contiguous, use numpy for a coalescing h2h copy into temporary buffer.\n"
                f"host_tmp: {host_tmp} <- Src: {src}"
            )
        transpose_squeeze_zeros_ones_layout(src_normalized, host_tmp.layout, src_axis_order)
    # now, host_tmp is contiguous, we can memcpy it to the device
    if dst_normalized.strides == src_normalized.strides:
        if logger is not None:
            logger.debug(
                f"The dst and src layouts match, launching direct H2D memcpy.\n"
                f"Dst: {dst} <- host: {host_tmp}"
            )
        with cython.nogil:
            memcpy_async(dst.data_ptr, host_tmp.data_ptr, size, stream_ptr)
            stream_sync(stream_ptr)
            return 0
    else:
        dev_tmp = _no_data_like(host_tmp, True)
        dev_tmp.data_device_id = dst.data_device_id
        _allocate_data(dev_tmp, size, None, device_memory_pool, stream, logger)
        if logger is not None:
            logger.debug(
                f"The dst and src layouts differ, we need a tmp dev buffer for memcpy.\n"
                f"memcpy: dev_tmp: {dev_tmp.layout} <- host: {host_tmp.layout}, followed by\n"
                f"d2d copy: dst: {dst.layout} <- dev: {dev_tmp.layout}"
            )
        memcpy_async(dev_tmp.data_ptr, host_tmp.data_ptr, size, stream_ptr)
        # if the dst layout is overlapping, we should land here
        # and the _copy_into_d2d will raise an error
        _copy_into_d2d(dst, dev_tmp, stream, True, logger)
        return 0


cpdef int copy_into(NDBuffer dst, NDBuffer src, object stream, object host_memory_pool=None, object device_memory_pool=None, object logger=None) except -1:
    """
    Copies data from src to dst. If both dst and src are on the same GPU, the call is asynchronous.
    Otherwise, the call is synchronous.
    """
    _check_shape_and_dtype(dst, src)
    if dst.data_device_id == NDBUFFER_CPU_DEVICE_ID and src.data_device_id == NDBUFFER_CPU_DEVICE_ID:
        _numpy.copyto(_as_nonowning_numpy_array(dst, readonly=False), _as_nonowning_numpy_array(src))
    elif dst.data_device_id == NDBUFFER_CPU_DEVICE_ID:
        return _copy_into_d2h(dst, src, stream, host_memory_pool, device_memory_pool, logger)
    elif src.data_device_id == NDBUFFER_CPU_DEVICE_ID:
        return _copy_into_h2d(dst, src, stream, host_memory_pool, device_memory_pool, logger)
    else:
        if src.data_device_id != dst.data_device_id:
            raise ValueError("The source and destination devices must be the same")
        return _copy_into_d2d(dst, src, stream, False, logger)


cpdef NDBuffer wrap_external(data, intptr_t ptr, str dtype_name, object shape, object strides, int device_id, int itemsize, bint strides_in_bytes=False):
    if device_id < 0 and device_id != NDBUFFER_CPU_DEVICE_ID:
        raise ValueError(f"Incorrect device id {device_id}")

    cdef NDBuffer out = NDBuffer()
    _set_flags(out)
    out.data = data
    out.data_ptr = ptr
    out.data_device_id = device_id
    out.dtype_name = dtype_name
    out.layout = create_layout(shape, strides, itemsize, strides_in_bytes)
    return out


cpdef NDBuffer empty(object shape, int device_id, str dtype_name, int itemsize, object axis_order=None, object strides=None, object host_memory_pool=None, object device_memory_pool=None, object stream=None, bint strides_in_bytes=False, object logger=None):
    if device_id < 0 and device_id != NDBUFFER_CPU_DEVICE_ID:
        raise ValueError(f"Incorrect device id {device_id}")

    cdef Layout layout = create_layout_without_strides(shape, itemsize)
    cdef NDBuffer out = NDBuffer()
    _set_flags(out)
    out.layout = layout
    out.data_device_id = device_id
    out.dtype_name = dtype_name

    # set strides
    cdef axis_order_t axis_order_vec
    if axis_order is not None:
        if axis_order == 'C':
            set_strides_in_order(layout.strides, layout.shape, OrderFlag.C_ORDER)
        elif axis_order == 'F':
            set_strides_in_order(layout.strides, layout.shape, OrderFlag.F_ORDER)
        else:
            tuple2vec(axis_order_vec, axis_order)
            set_strides_in_order(layout.strides, layout.shape, OrderFlag.CUSTOM_PERMUTATION, &axis_order_vec)
    elif strides is not None:
        if len(strides) != layout.ndim:
            raise ValueError("strides, if specified, must be a tuple and have the same length as shape")
        set_strides_tuple(layout, strides, strides_in_bytes)
    else:
        set_strides_in_order(layout.strides, layout.shape, OrderFlag.C_ORDER)

    _allocate_data(out, _size_in_bytes(out.layout), host_memory_pool, device_memory_pool, stream, logger)
    return out


cpdef NDBuffer empty_like(NDBuffer other, object axis_order='K', object device_id=None, object stream=None, object host_memory_pool=None, object device_memory_pool=None, object logger=None):
    cdef axis_order_t axis_order_vec
    cdef OrderFlag order_flag = OrderFlag.C_ORDER
    parse_py_axis_order(order_flag, axis_order_vec, other.layout, axis_order)
    cdef NDBuffer out = _no_data_dense_like(other, &axis_order_vec, order_flag)
    if device_id is not None:
        out.data_device_id = int(device_id)
    _allocate_data(out, _size_in_bytes(out.layout), host_memory_pool, device_memory_pool, stream, logger)
    return out


cpdef NDBuffer reshaped_view(NDBuffer other, object shape, object logger=None):
    cdef NDBuffer out = _no_data_like(other, True)
    out.layout = create_layout_without_strides(shape, other.layout.itemsize)
    out.data = other.data
    out.data_ptr = other.data_ptr
    if out.layout.volume != other.layout.volume:
        raise ValueError("The source and destination have different volumes")
    elif other.layout.volume == 0:
        zero_strides(out.layout.strides, out.layout.ndim)
        return out
    cdef shape_t squeezed_shape
    cdef strides_t squeezed_strides
    squeeze_layout(squeezed_shape, squeezed_strides, other.layout)
    if logger is not None:
        logger.debug(
            f"Input layout: squeezed to "
            f"shape: {squeezed_shape} <- {other.layout.shape}, "
            f"strides: {squeezed_strides} <- {other.layout.strides}"
        )
    if not split_strides(out.layout, squeezed_shape, squeezed_strides):
        raise ValueError("Cannot reshape the tensor without performing a copy")
    if logger is not None:
        logger.debug(
            f"Squeezed layout split to: "
            f"shape: {out.layout.shape} <- {squeezed_shape}, "
            f"strides: {out.layout.strides} <- {squeezed_strides}"
        )
    out.data = other.data
    out.data_ptr = other.data_ptr
    return out
