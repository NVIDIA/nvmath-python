# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from libc.stdint cimport int64_t, intptr_t, uint64_t, UINT64_MAX

import threading
import weakref

from .bindings cimport (
    mem_alloc_async, mem_free_async,
    get_device_current_memory_pool as _get_device_current_memory_pool,
    get_memory_pool_reserved_memory_size as _get_memory_pool_reserved_memory_size,
    get_memory_pool_used_memory_size as _get_memory_pool_used_memory_size,
    free_memory_pool_reserved_memory as _free_memory_pool_reserved_memory,
    set_memory_pool_release_threshold as _set_memory_pool_release_threshold,
    get_memory_pool_release_threshold as _get_memory_pool_release_threshold,
)

import cuda.core.experimental as ccx
from nvmath.internal.package_ifc import StreamHolder


@cython.final
cdef class MemAsyncAllocationFinalizer:

    def __cinit__(MemAsyncAllocationFinalizer self, MemAsyncPool pool, intptr_t ptr, int64_t size, ccx_stream, external_stream, logger=None):
        self.pool = pool
        self.ptr = ptr
        self.size = size
        # we got plain ccx.Stream object only or a StreamHolder wrapping ccx.Stream only
        if external_stream is ccx_stream:
            # We cannot use weakref here, as the ccx.Stream does not support
            # weakrefs. We store regular reference, potentially prolonging its lifetime.
            self.stream_obj = ccx_stream
        else:
            # The `stream.obj` just wraps the raw pointer that comes from
            # the `stream.external`. We expect that user passing some external
            # stream makes sure the python object does not outlive the underlying
            # cuda stream. If the `stream.external` was created with the package
            # of choice, the object owns the stream and their lifetimes are coupled.
            # Otherwise, if user wrapped raw pointer into `stream.external`, it is their
            # responsibility to make sure the alive object does not store dangling
            # pointer to invalidated stream.
            self.external_stream_ref = weakref.ref(external_stream)
            self.stream_ptr = int(ccx_stream.handle)
        if logger is not None:
            self.logger = logger
            logger.debug(
                "_RawCUDAMemoryManager (allocate memory): size = %d, ptr = %d, device_id = %d, stream = %s",
                size,
                ptr,
                pool.device_id,
                ccx_stream,
            )

    cdef close(MemAsyncAllocationFinalizer self, stream : ccx.Stream | None = None):
        if self.ptr == 0:
            return
        if stream is not None and stream.handle is not None:
            stream_ptr = int(stream.handle)
        elif self.external_stream_ref is None:
            stream_handle = self.stream_obj.handle
            if stream_handle is None:
                stream_ptr = self.pool.default_stream_ptr
            else:
                stream_ptr = int(stream_handle)
        else:
            # try to deallocate in allocation order if the originally passed
            # stream object is still alive, otherwise free on the default stream
            # which is correct but can be slower and more resource consuming, esp.
            # if allocations and deallocations happen in a loop without explicitly
            # synchronization in between
            external_stream = self.external_stream_ref()
            if external_stream is None:
                stream_ptr = self.pool.default_stream_ptr
            else:
                stream_ptr = self.stream_ptr
        mem_free_async(self.ptr, stream_ptr)
        if self.logger is not None:
            self.logger.debug(
                "_RawCUDAMemoryManager (release memory): size = %d, ptr = %d, device_id = %d, stream = %s",
                self.size,
                self.ptr,
                self.pool.device_id,
                stream_ptr,
            )
        self.ptr = 0


@cython.final
cdef class MemAsyncAllocation:

    def __cinit__(MemAsyncAllocation self, MemAsyncPool pool, int64_t size, stream: StreamHolder | ccx.Stream, logger=None):
        if isinstance(stream, ccx.Stream):
            ccx_stream = stream
            external_stream = stream
        elif isinstance(stream, StreamHolder):
            ccx_stream = stream.obj
            external_stream = stream.external
        elif stream is None:
            raise ValueError("stream is required for allocating GPU tensor")
        else:
            raise ValueError(f"Unsupported stream type: {type(stream)}")
        size = _round_up_allocation_size(size)
        cdef intptr_t ptr = mem_alloc_async(size, int(ccx_stream.handle))
        try:
            self.finalizer = MemAsyncAllocationFinalizer(pool, ptr, size, ccx_stream, external_stream, logger)
        except:
            mem_free_async(ptr, pool.default_stream_ptr)
            raise

    def __dealloc__(MemAsyncAllocation self):
        # even if the __cinit__ exits with an exception,
        # the __dealloc__ is still called, so the finalizer
        # will be None e.g. if the mem_alloc_async failed
        if self.finalizer is not None:
            self.finalizer.close()

    @property
    def ptr(self):
        return self.finalizer.ptr

    @property
    def handle(self):
        return self.finalizer.ptr

    @property
    def size(self):
        return self.finalizer.size

    def close(self, stream=None):
        self.finalizer.close(stream)


cdef int64_t _round_up_allocation_size(int64_t size) except? -1 nogil:
    """
    Rounds up the allocation size to the nearest multiple of 512 bytes.
    """
    return (size + 511) & ~511


@cython.final
cdef class MemAsyncPool:

    """
    MemAsyncPool is a wrapper around the cuda current (possibly default)
    asynchronous memory pool for a given device (introduced in CUDA 11.2).
    Using the current memory pool allows reusing the same pool between different
    libraries running in the same process. This is the same pool that is used
    by cupy when user opts-in for asynchronous memory allocation.
    """

    def __cinit__(MemAsyncPool self, object device):
        """
        Creates a new MemAsyncPool instance for the given device.
        NOTE: The MemAsyncPool should not be created directly, but rather obtained
        with call to `get_device_current_memory_pool`.
        """
        self.device_id = device.device_id
        self.default_stream = device.default_stream
        self.default_stream_ptr = int(self.default_stream.handle)

    cpdef allocate(MemAsyncPool self, int64_t size, stream: StreamHolder | ccx.Stream, logger=None):
        """
        Allocates memory from the device's current asynchronous memory pool.
        NOTE: To avoid overhead of switching current device context,
        it is the caller's responsibility to ensure that the current device
        is set to the `self.device_id` before calling this method.
        """
        return MemAsyncAllocation(self, size, stream, logger)

    cpdef set_limit(MemAsyncPool self, uint64_t limit):
        """
        NOTE: It is the caller's responsibility to ensure that the current device
        is set to the `self.device_id` before calling this method.
        """
        cdef intptr_t pool_ptr = _get_device_current_memory_pool(self.device_id)
        _set_memory_pool_release_threshold(pool_ptr, limit)

    cpdef uint64_t get_limit(MemAsyncPool self) except? -1:
        """
        NOTE: It is the caller's responsibility to ensure that the current device
        is set to the `self.device_id` before calling this method.
        """
        cdef intptr_t pool_ptr = _get_device_current_memory_pool(self.device_id)
        return _get_memory_pool_release_threshold(pool_ptr)

    cpdef uint64_t get_reserved_memory_size(MemAsyncPool self) except? -1:
        """
        NOTE: It is the caller's responsibility to ensure that the current device
        is set to the `self.device_id` before calling this method.
        """
        cdef intptr_t pool_ptr = _get_device_current_memory_pool(self.device_id)
        return _get_memory_pool_reserved_memory_size(pool_ptr)

    cpdef uint64_t get_used_memory_size(MemAsyncPool self) except? -1:
        """
        NOTE: It is the caller's responsibility to ensure that the current device
        is set to the `self.device_id` before calling this method.
        """
        cdef intptr_t pool_ptr = _get_device_current_memory_pool(self.device_id)
        return _get_memory_pool_used_memory_size(pool_ptr)

    cpdef free_reserved_memory(MemAsyncPool self):
        """
        NOTE: It is the caller's responsibility to ensure that the current device
        is set to the `self.device_id` before calling this method.
        """
        self.default_stream.sync()
        cdef intptr_t pool_ptr = _get_device_current_memory_pool(self.device_id)
        _free_memory_pool_reserved_memory(pool_ptr)


thread_local = threading.local()

cdef _create_memory_pool(int device_id):
    cdef uint64_t limit = UINT64_MAX
    cdef object current_device = ccx.Device()
    cdef object new_device = ccx.Device(device_id)
    # We need to set the current device to the one requested unconditionally,
    # to make sure context is initialized and set (pool memory creation is likely)
    # to be the first interactiion with the device in the process.
    # This adds some overhead, but _create_memory_pool is supposed to be
    # one-time operation (per perocess, per device).
    try:
        new_device.set_current()
        memory_pool = MemAsyncPool(new_device)
        # If the default 0 is kept and all the memory is freed back to the pool,
        # the pool releases memory back to OS, making subsequent allocations slower.
        # To ensure performant allocations, we set the limit to the maximum possible
        # value, to prevent this behavior.
        memory_pool.set_limit(limit)
        return memory_pool
    finally:
        current_device.set_current()


cdef _thread_local_memory_pools_cache():
    if not hasattr(thread_local, "device_memory_pools"):
        thread_local.device_memory_pools = {}
    return thread_local.device_memory_pools


cpdef get_device_current_memory_pool(int device_id):
    """
    Gets or creates MemAsyncPool instance for the given device.
    Caller does not need to ensure current device is set correctly - if
    the memory pools needs to be created, the function ensures
    setting the current device to the one requested.
    """
    cdef dict _device_memory_pools = _thread_local_memory_pools_cache()
    if device_id not in _device_memory_pools:
        _device_memory_pools[device_id] = _create_memory_pool(device_id)
    return _device_memory_pools[device_id]


cpdef free_reserved_memory():
    """
    Frees current async memory pool for all devices. Note, the
    memory is freed only from the device's current memory pool and only
    if get_device_current_memory_pool was called for that device by the
    calling thread.
    Internally, the function calls cuMemPoolTrimTo with 0 size, which should
    release back to OS all unused memory from the current memory pool.
    """
    cdef object current_device = ccx.Device()
    cdef object new_device
    try:
        for pool in _thread_local_memory_pools_cache().values():
            new_device = ccx.Device(pool.device_id)
            new_device.set_current()
            pool.free_reserved_memory()
    finally:
        current_device.set_current()


@cython.final
cdef class MemoryPointer:
    """
    MemoryPointer class defines an interface for memory pointers returned
    from user-provided memory resources. See `_allocate_data` in `ndbuffer.pyx`
    for an example.
    """

    def __cinit__(MemoryPointer self, intptr_t ptr, object owner=None):
        self.ptr = ptr
        self.owner = owner
