# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from libc.stdint cimport int64_t, intptr_t
from ._bindings cimport free_memory_pool_reserved_memory as _free_memory_pool_reserved_memory

import threading
import logging

try:
    from cuda.core import system
except ImportError:
    from cuda.core.experimental import system

from nvmath.internal._device_utils import get_device


_thread_local = threading.local()


cdef inline _get_local_mrs():
    try:
        return _thread_local.mem_resources
    except AttributeError:
        try:
            num_devices = system.get_num_devices()
        except AttributeError:
            # cuda.core < 0.5.0
            num_devices = system.num_devices
        _thread_local.mem_resources = mrs = [None] * num_devices
        return mrs


cpdef get_device_memory_resource(int device_id):
    """
    Returns cuda.core.MemoryResource instance for the given device.
    The instances are cached per thread and reused for subsequent calls.
    """
    mrs = _get_local_mrs()
    mr = mrs[device_id]
    if mr is None:
        mrs[device_id] = mr = get_device(device_id).memory_resource
    return mr


cdef inline int64_t _round_up_allocation_size(int64_t size) except? -1 nogil:
    """
    Rounds up the allocation size to the nearest multiple of 512 bytes.
    """
    return (size + 511) & ~511


cdef class _MemoryPointer:
    """
    Temporary internal NDBuffer allocation adapter class. NDBuffer expects
    custom allocator to return a cuda.core.Buffer instance. Until all supported
    cuda.core versions have unified support for wrapping external allocations
    with Buffer.from_handle, this class servers as an adapter/workaround.

    WARNING: This is internal tool subject to change/removal without notice.

    Internally, it is used conditionally if any of the following is needed:
        * wrap external allocations that don't come as cuda.core.Buffer (e.g. from cupy).
        This is needed because prior to cuda.core 0.5.0, it's not possible to
        pass reference to externall RAII object as ``owner`` parameter to Buffer.from_handle.
        * provide debug logging on deallocation. In the future, we can use
        Buffer.from_handle(owner=...) to inject deallocation callback (cuda.core >= 0.5.0)
        or weakref.finalize for Buffer (cuda.core >= 0.6.0).

    The only publicly exposed field is the handle - a base pointer to the allocated memory.
    """
    cdef public intptr_t handle
    cdef public object owner
    cdef object logger
    cdef str dealloc_message

    @classmethod
    def from_handle(cls, intptr_t handle, owner):
        cdef _MemoryPointer self = _MemoryPointer.__new__(_MemoryPointer)
        self._init_from_handle(handle, owner)
        return self

    def __del__(self):
        if self.logger is not None:
            self.logger.debug(self.dealloc_message)

    cdef inline _init_from_buffer(self, buffer, stream, logger, dealloc_message):
        self.handle = int(buffer.handle)
        self.owner = buffer
        self.logger = logger
        self.dealloc_message = dealloc_message

    cdef inline _init_from_handle(self, intptr_t handle, owner):
        self.handle = handle
        self.owner = owner


cpdef allocate_from_mr(mr, int64_t size, stream, int device_id, logger = None):
    """
    Common helper for allocating memory from a memory resource.
    It rounds-up the allocation size to the nearest multiple of 512 bytes
    and provides debug logging if requested.
    """
    cdef bint has_debug_logging = logger is not None and logger.isEnabledFor(logging.DEBUG)
    size = _round_up_allocation_size(size)
    cdef buffer = mr.allocate(size, stream)
    if not has_debug_logging:
        return buffer
    cdef intptr_t handle = int(buffer.handle)
    logger.debug(
        f"MemoryResource {mr} (allocate memory): size = {size}, "
        f"ptr = {handle}, device_id = {device_id}, "
        f"stream = {stream}"
    )
    cdef str dealloc_message = (
        f"MemoryResource {mr} (release memory): size = {size}, "
        f"ptr = {handle}, device_id = {device_id}, stream = {stream}"
    )
    cdef _MemoryPointer mem_ptr = _MemoryPointer.__new__(_MemoryPointer)
    mem_ptr._init_from_buffer(buffer, stream, logger, dealloc_message)
    return mem_ptr


cpdef free_reserved_memory(bint sync = True):
    """
    Asks driver to free reserved unused memory from
    memory resources cached by get_device_memory_resource.
    Internally, the function calls cuMemPoolTrimTo with 0 size, which should
    release back to OS all unused memory from the current memory pool.

    Note:
    1. The function does not attempt to discover memory pools existing
    outside of the `nvmath.internal.memory` thread-local cache. In other words,
    if the calling thread never called get_device_memory_resource(device_id),
    the device's memory won't be affected by this function.
    Moreover, there may be multiple memory pools per device,
    this function will only affect the memory pool associated with the
    memory resource returned by get_device_memory_resource(device_id).

    2. The cuda memory pool underlying the resource returned with
    get_device_memory_resource(device_id) may be shared with different
    packages, so this call may affect other packages'.
    """
    for device_id, mr in enumerate(_get_local_mrs()):
        if mr is not None:
            # the driver seems to be very conservative in identifying
            # unused memory, doing a full device sync helps
            if sync:
                get_device(device_id).sync()
            _free_memory_pool_reserved_memory(mr.handle)
