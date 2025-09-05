# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from libc.stdint cimport int64_t, intptr_t, uint64_t


@cython.final
cdef class MemAsyncAllocationFinalizer:

    cdef MemAsyncPool pool
    cdef intptr_t ptr
    cdef int64_t size
    cdef intptr_t stream_ptr
    cdef object stream_obj
    cdef object external_stream_ref
    cdef object logger

    cdef close(MemAsyncAllocationFinalizer self, stream=*)


@cython.final
cdef class MemAsyncAllocation:

    cdef MemAsyncAllocationFinalizer finalizer


@cython.final
cdef class MemAsyncPool:
    cdef readonly int device_id
    cdef readonly object default_stream
    cdef readonly intptr_t default_stream_ptr

    cpdef allocate(MemAsyncPool self, int64_t size, stream, logger=*)
    cpdef set_limit(MemAsyncPool self, uint64_t limit)
    cpdef uint64_t get_limit(MemAsyncPool self) except? -1
    cpdef uint64_t get_reserved_memory_size(MemAsyncPool self) except? -1
    cpdef uint64_t get_used_memory_size(MemAsyncPool self) except? -1
    cpdef free_reserved_memory(MemAsyncPool self)


cpdef get_device_current_memory_pool(int device_id)
cpdef free_reserved_memory()


@cython.final
cdef class MemoryPointer:
    cdef public intptr_t ptr
    cdef public object owner
