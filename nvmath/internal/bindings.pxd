# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t, int64_t, uint64_t

cdef struct Dim3:
    unsigned int x, y, z


cpdef int memcpy_async(intptr_t dst_ptr, intptr_t src_ptr, int64_t size, intptr_t stream) except -1 nogil
cpdef int stream_sync(intptr_t stream) except -1 nogil
cpdef intptr_t get_device_current_memory_pool(int device_id) except? 0 nogil
cpdef int set_memory_pool_release_threshold(intptr_t pool_ptr, uint64_t threshold) except -1 nogil
cpdef uint64_t get_memory_pool_release_threshold(intptr_t pool_ptr) except? -1 nogil
cpdef uint64_t get_memory_pool_reserved_memory_size(intptr_t pool_ptr) except? -1 nogil
cpdef uint64_t get_memory_pool_used_memory_size(intptr_t pool_ptr) except? -1 nogil
cpdef int free_memory_pool_reserved_memory(intptr_t pool_ptr) except -1 nogil
cpdef intptr_t mem_alloc_async(int64_t size, intptr_t stream_handle) except? -1 nogil
cpdef int mem_free_async(intptr_t dptr, intptr_t stream_handle) except -1 nogil
cpdef int launch_kernel(intptr_t f, intptr_t kernel_params, Dim3 grid_dim, Dim3 block_dim, unsigned int shared_mem_bytes, intptr_t stream_handle) except -1 nogil
# cdef only for the output is passed as in/out reference args
cdef int get_cc(int &major, int &minor, int device_id) except? -1 nogil
cpdef intptr_t get_function_from_module(intptr_t module, const char *name) except? 0 nogil
