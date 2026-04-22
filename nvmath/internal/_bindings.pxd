# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t, int64_t, uint64_t

cpdef memcpy_async(intptr_t dst_ptr, intptr_t src_ptr, int64_t size, intptr_t stream)
cpdef stream_sync(intptr_t stream)
cpdef uint64_t get_memory_pool_reserved_memory_size(pool) except? -1
cpdef uint64_t get_memory_pool_used_memory_size(pool) except? -1
cpdef free_memory_pool_reserved_memory(pool)
cpdef launch_kernel(intptr_t f, intptr_t kernel_params, int gx, int gy, int gz, int bx, int by, int bz, unsigned int shared_mem_bytes, intptr_t stream_handle)
