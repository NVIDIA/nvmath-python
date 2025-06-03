# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 3.1.7. Do not modify it directly.

from ..cynvshmem cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef int _nvshmemx_init_status() except?-42 nogil
cdef int _nvshmem_my_pe() except?-42 nogil
cdef int _nvshmem_n_pes() except?-42 nogil
cdef void* _nvshmem_malloc(size_t size) except?NULL nogil
cdef void* _nvshmem_calloc(size_t count, size_t size) except?NULL nogil
cdef void* _nvshmem_align(size_t alignment, size_t size) except?NULL nogil
cdef void _nvshmem_free(void* ptr) except* nogil
cdef void* _nvshmem_ptr(const void* dest, int pe) except?NULL nogil
cdef void _nvshmem_int_p(int* dest, const int value, int pe) except* nogil
cdef int _nvshmem_team_my_pe(nvshmem_team_t team) except?-42 nogil
cdef void _nvshmemx_barrier_all_on_stream(cudaStream_t stream) except* nogil
cdef void _nvshmemx_sync_all_on_stream(cudaStream_t stream) except* nogil
cdef int _nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t* attr) except?-42 nogil
cdef void _nvshmemx_hostlib_finalize() except* nogil
cdef int _nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks, const nvshmemx_uniqueid_t* uniqueid, nvshmemx_init_attr_t* attr) except?-42 nogil
cdef int _nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* uniqueid) except?-42 nogil
