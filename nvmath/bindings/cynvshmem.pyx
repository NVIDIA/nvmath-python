# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 3.1.7. Do not modify it directly.

cimport cython

from ._internal cimport nvshmem as _nvshmem


###############################################################################
# Wrapper functions
###############################################################################

cdef int nvshmemx_init_status() except?-42 nogil:
    return _nvshmem._nvshmemx_init_status()


cdef int nvshmem_my_pe() except?-42 nogil:
    return _nvshmem._nvshmem_my_pe()


cdef int nvshmem_n_pes() except?-42 nogil:
    return _nvshmem._nvshmem_n_pes()


cdef void* nvshmem_malloc(size_t size) except?NULL nogil:
    return _nvshmem._nvshmem_malloc(size)


cdef void* nvshmem_calloc(size_t count, size_t size) except?NULL nogil:
    return _nvshmem._nvshmem_calloc(count, size)


cdef void* nvshmem_align(size_t alignment, size_t size) except?NULL nogil:
    return _nvshmem._nvshmem_align(alignment, size)


@cython.show_performance_hints(False)
cdef void nvshmem_free(void* ptr) except* nogil:
    _nvshmem._nvshmem_free(ptr)


cdef void* nvshmem_ptr(const void* dest, int pe) except?NULL nogil:
    return _nvshmem._nvshmem_ptr(dest, pe)


@cython.show_performance_hints(False)
cdef void nvshmem_int_p(int* dest, const int value, int pe) except* nogil:
    _nvshmem._nvshmem_int_p(dest, value, pe)


cdef int nvshmem_team_my_pe(nvshmem_team_t team) except?-42 nogil:
    return _nvshmem._nvshmem_team_my_pe(team)


@cython.show_performance_hints(False)
cdef void nvshmemx_barrier_all_on_stream(cudaStream_t stream) except* nogil:
    _nvshmem._nvshmemx_barrier_all_on_stream(stream)


@cython.show_performance_hints(False)
cdef void nvshmemx_sync_all_on_stream(cudaStream_t stream) except* nogil:
    _nvshmem._nvshmemx_sync_all_on_stream(stream)


cdef int nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t* attr) except?-42 nogil:
    return _nvshmem._nvshmemx_hostlib_init_attr(flags, attr)


@cython.show_performance_hints(False)
cdef void nvshmemx_hostlib_finalize() except* nogil:
    _nvshmem._nvshmemx_hostlib_finalize()


cdef int nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks, const nvshmemx_uniqueid_t* uniqueid, nvshmemx_init_attr_t* attr) except?-42 nogil:
    return _nvshmem._nvshmemx_set_attr_uniqueid_args(myrank, nranks, uniqueid, attr)


cdef int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* uniqueid) except?-42 nogil:
    return _nvshmem._nvshmemx_get_uniqueid(uniqueid)
