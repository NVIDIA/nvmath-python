# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 3.1.7. Do not modify it directly.


from libc.stdint cimport int64_t
from libc.stdint cimport int32_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum _anon_enum0 "_anon_enum0":
    PROXY_GLOBAL_EXIT_NOT_REQUESTED "PROXY_GLOBAL_EXIT_NOT_REQUESTED" = 0
    PROXY_GLOBAL_EXIT_INIT "PROXY_GLOBAL_EXIT_INIT"
    PROXY_GLOBAL_EXIT_REQUESTED "PROXY_GLOBAL_EXIT_REQUESTED"
    PROXY_GLOBAL_EXIT_FINISHED "PROXY_GLOBAL_EXIT_FINISHED"
    PROXY_GLOBAL_EXIT_MAX_STATE "PROXY_GLOBAL_EXIT_MAX_STATE" = 32767

ctypedef enum _anon_enum1 "_anon_enum1":
    NVSHMEM_STATUS_NOT_INITIALIZED "NVSHMEM_STATUS_NOT_INITIALIZED" = 0
    NVSHMEM_STATUS_IS_BOOTSTRAPPED "NVSHMEM_STATUS_IS_BOOTSTRAPPED"
    NVSHMEM_STATUS_IS_INITIALIZED "NVSHMEM_STATUS_IS_INITIALIZED"
    NVSHMEM_STATUS_LIMITED_MPG "NVSHMEM_STATUS_LIMITED_MPG"
    NVSHMEM_STATUS_FULL_MPG "NVSHMEM_STATUS_FULL_MPG"
    NVSHMEM_STATUS_INVALID "NVSHMEM_STATUS_INVALID" = 32767

ctypedef enum _anon_enum2 "_anon_enum2":
    NVSHMEM_TEAM_INVALID "NVSHMEM_TEAM_INVALID" = -(1)
    NVSHMEM_TEAM_WORLD "NVSHMEM_TEAM_WORLD" = 0
    NVSHMEM_TEAM_WORLD_INDEX "NVSHMEM_TEAM_WORLD_INDEX" = 0
    NVSHMEM_TEAM_SHARED "NVSHMEM_TEAM_SHARED" = 1
    NVSHMEM_TEAM_SHARED_INDEX "NVSHMEM_TEAM_SHARED_INDEX" = 1
    NVSHMEMX_TEAM_NODE "NVSHMEMX_TEAM_NODE" = 2
    NVSHMEM_TEAM_NODE_INDEX "NVSHMEM_TEAM_NODE_INDEX" = 2
    NVSHMEMX_TEAM_SAME_MYPE_NODE "NVSHMEMX_TEAM_SAME_MYPE_NODE" = 3
    NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX "NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX" = 3
    NVSHMEMI_TEAM_SAME_GPU "NVSHMEMI_TEAM_SAME_GPU" = 4
    NVSHMEM_TEAM_SAME_GPU_INDEX "NVSHMEM_TEAM_SAME_GPU_INDEX" = 4
    NVSHMEMI_TEAM_GPU_LEADERS "NVSHMEMI_TEAM_GPU_LEADERS" = 5
    NVSHMEM_TEAM_GPU_LEADERS_INDEX "NVSHMEM_TEAM_GPU_LEADERS_INDEX" = 5
    NVSHMEM_TEAMS_MIN "NVSHMEM_TEAMS_MIN" = 6
    NVSHMEM_TEAM_INDEX_MAX "NVSHMEM_TEAM_INDEX_MAX" = 32767

ctypedef enum nvshmemx_status "nvshmemx_status":
    NVSHMEMX_SUCCESS "NVSHMEMX_SUCCESS" = 0
    NVSHMEMX_ERROR_INVALID_VALUE "NVSHMEMX_ERROR_INVALID_VALUE"
    NVSHMEMX_ERROR_OUT_OF_MEMORY "NVSHMEMX_ERROR_OUT_OF_MEMORY"
    NVSHMEMX_ERROR_NOT_SUPPORTED "NVSHMEMX_ERROR_NOT_SUPPORTED"
    NVSHMEMX_ERROR_SYMMETRY "NVSHMEMX_ERROR_SYMMETRY"
    NVSHMEMX_ERROR_GPU_NOT_SELECTED "NVSHMEMX_ERROR_GPU_NOT_SELECTED"
    NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED "NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED"
    NVSHMEMX_ERROR_INTERNAL "NVSHMEMX_ERROR_INTERNAL"
    NVSHMEMX_ERROR_SENTINEL "NVSHMEMX_ERROR_SENTINEL" = 32767

ctypedef enum flags "flags":
    NVSHMEMX_INIT_THREAD_PES "NVSHMEMX_INIT_THREAD_PES" = 1
    NVSHMEMX_INIT_WITH_MPI_COMM "NVSHMEMX_INIT_WITH_MPI_COMM" = (1 << 1)
    NVSHMEMX_INIT_WITH_SHMEM "NVSHMEMX_INIT_WITH_SHMEM" = (1 << 2)
    NVSHMEMX_INIT_WITH_UNIQUEID "NVSHMEMX_INIT_WITH_UNIQUEID" = (1 << 3)
    NVSHMEMX_INIT_MAX "NVSHMEMX_INIT_MAX" = (1 << 31)


# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """
    ctypedef void* cudaStream_t 'cudaStream_t'


ctypedef struct nvshmemx_uniqueid_v1 'nvshmemx_uniqueid_v1':
    int version
    char internal[124]
ctypedef int32_t nvshmem_team_t 'nvshmem_team_t'
ctypedef struct nvshmem_team_config_v1 'nvshmem_team_config_v1':
    int version
    int num_contexts
    char padding[56]
ctypedef nvshmemx_uniqueid_v1 nvshmemx_uniqueid_t 'nvshmemx_uniqueid_t'
ctypedef struct nvshmemx_uniqueid_args_v1 'nvshmemx_uniqueid_args_v1':
    int version
    nvshmemx_uniqueid_v1* id
    int myrank
    int nranks
ctypedef nvshmem_team_t nvshmemx_team_t 'nvshmemx_team_t'
ctypedef nvshmem_team_config_v1 nvshmem_team_config_t 'nvshmem_team_config_t'
ctypedef nvshmemx_uniqueid_args_v1 nvshmemx_uniqueid_args_t 'nvshmemx_uniqueid_args_t'
ctypedef struct nvshmemx_init_args_v1 'nvshmemx_init_args_v1':
    int version
    nvshmemx_uniqueid_args_t uid_args
    char content[96]
ctypedef nvshmemx_init_args_v1 nvshmemx_init_args_t 'nvshmemx_init_args_t'
ctypedef struct nvshmemx_init_attr_v1 'nvshmemx_init_attr_v1':
    int version
    void* mpi_comm
    nvshmemx_init_args_t args
ctypedef nvshmemx_init_attr_v1 nvshmemx_init_attr_t 'nvshmemx_init_attr_t'


###############################################################################
# Functions
###############################################################################

cdef int nvshmemx_init_status() except?-42 nogil
cdef int nvshmem_my_pe() except?-42 nogil
cdef int nvshmem_n_pes() except?-42 nogil
cdef void* nvshmem_malloc(size_t size) except?NULL nogil
cdef void* nvshmem_calloc(size_t count, size_t size) except?NULL nogil
cdef void* nvshmem_align(size_t alignment, size_t size) except?NULL nogil
cdef void nvshmem_free(void* ptr) except* nogil
cdef void* nvshmem_ptr(const void* dest, int pe) except?NULL nogil
cdef void nvshmem_int_p(int* dest, const int value, int pe) except* nogil
cdef int nvshmem_team_my_pe(nvshmem_team_t team) except?-42 nogil
cdef void nvshmemx_barrier_all_on_stream(cudaStream_t stream) except* nogil
cdef void nvshmemx_sync_all_on_stream(cudaStream_t stream) except* nogil
cdef int nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t* attr) except?-42 nogil
cdef void nvshmemx_hostlib_finalize() except* nogil
cdef int nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks, const nvshmemx_uniqueid_t* uniqueid, nvshmemx_init_attr_t* attr) except?-42 nogil
cdef int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* uniqueid) except?-42 nogil
