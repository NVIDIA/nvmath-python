# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 3.1.7. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cynvshmem cimport *


###############################################################################
# Types
###############################################################################

ctypedef nvshmem_team_config_v1 team_config
ctypedef nvshmemx_uniqueid_args_v1 uniqueid_args
ctypedef nvshmemx_init_args_v1 init_args
ctypedef nvshmemx_init_attr_v1 init_attr

ctypedef cudaStream_t Stream


###############################################################################
# Enum
###############################################################################

ctypedef _anon_enum0 __anon_enum0
ctypedef _anon_enum1 __anon_enum1
ctypedef _anon_enum2 __anon_enum2
ctypedef nvshmemx_status _Status
ctypedef flags _Flags


###############################################################################
# Functions
###############################################################################

cpdef int init_status() except? -1
cpdef int my_pe() except? -1
cpdef int n_pes() except? -1
cpdef intptr_t malloc(size_t size) except? 0
cpdef intptr_t calloc(size_t count, size_t size) except? 0
cpdef intptr_t align(size_t alignment, size_t size) except? 0
cpdef void free(intptr_t ptr) except*
cpdef intptr_t ptr(intptr_t dest, int pe) except? 0
cpdef void int_p(intptr_t dest, int value, int pe) except*
cpdef int team_my_pe(int32_t team) except? -1
cpdef void barrier_all_on_stream(intptr_t stream) except*
cpdef void sync_all_on_stream(intptr_t stream) except*
cpdef hostlib_init_attr(unsigned int flags, intptr_t attr)
cpdef void hostlib_finalize() except*
cpdef set_attr_uniqueid_args(int myrank, int nranks, intptr_t uniqueid, intptr_t attr)
cpdef get_uniqueid(intptr_t uniqueid)
