# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 2.11.4 to 2.28.3. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cynccl cimport *


###############################################################################
# Types
###############################################################################

ctypedef ncclComm_t Comm

ctypedef cudaStream_t Stream


###############################################################################
# Enum
###############################################################################

ctypedef ncclResult_t _Result


###############################################################################
# Functions
###############################################################################

cpdef int get_version() except? -1
cpdef get_unique_id(intptr_t unique_id)
cpdef intptr_t comm_init_rank(int nranks, comm_id, int rank) except? 0
cpdef comm_destroy(intptr_t comm)
cpdef comm_abort(intptr_t comm)
cpdef str get_error_string(int result)
cpdef int comm_count(intptr_t comm) except? -1
cpdef int comm_cu_device(intptr_t comm) except? -1
cpdef int comm_user_rank(intptr_t comm) except? -1
cpdef str get_last_error(intptr_t comm)
cpdef comm_finalize(intptr_t comm)
