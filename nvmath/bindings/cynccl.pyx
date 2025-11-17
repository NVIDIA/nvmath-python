# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 2.11.4 to 2.28.3. Do not modify it directly.

from ._internal cimport nccl as _nccl


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t ncclGetVersion(int* version) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetVersion(version)


cdef ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetUniqueId(uniqueId)


cdef ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommInitRank(comm, nranks, commId, rank)


cdef ncclResult_t ncclCommDestroy(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommDestroy(comm)


cdef ncclResult_t ncclCommAbort(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommAbort(comm)


cdef const char* ncclGetErrorString(ncclResult_t result) except?NULL nogil:
    return _nccl._ncclGetErrorString(result)


cdef ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommCount(comm, count)


cdef ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommCuDevice(comm, device)


cdef ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommUserRank(comm, rank)


cdef const char* ncclGetLastError(ncclComm_t comm) except?NULL nogil:
    return _nccl._ncclGetLastError(comm)


cdef ncclResult_t ncclCommFinalize(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommFinalize(comm)
