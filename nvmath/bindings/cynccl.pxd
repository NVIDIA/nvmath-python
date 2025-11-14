# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 2.11.4 to 2.28.3. Do not modify it directly.


from libc.stdint cimport int64_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum ncclResult_t "ncclResult_t":
    ncclSuccess "ncclSuccess" = 0
    ncclUnhandledCudaError "ncclUnhandledCudaError" = 1
    ncclSystemError "ncclSystemError" = 2
    ncclInternalError "ncclInternalError" = 3
    ncclInvalidArgument "ncclInvalidArgument" = 4
    ncclInvalidUsage "ncclInvalidUsage" = 5
    ncclRemoteError "ncclRemoteError" = 6
    ncclInProgress "ncclInProgress" = 7
    ncclNumResults "ncclNumResults" = 8
    _NCCLRESULT_T_INTERNAL_LOADING_ERROR "_NCCLRESULT_T_INTERNAL_LOADING_ERROR" = -42


# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """
    ctypedef void* cudaStream_t 'cudaStream_t'


ctypedef void* ncclComm_t 'ncclComm_t'
ctypedef struct ncclUniqueId 'ncclUniqueId':
    char internal[128]


###############################################################################
# Functions
###############################################################################

cdef ncclResult_t ncclGetVersion(int* version) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommDestroy(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommAbort(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef const char* ncclGetErrorString(ncclResult_t result) except?NULL nogil
cdef ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef const char* ncclGetLastError(ncclComm_t comm) except?NULL nogil
cdef ncclResult_t ncclCommFinalize(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
