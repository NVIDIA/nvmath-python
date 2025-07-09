# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.5.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int64_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum cudssOpType_t "cudssOpType_t":
    CUDSS_SUM "CUDSS_SUM"
    CUDSS_MAX "CUDSS_MAX"
    CUDSS_MIN "CUDSS_MIN"

ctypedef enum cudssConfigParam_t "cudssConfigParam_t":
    CUDSS_CONFIG_REORDERING_ALG "CUDSS_CONFIG_REORDERING_ALG"
    CUDSS_CONFIG_FACTORIZATION_ALG "CUDSS_CONFIG_FACTORIZATION_ALG"
    CUDSS_CONFIG_SOLVE_ALG "CUDSS_CONFIG_SOLVE_ALG"
    CUDSS_CONFIG_MATCHING_TYPE "CUDSS_CONFIG_MATCHING_TYPE"
    CUDSS_CONFIG_SOLVE_MODE "CUDSS_CONFIG_SOLVE_MODE"
    CUDSS_CONFIG_IR_N_STEPS "CUDSS_CONFIG_IR_N_STEPS"
    CUDSS_CONFIG_IR_TOL "CUDSS_CONFIG_IR_TOL"
    CUDSS_CONFIG_PIVOT_TYPE "CUDSS_CONFIG_PIVOT_TYPE"
    CUDSS_CONFIG_PIVOT_THRESHOLD "CUDSS_CONFIG_PIVOT_THRESHOLD"
    CUDSS_CONFIG_PIVOT_EPSILON "CUDSS_CONFIG_PIVOT_EPSILON"
    CUDSS_CONFIG_MAX_LU_NNZ "CUDSS_CONFIG_MAX_LU_NNZ"
    CUDSS_CONFIG_HYBRID_MODE "CUDSS_CONFIG_HYBRID_MODE"
    CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT "CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT"
    CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY "CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY"
    CUDSS_CONFIG_HOST_NTHREADS "CUDSS_CONFIG_HOST_NTHREADS"
    CUDSS_CONFIG_HYBRID_EXECUTE_MODE "CUDSS_CONFIG_HYBRID_EXECUTE_MODE"
    CUDSS_CONFIG_PIVOT_EPSILON_ALG "CUDSS_CONFIG_PIVOT_EPSILON_ALG"

ctypedef enum cudssDataParam_t "cudssDataParam_t":
    CUDSS_DATA_INFO "CUDSS_DATA_INFO"
    CUDSS_DATA_LU_NNZ "CUDSS_DATA_LU_NNZ"
    CUDSS_DATA_NPIVOTS "CUDSS_DATA_NPIVOTS"
    CUDSS_DATA_INERTIA "CUDSS_DATA_INERTIA"
    CUDSS_DATA_PERM_REORDER_ROW "CUDSS_DATA_PERM_REORDER_ROW"
    CUDSS_DATA_PERM_REORDER_COL "CUDSS_DATA_PERM_REORDER_COL"
    CUDSS_DATA_PERM_ROW "CUDSS_DATA_PERM_ROW"
    CUDSS_DATA_PERM_COL "CUDSS_DATA_PERM_COL"
    CUDSS_DATA_DIAG "CUDSS_DATA_DIAG"
    CUDSS_DATA_USER_PERM "CUDSS_DATA_USER_PERM"
    CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN "CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN"
    CUDSS_DATA_COMM "CUDSS_DATA_COMM"
    CUDSS_DATA_MEMORY_ESTIMATES "CUDSS_DATA_MEMORY_ESTIMATES"

ctypedef enum cudssPhase_t "cudssPhase_t":
    CUDSS_PHASE_ANALYSIS "CUDSS_PHASE_ANALYSIS" = 1
    CUDSS_PHASE_FACTORIZATION "CUDSS_PHASE_FACTORIZATION" = 2
    CUDSS_PHASE_REFACTORIZATION "CUDSS_PHASE_REFACTORIZATION" = 4
    CUDSS_PHASE_SOLVE "CUDSS_PHASE_SOLVE" = 8
    CUDSS_PHASE_SOLVE_FWD "CUDSS_PHASE_SOLVE_FWD" = 16
    CUDSS_PHASE_SOLVE_DIAG "CUDSS_PHASE_SOLVE_DIAG" = 32
    CUDSS_PHASE_SOLVE_BWD "CUDSS_PHASE_SOLVE_BWD" = 64

ctypedef enum cudssStatus_t "cudssStatus_t":
    CUDSS_STATUS_SUCCESS "CUDSS_STATUS_SUCCESS" = 0
    CUDSS_STATUS_NOT_INITIALIZED "CUDSS_STATUS_NOT_INITIALIZED" = 1
    CUDSS_STATUS_ALLOC_FAILED "CUDSS_STATUS_ALLOC_FAILED" = 2
    CUDSS_STATUS_INVALID_VALUE "CUDSS_STATUS_INVALID_VALUE" = 3
    CUDSS_STATUS_NOT_SUPPORTED "CUDSS_STATUS_NOT_SUPPORTED" = 4
    CUDSS_STATUS_EXECUTION_FAILED "CUDSS_STATUS_EXECUTION_FAILED" = 5
    CUDSS_STATUS_INTERNAL_ERROR "CUDSS_STATUS_INTERNAL_ERROR" = 6
    _CUDSSSTATUS_T_INTERNAL_LOADING_ERROR "_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cudssMatrixType_t "cudssMatrixType_t":
    CUDSS_MTYPE_GENERAL "CUDSS_MTYPE_GENERAL"
    CUDSS_MTYPE_SYMMETRIC "CUDSS_MTYPE_SYMMETRIC"
    CUDSS_MTYPE_HERMITIAN "CUDSS_MTYPE_HERMITIAN"
    CUDSS_MTYPE_SPD "CUDSS_MTYPE_SPD"
    CUDSS_MTYPE_HPD "CUDSS_MTYPE_HPD"

ctypedef enum cudssMatrixViewType_t "cudssMatrixViewType_t":
    CUDSS_MVIEW_FULL "CUDSS_MVIEW_FULL"
    CUDSS_MVIEW_LOWER "CUDSS_MVIEW_LOWER"
    CUDSS_MVIEW_UPPER "CUDSS_MVIEW_UPPER"

ctypedef enum cudssIndexBase_t "cudssIndexBase_t":
    CUDSS_BASE_ZERO "CUDSS_BASE_ZERO"
    CUDSS_BASE_ONE "CUDSS_BASE_ONE"

ctypedef enum cudssLayout_t "cudssLayout_t":
    CUDSS_LAYOUT_COL_MAJOR "CUDSS_LAYOUT_COL_MAJOR"
    CUDSS_LAYOUT_ROW_MAJOR "CUDSS_LAYOUT_ROW_MAJOR"

ctypedef enum cudssAlgType_t "cudssAlgType_t":
    CUDSS_ALG_DEFAULT "CUDSS_ALG_DEFAULT"
    CUDSS_ALG_1 "CUDSS_ALG_1"
    CUDSS_ALG_2 "CUDSS_ALG_2"
    CUDSS_ALG_3 "CUDSS_ALG_3"

ctypedef enum cudssPivotType_t "cudssPivotType_t":
    CUDSS_PIVOT_COL "CUDSS_PIVOT_COL"
    CUDSS_PIVOT_ROW "CUDSS_PIVOT_ROW"
    CUDSS_PIVOT_NONE "CUDSS_PIVOT_NONE"

ctypedef enum cudssMatrixFormat_t "cudssMatrixFormat_t":
    CUDSS_MFORMAT_DENSE "CUDSS_MFORMAT_DENSE" = 1
    CUDSS_MFORMAT_CSR "CUDSS_MFORMAT_CSR" = 2
    CUDSS_MFORMAT_BATCH "CUDSS_MFORMAT_BATCH" = 4


# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef int cudaDataType_t 'cudaDataType_t'
    ctypedef int cudaDataType 'cudaDataType'
    ctypedef int libraryPropertyType_t 'libraryPropertyType_t'
    ctypedef int libraryPropertyType 'libraryPropertyType'

    ctypedef struct cuComplex:
        float x
        float y
    ctypedef struct cuDoubleComplex:
        double x
        double y


ctypedef void* cudssHandle_t 'cudssHandle_t'
ctypedef void* cudssMatrix_t 'cudssMatrix_t'
ctypedef void* cudssData_t 'cudssData_t'
ctypedef void* cudssConfig_t 'cudssConfig_t'
ctypedef struct cudssDistributedInterface_t 'cudssDistributedInterface_t':
    int (*cudssCommRank)(void*, int*)
    int (*cudssCommSize)(void*, int*)
    int (*cudssSend)(const void*, int, cudaDataType_t, int, int, void*, cudaStream_t)
    int (*cudssRecv)(void*, int, cudaDataType_t, int, int, void*, cudaStream_t)
    int (*cudssBcast)(void*, int, cudaDataType_t, int, void*, cudaStream_t)
    int (*cudssReduce)(const void*, void*, int, cudaDataType_t, cudssOpType_t, int, void*, cudaStream_t)
    int (*cudssAllreduce)(const void*, void*, int, cudaDataType_t, cudssOpType_t, void*, cudaStream_t)
    int (*cudssScatterv)(const void*, const int*, const int*, cudaDataType_t, void*, int, cudaDataType_t, int, void*, cudaStream_t)
    int (*cudssCommSplit)(const void*, int, int, void*)
    int (*cudssCommFree)(void*)
ctypedef struct cudssThreadingInterface_t 'cudssThreadingInterface_t':
    int (*cudssGetMaxThreads)()
    void (*cudssParallelFor)(int, int, void*, cudss_thr_func_t)
ctypedef struct cudssDeviceMemHandler_t 'cudssDeviceMemHandler_t':
    void* ctx
    int (*device_alloc)(void*, void**, size_t, cudaStream_t)
    int (*device_free)(void*, void*, size_t, cudaStream_t)
    char name[64]


###############################################################################
# Functions
###############################################################################

cdef cudssStatus_t cudssConfigSet(cudssConfig_t config, cudssConfigParam_t param, void* value, size_t sizeInBytes) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssConfigGet(cudssConfig_t config, cudssConfigParam_t param, void* value, size_t sizeInBytes, size_t* sizeWritten) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssDataSet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param, void* value, size_t sizeInBytes) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssDataGet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param, void* value, size_t sizeInBytes, size_t* sizeWritten) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssExecute(cudssHandle_t handle, cudssPhase_t phase, cudssConfig_t solverConfig, cudssData_t solverData, cudssMatrix_t inputMatrix, cudssMatrix_t solution, cudssMatrix_t rhs) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssSetStream(cudssHandle_t handle, cudaStream_t stream) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssSetCommLayer(cudssHandle_t handle, const char* commLibFileName) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssSetThreadingLayer(cudssHandle_t handle, const char* thrLibFileName) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssConfigCreate(cudssConfig_t* solverConfig) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssConfigDestroy(cudssConfig_t solverConfig) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssDataCreate(cudssHandle_t handle, cudssData_t* solverData) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssDataDestroy(cudssHandle_t handle, cudssData_t solverData) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssCreate(cudssHandle_t* handle) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssDestroy(cudssHandle_t handle) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssGetProperty(libraryPropertyType propertyType, int* value) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixCreateDn(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t ld, void* values, cudaDataType_t valueType, cudssLayout_t layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixCreateCsr(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t nnz, void* rowStart, void* rowEnd, void* colIndices, void* values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixCreateBatchDn(cudssMatrix_t* matrix, int64_t batchCount, void* nrows, void* ncols, void* ld, void** values, cudaDataType_t indexType, cudaDataType_t valueType, cudssLayout_t layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixCreateBatchCsr(cudssMatrix_t* matrix, int64_t batchCount, void* nrows, void* ncols, void* nnz, void** rowStart, void** rowEnd, void** colIndices, void** values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixDestroy(cudssMatrix_t matrix) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixGetDn(cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* ld, void** values, cudaDataType_t* type, cudssLayout_t* layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixGetCsr(cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* nnz, void** rowStart, void** rowEnd, void** colIndices, void** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixSetValues(cudssMatrix_t matrix, void* values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixSetCsrPointers(cudssMatrix_t matrix, void* rowOffsets, void* rowEnd, void* colIndices, void* values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixGetBatchDn(cudssMatrix_t matrix, int64_t* batchCount, void** nrows, void** ncols, void** ld, void*** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssLayout_t* layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixGetBatchCsr(cudssMatrix_t matrix, int64_t* batchCount, void** nrows, void** ncols, void** nnz, void*** rowStart, void*** rowEnd, void*** colIndices, void*** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixSetBatchValues(cudssMatrix_t matrix, void** values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixSetBatchCsrPointers(cudssMatrix_t matrix, void** rowOffsets, void** rowEnd, void** colIndices, void** values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssMatrixGetFormat(cudssMatrix_t matrix, int* format) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssGetDeviceMemHandler(cudssHandle_t handle, cudssDeviceMemHandler_t* handler) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t cudssSetDeviceMemHandler(cudssHandle_t handle, const cudssDeviceMemHandler_t* handler) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
