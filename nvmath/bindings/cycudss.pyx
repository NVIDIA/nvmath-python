# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.5.0. Do not modify it directly.

from ._internal cimport cudss as _cudss


###############################################################################
# Wrapper functions
###############################################################################

cdef cudssStatus_t cudssConfigSet(cudssConfig_t config, cudssConfigParam_t param, void* value, size_t sizeInBytes) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssConfigSet(config, param, value, sizeInBytes)


cdef cudssStatus_t cudssConfigGet(cudssConfig_t config, cudssConfigParam_t param, void* value, size_t sizeInBytes, size_t* sizeWritten) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssConfigGet(config, param, value, sizeInBytes, sizeWritten)


cdef cudssStatus_t cudssDataSet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param, void* value, size_t sizeInBytes) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssDataSet(handle, data, param, value, sizeInBytes)


cdef cudssStatus_t cudssDataGet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param, void* value, size_t sizeInBytes, size_t* sizeWritten) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssDataGet(handle, data, param, value, sizeInBytes, sizeWritten)


cdef cudssStatus_t cudssExecute(cudssHandle_t handle, cudssPhase_t phase, cudssConfig_t solverConfig, cudssData_t solverData, cudssMatrix_t inputMatrix, cudssMatrix_t solution, cudssMatrix_t rhs) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssExecute(handle, phase, solverConfig, solverData, inputMatrix, solution, rhs)


cdef cudssStatus_t cudssSetStream(cudssHandle_t handle, cudaStream_t stream) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssSetStream(handle, stream)


cdef cudssStatus_t cudssSetCommLayer(cudssHandle_t handle, const char* commLibFileName) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssSetCommLayer(handle, commLibFileName)


cdef cudssStatus_t cudssSetThreadingLayer(cudssHandle_t handle, const char* thrLibFileName) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssSetThreadingLayer(handle, thrLibFileName)


cdef cudssStatus_t cudssConfigCreate(cudssConfig_t* solverConfig) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssConfigCreate(solverConfig)


cdef cudssStatus_t cudssConfigDestroy(cudssConfig_t solverConfig) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssConfigDestroy(solverConfig)


cdef cudssStatus_t cudssDataCreate(cudssHandle_t handle, cudssData_t* solverData) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssDataCreate(handle, solverData)


cdef cudssStatus_t cudssDataDestroy(cudssHandle_t handle, cudssData_t solverData) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssDataDestroy(handle, solverData)


cdef cudssStatus_t cudssCreate(cudssHandle_t* handle) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssCreate(handle)


cdef cudssStatus_t cudssDestroy(cudssHandle_t handle) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssDestroy(handle)


cdef cudssStatus_t cudssGetProperty(libraryPropertyType propertyType, int* value) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssGetProperty(propertyType, value)


cdef cudssStatus_t cudssMatrixCreateDn(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t ld, void* values, cudaDataType_t valueType, cudssLayout_t layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixCreateDn(matrix, nrows, ncols, ld, values, valueType, layout)


cdef cudssStatus_t cudssMatrixCreateCsr(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t nnz, void* rowStart, void* rowEnd, void* colIndices, void* values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixCreateCsr(matrix, nrows, ncols, nnz, rowStart, rowEnd, colIndices, values, indexType, valueType, mtype, mview, indexBase)


cdef cudssStatus_t cudssMatrixCreateBatchDn(cudssMatrix_t* matrix, int64_t batchCount, void* nrows, void* ncols, void* ld, void** values, cudaDataType_t indexType, cudaDataType_t valueType, cudssLayout_t layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixCreateBatchDn(matrix, batchCount, nrows, ncols, ld, values, indexType, valueType, layout)


cdef cudssStatus_t cudssMatrixCreateBatchCsr(cudssMatrix_t* matrix, int64_t batchCount, void* nrows, void* ncols, void* nnz, void** rowStart, void** rowEnd, void** colIndices, void** values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixCreateBatchCsr(matrix, batchCount, nrows, ncols, nnz, rowStart, rowEnd, colIndices, values, indexType, valueType, mtype, mview, indexBase)


cdef cudssStatus_t cudssMatrixDestroy(cudssMatrix_t matrix) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixDestroy(matrix)


cdef cudssStatus_t cudssMatrixGetDn(cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* ld, void** values, cudaDataType_t* type, cudssLayout_t* layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixGetDn(matrix, nrows, ncols, ld, values, type, layout)


cdef cudssStatus_t cudssMatrixGetCsr(cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* nnz, void** rowStart, void** rowEnd, void** colIndices, void** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixGetCsr(matrix, nrows, ncols, nnz, rowStart, rowEnd, colIndices, values, indexType, valueType, mtype, mview, indexBase)


cdef cudssStatus_t cudssMatrixSetValues(cudssMatrix_t matrix, void* values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixSetValues(matrix, values)


cdef cudssStatus_t cudssMatrixSetCsrPointers(cudssMatrix_t matrix, void* rowOffsets, void* rowEnd, void* colIndices, void* values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixSetCsrPointers(matrix, rowOffsets, rowEnd, colIndices, values)


cdef cudssStatus_t cudssMatrixGetBatchDn(cudssMatrix_t matrix, int64_t* batchCount, void** nrows, void** ncols, void** ld, void*** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssLayout_t* layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixGetBatchDn(matrix, batchCount, nrows, ncols, ld, values, indexType, valueType, layout)


cdef cudssStatus_t cudssMatrixGetBatchCsr(cudssMatrix_t matrix, int64_t* batchCount, void** nrows, void** ncols, void** nnz, void*** rowStart, void*** rowEnd, void*** colIndices, void*** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixGetBatchCsr(matrix, batchCount, nrows, ncols, nnz, rowStart, rowEnd, colIndices, values, indexType, valueType, mtype, mview, indexBase)


cdef cudssStatus_t cudssMatrixSetBatchValues(cudssMatrix_t matrix, void** values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixSetBatchValues(matrix, values)


cdef cudssStatus_t cudssMatrixSetBatchCsrPointers(cudssMatrix_t matrix, void** rowOffsets, void** rowEnd, void** colIndices, void** values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixSetBatchCsrPointers(matrix, rowOffsets, rowEnd, colIndices, values)


cdef cudssStatus_t cudssMatrixGetFormat(cudssMatrix_t matrix, int* format) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssMatrixGetFormat(matrix, format)


cdef cudssStatus_t cudssGetDeviceMemHandler(cudssHandle_t handle, cudssDeviceMemHandler_t* handler) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssGetDeviceMemHandler(handle, handler)


cdef cudssStatus_t cudssSetDeviceMemHandler(cudssHandle_t handle, const cudssDeviceMemHandler_t* handler) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cudss._cudssSetDeviceMemHandler(handle, handler)
