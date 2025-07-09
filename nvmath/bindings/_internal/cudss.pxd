# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.5.0. Do not modify it directly.

from ..cycudss cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef cudssStatus_t _cudssConfigSet(cudssConfig_t config, cudssConfigParam_t param, void* value, size_t sizeInBytes) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssConfigGet(cudssConfig_t config, cudssConfigParam_t param, void* value, size_t sizeInBytes, size_t* sizeWritten) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssDataSet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param, void* value, size_t sizeInBytes) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssDataGet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param, void* value, size_t sizeInBytes, size_t* sizeWritten) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssExecute(cudssHandle_t handle, cudssPhase_t phase, cudssConfig_t solverConfig, cudssData_t solverData, cudssMatrix_t inputMatrix, cudssMatrix_t solution, cudssMatrix_t rhs) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssSetStream(cudssHandle_t handle, cudaStream_t stream) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssSetCommLayer(cudssHandle_t handle, const char* commLibFileName) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssSetThreadingLayer(cudssHandle_t handle, const char* thrLibFileName) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssConfigCreate(cudssConfig_t* solverConfig) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssConfigDestroy(cudssConfig_t solverConfig) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssDataCreate(cudssHandle_t handle, cudssData_t* solverData) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssDataDestroy(cudssHandle_t handle, cudssData_t solverData) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssCreate(cudssHandle_t* handle) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssDestroy(cudssHandle_t handle) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssGetProperty(libraryPropertyType propertyType, int* value) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixCreateDn(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t ld, void* values, cudaDataType_t valueType, cudssLayout_t layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixCreateCsr(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t nnz, void* rowStart, void* rowEnd, void* colIndices, void* values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixCreateBatchDn(cudssMatrix_t* matrix, int64_t batchCount, void* nrows, void* ncols, void* ld, void** values, cudaDataType_t indexType, cudaDataType_t valueType, cudssLayout_t layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixCreateBatchCsr(cudssMatrix_t* matrix, int64_t batchCount, void* nrows, void* ncols, void* nnz, void** rowStart, void** rowEnd, void** colIndices, void** values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixDestroy(cudssMatrix_t matrix) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixGetDn(cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* ld, void** values, cudaDataType_t* type, cudssLayout_t* layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixGetCsr(cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* nnz, void** rowStart, void** rowEnd, void** colIndices, void** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixSetValues(cudssMatrix_t matrix, void* values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixSetCsrPointers(cudssMatrix_t matrix, void* rowOffsets, void* rowEnd, void* colIndices, void* values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixGetBatchDn(cudssMatrix_t matrix, int64_t* batchCount, void** nrows, void** ncols, void** ld, void*** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssLayout_t* layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixGetBatchCsr(cudssMatrix_t matrix, int64_t* batchCount, void** nrows, void** ncols, void** nnz, void*** rowStart, void*** rowEnd, void*** colIndices, void*** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixSetBatchValues(cudssMatrix_t matrix, void** values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixSetBatchCsrPointers(cudssMatrix_t matrix, void** rowOffsets, void** rowEnd, void** colIndices, void** values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssMatrixGetFormat(cudssMatrix_t matrix, int* format) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssGetDeviceMemHandler(cudssHandle_t handle, cudssDeviceMemHandler_t* handler) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cudssStatus_t _cudssSetDeviceMemHandler(cudssHandle_t handle, const cudssDeviceMemHandler_t* handler) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil
