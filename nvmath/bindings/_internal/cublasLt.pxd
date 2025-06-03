# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from ..cycublasLt cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef cublasStatus_t _cublasLtCreate(cublasLtHandle_t* lightHandle) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtDestroy(cublasLtHandle_t lightHandle) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef size_t _cublasLtGetVersion() except?0 nogil
cdef size_t _cublasLtGetCudartVersion() except?0 nogil
cdef cublasStatus_t _cublasLtGetProperty(libraryPropertyType type, int* value) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatrixTransform(cublasLtHandle_t lightHandle, cublasLtMatrixTransformDesc_t transformDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* beta, const void* B, cublasLtMatrixLayout_t Bdesc, void* C, cublasLtMatrixLayout_t Cdesc, cudaStream_t stream) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, const void* buf, size_t sizeInBytes) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t* transformDesc, cudaDataType scaleType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t transformDesc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatrixTransformDescSetAttribute(cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, const void* buf, size_t sizeInBytes) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatrixTransformDescGetAttribute(cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void* buf, size_t sizeInBytes) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulAlgoGetIds(cublasLtHandle_t lightHandle, cublasComputeType_t computeType, cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype, int requestedAlgoCount, int algoIdsArray[], int* returnAlgoCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle, cublasComputeType_t computeType, cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype, int algoId, cublasLtMatmulAlgo_t* algo) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulAlgoCheck(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo, cublasLtMatmulHeuristicResult_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoCapAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr, const void* buf, size_t sizeInBytes) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtLoggerSetFile(FILE* file) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtLoggerOpenFile(const char* logFile) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtLoggerSetLevel(int level) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtLoggerSetMask(int mask) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtLoggerForceDisable() except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef const char* _cublasLtGetStatusName(cublasStatus_t status) except?NULL nogil
cdef const char* _cublasLtGetStatusString(cublasStatus_t status) except?NULL nogil
cdef cublasStatus_t _cublasLtHeuristicsCacheGetCapacity(size_t* capacity) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasStatus_t _cublasLtHeuristicsCacheSetCapacity(size_t capacity) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef unsigned _cublasLtDisableCpuInstructionsSetMask(unsigned mask) except?0 nogil
