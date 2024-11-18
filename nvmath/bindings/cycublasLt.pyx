# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from ._internal cimport cublasLt as _cublasLt


###############################################################################
# Wrapper functions
###############################################################################

cdef cublasStatus_t cublasLtCreate(cublasLtHandle_t* lightHandle) except* nogil:
    return _cublasLt._cublasLtCreate(lightHandle)


cdef cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) except* nogil:
    return _cublasLt._cublasLtDestroy(lightHandle)


cdef size_t cublasLtGetVersion() except* nogil:
    return _cublasLt._cublasLtGetVersion()


cdef size_t cublasLtGetCudartVersion() except* nogil:
    return _cublasLt._cublasLtGetCudartVersion()


cdef cublasStatus_t cublasLtGetProperty(libraryPropertyType type, int* value) except* nogil:
    return _cublasLt._cublasLtGetProperty(type, value)


cdef cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream) except* nogil:
    return _cublasLt._cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream)


cdef cublasStatus_t cublasLtMatrixTransform(cublasLtHandle_t lightHandle, cublasLtMatrixTransformDesc_t transformDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* beta, const void* B, cublasLtMatrixLayout_t Bdesc, void* C, cublasLtMatrixLayout_t Cdesc, cudaStream_t stream) except* nogil:
    return _cublasLt._cublasLtMatrixTransform(lightHandle, transformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream)


cdef cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld) except* nogil:
    return _cublasLt._cublasLtMatrixLayoutCreate(matLayout, type, rows, cols, ld)


cdef cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) except* nogil:
    return _cublasLt._cublasLtMatrixLayoutDestroy(matLayout)


cdef cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    return _cublasLt._cublasLtMatrixLayoutSetAttribute(matLayout, attr, buf, sizeInBytes)


cdef cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    return _cublasLt._cublasLtMatrixLayoutGetAttribute(matLayout, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType) except* nogil:
    return _cublasLt._cublasLtMatmulDescCreate(matmulDesc, computeType, scaleType)


cdef cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) except* nogil:
    return _cublasLt._cublasLtMatmulDescDestroy(matmulDesc)


cdef cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    return _cublasLt._cublasLtMatmulDescSetAttribute(matmulDesc, attr, buf, sizeInBytes)


cdef cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    return _cublasLt._cublasLtMatmulDescGetAttribute(matmulDesc, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t* transformDesc, cudaDataType scaleType) except* nogil:
    return _cublasLt._cublasLtMatrixTransformDescCreate(transformDesc, scaleType)


cdef cublasStatus_t cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t transformDesc) except* nogil:
    return _cublasLt._cublasLtMatrixTransformDescDestroy(transformDesc)


cdef cublasStatus_t cublasLtMatrixTransformDescSetAttribute(cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    return _cublasLt._cublasLtMatrixTransformDescSetAttribute(transformDesc, attr, buf, sizeInBytes)


cdef cublasStatus_t cublasLtMatrixTransformDescGetAttribute(cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    return _cublasLt._cublasLtMatrixTransformDescGetAttribute(transformDesc, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref) except* nogil:
    return _cublasLt._cublasLtMatmulPreferenceCreate(pref)


cdef cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) except* nogil:
    return _cublasLt._cublasLtMatmulPreferenceDestroy(pref)


cdef cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    return _cublasLt._cublasLtMatmulPreferenceSetAttribute(pref, attr, buf, sizeInBytes)


cdef cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    return _cublasLt._cublasLtMatmulPreferenceGetAttribute(pref, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount) except* nogil:
    return _cublasLt._cublasLtMatmulAlgoGetHeuristic(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount)


cdef cublasStatus_t cublasLtMatmulAlgoGetIds(cublasLtHandle_t lightHandle, cublasComputeType_t computeType, cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype, int requestedAlgoCount, int algoIdsArray[], int* returnAlgoCount) except* nogil:
    return _cublasLt._cublasLtMatmulAlgoGetIds(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, requestedAlgoCount, algoIdsArray, returnAlgoCount)


cdef cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle, cublasComputeType_t computeType, cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype, int algoId, cublasLtMatmulAlgo_t* algo) except* nogil:
    return _cublasLt._cublasLtMatmulAlgoInit(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, algoId, algo)


cdef cublasStatus_t cublasLtMatmulAlgoCheck(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo, cublasLtMatmulHeuristicResult_t* result) except* nogil:
    return _cublasLt._cublasLtMatmulAlgoCheck(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, algo, result)


cdef cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoCapAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    return _cublasLt._cublasLtMatmulAlgoCapGetAttribute(algo, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    return _cublasLt._cublasLtMatmulAlgoConfigSetAttribute(algo, attr, buf, sizeInBytes)


cdef cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    return _cublasLt._cublasLtMatmulAlgoConfigGetAttribute(algo, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback) except* nogil:
    return _cublasLt._cublasLtLoggerSetCallback(callback)


cdef cublasStatus_t cublasLtLoggerSetFile(FILE* file) except* nogil:
    return _cublasLt._cublasLtLoggerSetFile(file)


cdef cublasStatus_t cublasLtLoggerOpenFile(const char* logFile) except* nogil:
    return _cublasLt._cublasLtLoggerOpenFile(logFile)


cdef cublasStatus_t cublasLtLoggerSetLevel(int level) except* nogil:
    return _cublasLt._cublasLtLoggerSetLevel(level)


cdef cublasStatus_t cublasLtLoggerSetMask(int mask) except* nogil:
    return _cublasLt._cublasLtLoggerSetMask(mask)


cdef cublasStatus_t cublasLtLoggerForceDisable() except* nogil:
    return _cublasLt._cublasLtLoggerForceDisable()


cdef const char* cublasLtGetStatusName(cublasStatus_t status) except* nogil:
    return _cublasLt._cublasLtGetStatusName(status)


cdef const char* cublasLtGetStatusString(cublasStatus_t status) except* nogil:
    return _cublasLt._cublasLtGetStatusString(status)


cdef cublasStatus_t cublasLtHeuristicsCacheGetCapacity(size_t* capacity) except* nogil:
    return _cublasLt._cublasLtHeuristicsCacheGetCapacity(capacity)


cdef cublasStatus_t cublasLtHeuristicsCacheSetCapacity(size_t capacity) except* nogil:
    return _cublasLt._cublasLtHeuristicsCacheSetCapacity(capacity)


cdef unsigned cublasLtDisableCpuInstructionsSetMask(unsigned mask) except* nogil:
    return _cublasLt._cublasLtDisableCpuInstructionsSetMask(mask)
