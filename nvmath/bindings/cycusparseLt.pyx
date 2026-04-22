# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.7.1 to 0.8.1, generator version 0.3.1.dev1565+g7fa82f8eb. Do not modify it directly.

from ._internal cimport cusparseLt as _cusparseLt


###############################################################################
# Wrapper functions
###############################################################################

cdef const char* cusparseLtGetErrorName(cusparseStatus_t status) except?NULL nogil:
    return _cusparseLt._cusparseLtGetErrorName(status)


cdef const char* cusparseLtGetErrorString(cusparseStatus_t status) except?NULL nogil:
    return _cusparseLt._cusparseLtGetErrorString(status)


cdef cusparseStatus_t cusparseLtInit(cusparseLtHandle_t* handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtInit(handle)


cdef cusparseStatus_t cusparseLtDestroy(const cusparseLtHandle_t* handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtDestroy(handle)


cdef cusparseStatus_t cusparseLtGetVersion(const cusparseLtHandle_t* handle, int* version) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtGetVersion(handle, version)


cdef cusparseStatus_t cusparseLtGetProperty(libraryPropertyType propertyType, int* value) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtGetProperty(propertyType, value)


cdef cusparseStatus_t cusparseLtDenseDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtDenseDescriptorInit(handle, matDescr, rows, cols, ld, alignment, valueType, order)


cdef cusparseStatus_t cusparseLtStructuredDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order, cusparseLtSparsity_t sparsity) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtStructuredDescriptorInit(handle, matDescr, rows, cols, ld, alignment, valueType, order, sparsity)


cdef cusparseStatus_t cusparseLtMatDescriptorDestroy(const cusparseLtMatDescriptor_t* matDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatDescriptorDestroy(matDescr)


cdef cusparseStatus_t cusparseLtMatDescSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matmulDescr, cusparseLtMatDescAttribute_t matAttribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatDescSetAttribute(handle, matmulDescr, matAttribute, data, dataSize)


cdef cusparseStatus_t cusparseLtMatDescGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* matmulDescr, cusparseLtMatDescAttribute_t matAttribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatDescGetAttribute(handle, matmulDescr, matAttribute, data, dataSize)


cdef cusparseStatus_t cusparseLtMatmulDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr, cusparseOperation_t opA, cusparseOperation_t opB, const cusparseLtMatDescriptor_t* matA, const cusparseLtMatDescriptor_t* matB, const cusparseLtMatDescriptor_t* matC, const cusparseLtMatDescriptor_t* matD, cusparseComputeType computeType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulDescriptorInit(handle, matmulDescr, opA, opB, matA, matB, matC, matD, computeType)


cdef cusparseStatus_t cusparseLtMatmulDescSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulDescAttribute_t matmulAttribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulDescSetAttribute(handle, matmulDescr, matmulAttribute, data, dataSize)


cdef cusparseStatus_t cusparseLtMatmulDescGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulDescAttribute_t matmulAttribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulDescGetAttribute(handle, matmulDescr, matmulAttribute, data, dataSize)


cdef cusparseStatus_t cusparseLtMatmulAlgSelectionInit(const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection, const cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulAlg_t alg) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulAlgSelectionInit(handle, algSelection, matmulDescr, alg)


cdef cusparseStatus_t cusparseLtMatmulAlgSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection, cusparseLtMatmulAlgAttribute_t attribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulAlgSetAttribute(handle, algSelection, attribute, data, dataSize)


cdef cusparseStatus_t cusparseLtMatmulAlgGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatmulAlgSelection_t* algSelection, cusparseLtMatmulAlgAttribute_t attribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulAlgGetAttribute(handle, algSelection, attribute, data, dataSize)


cdef cusparseStatus_t cusparseLtMatmulGetWorkspace(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* workspaceSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulGetWorkspace(handle, plan, workspaceSize)


cdef cusparseStatus_t cusparseLtMatmulPlanInit(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const cusparseLtMatmulDescriptor_t* matmulDescr, const cusparseLtMatmulAlgSelection_t* algSelection) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulPlanInit(handle, plan, matmulDescr, algSelection)


cdef cusparseStatus_t cusparseLtMatmulPlanDestroy(const cusparseLtMatmulPlan_t* plan) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulPlanDestroy(plan)


cdef cusparseStatus_t cusparseLtMatmul(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, cudaStream_t* streams, int32_t numStreams) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmul(handle, plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, streams, numStreams)


cdef cusparseStatus_t cusparseLtMatmulSearch(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, cudaStream_t* streams, int32_t numStreams) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulSearch(handle, plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, streams, numStreams)


cdef cusparseStatus_t cusparseLtSpMMAPrune(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtSpMMAPrune(handle, matmulDescr, d_in, d_out, pruneAlg, stream)


cdef cusparseStatus_t cusparseLtSpMMAPruneCheck(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, int* valid, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtSpMMAPruneCheck(handle, matmulDescr, d_in, valid, stream)


cdef cusparseStatus_t cusparseLtSpMMAPrune2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtSpMMAPrune2(handle, sparseMatDescr, isSparseA, op, d_in, d_out, pruneAlg, stream)


cdef cusparseStatus_t cusparseLtSpMMAPruneCheck2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_in, int* d_valid, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtSpMMAPruneCheck2(handle, sparseMatDescr, isSparseA, op, d_in, d_valid, stream)


cdef cusparseStatus_t cusparseLtSpMMACompressedSize(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* compressedSize, size_t* compressedBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtSpMMACompressedSize(handle, plan, compressedSize, compressedBufferSize)


cdef cusparseStatus_t cusparseLtSpMMACompress(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* d_dense, void* d_compressed, void* d_compressed_buffer, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtSpMMACompress(handle, plan, d_dense, d_compressed, d_compressed_buffer, stream)


cdef cusparseStatus_t cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, size_t* compressedSize, size_t* compressedBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtSpMMACompressedSize2(handle, sparseMatDescr, compressedSize, compressedBufferSize)


cdef cusparseStatus_t cusparseLtSpMMACompress2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_dense, void* d_compressed, void* d_compressed_buffer, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtSpMMACompress2(handle, sparseMatDescr, isSparseA, op, d_dense, d_compressed, d_compressed_buffer, stream)


cdef cusparseStatus_t cusparseLtMatmulAlgSelectionDestroy(const cusparseLtMatmulAlgSelection_t* algSelection) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cusparseLt._cusparseLtMatmulAlgSelectionDestroy(algSelection)
