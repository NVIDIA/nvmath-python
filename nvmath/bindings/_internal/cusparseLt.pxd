# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.7.1 to 0.8.1, generator version 0.3.1.dev1565+g7fa82f8eb. Do not modify it directly.

from libc.stdint cimport int64_t, uint8_t, uint32_t, int32_t
from ..cycusparseLt cimport *
from ..cycusparseLt cimport cudaDataType, cudaStream_t, libraryPropertyType, cusparseComputeType
from ..cycusparse cimport cusparseStatus_t, _CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR, cusparseOrder_t, cusparseOperation_t

cdef void* load_library(const int driver_ver) except* with gil

###############################################################################
# Wrapper functions
###############################################################################

cdef const char* _cusparseLtGetErrorName(cusparseStatus_t status) except?NULL nogil
cdef const char* _cusparseLtGetErrorString(cusparseStatus_t status) except?NULL nogil
cdef cusparseStatus_t _cusparseLtInit(cusparseLtHandle_t* handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtDestroy(const cusparseLtHandle_t* handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtGetVersion(const cusparseLtHandle_t* handle, int* version) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtGetProperty(libraryPropertyType propertyType, int* value) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtDenseDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtStructuredDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order, cusparseLtSparsity_t sparsity) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatDescriptorDestroy(const cusparseLtMatDescriptor_t* matDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatDescSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matmulDescr, cusparseLtMatDescAttribute_t matAttribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatDescGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* matmulDescr, cusparseLtMatDescAttribute_t matAttribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr, cusparseOperation_t opA, cusparseOperation_t opB, const cusparseLtMatDescriptor_t* matA, const cusparseLtMatDescriptor_t* matB, const cusparseLtMatDescriptor_t* matC, const cusparseLtMatDescriptor_t* matD, cusparseComputeType computeType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulDescSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulDescAttribute_t matmulAttribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulDescGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulDescAttribute_t matmulAttribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulAlgSelectionInit(const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection, const cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulAlg_t alg) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulAlgSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection, cusparseLtMatmulAlgAttribute_t attribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulAlgGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatmulAlgSelection_t* algSelection, cusparseLtMatmulAlgAttribute_t attribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulGetWorkspace(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* workspaceSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulPlanInit(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const cusparseLtMatmulDescriptor_t* matmulDescr, const cusparseLtMatmulAlgSelection_t* algSelection) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulPlanDestroy(const cusparseLtMatmulPlan_t* plan) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmul(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, cudaStream_t* streams, int32_t numStreams) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulSearch(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, cudaStream_t* streams, int32_t numStreams) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtSpMMAPrune(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtSpMMAPruneCheck(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, int* valid, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtSpMMAPrune2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtSpMMAPruneCheck2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_in, int* d_valid, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtSpMMACompressedSize(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* compressedSize, size_t* compressedBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtSpMMACompress(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* d_dense, void* d_compressed, void* d_compressed_buffer, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, size_t* compressedSize, size_t* compressedBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtSpMMACompress2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_dense, void* d_compressed, void* d_compressed_buffer, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t _cusparseLtMatmulAlgSelectionDestroy(const cusparseLtMatmulAlgSelection_t* algSelection) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
