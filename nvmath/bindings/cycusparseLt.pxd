# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.7.1 to 0.8.1, generator version 0.3.1.dev1565+g7fa82f8eb. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int64_t, uint8_t, uint32_t, int32_t
from libc.stdio cimport FILE
from .cycusparse cimport cusparseStatus_t, _CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR, cusparseOrder_t, cusparseOperation_t, cudaDataType, cudaDataType_t, cudaStream_t, libraryPropertyType, libraryPropertyType_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum cusparseLtSparsity_t "cusparseLtSparsity_t":
    CUSPARSELT_SPARSITY_50_PERCENT "CUSPARSELT_SPARSITY_50_PERCENT"

ctypedef enum cusparseLtMatDescAttribute_t "cusparseLtMatDescAttribute_t":
    CUSPARSELT_MAT_NUM_BATCHES "CUSPARSELT_MAT_NUM_BATCHES"
    CUSPARSELT_MAT_BATCH_STRIDE "CUSPARSELT_MAT_BATCH_STRIDE"

ctypedef enum cusparseComputeType "cusparseComputeType":
    CUSPARSE_COMPUTE_32I "CUSPARSE_COMPUTE_32I"
    CUSPARSE_COMPUTE_16F "CUSPARSE_COMPUTE_16F"
    CUSPARSE_COMPUTE_32F "CUSPARSE_COMPUTE_32F"

ctypedef enum cusparseLtMatmulDescAttribute_t "cusparseLtMatmulDescAttribute_t":
    CUSPARSELT_MATMUL_ACTIVATION_RELU "CUSPARSELT_MATMUL_ACTIVATION_RELU"
    CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND "CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND"
    CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD "CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD"
    CUSPARSELT_MATMUL_ACTIVATION_GELU "CUSPARSELT_MATMUL_ACTIVATION_GELU"
    CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING "CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING"
    CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING "CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING"
    CUSPARSELT_MATMUL_BETA_VECTOR_SCALING "CUSPARSELT_MATMUL_BETA_VECTOR_SCALING"
    CUSPARSELT_MATMUL_BIAS_STRIDE "CUSPARSELT_MATMUL_BIAS_STRIDE"
    CUSPARSELT_MATMUL_BIAS_POINTER "CUSPARSELT_MATMUL_BIAS_POINTER"
    CUSPARSELT_MATMUL_SPARSE_MAT_POINTER "CUSPARSELT_MATMUL_SPARSE_MAT_POINTER"
    CUSPARSELT_MATMUL_A_SCALE_MODE "CUSPARSELT_MATMUL_A_SCALE_MODE"
    CUSPARSELT_MATMUL_B_SCALE_MODE "CUSPARSELT_MATMUL_B_SCALE_MODE"
    CUSPARSELT_MATMUL_C_SCALE_MODE "CUSPARSELT_MATMUL_C_SCALE_MODE"
    CUSPARSELT_MATMUL_D_SCALE_MODE "CUSPARSELT_MATMUL_D_SCALE_MODE"
    CUSPARSELT_MATMUL_D_OUT_SCALE_MODE "CUSPARSELT_MATMUL_D_OUT_SCALE_MODE"
    CUSPARSELT_MATMUL_A_SCALE_POINTER "CUSPARSELT_MATMUL_A_SCALE_POINTER"
    CUSPARSELT_MATMUL_B_SCALE_POINTER "CUSPARSELT_MATMUL_B_SCALE_POINTER"
    CUSPARSELT_MATMUL_C_SCALE_POINTER "CUSPARSELT_MATMUL_C_SCALE_POINTER"
    CUSPARSELT_MATMUL_D_SCALE_POINTER "CUSPARSELT_MATMUL_D_SCALE_POINTER"
    CUSPARSELT_MATMUL_D_OUT_SCALE_POINTER "CUSPARSELT_MATMUL_D_OUT_SCALE_POINTER"

ctypedef enum cusparseLtMatmulMatrixScale_t "cusparseLtMatmulMatrixScale_t":
    CUSPARSELT_MATMUL_SCALE_NONE "CUSPARSELT_MATMUL_SCALE_NONE"
    CUSPARSELT_MATMUL_MATRIX_SCALE_SCALAR_32F "CUSPARSELT_MATMUL_MATRIX_SCALE_SCALAR_32F"
    CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3 "CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3"
    CUSPARSELT_MATMUL_MATRIX_SCALE_VEC64_UE8M0 "CUSPARSELT_MATMUL_MATRIX_SCALE_VEC64_UE8M0"

ctypedef enum cusparseLtMatmulAlg_t "cusparseLtMatmulAlg_t":
    CUSPARSELT_MATMUL_ALG_DEFAULT "CUSPARSELT_MATMUL_ALG_DEFAULT"

ctypedef enum cusparseLtMatmulAlgAttribute_t "cusparseLtMatmulAlgAttribute_t":
    CUSPARSELT_MATMUL_ALG_CONFIG_ID "CUSPARSELT_MATMUL_ALG_CONFIG_ID"
    CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID "CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID"
    CUSPARSELT_MATMUL_SEARCH_ITERATIONS "CUSPARSELT_MATMUL_SEARCH_ITERATIONS"
    CUSPARSELT_MATMUL_SPLIT_K "CUSPARSELT_MATMUL_SPLIT_K"
    CUSPARSELT_MATMUL_SPLIT_K_MODE "CUSPARSELT_MATMUL_SPLIT_K_MODE"
    CUSPARSELT_MATMUL_SPLIT_K_BUFFERS "CUSPARSELT_MATMUL_SPLIT_K_BUFFERS"

ctypedef enum cusparseLtSplitKMode_t "cusparseLtSplitKMode_t":
    CUSPARSELT_INVALID_MODE "CUSPARSELT_INVALID_MODE" = 0
    CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL "CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL" = 1
    CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS "CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS" = 2
    CUSPARSELT_HEURISTIC "CUSPARSELT_HEURISTIC"
    CUSPARSELT_DATAPARALLEL "CUSPARSELT_DATAPARALLEL"
    CUSPARSELT_SPLITK "CUSPARSELT_SPLITK"
    CUSPARSELT_STREAMK "CUSPARSELT_STREAMK"

ctypedef enum cusparseLtPruneAlg_t "cusparseLtPruneAlg_t":
    CUSPARSELT_PRUNE_SPMMA_TILE "CUSPARSELT_PRUNE_SPMMA_TILE" = 0
    CUSPARSELT_PRUNE_SPMMA_STRIP "CUSPARSELT_PRUNE_SPMMA_STRIP" = 1


# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """

    ctypedef struct cuComplex:
        float x
        float y
    ctypedef struct cuDoubleComplex:
        double x
        double y


ctypedef struct cusparseLtHandle_t 'cusparseLtHandle_t':
    uint8_t data[512]

ctypedef struct cusparseLtMatDescriptor_t 'cusparseLtMatDescriptor_t':
    uint8_t data[512]

ctypedef struct cusparseLtMatmulDescriptor_t 'cusparseLtMatmulDescriptor_t':
    uint8_t data[512]

ctypedef struct cusparseLtMatmulAlgSelection_t 'cusparseLtMatmulAlgSelection_t':
    uint8_t data[512]

ctypedef struct cusparseLtMatmulPlan_t 'cusparseLtMatmulPlan_t':
    uint8_t data[512]



###############################################################################
# Functions
###############################################################################

cdef const char* cusparseLtGetErrorName(cusparseStatus_t status) except?NULL nogil
cdef const char* cusparseLtGetErrorString(cusparseStatus_t status) except?NULL nogil
cdef cusparseStatus_t cusparseLtInit(cusparseLtHandle_t* handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtDestroy(const cusparseLtHandle_t* handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtGetVersion(const cusparseLtHandle_t* handle, int* version) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtGetProperty(libraryPropertyType propertyType, int* value) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtDenseDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtStructuredDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order, cusparseLtSparsity_t sparsity) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatDescriptorDestroy(const cusparseLtMatDescriptor_t* matDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatDescSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matmulDescr, cusparseLtMatDescAttribute_t matAttribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatDescGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* matmulDescr, cusparseLtMatDescAttribute_t matAttribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr, cusparseOperation_t opA, cusparseOperation_t opB, const cusparseLtMatDescriptor_t* matA, const cusparseLtMatDescriptor_t* matB, const cusparseLtMatDescriptor_t* matC, const cusparseLtMatDescriptor_t* matD, cusparseComputeType computeType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulDescSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulDescAttribute_t matmulAttribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulDescGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulDescAttribute_t matmulAttribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulAlgSelectionInit(const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection, const cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulAlg_t alg) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulAlgSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection, cusparseLtMatmulAlgAttribute_t attribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulAlgGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatmulAlgSelection_t* algSelection, cusparseLtMatmulAlgAttribute_t attribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulGetWorkspace(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* workspaceSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulPlanInit(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const cusparseLtMatmulDescriptor_t* matmulDescr, const cusparseLtMatmulAlgSelection_t* algSelection) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulPlanDestroy(const cusparseLtMatmulPlan_t* plan) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmul(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, cudaStream_t* streams, int32_t numStreams) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulSearch(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, cudaStream_t* streams, int32_t numStreams) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtSpMMAPrune(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtSpMMAPruneCheck(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, int* valid, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtSpMMAPrune2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtSpMMAPruneCheck2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_in, int* d_valid, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtSpMMACompressedSize(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* compressedSize, size_t* compressedBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtSpMMACompress(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* d_dense, void* d_compressed, void* d_compressed_buffer, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, size_t* compressedSize, size_t* compressedBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtSpMMACompress2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_dense, void* d_compressed, void* d_compressed_buffer, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLtMatmulAlgSelectionDestroy(const cusparseLtMatmulAlgSelection_t* algSelection) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
