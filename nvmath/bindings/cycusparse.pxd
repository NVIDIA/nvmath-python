# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int64_t
from libc.stdio cimport FILE


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum cusparseStatus_t "cusparseStatus_t":
    CUSPARSE_STATUS_SUCCESS "CUSPARSE_STATUS_SUCCESS" = 0
    CUSPARSE_STATUS_NOT_INITIALIZED "CUSPARSE_STATUS_NOT_INITIALIZED" = 1
    CUSPARSE_STATUS_ALLOC_FAILED "CUSPARSE_STATUS_ALLOC_FAILED" = 2
    CUSPARSE_STATUS_INVALID_VALUE "CUSPARSE_STATUS_INVALID_VALUE" = 3
    CUSPARSE_STATUS_ARCH_MISMATCH "CUSPARSE_STATUS_ARCH_MISMATCH" = 4
    CUSPARSE_STATUS_MAPPING_ERROR "CUSPARSE_STATUS_MAPPING_ERROR" = 5
    CUSPARSE_STATUS_EXECUTION_FAILED "CUSPARSE_STATUS_EXECUTION_FAILED" = 6
    CUSPARSE_STATUS_INTERNAL_ERROR "CUSPARSE_STATUS_INTERNAL_ERROR" = 7
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED" = 8
    CUSPARSE_STATUS_ZERO_PIVOT "CUSPARSE_STATUS_ZERO_PIVOT" = 9
    CUSPARSE_STATUS_NOT_SUPPORTED "CUSPARSE_STATUS_NOT_SUPPORTED" = 10
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES" = 11
    _CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR "_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cusparsePointerMode_t "cusparsePointerMode_t":
    CUSPARSE_POINTER_MODE_HOST "CUSPARSE_POINTER_MODE_HOST" = 0
    CUSPARSE_POINTER_MODE_DEVICE "CUSPARSE_POINTER_MODE_DEVICE" = 1

ctypedef enum cusparseAction_t "cusparseAction_t":
    CUSPARSE_ACTION_SYMBOLIC "CUSPARSE_ACTION_SYMBOLIC" = 0
    CUSPARSE_ACTION_NUMERIC "CUSPARSE_ACTION_NUMERIC" = 1

ctypedef enum cusparseMatrixType_t "cusparseMatrixType_t":
    CUSPARSE_MATRIX_TYPE_GENERAL "CUSPARSE_MATRIX_TYPE_GENERAL" = 0
    CUSPARSE_MATRIX_TYPE_SYMMETRIC "CUSPARSE_MATRIX_TYPE_SYMMETRIC" = 1
    CUSPARSE_MATRIX_TYPE_HERMITIAN "CUSPARSE_MATRIX_TYPE_HERMITIAN" = 2
    CUSPARSE_MATRIX_TYPE_TRIANGULAR "CUSPARSE_MATRIX_TYPE_TRIANGULAR" = 3
    _CUSPARSEMATRIXTYPE_T_INTERNAL_LOADING_ERROR "_CUSPARSEMATRIXTYPE_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cusparseFillMode_t "cusparseFillMode_t":
    CUSPARSE_FILL_MODE_LOWER "CUSPARSE_FILL_MODE_LOWER" = 0
    CUSPARSE_FILL_MODE_UPPER "CUSPARSE_FILL_MODE_UPPER" = 1
    _CUSPARSEFILLMODE_T_INTERNAL_LOADING_ERROR "_CUSPARSEFILLMODE_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cusparseDiagType_t "cusparseDiagType_t":
    CUSPARSE_DIAG_TYPE_NON_UNIT "CUSPARSE_DIAG_TYPE_NON_UNIT" = 0
    CUSPARSE_DIAG_TYPE_UNIT "CUSPARSE_DIAG_TYPE_UNIT" = 1
    _CUSPARSEDIAGTYPE_T_INTERNAL_LOADING_ERROR "_CUSPARSEDIAGTYPE_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cusparseIndexBase_t "cusparseIndexBase_t":
    CUSPARSE_INDEX_BASE_ZERO "CUSPARSE_INDEX_BASE_ZERO" = 0
    CUSPARSE_INDEX_BASE_ONE "CUSPARSE_INDEX_BASE_ONE" = 1
    _CUSPARSEINDEXBASE_T_INTERNAL_LOADING_ERROR "_CUSPARSEINDEXBASE_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cusparseOperation_t "cusparseOperation_t":
    CUSPARSE_OPERATION_NON_TRANSPOSE "CUSPARSE_OPERATION_NON_TRANSPOSE" = 0
    CUSPARSE_OPERATION_TRANSPOSE "CUSPARSE_OPERATION_TRANSPOSE" = 1
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE "CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE" = 2

ctypedef enum cusparseDirection_t "cusparseDirection_t":
    CUSPARSE_DIRECTION_ROW "CUSPARSE_DIRECTION_ROW" = 0
    CUSPARSE_DIRECTION_COLUMN "CUSPARSE_DIRECTION_COLUMN" = 1

ctypedef enum cusparseSolvePolicy_t "cusparseSolvePolicy_t":
    CUSPARSE_SOLVE_POLICY_NO_LEVEL "CUSPARSE_SOLVE_POLICY_NO_LEVEL" = 0
    CUSPARSE_SOLVE_POLICY_USE_LEVEL "CUSPARSE_SOLVE_POLICY_USE_LEVEL" = 1

ctypedef enum cusparseColorAlg_t "cusparseColorAlg_t":
    CUSPARSE_COLOR_ALG0 "CUSPARSE_COLOR_ALG0" = 0
    CUSPARSE_COLOR_ALG1 "CUSPARSE_COLOR_ALG1" = 1

ctypedef enum cusparseCsr2CscAlg_t "cusparseCsr2CscAlg_t":
    CUSPARSE_CSR2CSC_ALG_DEFAULT "CUSPARSE_CSR2CSC_ALG_DEFAULT" = 1
    CUSPARSE_CSR2CSC_ALG1 "CUSPARSE_CSR2CSC_ALG1" = 1
    CUSPARSE_CSR2CSC_ALG2 "CUSPARSE_CSR2CSC_ALG2" = 2

ctypedef enum cusparseFormat_t "cusparseFormat_t":
    CUSPARSE_FORMAT_CSR "CUSPARSE_FORMAT_CSR" = 1
    CUSPARSE_FORMAT_CSC "CUSPARSE_FORMAT_CSC" = 2
    CUSPARSE_FORMAT_COO "CUSPARSE_FORMAT_COO" = 3
    CUSPARSE_FORMAT_BLOCKED_ELL "CUSPARSE_FORMAT_BLOCKED_ELL" = 5
    CUSPARSE_FORMAT_BSR "CUSPARSE_FORMAT_BSR" = 6
    CUSPARSE_FORMAT_SLICED_ELLPACK "CUSPARSE_FORMAT_SLICED_ELLPACK" = 7
    CUSPARSE_FORMAT_COO_AOS "CUSPARSE_FORMAT_COO_AOS" = 4

ctypedef enum cusparseOrder_t "cusparseOrder_t":
    CUSPARSE_ORDER_COL "CUSPARSE_ORDER_COL" = 1
    CUSPARSE_ORDER_ROW "CUSPARSE_ORDER_ROW" = 2

ctypedef enum cusparseIndexType_t "cusparseIndexType_t":
    CUSPARSE_INDEX_16U "CUSPARSE_INDEX_16U" = 1
    CUSPARSE_INDEX_32I "CUSPARSE_INDEX_32I" = 2
    CUSPARSE_INDEX_64I "CUSPARSE_INDEX_64I" = 3

ctypedef enum cusparseSpMVAlg_t "cusparseSpMVAlg_t":
    CUSPARSE_SPMV_ALG_DEFAULT "CUSPARSE_SPMV_ALG_DEFAULT" = 0
    CUSPARSE_SPMV_CSR_ALG1 "CUSPARSE_SPMV_CSR_ALG1" = 2
    CUSPARSE_SPMV_CSR_ALG2 "CUSPARSE_SPMV_CSR_ALG2" = 3
    CUSPARSE_SPMV_COO_ALG1 "CUSPARSE_SPMV_COO_ALG1" = 1
    CUSPARSE_SPMV_COO_ALG2 "CUSPARSE_SPMV_COO_ALG2" = 4
    CUSPARSE_SPMV_SELL_ALG1 "CUSPARSE_SPMV_SELL_ALG1" = 5
    CUSPARSE_MV_ALG_DEFAULT "CUSPARSE_MV_ALG_DEFAULT" = 0
    CUSPARSE_COOMV_ALG "CUSPARSE_COOMV_ALG" = 1
    CUSPARSE_CSRMV_ALG1 "CUSPARSE_CSRMV_ALG1" = 2
    CUSPARSE_CSRMV_ALG2 "CUSPARSE_CSRMV_ALG2" = 3

ctypedef enum cusparseSpMMAlg_t "cusparseSpMMAlg_t":
    CUSPARSE_SPMM_ALG_DEFAULT "CUSPARSE_SPMM_ALG_DEFAULT" = 0
    CUSPARSE_SPMM_COO_ALG1 "CUSPARSE_SPMM_COO_ALG1" = 1
    CUSPARSE_SPMM_COO_ALG2 "CUSPARSE_SPMM_COO_ALG2" = 2
    CUSPARSE_SPMM_COO_ALG3 "CUSPARSE_SPMM_COO_ALG3" = 3
    CUSPARSE_SPMM_COO_ALG4 "CUSPARSE_SPMM_COO_ALG4" = 5
    CUSPARSE_SPMM_CSR_ALG1 "CUSPARSE_SPMM_CSR_ALG1" = 4
    CUSPARSE_SPMM_CSR_ALG2 "CUSPARSE_SPMM_CSR_ALG2" = 6
    CUSPARSE_SPMM_CSR_ALG3 "CUSPARSE_SPMM_CSR_ALG3" = 12
    CUSPARSE_SPMM_BLOCKED_ELL_ALG1 "CUSPARSE_SPMM_BLOCKED_ELL_ALG1" = 13
    CUSPARSE_SPMM_BSR_ALG1 "CUSPARSE_SPMM_BSR_ALG1" = 14
    CUSPARSE_MM_ALG_DEFAULT "CUSPARSE_MM_ALG_DEFAULT" = 0
    CUSPARSE_COOMM_ALG1 "CUSPARSE_COOMM_ALG1" = 1
    CUSPARSE_COOMM_ALG2 "CUSPARSE_COOMM_ALG2" = 2
    CUSPARSE_COOMM_ALG3 "CUSPARSE_COOMM_ALG3" = 3
    CUSPARSE_CSRMM_ALG1 "CUSPARSE_CSRMM_ALG1" = 4
    CUSPARSE_SPMMA_PREPROCESS "CUSPARSE_SPMMA_PREPROCESS" = 7
    CUSPARSE_SPMMA_ALG1 "CUSPARSE_SPMMA_ALG1" = 8
    CUSPARSE_SPMMA_ALG2 "CUSPARSE_SPMMA_ALG2" = 9
    CUSPARSE_SPMMA_ALG3 "CUSPARSE_SPMMA_ALG3" = 10
    CUSPARSE_SPMMA_ALG4 "CUSPARSE_SPMMA_ALG4" = 11

ctypedef enum cusparseSpGEMMAlg_t "cusparseSpGEMMAlg_t":
    CUSPARSE_SPGEMM_DEFAULT "CUSPARSE_SPGEMM_DEFAULT" = 0
    CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC "CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC" = 1
    CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC "CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC" = 2
    CUSPARSE_SPGEMM_ALG1 "CUSPARSE_SPGEMM_ALG1" = 3
    CUSPARSE_SPGEMM_ALG2 "CUSPARSE_SPGEMM_ALG2" = 4
    CUSPARSE_SPGEMM_ALG3 "CUSPARSE_SPGEMM_ALG3" = 5

ctypedef enum cusparseSparseToDenseAlg_t "cusparseSparseToDenseAlg_t":
    CUSPARSE_SPARSETODENSE_ALG_DEFAULT "CUSPARSE_SPARSETODENSE_ALG_DEFAULT" = 0

ctypedef enum cusparseDenseToSparseAlg_t "cusparseDenseToSparseAlg_t":
    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT "CUSPARSE_DENSETOSPARSE_ALG_DEFAULT" = 0

ctypedef enum cusparseSDDMMAlg_t "cusparseSDDMMAlg_t":
    CUSPARSE_SDDMM_ALG_DEFAULT "CUSPARSE_SDDMM_ALG_DEFAULT" = 0

ctypedef enum cusparseSpMatAttribute_t "cusparseSpMatAttribute_t":
    CUSPARSE_SPMAT_FILL_MODE "CUSPARSE_SPMAT_FILL_MODE"
    CUSPARSE_SPMAT_DIAG_TYPE "CUSPARSE_SPMAT_DIAG_TYPE"

ctypedef enum cusparseSpSVAlg_t "cusparseSpSVAlg_t":
    CUSPARSE_SPSV_ALG_DEFAULT "CUSPARSE_SPSV_ALG_DEFAULT" = 0

ctypedef enum cusparseSpSMAlg_t "cusparseSpSMAlg_t":
    CUSPARSE_SPSM_ALG_DEFAULT "CUSPARSE_SPSM_ALG_DEFAULT" = 0

ctypedef enum cusparseSpMMOpAlg_t "cusparseSpMMOpAlg_t":
    CUSPARSE_SPMM_OP_ALG_DEFAULT "CUSPARSE_SPMM_OP_ALG_DEFAULT"

ctypedef enum cusparseSpSVUpdate_t "cusparseSpSVUpdate_t":
    CUSPARSE_SPSV_UPDATE_GENERAL "CUSPARSE_SPSV_UPDATE_GENERAL" = 0
    CUSPARSE_SPSV_UPDATE_DIAGONAL "CUSPARSE_SPSV_UPDATE_DIAGONAL" = 1

ctypedef enum cusparseSpSMUpdate_t "cusparseSpSMUpdate_t":
    CUSPARSE_SPSM_UPDATE_GENERAL "CUSPARSE_SPSM_UPDATE_GENERAL" = 0
    CUSPARSE_SPSM_UPDATE_DIAGONAL "CUSPARSE_SPSM_UPDATE_DIAGONAL" = 1


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


ctypedef void* cusparseHandle_t 'cusparseHandle_t'
ctypedef void* cusparseMatDescr_t 'cusparseMatDescr_t'
ctypedef void* cusparseSpVecDescr_t 'cusparseSpVecDescr_t'
ctypedef void* cusparseDnVecDescr_t 'cusparseDnVecDescr_t'
ctypedef void* cusparseSpMatDescr_t 'cusparseSpMatDescr_t'
ctypedef void* cusparseDnMatDescr_t 'cusparseDnMatDescr_t'
ctypedef void* cusparseSpGEMMDescr_t 'cusparseSpGEMMDescr_t'
ctypedef void* cusparseSpSVDescr_t 'cusparseSpSVDescr_t'
ctypedef void* cusparseSpSMDescr_t 'cusparseSpSMDescr_t'
ctypedef void* cusparseSpMMOpPlan_t 'cusparseSpMMOpPlan_t'
ctypedef void* cusparseConstSpVecDescr_t 'cusparseConstSpVecDescr_t'
ctypedef void* cusparseConstDnVecDescr_t 'cusparseConstDnVecDescr_t'
ctypedef void* cusparseConstSpMatDescr_t 'cusparseConstSpMatDescr_t'
ctypedef void* cusparseConstDnMatDescr_t 'cusparseConstDnMatDescr_t'
ctypedef void (*cusparseLoggerCallback_t 'cusparseLoggerCallback_t')(
    int logLevel,
    const char* functionName,
    const char* message
)


###############################################################################
# Functions
###############################################################################

cdef cusparseStatus_t cusparseCreate(cusparseHandle_t* handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseGetProperty(libraryPropertyType type, int* value) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef const char* cusparseGetErrorName(cusparseStatus_t status) except?NULL nogil
cdef const char* cusparseGetErrorString(cusparseStatus_t status) except?NULL nogil
cdef cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t* mode) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* descrA) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA) except?_CUSPARSEMATRIXTYPE_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseFillMode_t cusparseGetMatFillMode(const cusparseMatDescr_t descrA) except?_CUSPARSEFILLMODE_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseDiagType_t cusparseGetMatDiagType(const cusparseMatDescr_t descrA) except?_CUSPARSEDIAGTYPE_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) except?_CUSPARSEINDEXBASE_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const float* alpha, const float* A, int lda, int nnz, const float* xVal, const int* xInd, const float* beta, float* y, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const double* alpha, const double* A, int lda, int nnz, const double* xVal, const int* xInd, const double* beta, double* y, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, int nnz, const cuComplex* xVal, const int* xInd, const cuComplex* beta, cuComplex* y, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, int nnz, const cuDoubleComplex* xVal, const int* xInd, const cuDoubleComplex* beta, cuDoubleComplex* y, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const float* x, const float* beta, float* y) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const double* x, const double* beta, double* y) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cuComplex* x, const cuComplex* beta, cuComplex* y) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const float* B, const int ldb, const float* beta, float* C, int ldc) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const double* B, const int ldb, const double* beta, double* C, int ldc) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const cuComplex* B, const int ldb, const cuComplex* beta, cuComplex* C, int ldc) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const cuDoubleComplex* B, const int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgtsv2(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgtsv2(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgtsv2(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgtsv2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, int batchStride, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, int batchStride, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* x, int batchCount, int batchStride, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* x, int batchCount, int batchStride, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgtsv2StridedBatch(cusparseHandle_t handle, int m, const float* dl, const float* d, const float* du, float* x, int batchCount, int batchStride, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgtsv2StridedBatch(cusparseHandle_t handle, int m, const double* dl, const double* d, const double* du, double* x, int batchCount, int batchStride, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* x, int batchCount, int batchStride, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* x, int batchCount, int batchStride, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float* dl, float* d, float* du, float* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double* dl, double* d, double* du, double* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuDoubleComplex* dl, cuDoubleComplex* d, cuDoubleComplex* du, cuDoubleComplex* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const float* ds, const float* dl, const float* d, const float* du, const float* dw, const float* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const double* ds, const double* dl, const double* d, const double* du, const double* dw, const double* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuComplex* ds, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* dw, const cuComplex* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuDoubleComplex* ds, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* dw, const cuDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float* ds, float* dl, float* d, float* du, float* dw, float* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double* ds, double* dl, double* d, double* du, double* dw, double* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex* ds, cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* dw, cuComplex* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuDoubleComplex* ds, cuDoubleComplex* dl, cuDoubleComplex* d, cuDoubleComplex* du, cuDoubleComplex* dw, cuDoubleComplex* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const cuComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseScsrgeam2(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDcsrgeam2(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCcsrgeam2(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZcsrgeam2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz, int m, int* csrSortedRowPtr, cusparseIndexBase_t idxBase) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcsr2coo(cusparseHandle_t handle, const int* csrSortedRowPtr, int nnz, int m, int* cooRowInd, cusparseIndexBase_t idxBase) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, float* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, double* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, cuComplex* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, cuDoubleComplex* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseScsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseScsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDcsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCcsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZcsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, int* bsrSortedRowPtrC, int rowBlockDim, int colBlockDim, int* nnzTotalDevHostPtr, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseScsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, cuComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXgebsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, int* bsrSortedRowPtrC, int rowBlockDimC, int colBlockDimC, int* nnzTotalDevHostPtr, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, cuComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseZgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* cooRowsA, const int* cooColsA, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA, int* P, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcoosortByColumn(cusparseHandle_t handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA, int* P, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* csrRowPtrA, const int* csrColIndA, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const int* csrRowPtrA, int* csrColIndA, int* P, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* cscColPtrA, const int* cscRowIndA, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseXcscsort(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const int* cscColPtrA, int* cscRowIndA, int* P, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCsr2cscEx2(cusparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, cudaDataType valType, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, void* buffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCsr2cscEx2_bufferSize(cusparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, cudaDataType valType, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices, void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDestroySpVec(cusparseConstSpVecDescr_t spVecDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpVecGetIndexBase(cusparseConstSpVecDescr_t spVecDescr, cusparseIndexBase_t* idxBase) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr, void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void* values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr, int64_t* size, void** values, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr, void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void* values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMatGetFormat(cusparseConstSpMatDescr_t spMatDescr, cusparseFormat_t* format) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMatGetIndexBase(cusparseConstSpMatDescr_t spMatDescr, cusparseIndexBase_t* idxBase) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr, void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void* values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMatGetSize(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMatGetStridedBatch(cusparseConstSpMatDescr_t spMatDescr, int* batchCount) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride, int64_t columnsValuesBatchStride) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets, void* csrColInd, void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCsrGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** csrRowOffsets, void** csrColInd, void** csrValues, cusparseIndexType_t* csrRowOffsetsType, cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd, void* csrValues) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooRowInd, void* cooColInd, void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd, void** cooColInd, void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, cudaDataType* type, cusparseOrder_t* order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr, void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void* values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDnMatGetStridedBatch(cusparseConstDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseAxpby(cusparseHandle_t handle, const void* alpha, cusparseConstSpVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseGather(cusparseHandle_t handle, cusparseConstDnVecDescr_t vecY, cusparseSpVecDescr_t vecX) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseScatter(cusparseHandle_t handle, cusparseConstSpVecDescr_t vecX, cusparseDnVecDescr_t vecY) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX, cusparseConstSpVecDescr_t vecX, cusparseConstDnVecDescr_t vecY, const void* result, cudaDataType computeType, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX, cusparseConstSpVecDescr_t vecX, cusparseConstDnVecDescr_t vecY, void* result, cudaDataType computeType, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMM_workEstimation(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMM_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMM_copy(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cscColOffsets, void* cscRowInd, void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCscSetPointers(cusparseSpMatDescr_t spMatDescr, void* cscColOffsets, void* cscRowInd, void* cscValues) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCooSetPointers(cusparseSpMatDescr_t spMatDescr, void* cooRows, void* cooColumns, void* cooValues) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t handle, cusparseConstSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSparseToDense(cusparseHandle_t handle, cusparseConstSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseDenseToSparse_convert(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateBlockedEll(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseBlockedEllGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, void** ellColInd, void** ellValue, cusparseIndexType_t* ellIdxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSDDMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSDDMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMatGetAttribute(cusparseConstSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t* descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMMreuse_workEstimation(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMMreuse_nnz(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2, size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize4, void* externalBuffer4) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMMreuse_copy(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize5, void* externalBuffer5) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMMreuse_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLoggerSetCallback(cusparseLoggerCallback_t callback) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLoggerSetFile(FILE* file) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLoggerOpenFile(const char* logFile) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLoggerSetLevel(int level) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLoggerSetMask(int mask) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseLoggerForceDisable() except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMMOp_createPlan(cusparseHandle_t handle, cusparseSpMMOpPlan_t* plan, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMOpAlg_t alg, const void* addOperationNvvmBuffer, size_t addOperationBufferSize, const void* mulOperationNvvmBuffer, size_t mulOperationBufferSize, const void* epilogueNvvmBuffer, size_t epilogueBufferSize, size_t* SpMMWorkspaceSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMMOp(cusparseSpMMOpPlan_t plan, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMMOp_destroyPlan(cusparseSpMMOpPlan_t plan) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCscGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cscColOffsets, void** cscRowInd, void** cscValues, cusparseIndexType_t* cscColOffsetsType, cusparseIndexType_t* cscRowIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateConstSpVec(cusparseConstSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, const void* indices, const void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstSpVecGet(cusparseConstSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, const void** indices, const void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstSpVecGetValues(cusparseConstSpVecDescr_t spVecDescr, const void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateConstDnVec(cusparseConstDnVecDescr_t* dnVecDescr, int64_t size, const void* values, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstDnVecGet(cusparseConstDnVecDescr_t dnVecDescr, int64_t* size, const void** values, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstDnVecGetValues(cusparseConstDnVecDescr_t dnVecDescr, const void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstSpMatGetValues(cusparseConstSpMatDescr_t spMatDescr, const void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateConstCsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* csrRowOffsets, const void* csrColInd, const void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateConstCsc(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cscColOffsets, const void* cscRowInd, const void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstCsrGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** csrRowOffsets, const void** csrColInd, const void** csrValues, cusparseIndexType_t* csrRowOffsetsType, cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstCscGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** cscColOffsets, const void** cscRowInd, const void** cscValues, cusparseIndexType_t* cscColOffsetsType, cusparseIndexType_t* cscRowIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateConstCoo(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cooRowInd, const void* cooColInd, const void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstCooGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** cooRowInd, const void** cooColInd, const void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateConstBlockedEll(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, const void* ellColInd, const void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstBlockedEllGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, const void** ellColInd, const void** ellValue, cusparseIndexType_t* ellIdxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateConstDnMat(cusparseConstDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, const void* values, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstDnMatGet(cusparseConstDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, const void** values, cudaDataType* type, cusparseOrder_t* order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseConstDnMatGetValues(cusparseConstDnMatDescr_t dnMatDescr, const void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMM_getNumProducts(cusparseSpGEMMDescr_t spgemmDescr, int64_t* num_prods) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpGEMM_estimateMemory(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, float chunk_fraction, size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize2) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseBsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride, int64_t columnsBatchStride, int64_t ValuesBatchStride) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateBsr(cusparseSpMatDescr_t* spMatDescr, int64_t brows, int64_t bcols, int64_t bnnz, int64_t rowBlockSize, int64_t colBlockSize, void* bsrRowOffsets, void* bsrColInd, void* bsrValues, cusparseIndexType_t bsrRowOffsetsType, cusparseIndexType_t bsrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateConstBsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t brows, int64_t bcols, int64_t bnnz, int64_t rowBlockDim, int64_t colBlockDim, const void* bsrRowOffsets, const void* bsrColInd, const void* bsrValues, cusparseIndexType_t bsrRowOffsetsType, cusparseIndexType_t bsrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateSlicedEll(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, int64_t sellValuesSize, int64_t sliceSize, void* sellSliceOffsets, void* sellColInd, void* sellValues, cusparseIndexType_t sellSliceOffsetsType, cusparseIndexType_t sellColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseCreateConstSlicedEll(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, int64_t sellValuesSize, int64_t sliceSize, const void* sellSliceOffsets, const void* sellColInd, const void* sellValues, cusparseIndexType_t sellSliceOffsetsType, cusparseIndexType_t sellColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSV_updateMatrix(cusparseHandle_t handle, cusparseSpSVDescr_t spsvDescr, void* newValues, cusparseSpSVUpdate_t updatePart) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpMV_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cusparseStatus_t cusparseSpSM_updateMatrix(cusparseHandle_t handle, cusparseSpSMDescr_t spsmDescr, void* newValues, cusparseSpSMUpdate_t updatePart) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil
