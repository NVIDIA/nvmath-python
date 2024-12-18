# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

cimport cython  # NOQA

from enum import IntEnum as _IntEnum

import numpy as _numpy


###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """See `cusparseStatus_t`."""
    SUCCESS = CUSPARSE_STATUS_SUCCESS
    NOT_INITIALIZED = CUSPARSE_STATUS_NOT_INITIALIZED
    ALLOC_FAILED = CUSPARSE_STATUS_ALLOC_FAILED
    INVALID_VALUE = CUSPARSE_STATUS_INVALID_VALUE
    ARCH_MISMATCH = CUSPARSE_STATUS_ARCH_MISMATCH
    MAPPING_ERROR = CUSPARSE_STATUS_MAPPING_ERROR
    EXECUTION_FAILED = CUSPARSE_STATUS_EXECUTION_FAILED
    INTERNAL_ERROR = CUSPARSE_STATUS_INTERNAL_ERROR
    MATRIX_TYPE_NOT_SUPPORTED = CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
    ZERO_PIVOT = CUSPARSE_STATUS_ZERO_PIVOT
    NOT_SUPPORTED = CUSPARSE_STATUS_NOT_SUPPORTED
    INSUFFICIENT_RESOURCES = CUSPARSE_STATUS_INSUFFICIENT_RESOURCES

class PointerMode(_IntEnum):
    """See `cusparsePointerMode_t`."""
    HOST = CUSPARSE_POINTER_MODE_HOST
    DEVICE = CUSPARSE_POINTER_MODE_DEVICE

class Action(_IntEnum):
    """See `cusparseAction_t`."""
    SYMBOLIC = CUSPARSE_ACTION_SYMBOLIC
    NUMERIC = CUSPARSE_ACTION_NUMERIC

class MatrixType(_IntEnum):
    """See `cusparseMatrixType_t`."""
    GENERAL = CUSPARSE_MATRIX_TYPE_GENERAL
    SYMMETRIC = CUSPARSE_MATRIX_TYPE_SYMMETRIC
    HERMITIAN = CUSPARSE_MATRIX_TYPE_HERMITIAN
    TRIANGULAR = CUSPARSE_MATRIX_TYPE_TRIANGULAR

class FillMode(_IntEnum):
    """See `cusparseFillMode_t`."""
    LOWER = CUSPARSE_FILL_MODE_LOWER
    UPPER = CUSPARSE_FILL_MODE_UPPER

class DiagType(_IntEnum):
    """See `cusparseDiagType_t`."""
    NON_UNIT = CUSPARSE_DIAG_TYPE_NON_UNIT
    UNIT = CUSPARSE_DIAG_TYPE_UNIT

class IndexBase(_IntEnum):
    """See `cusparseIndexBase_t`."""
    ZERO = CUSPARSE_INDEX_BASE_ZERO
    ONE = CUSPARSE_INDEX_BASE_ONE

class Operation(_IntEnum):
    """See `cusparseOperation_t`."""
    NON_TRANSPOSE = CUSPARSE_OPERATION_NON_TRANSPOSE
    TRANSPOSE = CUSPARSE_OPERATION_TRANSPOSE
    CONJUGATE_TRANSPOSE = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE

class Direction(_IntEnum):
    """See `cusparseDirection_t`."""
    ROW = CUSPARSE_DIRECTION_ROW
    COLUMN = CUSPARSE_DIRECTION_COLUMN

class SolvePolicy(_IntEnum):
    """See `cusparseSolvePolicy_t`."""
    NO_LEVEL = CUSPARSE_SOLVE_POLICY_NO_LEVEL
    USE_LEVEL = CUSPARSE_SOLVE_POLICY_USE_LEVEL

class ColorAlg(_IntEnum):
    """See `cusparseColorAlg_t`."""
    COLOR_ALG0 = CUSPARSE_COLOR_ALG0
    COLOR_ALG1 = CUSPARSE_COLOR_ALG1

class Csr2CscAlg(_IntEnum):
    """See `cusparseCsr2CscAlg_t`."""
    DEFAULT = CUSPARSE_CSR2CSC_ALG_DEFAULT
    ALG1 = CUSPARSE_CSR2CSC_ALG1
    ALG2 = CUSPARSE_CSR2CSC_ALG2

class Format(_IntEnum):
    """See `cusparseFormat_t`."""
    CSR = CUSPARSE_FORMAT_CSR
    CSC = CUSPARSE_FORMAT_CSC
    COO = CUSPARSE_FORMAT_COO
    BLOCKED_ELL = CUSPARSE_FORMAT_BLOCKED_ELL
    BSR = CUSPARSE_FORMAT_BSR
    SLICED_ELLPACK = CUSPARSE_FORMAT_SLICED_ELLPACK
    COO_AOS = CUSPARSE_FORMAT_COO_AOS

class Order(_IntEnum):
    """See `cusparseOrder_t`."""
    COL = CUSPARSE_ORDER_COL
    ROW = CUSPARSE_ORDER_ROW

class IndexType(_IntEnum):
    """See `cusparseIndexType_t`."""
    INDEX_16U = CUSPARSE_INDEX_16U
    INDEX_32I = CUSPARSE_INDEX_32I
    INDEX_64I = CUSPARSE_INDEX_64I

class SpMVAlg(_IntEnum):
    """See `cusparseSpMVAlg_t`."""
    DEFAULT = CUSPARSE_SPMV_ALG_DEFAULT
    CSR_ALG1 = CUSPARSE_SPMV_CSR_ALG1
    CSR_ALG2 = CUSPARSE_SPMV_CSR_ALG2
    COO_ALG1 = CUSPARSE_SPMV_COO_ALG1
    COO_ALG2 = CUSPARSE_SPMV_COO_ALG2
    SELL_ALG1 = CUSPARSE_SPMV_SELL_ALG1
    MV_ALG_DEFAULT = CUSPARSE_MV_ALG_DEFAULT
    COOMV_ALG = CUSPARSE_COOMV_ALG
    CSRMV_ALG1 = CUSPARSE_CSRMV_ALG1
    CSRMV_ALG2 = CUSPARSE_CSRMV_ALG2

class SpMMAlg(_IntEnum):
    """See `cusparseSpMMAlg_t`."""
    DEFAULT = CUSPARSE_SPMM_ALG_DEFAULT
    COO_ALG1 = CUSPARSE_SPMM_COO_ALG1
    COO_ALG2 = CUSPARSE_SPMM_COO_ALG2
    COO_ALG3 = CUSPARSE_SPMM_COO_ALG3
    COO_ALG4 = CUSPARSE_SPMM_COO_ALG4
    CSR_ALG1 = CUSPARSE_SPMM_CSR_ALG1
    CSR_ALG2 = CUSPARSE_SPMM_CSR_ALG2
    CSR_ALG3 = CUSPARSE_SPMM_CSR_ALG3
    BLOCKED_ELL_ALG1 = CUSPARSE_SPMM_BLOCKED_ELL_ALG1
    BSR_ALG1 = CUSPARSE_SPMM_BSR_ALG1
    MM_ALG_DEFAULT = CUSPARSE_MM_ALG_DEFAULT
    COOMM_ALG1 = CUSPARSE_COOMM_ALG1
    COOMM_ALG2 = CUSPARSE_COOMM_ALG2
    COOMM_ALG3 = CUSPARSE_COOMM_ALG3
    CSRMM_ALG1 = CUSPARSE_CSRMM_ALG1
    SPMMA_PREPROCESS = CUSPARSE_SPMMA_PREPROCESS
    SPMMA_ALG1 = CUSPARSE_SPMMA_ALG1
    SPMMA_ALG2 = CUSPARSE_SPMMA_ALG2
    SPMMA_ALG3 = CUSPARSE_SPMMA_ALG3
    SPMMA_ALG4 = CUSPARSE_SPMMA_ALG4

class SpGEMMAlg(_IntEnum):
    """See `cusparseSpGEMMAlg_t`."""
    DEFAULT = CUSPARSE_SPGEMM_DEFAULT
    CSR_ALG_DETERMINITIC = CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC
    CSR_ALG_NONDETERMINITIC = CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC
    ALG1 = CUSPARSE_SPGEMM_ALG1
    ALG2 = CUSPARSE_SPGEMM_ALG2
    ALG3 = CUSPARSE_SPGEMM_ALG3

class SparseToDenseAlg(_IntEnum):
    """See `cusparseSparseToDenseAlg_t`."""
    DEFAULT = CUSPARSE_SPARSETODENSE_ALG_DEFAULT

class DenseToSparseAlg(_IntEnum):
    """See `cusparseDenseToSparseAlg_t`."""
    DEFAULT = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT

class SDDMMAlg(_IntEnum):
    """See `cusparseSDDMMAlg_t`."""
    DEFAULT = CUSPARSE_SDDMM_ALG_DEFAULT

class SpMatAttribute(_IntEnum):
    """See `cusparseSpMatAttribute_t`."""
    FILL_MODE = CUSPARSE_SPMAT_FILL_MODE
    DIAG_TYPE = CUSPARSE_SPMAT_DIAG_TYPE

class SpSVAlg(_IntEnum):
    """See `cusparseSpSVAlg_t`."""
    DEFAULT = CUSPARSE_SPSV_ALG_DEFAULT

class SpSMAlg(_IntEnum):
    """See `cusparseSpSMAlg_t`."""
    DEFAULT = CUSPARSE_SPSM_ALG_DEFAULT

class SpMMOpAlg(_IntEnum):
    """See `cusparseSpMMOpAlg_t`."""
    DEFAULT = CUSPARSE_SPMM_OP_ALG_DEFAULT

class SpSVUpdate(_IntEnum):
    """See `cusparseSpSVUpdate_t`."""
    GENERAL = CUSPARSE_SPSV_UPDATE_GENERAL
    DIAGONAL = CUSPARSE_SPSV_UPDATE_DIAGONAL

class SpSMUpdate(_IntEnum):
    """See `cusparseSpSMUpdate_t`."""
    UPDATE_GENERAL = CUSPARSE_SPSM_UPDATE_GENERAL
    UPDATE_DIAGONAL = CUSPARSE_SPSM_UPDATE_DIAGONAL


###############################################################################
# Error handling
###############################################################################

cdef class cuSPARSEError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value})"
        super(cuSPARSEError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuSPARSEError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create() except? 0:
    """See `cusparseCreate`."""
    cdef Handle handle
    with nogil:
        status = cusparseCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """See `cusparseDestroy`."""
    with nogil:
        status = cusparseDestroy(<Handle>handle)
    check_status(status)


cpdef int get_version(intptr_t handle) except? -1:
    """See `cusparseGetVersion`."""
    cdef int version
    with nogil:
        status = cusparseGetVersion(<Handle>handle, &version)
    check_status(status)
    return version


cpdef int get_property(int type) except? -1:
    """See `cusparseGetProperty`."""
    cdef int value
    with nogil:
        status = cusparseGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


cpdef str get_error_name(int status):
    """See `cusparseGetErrorName`."""
    cdef bytes _output_
    _output_ = cusparseGetErrorName(<_Status>status)
    return _output_.decode()


cpdef str get_error_string(int status):
    """See `cusparseGetErrorString`."""
    cdef bytes _output_
    _output_ = cusparseGetErrorString(<_Status>status)
    return _output_.decode()


cpdef set_stream(intptr_t handle, intptr_t stream_id):
    """See `cusparseSetStream`."""
    with nogil:
        status = cusparseSetStream(<Handle>handle, <Stream>stream_id)
    check_status(status)


cpdef intptr_t get_stream(intptr_t handle) except? 0:
    """See `cusparseGetStream`."""
    cdef Stream stream_id
    with nogil:
        status = cusparseGetStream(<Handle>handle, &stream_id)
    check_status(status)
    return <intptr_t>stream_id


cpdef int get_pointer_mode(intptr_t handle) except? -1:
    """See `cusparseGetPointerMode`."""
    cdef _PointerMode mode
    with nogil:
        status = cusparseGetPointerMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


cpdef set_pointer_mode(intptr_t handle, int mode):
    """See `cusparseSetPointerMode`."""
    with nogil:
        status = cusparseSetPointerMode(<Handle>handle, <_PointerMode>mode)
    check_status(status)


cpdef intptr_t create_mat_descr() except? 0:
    """See `cusparseCreateMatDescr`."""
    cdef MatDescr descr_a
    with nogil:
        status = cusparseCreateMatDescr(&descr_a)
    check_status(status)
    return <intptr_t>descr_a


cpdef destroy_mat_descr(intptr_t descr_a):
    """See `cusparseDestroyMatDescr`."""
    with nogil:
        status = cusparseDestroyMatDescr(<MatDescr>descr_a)
    check_status(status)


cpdef set_mat_type(intptr_t descr_a, int type):
    """See `cusparseSetMatType`."""
    with nogil:
        status = cusparseSetMatType(<MatDescr>descr_a, <_MatrixType>type)
    check_status(status)


cpdef int get_mat_type(intptr_t descr_a) except? -1:
    """See `cusparseGetMatType`."""
    return <int>cusparseGetMatType(<const MatDescr>descr_a)


cpdef set_mat_fill_mode(intptr_t descr_a, int fill_mode):
    """See `cusparseSetMatFillMode`."""
    with nogil:
        status = cusparseSetMatFillMode(<MatDescr>descr_a, <_FillMode>fill_mode)
    check_status(status)


cpdef int get_mat_fill_mode(intptr_t descr_a) except? -1:
    """See `cusparseGetMatFillMode`."""
    return <int>cusparseGetMatFillMode(<const MatDescr>descr_a)


cpdef set_mat_diag_type(intptr_t descr_a, int diag_type):
    """See `cusparseSetMatDiagType`."""
    with nogil:
        status = cusparseSetMatDiagType(<MatDescr>descr_a, <_DiagType>diag_type)
    check_status(status)


cpdef int get_mat_diag_type(intptr_t descr_a) except? -1:
    """See `cusparseGetMatDiagType`."""
    return <int>cusparseGetMatDiagType(<const MatDescr>descr_a)


cpdef set_mat_index_base(intptr_t descr_a, int base):
    """See `cusparseSetMatIndexBase`."""
    with nogil:
        status = cusparseSetMatIndexBase(<MatDescr>descr_a, <_IndexBase>base)
    check_status(status)


cpdef int get_mat_index_base(intptr_t descr_a) except? -1:
    """See `cusparseGetMatIndexBase`."""
    return <int>cusparseGetMatIndexBase(<const MatDescr>descr_a)


cpdef sgemvi(intptr_t handle, int trans_a, int m, int n, intptr_t alpha, intptr_t a, int lda, int nnz, intptr_t x_val, intptr_t x_ind, intptr_t beta, intptr_t y, int idx_base, intptr_t p_buffer):
    """See `cusparseSgemvi`."""
    with nogil:
        status = cusparseSgemvi(<Handle>handle, <_Operation>trans_a, m, n, <const float*>alpha, <const float*>a, lda, nnz, <const float*>x_val, <const int*>x_ind, <const float*>beta, <float*>y, <_IndexBase>idx_base, <void*>p_buffer)
    check_status(status)


cpdef int sgemvi_buffer_size(intptr_t handle, int trans_a, int m, int n, int nnz) except? -1:
    """See `cusparseSgemvi_bufferSize`."""
    cdef int p_buffer_size
    with nogil:
        status = cusparseSgemvi_bufferSize(<Handle>handle, <_Operation>trans_a, m, n, nnz, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef dgemvi(intptr_t handle, int trans_a, int m, int n, intptr_t alpha, intptr_t a, int lda, int nnz, intptr_t x_val, intptr_t x_ind, intptr_t beta, intptr_t y, int idx_base, intptr_t p_buffer):
    """See `cusparseDgemvi`."""
    with nogil:
        status = cusparseDgemvi(<Handle>handle, <_Operation>trans_a, m, n, <const double*>alpha, <const double*>a, lda, nnz, <const double*>x_val, <const int*>x_ind, <const double*>beta, <double*>y, <_IndexBase>idx_base, <void*>p_buffer)
    check_status(status)


cpdef int dgemvi_buffer_size(intptr_t handle, int trans_a, int m, int n, int nnz) except? -1:
    """See `cusparseDgemvi_bufferSize`."""
    cdef int p_buffer_size
    with nogil:
        status = cusparseDgemvi_bufferSize(<Handle>handle, <_Operation>trans_a, m, n, nnz, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef cgemvi(intptr_t handle, int trans_a, int m, int n, intptr_t alpha, intptr_t a, int lda, int nnz, intptr_t x_val, intptr_t x_ind, intptr_t beta, intptr_t y, int idx_base, intptr_t p_buffer):
    """See `cusparseCgemvi`."""
    with nogil:
        status = cusparseCgemvi(<Handle>handle, <_Operation>trans_a, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, nnz, <const cuComplex*>x_val, <const int*>x_ind, <const cuComplex*>beta, <cuComplex*>y, <_IndexBase>idx_base, <void*>p_buffer)
    check_status(status)


cpdef int cgemvi_buffer_size(intptr_t handle, int trans_a, int m, int n, int nnz) except? -1:
    """See `cusparseCgemvi_bufferSize`."""
    cdef int p_buffer_size
    with nogil:
        status = cusparseCgemvi_bufferSize(<Handle>handle, <_Operation>trans_a, m, n, nnz, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef zgemvi(intptr_t handle, int trans_a, int m, int n, intptr_t alpha, intptr_t a, int lda, int nnz, intptr_t x_val, intptr_t x_ind, intptr_t beta, intptr_t y, int idx_base, intptr_t p_buffer):
    """See `cusparseZgemvi`."""
    with nogil:
        status = cusparseZgemvi(<Handle>handle, <_Operation>trans_a, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, nnz, <const cuDoubleComplex*>x_val, <const int*>x_ind, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, <_IndexBase>idx_base, <void*>p_buffer)
    check_status(status)


cpdef int zgemvi_buffer_size(intptr_t handle, int trans_a, int m, int n, int nnz) except? -1:
    """See `cusparseZgemvi_bufferSize`."""
    cdef int p_buffer_size
    with nogil:
        status = cusparseZgemvi_bufferSize(<Handle>handle, <_Operation>trans_a, m, n, nnz, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef sbsrmv(intptr_t handle, int dir_a, int trans_a, int mb, int nb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t x, intptr_t beta, intptr_t y):
    """See `cusparseSbsrmv`."""
    with nogil:
        status = cusparseSbsrmv(<Handle>handle, <_Direction>dir_a, <_Operation>trans_a, mb, nb, nnzb, <const float*>alpha, <const MatDescr>descr_a, <const float*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, block_dim, <const float*>x, <const float*>beta, <float*>y)
    check_status(status)


cpdef dbsrmv(intptr_t handle, int dir_a, int trans_a, int mb, int nb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t x, intptr_t beta, intptr_t y):
    """See `cusparseDbsrmv`."""
    with nogil:
        status = cusparseDbsrmv(<Handle>handle, <_Direction>dir_a, <_Operation>trans_a, mb, nb, nnzb, <const double*>alpha, <const MatDescr>descr_a, <const double*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, block_dim, <const double*>x, <const double*>beta, <double*>y)
    check_status(status)


cpdef cbsrmv(intptr_t handle, int dir_a, int trans_a, int mb, int nb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t x, intptr_t beta, intptr_t y):
    """See `cusparseCbsrmv`."""
    with nogil:
        status = cusparseCbsrmv(<Handle>handle, <_Direction>dir_a, <_Operation>trans_a, mb, nb, nnzb, <const cuComplex*>alpha, <const MatDescr>descr_a, <const cuComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, block_dim, <const cuComplex*>x, <const cuComplex*>beta, <cuComplex*>y)
    check_status(status)


cpdef zbsrmv(intptr_t handle, int dir_a, int trans_a, int mb, int nb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t x, intptr_t beta, intptr_t y):
    """See `cusparseZbsrmv`."""
    with nogil:
        status = cusparseZbsrmv(<Handle>handle, <_Direction>dir_a, <_Operation>trans_a, mb, nb, nnzb, <const cuDoubleComplex*>alpha, <const MatDescr>descr_a, <const cuDoubleComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, block_dim, <const cuDoubleComplex*>x, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y)
    check_status(status)


cpdef sbsrmm(intptr_t handle, int dir_a, int trans_a, int trans_b, int mb, int n, int kb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_size, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cusparseSbsrmm`."""
    with nogil:
        status = cusparseSbsrmm(<Handle>handle, <_Direction>dir_a, <_Operation>trans_a, <_Operation>trans_b, mb, n, kb, nnzb, <const float*>alpha, <const MatDescr>descr_a, <const float*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, <const int>block_size, <const float*>b, <const int>ldb, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dbsrmm(intptr_t handle, int dir_a, int trans_a, int trans_b, int mb, int n, int kb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_size, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cusparseDbsrmm`."""
    with nogil:
        status = cusparseDbsrmm(<Handle>handle, <_Direction>dir_a, <_Operation>trans_a, <_Operation>trans_b, mb, n, kb, nnzb, <const double*>alpha, <const MatDescr>descr_a, <const double*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, <const int>block_size, <const double*>b, <const int>ldb, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef cbsrmm(intptr_t handle, int dir_a, int trans_a, int trans_b, int mb, int n, int kb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_size, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cusparseCbsrmm`."""
    with nogil:
        status = cusparseCbsrmm(<Handle>handle, <_Direction>dir_a, <_Operation>trans_a, <_Operation>trans_b, mb, n, kb, nnzb, <const cuComplex*>alpha, <const MatDescr>descr_a, <const cuComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, <const int>block_size, <const cuComplex*>b, <const int>ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zbsrmm(intptr_t handle, int dir_a, int trans_a, int trans_b, int mb, int n, int kb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_size, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cusparseZbsrmm`."""
    with nogil:
        status = cusparseZbsrmm(<Handle>handle, <_Direction>dir_a, <_Operation>trans_a, <_Operation>trans_b, mb, n, kb, nnzb, <const cuDoubleComplex*>alpha, <const MatDescr>descr_a, <const cuDoubleComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, <const int>block_size, <const cuDoubleComplex*>b, <const int>ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef size_t sgtsv2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0:
    """See `cusparseSgtsv2_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseSgtsv2_bufferSizeExt(<Handle>handle, m, n, <const float*>dl, <const float*>d, <const float*>du, <const float*>b, ldb, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef size_t dgtsv2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0:
    """See `cusparseDgtsv2_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseDgtsv2_bufferSizeExt(<Handle>handle, m, n, <const double*>dl, <const double*>d, <const double*>du, <const double*>b, ldb, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef size_t cgtsv2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0:
    """See `cusparseCgtsv2_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseCgtsv2_bufferSizeExt(<Handle>handle, m, n, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>b, ldb, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef size_t zgtsv2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0:
    """See `cusparseZgtsv2_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseZgtsv2_bufferSizeExt(<Handle>handle, m, n, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <const cuDoubleComplex*>b, ldb, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef sgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer):
    """See `cusparseSgtsv2`."""
    with nogil:
        status = cusparseSgtsv2(<Handle>handle, m, n, <const float*>dl, <const float*>d, <const float*>du, <float*>b, ldb, <void*>p_buffer)
    check_status(status)


cpdef dgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer):
    """See `cusparseDgtsv2`."""
    with nogil:
        status = cusparseDgtsv2(<Handle>handle, m, n, <const double*>dl, <const double*>d, <const double*>du, <double*>b, ldb, <void*>p_buffer)
    check_status(status)


cpdef cgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer):
    """See `cusparseCgtsv2`."""
    with nogil:
        status = cusparseCgtsv2(<Handle>handle, m, n, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <cuComplex*>b, ldb, <void*>p_buffer)
    check_status(status)


cpdef zgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer):
    """See `cusparseZgtsv2`."""
    with nogil:
        status = cusparseZgtsv2(<Handle>handle, m, n, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <cuDoubleComplex*>b, ldb, <void*>p_buffer)
    check_status(status)


cpdef size_t sgtsv2_nopivot_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0:
    """See `cusparseSgtsv2_nopivot_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseSgtsv2_nopivot_bufferSizeExt(<Handle>handle, m, n, <const float*>dl, <const float*>d, <const float*>du, <const float*>b, ldb, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef size_t dgtsv2_nopivot_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0:
    """See `cusparseDgtsv2_nopivot_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseDgtsv2_nopivot_bufferSizeExt(<Handle>handle, m, n, <const double*>dl, <const double*>d, <const double*>du, <const double*>b, ldb, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef size_t cgtsv2_nopivot_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0:
    """See `cusparseCgtsv2_nopivot_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseCgtsv2_nopivot_bufferSizeExt(<Handle>handle, m, n, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>b, ldb, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef size_t zgtsv2_nopivot_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0:
    """See `cusparseZgtsv2_nopivot_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseZgtsv2_nopivot_bufferSizeExt(<Handle>handle, m, n, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <const cuDoubleComplex*>b, ldb, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef sgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer):
    """See `cusparseSgtsv2_nopivot`."""
    with nogil:
        status = cusparseSgtsv2_nopivot(<Handle>handle, m, n, <const float*>dl, <const float*>d, <const float*>du, <float*>b, ldb, <void*>p_buffer)
    check_status(status)


cpdef dgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer):
    """See `cusparseDgtsv2_nopivot`."""
    with nogil:
        status = cusparseDgtsv2_nopivot(<Handle>handle, m, n, <const double*>dl, <const double*>d, <const double*>du, <double*>b, ldb, <void*>p_buffer)
    check_status(status)


cpdef cgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer):
    """See `cusparseCgtsv2_nopivot`."""
    with nogil:
        status = cusparseCgtsv2_nopivot(<Handle>handle, m, n, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <cuComplex*>b, ldb, <void*>p_buffer)
    check_status(status)


cpdef zgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer):
    """See `cusparseZgtsv2_nopivot`."""
    with nogil:
        status = cusparseZgtsv2_nopivot(<Handle>handle, m, n, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <cuDoubleComplex*>b, ldb, <void*>p_buffer)
    check_status(status)


cpdef size_t sgtsv2strided_batch_buffer_size_ext(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride) except? 0:
    """See `cusparseSgtsv2StridedBatch_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseSgtsv2StridedBatch_bufferSizeExt(<Handle>handle, m, <const float*>dl, <const float*>d, <const float*>du, <const float*>x, batch_count, batch_stride, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef size_t dgtsv2strided_batch_buffer_size_ext(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride) except? 0:
    """See `cusparseDgtsv2StridedBatch_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseDgtsv2StridedBatch_bufferSizeExt(<Handle>handle, m, <const double*>dl, <const double*>d, <const double*>du, <const double*>x, batch_count, batch_stride, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef size_t cgtsv2strided_batch_buffer_size_ext(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride) except? 0:
    """See `cusparseCgtsv2StridedBatch_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseCgtsv2StridedBatch_bufferSizeExt(<Handle>handle, m, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>x, batch_count, batch_stride, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef size_t zgtsv2strided_batch_buffer_size_ext(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride) except? 0:
    """See `cusparseZgtsv2StridedBatch_bufferSizeExt`."""
    cdef size_t buffer_size_in_bytes
    with nogil:
        status = cusparseZgtsv2StridedBatch_bufferSizeExt(<Handle>handle, m, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <const cuDoubleComplex*>x, batch_count, batch_stride, &buffer_size_in_bytes)
    check_status(status)
    return buffer_size_in_bytes


cpdef sgtsv2strided_batch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride, intptr_t p_buffer):
    """See `cusparseSgtsv2StridedBatch`."""
    with nogil:
        status = cusparseSgtsv2StridedBatch(<Handle>handle, m, <const float*>dl, <const float*>d, <const float*>du, <float*>x, batch_count, batch_stride, <void*>p_buffer)
    check_status(status)


cpdef dgtsv2strided_batch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride, intptr_t p_buffer):
    """See `cusparseDgtsv2StridedBatch`."""
    with nogil:
        status = cusparseDgtsv2StridedBatch(<Handle>handle, m, <const double*>dl, <const double*>d, <const double*>du, <double*>x, batch_count, batch_stride, <void*>p_buffer)
    check_status(status)


cpdef cgtsv2strided_batch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride, intptr_t p_buffer):
    """See `cusparseCgtsv2StridedBatch`."""
    with nogil:
        status = cusparseCgtsv2StridedBatch(<Handle>handle, m, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <cuComplex*>x, batch_count, batch_stride, <void*>p_buffer)
    check_status(status)


cpdef zgtsv2strided_batch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride, intptr_t p_buffer):
    """See `cusparseZgtsv2StridedBatch`."""
    with nogil:
        status = cusparseZgtsv2StridedBatch(<Handle>handle, m, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <cuDoubleComplex*>x, batch_count, batch_stride, <void*>p_buffer)
    check_status(status)


cpdef size_t sgtsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count) except? 0:
    """See `cusparseSgtsvInterleavedBatch_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseSgtsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const float*>dl, <const float*>d, <const float*>du, <const float*>x, batch_count, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t dgtsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count) except? 0:
    """See `cusparseDgtsvInterleavedBatch_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseDgtsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const double*>dl, <const double*>d, <const double*>du, <const double*>x, batch_count, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t cgtsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count) except? 0:
    """See `cusparseCgtsvInterleavedBatch_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseCgtsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>x, batch_count, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t zgtsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count) except? 0:
    """See `cusparseZgtsvInterleavedBatch_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseZgtsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <const cuDoubleComplex*>x, batch_count, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef sgtsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, intptr_t p_buffer):
    """See `cusparseSgtsvInterleavedBatch`."""
    with nogil:
        status = cusparseSgtsvInterleavedBatch(<Handle>handle, algo, m, <float*>dl, <float*>d, <float*>du, <float*>x, batch_count, <void*>p_buffer)
    check_status(status)


cpdef dgtsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, intptr_t p_buffer):
    """See `cusparseDgtsvInterleavedBatch`."""
    with nogil:
        status = cusparseDgtsvInterleavedBatch(<Handle>handle, algo, m, <double*>dl, <double*>d, <double*>du, <double*>x, batch_count, <void*>p_buffer)
    check_status(status)


cpdef cgtsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, intptr_t p_buffer):
    """See `cusparseCgtsvInterleavedBatch`."""
    with nogil:
        status = cusparseCgtsvInterleavedBatch(<Handle>handle, algo, m, <cuComplex*>dl, <cuComplex*>d, <cuComplex*>du, <cuComplex*>x, batch_count, <void*>p_buffer)
    check_status(status)


cpdef zgtsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, intptr_t p_buffer):
    """See `cusparseZgtsvInterleavedBatch`."""
    with nogil:
        status = cusparseZgtsvInterleavedBatch(<Handle>handle, algo, m, <cuDoubleComplex*>dl, <cuDoubleComplex*>d, <cuDoubleComplex*>du, <cuDoubleComplex*>x, batch_count, <void*>p_buffer)
    check_status(status)


cpdef size_t sgpsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count) except? 0:
    """See `cusparseSgpsvInterleavedBatch_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseSgpsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const float*>ds, <const float*>dl, <const float*>d, <const float*>du, <const float*>dw, <const float*>x, batch_count, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t dgpsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count) except? 0:
    """See `cusparseDgpsvInterleavedBatch_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseDgpsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const double*>ds, <const double*>dl, <const double*>d, <const double*>du, <const double*>dw, <const double*>x, batch_count, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t cgpsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count) except? 0:
    """See `cusparseCgpsvInterleavedBatch_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseCgpsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const cuComplex*>ds, <const cuComplex*>dl, <const cuComplex*>d, <const cuComplex*>du, <const cuComplex*>dw, <const cuComplex*>x, batch_count, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t zgpsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count) except? 0:
    """See `cusparseZgpsvInterleavedBatch_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseZgpsvInterleavedBatch_bufferSizeExt(<Handle>handle, algo, m, <const cuDoubleComplex*>ds, <const cuDoubleComplex*>dl, <const cuDoubleComplex*>d, <const cuDoubleComplex*>du, <const cuDoubleComplex*>dw, <const cuDoubleComplex*>x, batch_count, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef sgpsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count, intptr_t p_buffer):
    """See `cusparseSgpsvInterleavedBatch`."""
    with nogil:
        status = cusparseSgpsvInterleavedBatch(<Handle>handle, algo, m, <float*>ds, <float*>dl, <float*>d, <float*>du, <float*>dw, <float*>x, batch_count, <void*>p_buffer)
    check_status(status)


cpdef dgpsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count, intptr_t p_buffer):
    """See `cusparseDgpsvInterleavedBatch`."""
    with nogil:
        status = cusparseDgpsvInterleavedBatch(<Handle>handle, algo, m, <double*>ds, <double*>dl, <double*>d, <double*>du, <double*>dw, <double*>x, batch_count, <void*>p_buffer)
    check_status(status)


cpdef cgpsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count, intptr_t p_buffer):
    """See `cusparseCgpsvInterleavedBatch`."""
    with nogil:
        status = cusparseCgpsvInterleavedBatch(<Handle>handle, algo, m, <cuComplex*>ds, <cuComplex*>dl, <cuComplex*>d, <cuComplex*>du, <cuComplex*>dw, <cuComplex*>x, batch_count, <void*>p_buffer)
    check_status(status)


cpdef zgpsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count, intptr_t p_buffer):
    """See `cusparseZgpsvInterleavedBatch`."""
    with nogil:
        status = cusparseZgpsvInterleavedBatch(<Handle>handle, algo, m, <cuDoubleComplex*>ds, <cuDoubleComplex*>dl, <cuDoubleComplex*>d, <cuDoubleComplex*>du, <cuDoubleComplex*>dw, <cuDoubleComplex*>x, batch_count, <void*>p_buffer)
    check_status(status)


cpdef size_t scsrgeam2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c) except? 0:
    """See `cusparseScsrgeam2_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseScsrgeam2_bufferSizeExt(<Handle>handle, m, n, <const float*>alpha, <const MatDescr>descr_a, nnz_a, <const float*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const float*>beta, <const MatDescr>descr_b, nnz_b, <const float*>csr_sorted_val_b, <const int*>csr_sorted_row_ptr_b, <const int*>csr_sorted_col_ind_b, <const MatDescr>descr_c, <const float*>csr_sorted_val_c, <const int*>csr_sorted_row_ptr_c, <const int*>csr_sorted_col_ind_c, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t dcsrgeam2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c) except? 0:
    """See `cusparseDcsrgeam2_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseDcsrgeam2_bufferSizeExt(<Handle>handle, m, n, <const double*>alpha, <const MatDescr>descr_a, nnz_a, <const double*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const double*>beta, <const MatDescr>descr_b, nnz_b, <const double*>csr_sorted_val_b, <const int*>csr_sorted_row_ptr_b, <const int*>csr_sorted_col_ind_b, <const MatDescr>descr_c, <const double*>csr_sorted_val_c, <const int*>csr_sorted_row_ptr_c, <const int*>csr_sorted_col_ind_c, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t ccsrgeam2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c) except? 0:
    """See `cusparseCcsrgeam2_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseCcsrgeam2_bufferSizeExt(<Handle>handle, m, n, <const cuComplex*>alpha, <const MatDescr>descr_a, nnz_a, <const cuComplex*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const cuComplex*>beta, <const MatDescr>descr_b, nnz_b, <const cuComplex*>csr_sorted_val_b, <const int*>csr_sorted_row_ptr_b, <const int*>csr_sorted_col_ind_b, <const MatDescr>descr_c, <const cuComplex*>csr_sorted_val_c, <const int*>csr_sorted_row_ptr_c, <const int*>csr_sorted_col_ind_c, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t zcsrgeam2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c) except? 0:
    """See `cusparseZcsrgeam2_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseZcsrgeam2_bufferSizeExt(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const MatDescr>descr_a, nnz_a, <const cuDoubleComplex*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const cuDoubleComplex*>beta, <const MatDescr>descr_b, nnz_b, <const cuDoubleComplex*>csr_sorted_val_b, <const int*>csr_sorted_row_ptr_b, <const int*>csr_sorted_col_ind_b, <const MatDescr>descr_c, <const cuDoubleComplex*>csr_sorted_val_c, <const int*>csr_sorted_row_ptr_c, <const int*>csr_sorted_col_ind_c, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef xcsrgeam2nnz(intptr_t handle, int m, int n, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_row_ptr_c, intptr_t nnz_total_dev_host_ptr, intptr_t workspace):
    """See `cusparseXcsrgeam2Nnz`."""
    with nogil:
        status = cusparseXcsrgeam2Nnz(<Handle>handle, m, n, <const MatDescr>descr_a, nnz_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const MatDescr>descr_b, nnz_b, <const int*>csr_sorted_row_ptr_b, <const int*>csr_sorted_col_ind_b, <const MatDescr>descr_c, <int*>csr_sorted_row_ptr_c, <int*>nnz_total_dev_host_ptr, <void*>workspace)
    check_status(status)


cpdef scsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c, intptr_t p_buffer):
    """See `cusparseScsrgeam2`."""
    with nogil:
        status = cusparseScsrgeam2(<Handle>handle, m, n, <const float*>alpha, <const MatDescr>descr_a, nnz_a, <const float*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const float*>beta, <const MatDescr>descr_b, nnz_b, <const float*>csr_sorted_val_b, <const int*>csr_sorted_row_ptr_b, <const int*>csr_sorted_col_ind_b, <const MatDescr>descr_c, <float*>csr_sorted_val_c, <int*>csr_sorted_row_ptr_c, <int*>csr_sorted_col_ind_c, <void*>p_buffer)
    check_status(status)


cpdef dcsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c, intptr_t p_buffer):
    """See `cusparseDcsrgeam2`."""
    with nogil:
        status = cusparseDcsrgeam2(<Handle>handle, m, n, <const double*>alpha, <const MatDescr>descr_a, nnz_a, <const double*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const double*>beta, <const MatDescr>descr_b, nnz_b, <const double*>csr_sorted_val_b, <const int*>csr_sorted_row_ptr_b, <const int*>csr_sorted_col_ind_b, <const MatDescr>descr_c, <double*>csr_sorted_val_c, <int*>csr_sorted_row_ptr_c, <int*>csr_sorted_col_ind_c, <void*>p_buffer)
    check_status(status)


cpdef ccsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c, intptr_t p_buffer):
    """See `cusparseCcsrgeam2`."""
    with nogil:
        status = cusparseCcsrgeam2(<Handle>handle, m, n, <const cuComplex*>alpha, <const MatDescr>descr_a, nnz_a, <const cuComplex*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const cuComplex*>beta, <const MatDescr>descr_b, nnz_b, <const cuComplex*>csr_sorted_val_b, <const int*>csr_sorted_row_ptr_b, <const int*>csr_sorted_col_ind_b, <const MatDescr>descr_c, <cuComplex*>csr_sorted_val_c, <int*>csr_sorted_row_ptr_c, <int*>csr_sorted_col_ind_c, <void*>p_buffer)
    check_status(status)


cpdef zcsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c, intptr_t p_buffer):
    """See `cusparseZcsrgeam2`."""
    with nogil:
        status = cusparseZcsrgeam2(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const MatDescr>descr_a, nnz_a, <const cuDoubleComplex*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const cuDoubleComplex*>beta, <const MatDescr>descr_b, nnz_b, <const cuDoubleComplex*>csr_sorted_val_b, <const int*>csr_sorted_row_ptr_b, <const int*>csr_sorted_col_ind_b, <const MatDescr>descr_c, <cuDoubleComplex*>csr_sorted_val_c, <int*>csr_sorted_row_ptr_c, <int*>csr_sorted_col_ind_c, <void*>p_buffer)
    check_status(status)


cpdef snnz(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t a, int lda, intptr_t nnz_per_row_col, intptr_t nnz_total_dev_host_ptr):
    """See `cusparseSnnz`."""
    with nogil:
        status = cusparseSnnz(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const float*>a, lda, <int*>nnz_per_row_col, <int*>nnz_total_dev_host_ptr)
    check_status(status)


cpdef dnnz(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t a, int lda, intptr_t nnz_per_row_col, intptr_t nnz_total_dev_host_ptr):
    """See `cusparseDnnz`."""
    with nogil:
        status = cusparseDnnz(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const double*>a, lda, <int*>nnz_per_row_col, <int*>nnz_total_dev_host_ptr)
    check_status(status)


cpdef cnnz(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t a, int lda, intptr_t nnz_per_row_col, intptr_t nnz_total_dev_host_ptr):
    """See `cusparseCnnz`."""
    with nogil:
        status = cusparseCnnz(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const cuComplex*>a, lda, <int*>nnz_per_row_col, <int*>nnz_total_dev_host_ptr)
    check_status(status)


cpdef znnz(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t a, int lda, intptr_t nnz_per_row_col, intptr_t nnz_total_dev_host_ptr):
    """See `cusparseZnnz`."""
    with nogil:
        status = cusparseZnnz(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const cuDoubleComplex*>a, lda, <int*>nnz_per_row_col, <int*>nnz_total_dev_host_ptr)
    check_status(status)


cpdef xcoo2csr(intptr_t handle, intptr_t coo_row_ind, int nnz, int m, intptr_t csr_sorted_row_ptr, int idx_base):
    """See `cusparseXcoo2csr`."""
    with nogil:
        status = cusparseXcoo2csr(<Handle>handle, <const int*>coo_row_ind, nnz, m, <int*>csr_sorted_row_ptr, <_IndexBase>idx_base)
    check_status(status)


cpdef xcsr2coo(intptr_t handle, intptr_t csr_sorted_row_ptr, int nnz, int m, intptr_t coo_row_ind, int idx_base):
    """See `cusparseXcsr2coo`."""
    with nogil:
        status = cusparseXcsr2coo(<Handle>handle, <const int*>csr_sorted_row_ptr, nnz, m, <int*>coo_row_ind, <_IndexBase>idx_base)
    check_status(status)


cpdef sbsr2csr(intptr_t handle, int dir_a, int mb, int nb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c):
    """See `cusparseSbsr2csr`."""
    with nogil:
        status = cusparseSbsr2csr(<Handle>handle, <_Direction>dir_a, mb, nb, <const MatDescr>descr_a, <const float*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, block_dim, <const MatDescr>descr_c, <float*>csr_sorted_val_c, <int*>csr_sorted_row_ptr_c, <int*>csr_sorted_col_ind_c)
    check_status(status)


cpdef dbsr2csr(intptr_t handle, int dir_a, int mb, int nb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c):
    """See `cusparseDbsr2csr`."""
    with nogil:
        status = cusparseDbsr2csr(<Handle>handle, <_Direction>dir_a, mb, nb, <const MatDescr>descr_a, <const double*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, block_dim, <const MatDescr>descr_c, <double*>csr_sorted_val_c, <int*>csr_sorted_row_ptr_c, <int*>csr_sorted_col_ind_c)
    check_status(status)


cpdef cbsr2csr(intptr_t handle, int dir_a, int mb, int nb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c):
    """See `cusparseCbsr2csr`."""
    with nogil:
        status = cusparseCbsr2csr(<Handle>handle, <_Direction>dir_a, mb, nb, <const MatDescr>descr_a, <const cuComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, block_dim, <const MatDescr>descr_c, <cuComplex*>csr_sorted_val_c, <int*>csr_sorted_row_ptr_c, <int*>csr_sorted_col_ind_c)
    check_status(status)


cpdef zbsr2csr(intptr_t handle, int dir_a, int mb, int nb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c):
    """See `cusparseZbsr2csr`."""
    with nogil:
        status = cusparseZbsr2csr(<Handle>handle, <_Direction>dir_a, mb, nb, <const MatDescr>descr_a, <const cuDoubleComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, block_dim, <const MatDescr>descr_c, <cuDoubleComplex*>csr_sorted_val_c, <int*>csr_sorted_row_ptr_c, <int*>csr_sorted_col_ind_c)
    check_status(status)


cpdef int sgebsr2gebsc_buffer_size(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? -1:
    """See `cusparseSgebsr2gebsc_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseSgebsr2gebsc_bufferSize(<Handle>handle, mb, nb, nnzb, <const float*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef int dgebsr2gebsc_buffer_size(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? -1:
    """See `cusparseDgebsr2gebsc_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseDgebsr2gebsc_bufferSize(<Handle>handle, mb, nb, nnzb, <const double*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef int cgebsr2gebsc_buffer_size(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? -1:
    """See `cusparseCgebsr2gebsc_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseCgebsr2gebsc_bufferSize(<Handle>handle, mb, nb, nnzb, <const cuComplex*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef int zgebsr2gebsc_buffer_size(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? -1:
    """See `cusparseZgebsr2gebsc_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseZgebsr2gebsc_bufferSize(<Handle>handle, mb, nb, nnzb, <const cuDoubleComplex*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t sgebsr2gebsc_buffer_size_ext(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? 0:
    """See `cusparseSgebsr2gebsc_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseSgebsr2gebsc_bufferSizeExt(<Handle>handle, mb, nb, nnzb, <const float*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef size_t dgebsr2gebsc_buffer_size_ext(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? 0:
    """See `cusparseDgebsr2gebsc_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseDgebsr2gebsc_bufferSizeExt(<Handle>handle, mb, nb, nnzb, <const double*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef size_t cgebsr2gebsc_buffer_size_ext(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? 0:
    """See `cusparseCgebsr2gebsc_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseCgebsr2gebsc_bufferSizeExt(<Handle>handle, mb, nb, nnzb, <const cuComplex*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef size_t zgebsr2gebsc_buffer_size_ext(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? 0:
    """See `cusparseZgebsr2gebsc_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseZgebsr2gebsc_bufferSizeExt(<Handle>handle, mb, nb, nnzb, <const cuDoubleComplex*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef sgebsr2gebsc(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim, intptr_t bsc_val, intptr_t bsc_row_ind, intptr_t bsc_col_ptr, int copy_values, int idx_base, intptr_t p_buffer):
    """See `cusparseSgebsr2gebsc`."""
    with nogil:
        status = cusparseSgebsr2gebsc(<Handle>handle, mb, nb, nnzb, <const float*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, <float*>bsc_val, <int*>bsc_row_ind, <int*>bsc_col_ptr, <_Action>copy_values, <_IndexBase>idx_base, <void*>p_buffer)
    check_status(status)


cpdef dgebsr2gebsc(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim, intptr_t bsc_val, intptr_t bsc_row_ind, intptr_t bsc_col_ptr, int copy_values, int idx_base, intptr_t p_buffer):
    """See `cusparseDgebsr2gebsc`."""
    with nogil:
        status = cusparseDgebsr2gebsc(<Handle>handle, mb, nb, nnzb, <const double*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, <double*>bsc_val, <int*>bsc_row_ind, <int*>bsc_col_ptr, <_Action>copy_values, <_IndexBase>idx_base, <void*>p_buffer)
    check_status(status)


cpdef cgebsr2gebsc(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim, intptr_t bsc_val, intptr_t bsc_row_ind, intptr_t bsc_col_ptr, int copy_values, int idx_base, intptr_t p_buffer):
    """See `cusparseCgebsr2gebsc`."""
    with nogil:
        status = cusparseCgebsr2gebsc(<Handle>handle, mb, nb, nnzb, <const cuComplex*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, <cuComplex*>bsc_val, <int*>bsc_row_ind, <int*>bsc_col_ptr, <_Action>copy_values, <_IndexBase>idx_base, <void*>p_buffer)
    check_status(status)


cpdef zgebsr2gebsc(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim, intptr_t bsc_val, intptr_t bsc_row_ind, intptr_t bsc_col_ptr, int copy_values, int idx_base, intptr_t p_buffer):
    """See `cusparseZgebsr2gebsc`."""
    with nogil:
        status = cusparseZgebsr2gebsc(<Handle>handle, mb, nb, nnzb, <const cuDoubleComplex*>bsr_sorted_val, <const int*>bsr_sorted_row_ptr, <const int*>bsr_sorted_col_ind, row_block_dim, col_block_dim, <cuDoubleComplex*>bsc_val, <int*>bsc_row_ind, <int*>bsc_col_ptr, <_Action>copy_values, <_IndexBase>idx_base, <void*>p_buffer)
    check_status(status)


cpdef int scsr2gebsr_buffer_size(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? -1:
    """See `cusparseScsr2gebsr_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseScsr2gebsr_bufferSize(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const float*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, row_block_dim, col_block_dim, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef int dcsr2gebsr_buffer_size(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? -1:
    """See `cusparseDcsr2gebsr_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseDcsr2gebsr_bufferSize(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const double*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, row_block_dim, col_block_dim, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef int ccsr2gebsr_buffer_size(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? -1:
    """See `cusparseCcsr2gebsr_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseCcsr2gebsr_bufferSize(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const cuComplex*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, row_block_dim, col_block_dim, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef int zcsr2gebsr_buffer_size(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? -1:
    """See `cusparseZcsr2gebsr_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseZcsr2gebsr_bufferSize(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const cuDoubleComplex*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, row_block_dim, col_block_dim, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t scsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? 0:
    """See `cusparseScsr2gebsr_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseScsr2gebsr_bufferSizeExt(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const float*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, row_block_dim, col_block_dim, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef size_t dcsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? 0:
    """See `cusparseDcsr2gebsr_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseDcsr2gebsr_bufferSizeExt(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const double*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, row_block_dim, col_block_dim, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef size_t ccsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? 0:
    """See `cusparseCcsr2gebsr_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseCcsr2gebsr_bufferSizeExt(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const cuComplex*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, row_block_dim, col_block_dim, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef size_t zcsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? 0:
    """See `cusparseZcsr2gebsr_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseZcsr2gebsr_bufferSizeExt(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const cuDoubleComplex*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, row_block_dim, col_block_dim, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef xcsr2gebsr_nnz(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_c, intptr_t bsr_sorted_row_ptr_c, int row_block_dim, int col_block_dim, intptr_t nnz_total_dev_host_ptr, intptr_t p_buffer):
    """See `cusparseXcsr2gebsrNnz`."""
    with nogil:
        status = cusparseXcsr2gebsrNnz(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const MatDescr>descr_c, <int*>bsr_sorted_row_ptr_c, row_block_dim, col_block_dim, <int*>nnz_total_dev_host_ptr, <void*>p_buffer)
    check_status(status)


cpdef scsr2gebsr(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim, int col_block_dim, intptr_t p_buffer):
    """See `cusparseScsr2gebsr`."""
    with nogil:
        status = cusparseScsr2gebsr(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const float*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const MatDescr>descr_c, <float*>bsr_sorted_val_c, <int*>bsr_sorted_row_ptr_c, <int*>bsr_sorted_col_ind_c, row_block_dim, col_block_dim, <void*>p_buffer)
    check_status(status)


cpdef dcsr2gebsr(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim, int col_block_dim, intptr_t p_buffer):
    """See `cusparseDcsr2gebsr`."""
    with nogil:
        status = cusparseDcsr2gebsr(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const double*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const MatDescr>descr_c, <double*>bsr_sorted_val_c, <int*>bsr_sorted_row_ptr_c, <int*>bsr_sorted_col_ind_c, row_block_dim, col_block_dim, <void*>p_buffer)
    check_status(status)


cpdef ccsr2gebsr(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim, int col_block_dim, intptr_t p_buffer):
    """See `cusparseCcsr2gebsr`."""
    with nogil:
        status = cusparseCcsr2gebsr(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const cuComplex*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const MatDescr>descr_c, <cuComplex*>bsr_sorted_val_c, <int*>bsr_sorted_row_ptr_c, <int*>bsr_sorted_col_ind_c, row_block_dim, col_block_dim, <void*>p_buffer)
    check_status(status)


cpdef zcsr2gebsr(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim, int col_block_dim, intptr_t p_buffer):
    """See `cusparseZcsr2gebsr`."""
    with nogil:
        status = cusparseZcsr2gebsr(<Handle>handle, <_Direction>dir_a, m, n, <const MatDescr>descr_a, <const cuDoubleComplex*>csr_sorted_val_a, <const int*>csr_sorted_row_ptr_a, <const int*>csr_sorted_col_ind_a, <const MatDescr>descr_c, <cuDoubleComplex*>bsr_sorted_val_c, <int*>bsr_sorted_row_ptr_c, <int*>bsr_sorted_col_ind_c, row_block_dim, col_block_dim, <void*>p_buffer)
    check_status(status)


cpdef int sgebsr2gebsr_buffer_size(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? -1:
    """See `cusparseSgebsr2gebsr_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseSgebsr2gebsr_bufferSize(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const float*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, row_block_dim_c, col_block_dim_c, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef int dgebsr2gebsr_buffer_size(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? -1:
    """See `cusparseDgebsr2gebsr_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseDgebsr2gebsr_bufferSize(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const double*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, row_block_dim_c, col_block_dim_c, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef int cgebsr2gebsr_buffer_size(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? -1:
    """See `cusparseCgebsr2gebsr_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseCgebsr2gebsr_bufferSize(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const cuComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, row_block_dim_c, col_block_dim_c, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef int zgebsr2gebsr_buffer_size(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? -1:
    """See `cusparseZgebsr2gebsr_bufferSize`."""
    cdef int p_buffer_size_in_bytes
    with nogil:
        status = cusparseZgebsr2gebsr_bufferSize(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const cuDoubleComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, row_block_dim_c, col_block_dim_c, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef size_t sgebsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? 0:
    """See `cusparseSgebsr2gebsr_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseSgebsr2gebsr_bufferSizeExt(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const float*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, row_block_dim_c, col_block_dim_c, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef size_t dgebsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? 0:
    """See `cusparseDgebsr2gebsr_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseDgebsr2gebsr_bufferSizeExt(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const double*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, row_block_dim_c, col_block_dim_c, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef size_t cgebsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? 0:
    """See `cusparseCgebsr2gebsr_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseCgebsr2gebsr_bufferSizeExt(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const cuComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, row_block_dim_c, col_block_dim_c, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef size_t zgebsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? 0:
    """See `cusparseZgebsr2gebsr_bufferSizeExt`."""
    cdef size_t p_buffer_size
    with nogil:
        status = cusparseZgebsr2gebsr_bufferSizeExt(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const cuDoubleComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, row_block_dim_c, col_block_dim_c, &p_buffer_size)
    check_status(status)
    return p_buffer_size


cpdef xgebsr2gebsr_nnz(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, intptr_t descr_c, intptr_t bsr_sorted_row_ptr_c, int row_block_dim_c, int col_block_dim_c, intptr_t nnz_total_dev_host_ptr, intptr_t p_buffer):
    """See `cusparseXgebsr2gebsrNnz`."""
    with nogil:
        status = cusparseXgebsr2gebsrNnz(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, <const MatDescr>descr_c, <int*>bsr_sorted_row_ptr_c, row_block_dim_c, col_block_dim_c, <int*>nnz_total_dev_host_ptr, <void*>p_buffer)
    check_status(status)


cpdef sgebsr2gebsr(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim_c, int col_block_dim_c, intptr_t p_buffer):
    """See `cusparseSgebsr2gebsr`."""
    with nogil:
        status = cusparseSgebsr2gebsr(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const float*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, <const MatDescr>descr_c, <float*>bsr_sorted_val_c, <int*>bsr_sorted_row_ptr_c, <int*>bsr_sorted_col_ind_c, row_block_dim_c, col_block_dim_c, <void*>p_buffer)
    check_status(status)


cpdef dgebsr2gebsr(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim_c, int col_block_dim_c, intptr_t p_buffer):
    """See `cusparseDgebsr2gebsr`."""
    with nogil:
        status = cusparseDgebsr2gebsr(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const double*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, <const MatDescr>descr_c, <double*>bsr_sorted_val_c, <int*>bsr_sorted_row_ptr_c, <int*>bsr_sorted_col_ind_c, row_block_dim_c, col_block_dim_c, <void*>p_buffer)
    check_status(status)


cpdef cgebsr2gebsr(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim_c, int col_block_dim_c, intptr_t p_buffer):
    """See `cusparseCgebsr2gebsr`."""
    with nogil:
        status = cusparseCgebsr2gebsr(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const cuComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, <const MatDescr>descr_c, <cuComplex*>bsr_sorted_val_c, <int*>bsr_sorted_row_ptr_c, <int*>bsr_sorted_col_ind_c, row_block_dim_c, col_block_dim_c, <void*>p_buffer)
    check_status(status)


cpdef zgebsr2gebsr(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim_c, int col_block_dim_c, intptr_t p_buffer):
    """See `cusparseZgebsr2gebsr`."""
    with nogil:
        status = cusparseZgebsr2gebsr(<Handle>handle, <_Direction>dir_a, mb, nb, nnzb, <const MatDescr>descr_a, <const cuDoubleComplex*>bsr_sorted_val_a, <const int*>bsr_sorted_row_ptr_a, <const int*>bsr_sorted_col_ind_a, row_block_dim_a, col_block_dim_a, <const MatDescr>descr_c, <cuDoubleComplex*>bsr_sorted_val_c, <int*>bsr_sorted_row_ptr_c, <int*>bsr_sorted_col_ind_c, row_block_dim_c, col_block_dim_c, <void*>p_buffer)
    check_status(status)


cpdef size_t xcoosort_buffer_size_ext(intptr_t handle, int m, int n, int nnz, intptr_t coo_rows_a, intptr_t coo_cols_a) except? 0:
    """See `cusparseXcoosort_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseXcoosort_bufferSizeExt(<Handle>handle, m, n, nnz, <const int*>coo_rows_a, <const int*>coo_cols_a, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef xcoosort_by_row(intptr_t handle, int m, int n, int nnz, intptr_t coo_rows_a, intptr_t coo_cols_a, intptr_t p, intptr_t p_buffer):
    """See `cusparseXcoosortByRow`."""
    with nogil:
        status = cusparseXcoosortByRow(<Handle>handle, m, n, nnz, <int*>coo_rows_a, <int*>coo_cols_a, <int*>p, <void*>p_buffer)
    check_status(status)


cpdef xcoosort_by_column(intptr_t handle, int m, int n, int nnz, intptr_t coo_rows_a, intptr_t coo_cols_a, intptr_t p, intptr_t p_buffer):
    """See `cusparseXcoosortByColumn`."""
    with nogil:
        status = cusparseXcoosortByColumn(<Handle>handle, m, n, nnz, <int*>coo_rows_a, <int*>coo_cols_a, <int*>p, <void*>p_buffer)
    check_status(status)


cpdef size_t xcsrsort_buffer_size_ext(intptr_t handle, int m, int n, int nnz, intptr_t csr_row_ptr_a, intptr_t csr_col_ind_a) except? 0:
    """See `cusparseXcsrsort_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseXcsrsort_bufferSizeExt(<Handle>handle, m, n, nnz, <const int*>csr_row_ptr_a, <const int*>csr_col_ind_a, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef xcsrsort(intptr_t handle, int m, int n, int nnz, intptr_t descr_a, intptr_t csr_row_ptr_a, intptr_t csr_col_ind_a, intptr_t p, intptr_t p_buffer):
    """See `cusparseXcsrsort`."""
    with nogil:
        status = cusparseXcsrsort(<Handle>handle, m, n, nnz, <const MatDescr>descr_a, <const int*>csr_row_ptr_a, <int*>csr_col_ind_a, <int*>p, <void*>p_buffer)
    check_status(status)


cpdef size_t xcscsort_buffer_size_ext(intptr_t handle, int m, int n, int nnz, intptr_t csc_col_ptr_a, intptr_t csc_row_ind_a) except? 0:
    """See `cusparseXcscsort_bufferSizeExt`."""
    cdef size_t p_buffer_size_in_bytes
    with nogil:
        status = cusparseXcscsort_bufferSizeExt(<Handle>handle, m, n, nnz, <const int*>csc_col_ptr_a, <const int*>csc_row_ind_a, &p_buffer_size_in_bytes)
    check_status(status)
    return p_buffer_size_in_bytes


cpdef xcscsort(intptr_t handle, int m, int n, int nnz, intptr_t descr_a, intptr_t csc_col_ptr_a, intptr_t csc_row_ind_a, intptr_t p, intptr_t p_buffer):
    """See `cusparseXcscsort`."""
    with nogil:
        status = cusparseXcscsort(<Handle>handle, m, n, nnz, <const MatDescr>descr_a, <const int*>csc_col_ptr_a, <int*>csc_row_ind_a, <int*>p, <void*>p_buffer)
    check_status(status)


cpdef csr2csc_ex2(intptr_t handle, int m, int n, int nnz, intptr_t csr_val, intptr_t csr_row_ptr, intptr_t csr_col_ind, intptr_t csc_val, intptr_t csc_col_ptr, intptr_t csc_row_ind, int val_type, int copy_values, int idx_base, int alg, intptr_t buffer):
    """See `cusparseCsr2cscEx2`."""
    with nogil:
        status = cusparseCsr2cscEx2(<Handle>handle, m, n, nnz, <const void*>csr_val, <const int*>csr_row_ptr, <const int*>csr_col_ind, <void*>csc_val, <int*>csc_col_ptr, <int*>csc_row_ind, <DataType>val_type, <_Action>copy_values, <_IndexBase>idx_base, <_Csr2CscAlg>alg, <void*>buffer)
    check_status(status)


cpdef size_t csr2csc_ex2_buffer_size(intptr_t handle, int m, int n, int nnz, intptr_t csr_val, intptr_t csr_row_ptr, intptr_t csr_col_ind, intptr_t csc_val, intptr_t csc_col_ptr, intptr_t csc_row_ind, int val_type, int copy_values, int idx_base, int alg) except? 0:
    """See `cusparseCsr2cscEx2_bufferSize`."""
    cdef size_t buffer_size
    with nogil:
        status = cusparseCsr2cscEx2_bufferSize(<Handle>handle, m, n, nnz, <const void*>csr_val, <const int*>csr_row_ptr, <const int*>csr_col_ind, <void*>csc_val, <int*>csc_col_ptr, <int*>csc_row_ind, <DataType>val_type, <_Action>copy_values, <_IndexBase>idx_base, <_Csr2CscAlg>alg, &buffer_size)
    check_status(status)
    return buffer_size


cpdef intptr_t create_sp_vec(int64_t size, int64_t nnz, intptr_t indices, intptr_t values, int idx_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateSpVec`."""
    cdef SpVecDescr sp_vec_descr
    with nogil:
        status = cusparseCreateSpVec(&sp_vec_descr, size, nnz, <void*>indices, <void*>values, <_IndexType>idx_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_vec_descr


cpdef destroy_sp_vec(intptr_t sp_vec_descr):
    """See `cusparseDestroySpVec`."""
    with nogil:
        status = cusparseDestroySpVec(<ConstSpVecDescr>sp_vec_descr)
    check_status(status)


cpdef tuple sp_vec_get(intptr_t sp_vec_descr):
    """See `cusparseSpVecGet`."""
    cdef int64_t size
    cdef int64_t nnz
    cdef void* indices
    cdef void* values
    cdef _IndexType idx_type
    cdef _IndexBase idx_base
    cdef DataType value_type
    with nogil:
        status = cusparseSpVecGet(<SpVecDescr>sp_vec_descr, &size, &nnz, &indices, &values, &idx_type, &idx_base, &value_type)
    check_status(status)
    return (size, nnz, <intptr_t>indices, <intptr_t>values, <int>idx_type, <int>idx_base, <int>value_type)


cpdef int sp_vec_get_index_base(intptr_t sp_vec_descr) except? -1:
    """See `cusparseSpVecGetIndexBase`."""
    cdef _IndexBase idx_base
    with nogil:
        status = cusparseSpVecGetIndexBase(<ConstSpVecDescr>sp_vec_descr, &idx_base)
    check_status(status)
    return <int>idx_base


cpdef intptr_t sp_vec_get_values(intptr_t sp_vec_descr) except? -1:
    """See `cusparseSpVecGetValues`."""
    cdef void* values
    with nogil:
        status = cusparseSpVecGetValues(<SpVecDescr>sp_vec_descr, &values)
    check_status(status)
    return <intptr_t>values


cpdef sp_vec_set_values(intptr_t sp_vec_descr, intptr_t values):
    """See `cusparseSpVecSetValues`."""
    with nogil:
        status = cusparseSpVecSetValues(<SpVecDescr>sp_vec_descr, <void*>values)
    check_status(status)


cpdef intptr_t create_dn_vec(int64_t size, intptr_t values, int value_type) except? 0:
    """See `cusparseCreateDnVec`."""
    cdef DnVecDescr dn_vec_descr
    with nogil:
        status = cusparseCreateDnVec(&dn_vec_descr, size, <void*>values, <DataType>value_type)
    check_status(status)
    return <intptr_t>dn_vec_descr


cpdef destroy_dn_vec(intptr_t dn_vec_descr):
    """See `cusparseDestroyDnVec`."""
    with nogil:
        status = cusparseDestroyDnVec(<ConstDnVecDescr>dn_vec_descr)
    check_status(status)


cpdef tuple dn_vec_get(intptr_t dn_vec_descr):
    """See `cusparseDnVecGet`."""
    cdef int64_t size
    cdef void* values
    cdef DataType value_type
    with nogil:
        status = cusparseDnVecGet(<DnVecDescr>dn_vec_descr, &size, &values, &value_type)
    check_status(status)
    return (size, <intptr_t>values, <int>value_type)


cpdef intptr_t dn_vec_get_values(intptr_t dn_vec_descr) except? -1:
    """See `cusparseDnVecGetValues`."""
    cdef void* values
    with nogil:
        status = cusparseDnVecGetValues(<DnVecDescr>dn_vec_descr, &values)
    check_status(status)
    return <intptr_t>values


cpdef dn_vec_set_values(intptr_t dn_vec_descr, intptr_t values):
    """See `cusparseDnVecSetValues`."""
    with nogil:
        status = cusparseDnVecSetValues(<DnVecDescr>dn_vec_descr, <void*>values)
    check_status(status)


cpdef destroy_sp_mat(intptr_t sp_mat_descr):
    """See `cusparseDestroySpMat`."""
    with nogil:
        status = cusparseDestroySpMat(<ConstSpMatDescr>sp_mat_descr)
    check_status(status)


cpdef int sp_mat_get_format(intptr_t sp_mat_descr) except? -1:
    """See `cusparseSpMatGetFormat`."""
    cdef _Format format
    with nogil:
        status = cusparseSpMatGetFormat(<ConstSpMatDescr>sp_mat_descr, &format)
    check_status(status)
    return <int>format


cpdef int sp_mat_get_index_base(intptr_t sp_mat_descr) except? -1:
    """See `cusparseSpMatGetIndexBase`."""
    cdef _IndexBase idx_base
    with nogil:
        status = cusparseSpMatGetIndexBase(<ConstSpMatDescr>sp_mat_descr, &idx_base)
    check_status(status)
    return <int>idx_base


cpdef intptr_t sp_mat_get_values(intptr_t sp_mat_descr) except? -1:
    """See `cusparseSpMatGetValues`."""
    cdef void* values
    with nogil:
        status = cusparseSpMatGetValues(<SpMatDescr>sp_mat_descr, &values)
    check_status(status)
    return <intptr_t>values


cpdef sp_mat_set_values(intptr_t sp_mat_descr, intptr_t values):
    """See `cusparseSpMatSetValues`."""
    with nogil:
        status = cusparseSpMatSetValues(<SpMatDescr>sp_mat_descr, <void*>values)
    check_status(status)


cpdef tuple sp_mat_get_size(intptr_t sp_mat_descr):
    """See `cusparseSpMatGetSize`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t nnz
    with nogil:
        status = cusparseSpMatGetSize(<ConstSpMatDescr>sp_mat_descr, &rows, &cols, &nnz)
    check_status(status)
    return (rows, cols, nnz)


cpdef int sp_mat_get_strided_batch(intptr_t sp_mat_descr) except? -1:
    """See `cusparseSpMatGetStridedBatch`."""
    cdef int batch_count
    with nogil:
        status = cusparseSpMatGetStridedBatch(<ConstSpMatDescr>sp_mat_descr, &batch_count)
    check_status(status)
    return batch_count


cpdef coo_set_strided_batch(intptr_t sp_mat_descr, int batch_count, int64_t batch_stride):
    """See `cusparseCooSetStridedBatch`."""
    with nogil:
        status = cusparseCooSetStridedBatch(<SpMatDescr>sp_mat_descr, batch_count, batch_stride)
    check_status(status)


cpdef csr_set_strided_batch(intptr_t sp_mat_descr, int batch_count, int64_t offsets_batch_stride, int64_t columns_values_batch_stride):
    """See `cusparseCsrSetStridedBatch`."""
    with nogil:
        status = cusparseCsrSetStridedBatch(<SpMatDescr>sp_mat_descr, batch_count, offsets_batch_stride, columns_values_batch_stride)
    check_status(status)


cpdef intptr_t create_csr(int64_t rows, int64_t cols, int64_t nnz, intptr_t csr_row_offsets, intptr_t csr_col_ind, intptr_t csr_values, int csr_row_offsets_type, int csr_col_ind_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateCsr`."""
    cdef SpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateCsr(&sp_mat_descr, rows, cols, nnz, <void*>csr_row_offsets, <void*>csr_col_ind, <void*>csr_values, <_IndexType>csr_row_offsets_type, <_IndexType>csr_col_ind_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef tuple csr_get(intptr_t sp_mat_descr):
    """See `cusparseCsrGet`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t nnz
    cdef void* csr_row_offsets
    cdef void* csr_col_ind
    cdef void* csr_values
    cdef _IndexType csr_row_offsets_type
    cdef _IndexType csr_col_ind_type
    cdef _IndexBase idx_base
    cdef DataType value_type
    with nogil:
        status = cusparseCsrGet(<SpMatDescr>sp_mat_descr, &rows, &cols, &nnz, &csr_row_offsets, &csr_col_ind, &csr_values, &csr_row_offsets_type, &csr_col_ind_type, &idx_base, &value_type)
    check_status(status)
    return (rows, cols, nnz, <intptr_t>csr_row_offsets, <intptr_t>csr_col_ind, <intptr_t>csr_values, <int>csr_row_offsets_type, <int>csr_col_ind_type, <int>idx_base, <int>value_type)


cpdef csr_set_pointers(intptr_t sp_mat_descr, intptr_t csr_row_offsets, intptr_t csr_col_ind, intptr_t csr_values):
    """See `cusparseCsrSetPointers`."""
    with nogil:
        status = cusparseCsrSetPointers(<SpMatDescr>sp_mat_descr, <void*>csr_row_offsets, <void*>csr_col_ind, <void*>csr_values)
    check_status(status)


cpdef intptr_t create_coo(int64_t rows, int64_t cols, int64_t nnz, intptr_t coo_row_ind, intptr_t coo_col_ind, intptr_t coo_values, int coo_idx_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateCoo`."""
    cdef SpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateCoo(&sp_mat_descr, rows, cols, nnz, <void*>coo_row_ind, <void*>coo_col_ind, <void*>coo_values, <_IndexType>coo_idx_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef tuple coo_get(intptr_t sp_mat_descr):
    """See `cusparseCooGet`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t nnz
    cdef void* coo_row_ind
    cdef void* coo_col_ind
    cdef void* coo_values
    cdef _IndexType idx_type
    cdef _IndexBase idx_base
    cdef DataType value_type
    with nogil:
        status = cusparseCooGet(<SpMatDescr>sp_mat_descr, &rows, &cols, &nnz, &coo_row_ind, &coo_col_ind, &coo_values, &idx_type, &idx_base, &value_type)
    check_status(status)
    return (rows, cols, nnz, <intptr_t>coo_row_ind, <intptr_t>coo_col_ind, <intptr_t>coo_values, <int>idx_type, <int>idx_base, <int>value_type)


cpdef intptr_t create_dn_mat(int64_t rows, int64_t cols, int64_t ld, intptr_t values, int value_type, int order) except? 0:
    """See `cusparseCreateDnMat`."""
    cdef DnMatDescr dn_mat_descr
    with nogil:
        status = cusparseCreateDnMat(&dn_mat_descr, rows, cols, ld, <void*>values, <DataType>value_type, <_Order>order)
    check_status(status)
    return <intptr_t>dn_mat_descr


cpdef destroy_dn_mat(intptr_t dn_mat_descr):
    """See `cusparseDestroyDnMat`."""
    with nogil:
        status = cusparseDestroyDnMat(<ConstDnMatDescr>dn_mat_descr)
    check_status(status)


cpdef tuple dn_mat_get(intptr_t dn_mat_descr):
    """See `cusparseDnMatGet`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t ld
    cdef void* values
    cdef DataType type
    cdef _Order order
    with nogil:
        status = cusparseDnMatGet(<DnMatDescr>dn_mat_descr, &rows, &cols, &ld, &values, &type, &order)
    check_status(status)
    return (rows, cols, ld, <intptr_t>values, <int>type, <int>order)


cpdef intptr_t dn_mat_get_values(intptr_t dn_mat_descr) except? -1:
    """See `cusparseDnMatGetValues`."""
    cdef void* values
    with nogil:
        status = cusparseDnMatGetValues(<DnMatDescr>dn_mat_descr, &values)
    check_status(status)
    return <intptr_t>values


cpdef dn_mat_set_values(intptr_t dn_mat_descr, intptr_t values):
    """See `cusparseDnMatSetValues`."""
    with nogil:
        status = cusparseDnMatSetValues(<DnMatDescr>dn_mat_descr, <void*>values)
    check_status(status)


cpdef dn_mat_set_strided_batch(intptr_t dn_mat_descr, int batch_count, int64_t batch_stride):
    """See `cusparseDnMatSetStridedBatch`."""
    with nogil:
        status = cusparseDnMatSetStridedBatch(<DnMatDescr>dn_mat_descr, batch_count, batch_stride)
    check_status(status)


cpdef tuple dn_mat_get_strided_batch(intptr_t dn_mat_descr):
    """See `cusparseDnMatGetStridedBatch`."""
    cdef int batch_count
    cdef int64_t batch_stride
    with nogil:
        status = cusparseDnMatGetStridedBatch(<ConstDnMatDescr>dn_mat_descr, &batch_count, &batch_stride)
    check_status(status)
    return (batch_count, batch_stride)


cpdef axpby(intptr_t handle, intptr_t alpha, intptr_t vec_x, intptr_t beta, intptr_t vec_y):
    """See `cusparseAxpby`."""
    with nogil:
        status = cusparseAxpby(<Handle>handle, <const void*>alpha, <ConstSpVecDescr>vec_x, <const void*>beta, <DnVecDescr>vec_y)
    check_status(status)


cpdef gather(intptr_t handle, intptr_t vec_y, intptr_t vec_x):
    """See `cusparseGather`."""
    with nogil:
        status = cusparseGather(<Handle>handle, <ConstDnVecDescr>vec_y, <SpVecDescr>vec_x)
    check_status(status)


cpdef scatter(intptr_t handle, intptr_t vec_x, intptr_t vec_y):
    """See `cusparseScatter`."""
    with nogil:
        status = cusparseScatter(<Handle>handle, <ConstSpVecDescr>vec_x, <DnVecDescr>vec_y)
    check_status(status)


cpdef size_t sp_vv_buffer_size(intptr_t handle, int op_x, intptr_t vec_x, intptr_t vec_y, intptr_t result, int compute_type) except? 0:
    """See `cusparseSpVV_bufferSize`."""
    cdef size_t buffer_size
    with nogil:
        status = cusparseSpVV_bufferSize(<Handle>handle, <_Operation>op_x, <ConstSpVecDescr>vec_x, <ConstDnVecDescr>vec_y, <const void*>result, <DataType>compute_type, &buffer_size)
    check_status(status)
    return buffer_size


cpdef sp_vv(intptr_t handle, int op_x, intptr_t vec_x, intptr_t vec_y, intptr_t result, int compute_type, intptr_t external_buffer):
    """See `cusparseSpVV`."""
    with nogil:
        status = cusparseSpVV(<Handle>handle, <_Operation>op_x, <ConstSpVecDescr>vec_x, <ConstDnVecDescr>vec_y, <void*>result, <DataType>compute_type, <void*>external_buffer)
    check_status(status)


cpdef sp_mv(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t beta, intptr_t vec_y, int compute_type, int alg, intptr_t external_buffer):
    """See `cusparseSpMV`."""
    with nogil:
        status = cusparseSpMV(<Handle>handle, <_Operation>op_a, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnVecDescr>vec_x, <const void*>beta, <DnVecDescr>vec_y, <DataType>compute_type, <_SpMVAlg>alg, <void*>external_buffer)
    check_status(status)


cpdef size_t sp_mv_buffer_size(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t beta, intptr_t vec_y, int compute_type, int alg) except? 0:
    """See `cusparseSpMV_bufferSize`."""
    cdef size_t buffer_size
    with nogil:
        status = cusparseSpMV_bufferSize(<Handle>handle, <_Operation>op_a, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnVecDescr>vec_x, <const void*>beta, <DnVecDescr>vec_y, <DataType>compute_type, <_SpMVAlg>alg, &buffer_size)
    check_status(status)
    return buffer_size


cpdef sp_mm(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t external_buffer):
    """See `cusparseSpMM`."""
    with nogil:
        status = cusparseSpMM(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnMatDescr>mat_b, <const void*>beta, <DnMatDescr>mat_c, <DataType>compute_type, <_SpMMAlg>alg, <void*>external_buffer)
    check_status(status)


cpdef size_t sp_mm_buffer_size(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg) except? 0:
    """See `cusparseSpMM_bufferSize`."""
    cdef size_t buffer_size
    with nogil:
        status = cusparseSpMM_bufferSize(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnMatDescr>mat_b, <const void*>beta, <DnMatDescr>mat_c, <DataType>compute_type, <_SpMMAlg>alg, &buffer_size)
    check_status(status)
    return buffer_size


cpdef intptr_t sp_gemm_create_descr() except? 0:
    """See `cusparseSpGEMM_createDescr`."""
    cdef SpGEMMDescr descr
    with nogil:
        status = cusparseSpGEMM_createDescr(&descr)
    check_status(status)
    return <intptr_t>descr


cpdef sp_gemm_destroy_descr(intptr_t descr):
    """See `cusparseSpGEMM_destroyDescr`."""
    with nogil:
        status = cusparseSpGEMM_destroyDescr(<SpGEMMDescr>descr)
    check_status(status)


cpdef size_t sp_gemm_work_estimation(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t spgemm_descr, intptr_t external_buffer1) except? 0:
    """See `cusparseSpGEMM_workEstimation`."""
    cdef size_t buffer_size1
    with nogil:
        status = cusparseSpGEMM_workEstimation(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstSpMatDescr>mat_b, <const void*>beta, <SpMatDescr>mat_c, <DataType>compute_type, <_SpGEMMAlg>alg, <SpGEMMDescr>spgemm_descr, &buffer_size1, <void*>external_buffer1)
    check_status(status)
    return buffer_size1


cpdef sp_gemm_compute(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t spgemm_descr, intptr_t buffer_size2, intptr_t external_buffer2):
    """See `cusparseSpGEMM_compute`."""
    with nogil:
        status = cusparseSpGEMM_compute(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstSpMatDescr>mat_b, <const void*>beta, <SpMatDescr>mat_c, <DataType>compute_type, <_SpGEMMAlg>alg, <SpGEMMDescr>spgemm_descr, <size_t*>buffer_size2, <void*>external_buffer2)
    check_status(status)


cpdef sp_gemm_copy(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t spgemm_descr):
    """See `cusparseSpGEMM_copy`."""
    with nogil:
        status = cusparseSpGEMM_copy(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstSpMatDescr>mat_b, <const void*>beta, <SpMatDescr>mat_c, <DataType>compute_type, <_SpGEMMAlg>alg, <SpGEMMDescr>spgemm_descr)
    check_status(status)


cpdef intptr_t create_csc(int64_t rows, int64_t cols, int64_t nnz, intptr_t csc_col_offsets, intptr_t csc_row_ind, intptr_t csc_values, int csc_col_offsets_type, int csc_row_ind_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateCsc`."""
    cdef SpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateCsc(&sp_mat_descr, rows, cols, nnz, <void*>csc_col_offsets, <void*>csc_row_ind, <void*>csc_values, <_IndexType>csc_col_offsets_type, <_IndexType>csc_row_ind_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef csc_set_pointers(intptr_t sp_mat_descr, intptr_t csc_col_offsets, intptr_t csc_row_ind, intptr_t csc_values):
    """See `cusparseCscSetPointers`."""
    with nogil:
        status = cusparseCscSetPointers(<SpMatDescr>sp_mat_descr, <void*>csc_col_offsets, <void*>csc_row_ind, <void*>csc_values)
    check_status(status)


cpdef coo_set_pointers(intptr_t sp_mat_descr, intptr_t coo_rows, intptr_t coo_columns, intptr_t coo_values):
    """See `cusparseCooSetPointers`."""
    with nogil:
        status = cusparseCooSetPointers(<SpMatDescr>sp_mat_descr, <void*>coo_rows, <void*>coo_columns, <void*>coo_values)
    check_status(status)


cpdef size_t sparse_to_dense_buffer_size(intptr_t handle, intptr_t mat_a, intptr_t mat_b, int alg) except? 0:
    """See `cusparseSparseToDense_bufferSize`."""
    cdef size_t buffer_size
    with nogil:
        status = cusparseSparseToDense_bufferSize(<Handle>handle, <ConstSpMatDescr>mat_a, <DnMatDescr>mat_b, <_SparseToDenseAlg>alg, &buffer_size)
    check_status(status)
    return buffer_size


cpdef sparse_to_dense(intptr_t handle, intptr_t mat_a, intptr_t mat_b, int alg, intptr_t external_buffer):
    """See `cusparseSparseToDense`."""
    with nogil:
        status = cusparseSparseToDense(<Handle>handle, <ConstSpMatDescr>mat_a, <DnMatDescr>mat_b, <_SparseToDenseAlg>alg, <void*>external_buffer)
    check_status(status)


cpdef size_t dense_to_sparse_buffer_size(intptr_t handle, intptr_t mat_a, intptr_t mat_b, int alg) except? 0:
    """See `cusparseDenseToSparse_bufferSize`."""
    cdef size_t buffer_size
    with nogil:
        status = cusparseDenseToSparse_bufferSize(<Handle>handle, <ConstDnMatDescr>mat_a, <SpMatDescr>mat_b, <_DenseToSparseAlg>alg, &buffer_size)
    check_status(status)
    return buffer_size


cpdef dense_to_sparse_analysis(intptr_t handle, intptr_t mat_a, intptr_t mat_b, int alg, intptr_t external_buffer):
    """See `cusparseDenseToSparse_analysis`."""
    with nogil:
        status = cusparseDenseToSparse_analysis(<Handle>handle, <ConstDnMatDescr>mat_a, <SpMatDescr>mat_b, <_DenseToSparseAlg>alg, <void*>external_buffer)
    check_status(status)


cpdef dense_to_sparse_convert(intptr_t handle, intptr_t mat_a, intptr_t mat_b, int alg, intptr_t external_buffer):
    """See `cusparseDenseToSparse_convert`."""
    with nogil:
        status = cusparseDenseToSparse_convert(<Handle>handle, <ConstDnMatDescr>mat_a, <SpMatDescr>mat_b, <_DenseToSparseAlg>alg, <void*>external_buffer)
    check_status(status)


cpdef intptr_t create_blocked_ell(int64_t rows, int64_t cols, int64_t ell_block_size, int64_t ell_cols, intptr_t ell_col_ind, intptr_t ell_value, int ell_idx_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateBlockedEll`."""
    cdef SpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateBlockedEll(&sp_mat_descr, rows, cols, ell_block_size, ell_cols, <void*>ell_col_ind, <void*>ell_value, <_IndexType>ell_idx_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef tuple blocked_ell_get(intptr_t sp_mat_descr):
    """See `cusparseBlockedEllGet`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t ell_block_size
    cdef int64_t ell_cols
    cdef void* ell_col_ind
    cdef void* ell_value
    cdef _IndexType ell_idx_type
    cdef _IndexBase idx_base
    cdef DataType value_type
    with nogil:
        status = cusparseBlockedEllGet(<SpMatDescr>sp_mat_descr, &rows, &cols, &ell_block_size, &ell_cols, &ell_col_ind, &ell_value, &ell_idx_type, &idx_base, &value_type)
    check_status(status)
    return (rows, cols, ell_block_size, ell_cols, <intptr_t>ell_col_ind, <intptr_t>ell_value, <int>ell_idx_type, <int>idx_base, <int>value_type)


cpdef sp_mm_preprocess(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t external_buffer):
    """See `cusparseSpMM_preprocess`."""
    with nogil:
        status = cusparseSpMM_preprocess(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnMatDescr>mat_b, <const void*>beta, <DnMatDescr>mat_c, <DataType>compute_type, <_SpMMAlg>alg, <void*>external_buffer)
    check_status(status)


cpdef sddmm_buffer_size(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t buffer_size):
    """See `cusparseSDDMM_bufferSize`."""
    with nogil:
        status = cusparseSDDMM_bufferSize(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstDnMatDescr>mat_a, <ConstDnMatDescr>mat_b, <const void*>beta, <SpMatDescr>mat_c, <DataType>compute_type, <_SDDMMAlg>alg, <size_t*>buffer_size)
    check_status(status)


cpdef sddmm_preprocess(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t external_buffer):
    """See `cusparseSDDMM_preprocess`."""
    with nogil:
        status = cusparseSDDMM_preprocess(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstDnMatDescr>mat_a, <ConstDnMatDescr>mat_b, <const void*>beta, <SpMatDescr>mat_c, <DataType>compute_type, <_SDDMMAlg>alg, <void*>external_buffer)
    check_status(status)


cpdef sddmm(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t external_buffer):
    """See `cusparseSDDMM`."""
    with nogil:
        status = cusparseSDDMM(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstDnMatDescr>mat_a, <ConstDnMatDescr>mat_b, <const void*>beta, <SpMatDescr>mat_c, <DataType>compute_type, <_SDDMMAlg>alg, <void*>external_buffer)
    check_status(status)


######################### Python specific utility #########################

cdef dict sp_mat_attribute_sizes = {
    CUSPARSE_SPMAT_FILL_MODE: _numpy.int32,
    CUSPARSE_SPMAT_DIAG_TYPE: _numpy.int32,
}

cpdef get_sp_mat_attribute_dtype(int attr):
    """Get the Python data type of the corresponding SpMatAttribute attribute.

    Args:
        attr (SpMatAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`sp_mat_get_attribute`, :func:`sp_mat_set_attribute`.
    """
    return sp_mat_attribute_sizes[attr]

###########################################################################


cpdef sp_mat_get_attribute(intptr_t sp_mat_descr, int attribute, intptr_t data, size_t data_size):
    """See `cusparseSpMatGetAttribute`."""
    with nogil:
        status = cusparseSpMatGetAttribute(<ConstSpMatDescr>sp_mat_descr, <_SpMatAttribute>attribute, <void*>data, data_size)
    check_status(status)


cpdef sp_mat_set_attribute(intptr_t sp_mat_descr, int attribute, intptr_t data, size_t data_size):
    """See `cusparseSpMatSetAttribute`."""
    with nogil:
        status = cusparseSpMatSetAttribute(<SpMatDescr>sp_mat_descr, <_SpMatAttribute>attribute, <void*>data, data_size)
    check_status(status)


cpdef intptr_t sp_sv_create_descr() except? 0:
    """See `cusparseSpSV_createDescr`."""
    cdef SpSVDescr descr
    with nogil:
        status = cusparseSpSV_createDescr(&descr)
    check_status(status)
    return <intptr_t>descr


cpdef sp_sv_destroy_descr(intptr_t descr):
    """See `cusparseSpSV_destroyDescr`."""
    with nogil:
        status = cusparseSpSV_destroyDescr(<SpSVDescr>descr)
    check_status(status)


cpdef size_t sp_sv_buffer_size(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t vec_y, int compute_type, int alg, intptr_t spsv_descr) except? 0:
    """See `cusparseSpSV_bufferSize`."""
    cdef size_t buffer_size
    with nogil:
        status = cusparseSpSV_bufferSize(<Handle>handle, <_Operation>op_a, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnVecDescr>vec_x, <DnVecDescr>vec_y, <DataType>compute_type, <_SpSVAlg>alg, <SpSVDescr>spsv_descr, &buffer_size)
    check_status(status)
    return buffer_size


cpdef sp_sv_analysis(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t vec_y, int compute_type, int alg, intptr_t spsv_descr, intptr_t external_buffer):
    """See `cusparseSpSV_analysis`."""
    with nogil:
        status = cusparseSpSV_analysis(<Handle>handle, <_Operation>op_a, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnVecDescr>vec_x, <DnVecDescr>vec_y, <DataType>compute_type, <_SpSVAlg>alg, <SpSVDescr>spsv_descr, <void*>external_buffer)
    check_status(status)


cpdef sp_sv_solve(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t vec_y, int compute_type, int alg, intptr_t spsv_descr):
    """See `cusparseSpSV_solve`."""
    with nogil:
        status = cusparseSpSV_solve(<Handle>handle, <_Operation>op_a, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnVecDescr>vec_x, <DnVecDescr>vec_y, <DataType>compute_type, <_SpSVAlg>alg, <SpSVDescr>spsv_descr)
    check_status(status)


cpdef intptr_t sp_sm_create_descr() except? 0:
    """See `cusparseSpSM_createDescr`."""
    cdef SpSMDescr descr
    with nogil:
        status = cusparseSpSM_createDescr(&descr)
    check_status(status)
    return <intptr_t>descr


cpdef sp_sm_destroy_descr(intptr_t descr):
    """See `cusparseSpSM_destroyDescr`."""
    with nogil:
        status = cusparseSpSM_destroyDescr(<SpSMDescr>descr)
    check_status(status)


cpdef size_t sp_sm_buffer_size(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int compute_type, int alg, intptr_t spsm_descr) except? 0:
    """See `cusparseSpSM_bufferSize`."""
    cdef size_t buffer_size
    with nogil:
        status = cusparseSpSM_bufferSize(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnMatDescr>mat_b, <DnMatDescr>mat_c, <DataType>compute_type, <_SpSMAlg>alg, <SpSMDescr>spsm_descr, &buffer_size)
    check_status(status)
    return buffer_size


cpdef sp_sm_analysis(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int compute_type, int alg, intptr_t spsm_descr, intptr_t external_buffer):
    """See `cusparseSpSM_analysis`."""
    with nogil:
        status = cusparseSpSM_analysis(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnMatDescr>mat_b, <DnMatDescr>mat_c, <DataType>compute_type, <_SpSMAlg>alg, <SpSMDescr>spsm_descr, <void*>external_buffer)
    check_status(status)


cpdef sp_sm_solve(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int compute_type, int alg, intptr_t spsm_descr):
    """See `cusparseSpSM_solve`."""
    with nogil:
        status = cusparseSpSM_solve(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnMatDescr>mat_b, <DnMatDescr>mat_c, <DataType>compute_type, <_SpSMAlg>alg, <SpSMDescr>spsm_descr)
    check_status(status)


cpdef size_t sp_gemm_reuse_work_estimation(intptr_t handle, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int alg, intptr_t spgemm_descr, intptr_t external_buffer1) except? 0:
    """See `cusparseSpGEMMreuse_workEstimation`."""
    cdef size_t buffer_size1
    with nogil:
        status = cusparseSpGEMMreuse_workEstimation(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <ConstSpMatDescr>mat_a, <ConstSpMatDescr>mat_b, <SpMatDescr>mat_c, <_SpGEMMAlg>alg, <SpGEMMDescr>spgemm_descr, &buffer_size1, <void*>external_buffer1)
    check_status(status)
    return buffer_size1


cpdef tuple sp_gemm_reuse_nnz(intptr_t handle, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int alg, intptr_t spgemm_descr, intptr_t external_buffer2, intptr_t external_buffer3, intptr_t external_buffer4):
    """See `cusparseSpGEMMreuse_nnz`."""
    cdef size_t buffer_size2
    cdef size_t buffer_size3
    cdef size_t buffer_size4
    with nogil:
        status = cusparseSpGEMMreuse_nnz(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <ConstSpMatDescr>mat_a, <ConstSpMatDescr>mat_b, <SpMatDescr>mat_c, <_SpGEMMAlg>alg, <SpGEMMDescr>spgemm_descr, &buffer_size2, <void*>external_buffer2, &buffer_size3, <void*>external_buffer3, &buffer_size4, <void*>external_buffer4)
    check_status(status)
    return (buffer_size2, buffer_size3, buffer_size4)


cpdef size_t sp_gemm_reuse_copy(intptr_t handle, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int alg, intptr_t spgemm_descr, intptr_t external_buffer5) except? 0:
    """See `cusparseSpGEMMreuse_copy`."""
    cdef size_t buffer_size5
    with nogil:
        status = cusparseSpGEMMreuse_copy(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <ConstSpMatDescr>mat_a, <ConstSpMatDescr>mat_b, <SpMatDescr>mat_c, <_SpGEMMAlg>alg, <SpGEMMDescr>spgemm_descr, &buffer_size5, <void*>external_buffer5)
    check_status(status)
    return buffer_size5


cpdef sp_gemm_reuse_compute(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t spgemm_descr):
    """See `cusparseSpGEMMreuse_compute`."""
    with nogil:
        status = cusparseSpGEMMreuse_compute(<Handle>handle, <_Operation>op_a, <_Operation>op_b, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstSpMatDescr>mat_b, <const void*>beta, <SpMatDescr>mat_c, <DataType>compute_type, <_SpGEMMAlg>alg, <SpGEMMDescr>spgemm_descr)
    check_status(status)


cpdef logger_open_file(log_file):
    """See `cusparseLoggerOpenFile`."""
    if not isinstance(log_file, str):
        raise TypeError("log_file must be a Python str")
    cdef bytes _temp_log_file_ = (<str>log_file).encode()
    cdef char* _log_file_ = _temp_log_file_
    with nogil:
        status = cusparseLoggerOpenFile(<const char*>_log_file_)
    check_status(status)


cpdef logger_set_level(int level):
    """See `cusparseLoggerSetLevel`."""
    with nogil:
        status = cusparseLoggerSetLevel(level)
    check_status(status)


cpdef logger_set_mask(int mask):
    """See `cusparseLoggerSetMask`."""
    with nogil:
        status = cusparseLoggerSetMask(mask)
    check_status(status)


cpdef logger_force_disable():
    """See `cusparseLoggerForceDisable`."""
    with nogil:
        status = cusparseLoggerForceDisable()
    check_status(status)


cpdef tuple sp_mm_op_create_plan(intptr_t handle, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int compute_type, int alg, intptr_t add_operation_nvvm_buffer, size_t add_operation_buffer_size, intptr_t mul_operation_nvvm_buffer, size_t mul_operation_buffer_size, intptr_t epilogue_nvvm_buffer, size_t epilogue_buffer_size):
    """See `cusparseSpMMOp_createPlan`."""
    cdef SpMMOpPlan plan
    cdef size_t sp_mm_workspace_size
    with nogil:
        status = cusparseSpMMOp_createPlan(<Handle>handle, &plan, <_Operation>op_a, <_Operation>op_b, <ConstSpMatDescr>mat_a, <ConstDnMatDescr>mat_b, <DnMatDescr>mat_c, <DataType>compute_type, <_SpMMOpAlg>alg, <const void*>add_operation_nvvm_buffer, add_operation_buffer_size, <const void*>mul_operation_nvvm_buffer, mul_operation_buffer_size, <const void*>epilogue_nvvm_buffer, epilogue_buffer_size, &sp_mm_workspace_size)
    check_status(status)
    return (<intptr_t>plan, sp_mm_workspace_size)


cpdef sp_mm_op(intptr_t plan, intptr_t external_buffer):
    """See `cusparseSpMMOp`."""
    with nogil:
        status = cusparseSpMMOp(<SpMMOpPlan>plan, <void*>external_buffer)
    check_status(status)


cpdef sp_mm_op_destroy_plan(intptr_t plan):
    """See `cusparseSpMMOp_destroyPlan`."""
    with nogil:
        status = cusparseSpMMOp_destroyPlan(<SpMMOpPlan>plan)
    check_status(status)


cpdef tuple csc_get(intptr_t sp_mat_descr):
    """See `cusparseCscGet`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t nnz
    cdef void* csc_col_offsets
    cdef void* csc_row_ind
    cdef void* csc_values
    cdef _IndexType csc_col_offsets_type
    cdef _IndexType csc_row_ind_type
    cdef _IndexBase idx_base
    cdef DataType value_type
    with nogil:
        status = cusparseCscGet(<SpMatDescr>sp_mat_descr, &rows, &cols, &nnz, &csc_col_offsets, &csc_row_ind, &csc_values, &csc_col_offsets_type, &csc_row_ind_type, &idx_base, &value_type)
    check_status(status)
    return (rows, cols, nnz, <intptr_t>csc_col_offsets, <intptr_t>csc_row_ind, <intptr_t>csc_values, <int>csc_col_offsets_type, <int>csc_row_ind_type, <int>idx_base, <int>value_type)


cpdef intptr_t create_const_sp_vec(int64_t size, int64_t nnz, intptr_t indices, intptr_t values, int idx_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateConstSpVec`."""
    cdef ConstSpVecDescr sp_vec_descr
    with nogil:
        status = cusparseCreateConstSpVec(&sp_vec_descr, size, nnz, <const void*>indices, <const void*>values, <_IndexType>idx_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_vec_descr


cpdef tuple const_sp_vec_get(intptr_t sp_vec_descr):
    """See `cusparseConstSpVecGet`."""
    cdef int64_t size
    cdef int64_t nnz
    cdef const void* indices
    cdef const void* values
    cdef _IndexType idx_type
    cdef _IndexBase idx_base
    cdef DataType value_type
    with nogil:
        status = cusparseConstSpVecGet(<ConstSpVecDescr>sp_vec_descr, &size, &nnz, &indices, &values, &idx_type, &idx_base, &value_type)
    check_status(status)
    return (size, nnz, <intptr_t>indices, <intptr_t>values, <int>idx_type, <int>idx_base, <int>value_type)


cpdef intptr_t const_sp_vec_get_values(intptr_t sp_vec_descr) except? -1:
    """See `cusparseConstSpVecGetValues`."""
    cdef const void* values
    with nogil:
        status = cusparseConstSpVecGetValues(<ConstSpVecDescr>sp_vec_descr, &values)
    check_status(status)
    return <intptr_t>values


cpdef intptr_t create_const_dn_vec(int64_t size, intptr_t values, int value_type) except? 0:
    """See `cusparseCreateConstDnVec`."""
    cdef ConstDnVecDescr dn_vec_descr
    with nogil:
        status = cusparseCreateConstDnVec(&dn_vec_descr, size, <const void*>values, <DataType>value_type)
    check_status(status)
    return <intptr_t>dn_vec_descr


cpdef tuple const_dn_vec_get(intptr_t dn_vec_descr):
    """See `cusparseConstDnVecGet`."""
    cdef int64_t size
    cdef const void* values
    cdef DataType value_type
    with nogil:
        status = cusparseConstDnVecGet(<ConstDnVecDescr>dn_vec_descr, &size, &values, &value_type)
    check_status(status)
    return (size, <intptr_t>values, <int>value_type)


cpdef intptr_t const_dn_vec_get_values(intptr_t dn_vec_descr) except? -1:
    """See `cusparseConstDnVecGetValues`."""
    cdef const void* values
    with nogil:
        status = cusparseConstDnVecGetValues(<ConstDnVecDescr>dn_vec_descr, &values)
    check_status(status)
    return <intptr_t>values


cpdef intptr_t const_sp_mat_get_values(intptr_t sp_mat_descr) except? -1:
    """See `cusparseConstSpMatGetValues`."""
    cdef const void* values
    with nogil:
        status = cusparseConstSpMatGetValues(<ConstSpMatDescr>sp_mat_descr, &values)
    check_status(status)
    return <intptr_t>values


cpdef intptr_t create_const_csr(int64_t rows, int64_t cols, int64_t nnz, intptr_t csr_row_offsets, intptr_t csr_col_ind, intptr_t csr_values, int csr_row_offsets_type, int csr_col_ind_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateConstCsr`."""
    cdef ConstSpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateConstCsr(&sp_mat_descr, rows, cols, nnz, <const void*>csr_row_offsets, <const void*>csr_col_ind, <const void*>csr_values, <_IndexType>csr_row_offsets_type, <_IndexType>csr_col_ind_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef intptr_t create_const_csc(int64_t rows, int64_t cols, int64_t nnz, intptr_t csc_col_offsets, intptr_t csc_row_ind, intptr_t csc_values, int csc_col_offsets_type, int csc_row_ind_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateConstCsc`."""
    cdef ConstSpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateConstCsc(&sp_mat_descr, rows, cols, nnz, <const void*>csc_col_offsets, <const void*>csc_row_ind, <const void*>csc_values, <_IndexType>csc_col_offsets_type, <_IndexType>csc_row_ind_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef tuple const_csr_get(intptr_t sp_mat_descr):
    """See `cusparseConstCsrGet`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t nnz
    cdef const void* csr_row_offsets
    cdef const void* csr_col_ind
    cdef const void* csr_values
    cdef _IndexType csr_row_offsets_type
    cdef _IndexType csr_col_ind_type
    cdef _IndexBase idx_base
    cdef DataType value_type
    with nogil:
        status = cusparseConstCsrGet(<ConstSpMatDescr>sp_mat_descr, &rows, &cols, &nnz, &csr_row_offsets, &csr_col_ind, &csr_values, &csr_row_offsets_type, &csr_col_ind_type, &idx_base, &value_type)
    check_status(status)
    return (rows, cols, nnz, <intptr_t>csr_row_offsets, <intptr_t>csr_col_ind, <intptr_t>csr_values, <int>csr_row_offsets_type, <int>csr_col_ind_type, <int>idx_base, <int>value_type)


cpdef tuple const_csc_get(intptr_t sp_mat_descr):
    """See `cusparseConstCscGet`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t nnz
    cdef const void* csc_col_offsets
    cdef const void* csc_row_ind
    cdef const void* csc_values
    cdef _IndexType csc_col_offsets_type
    cdef _IndexType csc_row_ind_type
    cdef _IndexBase idx_base
    cdef DataType value_type
    with nogil:
        status = cusparseConstCscGet(<ConstSpMatDescr>sp_mat_descr, &rows, &cols, &nnz, &csc_col_offsets, &csc_row_ind, &csc_values, &csc_col_offsets_type, &csc_row_ind_type, &idx_base, &value_type)
    check_status(status)
    return (rows, cols, nnz, <intptr_t>csc_col_offsets, <intptr_t>csc_row_ind, <intptr_t>csc_values, <int>csc_col_offsets_type, <int>csc_row_ind_type, <int>idx_base, <int>value_type)


cpdef intptr_t create_const_coo(int64_t rows, int64_t cols, int64_t nnz, intptr_t coo_row_ind, intptr_t coo_col_ind, intptr_t coo_values, int coo_idx_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateConstCoo`."""
    cdef ConstSpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateConstCoo(&sp_mat_descr, rows, cols, nnz, <const void*>coo_row_ind, <const void*>coo_col_ind, <const void*>coo_values, <_IndexType>coo_idx_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef tuple const_coo_get(intptr_t sp_mat_descr):
    """See `cusparseConstCooGet`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t nnz
    cdef const void* coo_row_ind
    cdef const void* coo_col_ind
    cdef const void* coo_values
    cdef _IndexType idx_type
    cdef _IndexBase idx_base
    cdef DataType value_type
    with nogil:
        status = cusparseConstCooGet(<ConstSpMatDescr>sp_mat_descr, &rows, &cols, &nnz, &coo_row_ind, &coo_col_ind, &coo_values, &idx_type, &idx_base, &value_type)
    check_status(status)
    return (rows, cols, nnz, <intptr_t>coo_row_ind, <intptr_t>coo_col_ind, <intptr_t>coo_values, <int>idx_type, <int>idx_base, <int>value_type)


cpdef intptr_t create_const_blocked_ell(int64_t rows, int64_t cols, int64_t ell_block_size, int64_t ell_cols, intptr_t ell_col_ind, intptr_t ell_value, int ell_idx_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateConstBlockedEll`."""
    cdef ConstSpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateConstBlockedEll(&sp_mat_descr, rows, cols, ell_block_size, ell_cols, <const void*>ell_col_ind, <const void*>ell_value, <_IndexType>ell_idx_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef tuple const_blocked_ell_get(intptr_t sp_mat_descr):
    """See `cusparseConstBlockedEllGet`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t ell_block_size
    cdef int64_t ell_cols
    cdef const void* ell_col_ind
    cdef const void* ell_value
    cdef _IndexType ell_idx_type
    cdef _IndexBase idx_base
    cdef DataType value_type
    with nogil:
        status = cusparseConstBlockedEllGet(<ConstSpMatDescr>sp_mat_descr, &rows, &cols, &ell_block_size, &ell_cols, &ell_col_ind, &ell_value, &ell_idx_type, &idx_base, &value_type)
    check_status(status)
    return (rows, cols, ell_block_size, ell_cols, <intptr_t>ell_col_ind, <intptr_t>ell_value, <int>ell_idx_type, <int>idx_base, <int>value_type)


cpdef intptr_t create_const_dn_mat(int64_t rows, int64_t cols, int64_t ld, intptr_t values, int value_type, int order) except? 0:
    """See `cusparseCreateConstDnMat`."""
    cdef ConstDnMatDescr dn_mat_descr
    with nogil:
        status = cusparseCreateConstDnMat(&dn_mat_descr, rows, cols, ld, <const void*>values, <DataType>value_type, <_Order>order)
    check_status(status)
    return <intptr_t>dn_mat_descr


cpdef tuple const_dn_mat_get(intptr_t dn_mat_descr):
    """See `cusparseConstDnMatGet`."""
    cdef int64_t rows
    cdef int64_t cols
    cdef int64_t ld
    cdef const void* values
    cdef DataType type
    cdef _Order order
    with nogil:
        status = cusparseConstDnMatGet(<ConstDnMatDescr>dn_mat_descr, &rows, &cols, &ld, &values, &type, &order)
    check_status(status)
    return (rows, cols, ld, <intptr_t>values, <int>type, <int>order)


cpdef intptr_t const_dn_mat_get_values(intptr_t dn_mat_descr) except? -1:
    """See `cusparseConstDnMatGetValues`."""
    cdef const void* values
    with nogil:
        status = cusparseConstDnMatGetValues(<ConstDnMatDescr>dn_mat_descr, &values)
    check_status(status)
    return <intptr_t>values


cpdef int64_t sp_gemm_get_num_products(intptr_t spgemm_descr) except? -1:
    """See `cusparseSpGEMM_getNumProducts`."""
    cdef int64_t num_prods
    with nogil:
        status = cusparseSpGEMM_getNumProducts(<SpGEMMDescr>spgemm_descr, &num_prods)
    check_status(status)
    return num_prods


cpdef bsr_set_strided_batch(intptr_t sp_mat_descr, int batch_count, int64_t offsets_batch_stride, int64_t columns_batch_stride, int64_t values_batch_stride):
    """See `cusparseBsrSetStridedBatch`."""
    with nogil:
        status = cusparseBsrSetStridedBatch(<SpMatDescr>sp_mat_descr, batch_count, offsets_batch_stride, columns_batch_stride, values_batch_stride)
    check_status(status)


cpdef intptr_t create_bsr(int64_t brows, int64_t bcols, int64_t bnnz, int64_t row_block_size, int64_t col_block_size, intptr_t bsr_row_offsets, intptr_t bsr_col_ind, intptr_t bsr_values, int bsr_row_offsets_type, int bsr_col_ind_type, int idx_base, int value_type, int order) except? 0:
    """See `cusparseCreateBsr`."""
    cdef SpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateBsr(&sp_mat_descr, brows, bcols, bnnz, row_block_size, col_block_size, <void*>bsr_row_offsets, <void*>bsr_col_ind, <void*>bsr_values, <_IndexType>bsr_row_offsets_type, <_IndexType>bsr_col_ind_type, <_IndexBase>idx_base, <DataType>value_type, <_Order>order)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef intptr_t create_const_bsr(int64_t brows, int64_t bcols, int64_t bnnz, int64_t row_block_dim, int64_t col_block_dim, intptr_t bsr_row_offsets, intptr_t bsr_col_ind, intptr_t bsr_values, int bsr_row_offsets_type, int bsr_col_ind_type, int idx_base, int value_type, int order) except? 0:
    """See `cusparseCreateConstBsr`."""
    cdef ConstSpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateConstBsr(&sp_mat_descr, brows, bcols, bnnz, row_block_dim, col_block_dim, <const void*>bsr_row_offsets, <const void*>bsr_col_ind, <const void*>bsr_values, <_IndexType>bsr_row_offsets_type, <_IndexType>bsr_col_ind_type, <_IndexBase>idx_base, <DataType>value_type, <_Order>order)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef intptr_t create_sliced_ell(int64_t rows, int64_t cols, int64_t nnz, int64_t sell_values_size, int64_t slice_size, intptr_t sell_slice_offsets, intptr_t sell_col_ind, intptr_t sell_values, int sell_slice_offsets_type, int sell_col_ind_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateSlicedEll`."""
    cdef SpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateSlicedEll(&sp_mat_descr, rows, cols, nnz, sell_values_size, slice_size, <void*>sell_slice_offsets, <void*>sell_col_ind, <void*>sell_values, <_IndexType>sell_slice_offsets_type, <_IndexType>sell_col_ind_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef intptr_t create_const_sliced_ell(int64_t rows, int64_t cols, int64_t nnz, int64_t sell_values_size, int64_t slice_size, intptr_t sell_slice_offsets, intptr_t sell_col_ind, intptr_t sell_values, int sell_slice_offsets_type, int sell_col_ind_type, int idx_base, int value_type) except? 0:
    """See `cusparseCreateConstSlicedEll`."""
    cdef ConstSpMatDescr sp_mat_descr
    with nogil:
        status = cusparseCreateConstSlicedEll(&sp_mat_descr, rows, cols, nnz, sell_values_size, slice_size, <const void*>sell_slice_offsets, <const void*>sell_col_ind, <const void*>sell_values, <_IndexType>sell_slice_offsets_type, <_IndexType>sell_col_ind_type, <_IndexBase>idx_base, <DataType>value_type)
    check_status(status)
    return <intptr_t>sp_mat_descr


cpdef sp_sv_update_matrix(intptr_t handle, intptr_t spsv_descr, intptr_t new_values, int update_part):
    """See `cusparseSpSV_updateMatrix`."""
    with nogil:
        status = cusparseSpSV_updateMatrix(<Handle>handle, <SpSVDescr>spsv_descr, <void*>new_values, <_SpSVUpdate>update_part)
    check_status(status)


cpdef sp_mv_preprocess(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t beta, intptr_t vec_y, int compute_type, int alg, intptr_t external_buffer):
    """See `cusparseSpMV_preprocess`."""
    with nogil:
        status = cusparseSpMV_preprocess(<Handle>handle, <_Operation>op_a, <const void*>alpha, <ConstSpMatDescr>mat_a, <ConstDnVecDescr>vec_x, <const void*>beta, <DnVecDescr>vec_y, <DataType>compute_type, <_SpMVAlg>alg, <void*>external_buffer)
    check_status(status)


cpdef sp_sm_update_matrix(intptr_t handle, intptr_t spsm_descr, intptr_t new_values, int update_part):
    """See `cusparseSpSM_updateMatrix`."""
    with nogil:
        status = cusparseSpSM_updateMatrix(<Handle>handle, <SpSMDescr>spsm_descr, <void*>new_values, <_SpSMUpdate>update_part)
    check_status(status)
