# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

cimport cython

from libc.stdint cimport intptr_t

from .cycusparse cimport *


###############################################################################
# Types
###############################################################################

ctypedef cusparseHandle_t Handle
ctypedef cusparseMatDescr_t MatDescr
ctypedef cusparseSpVecDescr_t SpVecDescr
ctypedef cusparseDnVecDescr_t DnVecDescr
ctypedef cusparseSpMatDescr_t SpMatDescr
ctypedef cusparseDnMatDescr_t DnMatDescr
ctypedef cusparseSpGEMMDescr_t SpGEMMDescr
ctypedef cusparseSpSVDescr_t SpSVDescr
ctypedef cusparseSpSMDescr_t SpSMDescr
ctypedef cusparseSpMMOpPlan_t SpMMOpPlan
ctypedef cusparseConstSpVecDescr_t ConstSpVecDescr
ctypedef cusparseConstDnVecDescr_t ConstDnVecDescr
ctypedef cusparseConstSpMatDescr_t ConstSpMatDescr
ctypedef cusparseConstDnMatDescr_t ConstDnMatDescr
ctypedef cusparseLoggerCallback_t LoggerCallback

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef cusparseStatus_t _Status
ctypedef cusparsePointerMode_t _PointerMode
ctypedef cusparseAction_t _Action
ctypedef cusparseMatrixType_t _MatrixType
ctypedef cusparseFillMode_t _FillMode
ctypedef cusparseDiagType_t _DiagType
ctypedef cusparseIndexBase_t _IndexBase
ctypedef cusparseOperation_t _Operation
ctypedef cusparseDirection_t _Direction
ctypedef cusparseSolvePolicy_t _SolvePolicy
ctypedef cusparseColorAlg_t _ColorAlg
ctypedef cusparseCsr2CscAlg_t _Csr2CscAlg
ctypedef cusparseFormat_t _Format
ctypedef cusparseOrder_t _Order
ctypedef cusparseIndexType_t _IndexType
ctypedef cusparseSpMVAlg_t _SpMVAlg
ctypedef cusparseSpMMAlg_t _SpMMAlg
ctypedef cusparseSpGEMMAlg_t _SpGEMMAlg
ctypedef cusparseSparseToDenseAlg_t _SparseToDenseAlg
ctypedef cusparseDenseToSparseAlg_t _DenseToSparseAlg
ctypedef cusparseSDDMMAlg_t _SDDMMAlg
ctypedef cusparseSpMatAttribute_t _SpMatAttribute
ctypedef cusparseSpSVAlg_t _SpSVAlg
ctypedef cusparseSpSMAlg_t _SpSMAlg
ctypedef cusparseSpMMOpAlg_t _SpMMOpAlg
ctypedef cusparseSpSVUpdate_t _SpSVUpdate
ctypedef cusparseSpSMUpdate_t _SpSMUpdate


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef int get_version(intptr_t handle) except? -1
cpdef int get_property(int type) except? -1
cpdef str get_error_name(int status)
cpdef str get_error_string(int status)
cpdef set_stream(intptr_t handle, intptr_t stream_id)
cpdef intptr_t get_stream(intptr_t handle) except? 0
cpdef int get_pointer_mode(intptr_t handle) except? -1
cpdef set_pointer_mode(intptr_t handle, int mode)
cpdef intptr_t create_mat_descr() except? 0
cpdef destroy_mat_descr(intptr_t descr_a)
cpdef set_mat_type(intptr_t descr_a, int type)
cpdef int get_mat_type(intptr_t descr_a) except? -1
cpdef set_mat_fill_mode(intptr_t descr_a, int fill_mode)
cpdef int get_mat_fill_mode(intptr_t descr_a) except? -1
cpdef set_mat_diag_type(intptr_t descr_a, int diag_type)
cpdef int get_mat_diag_type(intptr_t descr_a) except? -1
cpdef set_mat_index_base(intptr_t descr_a, int base)
cpdef int get_mat_index_base(intptr_t descr_a) except? -1
cpdef sgemvi(intptr_t handle, int trans_a, int m, int n, intptr_t alpha, intptr_t a, int lda, int nnz, intptr_t x_val, intptr_t x_ind, intptr_t beta, intptr_t y, int idx_base, intptr_t p_buffer)
cpdef int sgemvi_buffer_size(intptr_t handle, int trans_a, int m, int n, int nnz) except? -1
cpdef dgemvi(intptr_t handle, int trans_a, int m, int n, intptr_t alpha, intptr_t a, int lda, int nnz, intptr_t x_val, intptr_t x_ind, intptr_t beta, intptr_t y, int idx_base, intptr_t p_buffer)
cpdef int dgemvi_buffer_size(intptr_t handle, int trans_a, int m, int n, int nnz) except? -1
cpdef cgemvi(intptr_t handle, int trans_a, int m, int n, intptr_t alpha, intptr_t a, int lda, int nnz, intptr_t x_val, intptr_t x_ind, intptr_t beta, intptr_t y, int idx_base, intptr_t p_buffer)
cpdef int cgemvi_buffer_size(intptr_t handle, int trans_a, int m, int n, int nnz) except? -1
cpdef zgemvi(intptr_t handle, int trans_a, int m, int n, intptr_t alpha, intptr_t a, int lda, int nnz, intptr_t x_val, intptr_t x_ind, intptr_t beta, intptr_t y, int idx_base, intptr_t p_buffer)
cpdef int zgemvi_buffer_size(intptr_t handle, int trans_a, int m, int n, int nnz) except? -1
cpdef sbsrmv(intptr_t handle, int dir_a, int trans_a, int mb, int nb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t x, intptr_t beta, intptr_t y)
cpdef dbsrmv(intptr_t handle, int dir_a, int trans_a, int mb, int nb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t x, intptr_t beta, intptr_t y)
cpdef cbsrmv(intptr_t handle, int dir_a, int trans_a, int mb, int nb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t x, intptr_t beta, intptr_t y)
cpdef zbsrmv(intptr_t handle, int dir_a, int trans_a, int mb, int nb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t x, intptr_t beta, intptr_t y)
cpdef sbsrmm(intptr_t handle, int dir_a, int trans_a, int trans_b, int mb, int n, int kb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_size, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc)
cpdef dbsrmm(intptr_t handle, int dir_a, int trans_a, int trans_b, int mb, int n, int kb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_size, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc)
cpdef cbsrmm(intptr_t handle, int dir_a, int trans_a, int trans_b, int mb, int n, int kb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_size, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc)
cpdef zbsrmm(intptr_t handle, int dir_a, int trans_a, int trans_b, int mb, int n, int kb, int nnzb, intptr_t alpha, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_size, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc)
cpdef size_t sgtsv2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0
cpdef size_t dgtsv2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0
cpdef size_t cgtsv2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0
cpdef size_t zgtsv2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0
cpdef sgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer)
cpdef dgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer)
cpdef cgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer)
cpdef zgtsv2(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer)
cpdef size_t sgtsv2_nopivot_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0
cpdef size_t dgtsv2_nopivot_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0
cpdef size_t cgtsv2_nopivot_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0
cpdef size_t zgtsv2_nopivot_buffer_size_ext(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb) except? 0
cpdef sgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer)
cpdef dgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer)
cpdef cgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer)
cpdef zgtsv2_nopivot(intptr_t handle, int m, int n, intptr_t dl, intptr_t d, intptr_t du, intptr_t b, int ldb, intptr_t p_buffer)
cpdef size_t sgtsv2strided_batch_buffer_size_ext(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride) except? 0
cpdef size_t dgtsv2strided_batch_buffer_size_ext(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride) except? 0
cpdef size_t cgtsv2strided_batch_buffer_size_ext(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride) except? 0
cpdef size_t zgtsv2strided_batch_buffer_size_ext(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride) except? 0
cpdef sgtsv2strided_batch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride, intptr_t p_buffer)
cpdef dgtsv2strided_batch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride, intptr_t p_buffer)
cpdef cgtsv2strided_batch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride, intptr_t p_buffer)
cpdef zgtsv2strided_batch(intptr_t handle, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, int batch_stride, intptr_t p_buffer)
cpdef size_t sgtsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count) except? 0
cpdef size_t dgtsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count) except? 0
cpdef size_t cgtsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count) except? 0
cpdef size_t zgtsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count) except? 0
cpdef sgtsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, intptr_t p_buffer)
cpdef dgtsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, intptr_t p_buffer)
cpdef cgtsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, intptr_t p_buffer)
cpdef zgtsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t dl, intptr_t d, intptr_t du, intptr_t x, int batch_count, intptr_t p_buffer)
cpdef size_t sgpsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count) except? 0
cpdef size_t dgpsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count) except? 0
cpdef size_t cgpsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count) except? 0
cpdef size_t zgpsv_interleaved_batch_buffer_size_ext(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count) except? 0
cpdef sgpsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count, intptr_t p_buffer)
cpdef dgpsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count, intptr_t p_buffer)
cpdef cgpsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count, intptr_t p_buffer)
cpdef zgpsv_interleaved_batch(intptr_t handle, int algo, int m, intptr_t ds, intptr_t dl, intptr_t d, intptr_t du, intptr_t dw, intptr_t x, int batch_count, intptr_t p_buffer)
cpdef size_t scsrgeam2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c) except? 0
cpdef size_t dcsrgeam2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c) except? 0
cpdef size_t ccsrgeam2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c) except? 0
cpdef size_t zcsrgeam2_buffer_size_ext(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c) except? 0
cpdef xcsrgeam2nnz(intptr_t handle, int m, int n, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_row_ptr_c, intptr_t nnz_total_dev_host_ptr, intptr_t workspace)
cpdef scsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c, intptr_t p_buffer)
cpdef dcsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c, intptr_t p_buffer)
cpdef ccsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c, intptr_t p_buffer)
cpdef zcsrgeam2(intptr_t handle, int m, int n, intptr_t alpha, intptr_t descr_a, int nnz_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t beta, intptr_t descr_b, int nnz_b, intptr_t csr_sorted_val_b, intptr_t csr_sorted_row_ptr_b, intptr_t csr_sorted_col_ind_b, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c, intptr_t p_buffer)
cpdef snnz(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t a, int lda, intptr_t nnz_per_row_col, intptr_t nnz_total_dev_host_ptr)
cpdef dnnz(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t a, int lda, intptr_t nnz_per_row_col, intptr_t nnz_total_dev_host_ptr)
cpdef cnnz(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t a, int lda, intptr_t nnz_per_row_col, intptr_t nnz_total_dev_host_ptr)
cpdef znnz(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t a, int lda, intptr_t nnz_per_row_col, intptr_t nnz_total_dev_host_ptr)
cpdef xcoo2csr(intptr_t handle, intptr_t coo_row_ind, int nnz, int m, intptr_t csr_sorted_row_ptr, int idx_base)
cpdef xcsr2coo(intptr_t handle, intptr_t csr_sorted_row_ptr, int nnz, int m, intptr_t coo_row_ind, int idx_base)
cpdef sbsr2csr(intptr_t handle, int dir_a, int mb, int nb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c)
cpdef dbsr2csr(intptr_t handle, int dir_a, int mb, int nb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c)
cpdef cbsr2csr(intptr_t handle, int dir_a, int mb, int nb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c)
cpdef zbsr2csr(intptr_t handle, int dir_a, int mb, int nb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int block_dim, intptr_t descr_c, intptr_t csr_sorted_val_c, intptr_t csr_sorted_row_ptr_c, intptr_t csr_sorted_col_ind_c)
cpdef int sgebsr2gebsc_buffer_size(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? -1
cpdef int dgebsr2gebsc_buffer_size(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? -1
cpdef int cgebsr2gebsc_buffer_size(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? -1
cpdef int zgebsr2gebsc_buffer_size(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? -1
cpdef size_t sgebsr2gebsc_buffer_size_ext(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? 0
cpdef size_t dgebsr2gebsc_buffer_size_ext(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? 0
cpdef size_t cgebsr2gebsc_buffer_size_ext(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? 0
cpdef size_t zgebsr2gebsc_buffer_size_ext(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim) except? 0
cpdef sgebsr2gebsc(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim, intptr_t bsc_val, intptr_t bsc_row_ind, intptr_t bsc_col_ptr, int copy_values, int idx_base, intptr_t p_buffer)
cpdef dgebsr2gebsc(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim, intptr_t bsc_val, intptr_t bsc_row_ind, intptr_t bsc_col_ptr, int copy_values, int idx_base, intptr_t p_buffer)
cpdef cgebsr2gebsc(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim, intptr_t bsc_val, intptr_t bsc_row_ind, intptr_t bsc_col_ptr, int copy_values, int idx_base, intptr_t p_buffer)
cpdef zgebsr2gebsc(intptr_t handle, int mb, int nb, int nnzb, intptr_t bsr_sorted_val, intptr_t bsr_sorted_row_ptr, intptr_t bsr_sorted_col_ind, int row_block_dim, int col_block_dim, intptr_t bsc_val, intptr_t bsc_row_ind, intptr_t bsc_col_ptr, int copy_values, int idx_base, intptr_t p_buffer)
cpdef int scsr2gebsr_buffer_size(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? -1
cpdef int dcsr2gebsr_buffer_size(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? -1
cpdef int ccsr2gebsr_buffer_size(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? -1
cpdef int zcsr2gebsr_buffer_size(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? -1
cpdef size_t scsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? 0
cpdef size_t dcsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? 0
cpdef size_t ccsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? 0
cpdef size_t zcsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, int row_block_dim, int col_block_dim) except? 0
cpdef xcsr2gebsr_nnz(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_c, intptr_t bsr_sorted_row_ptr_c, int row_block_dim, int col_block_dim, intptr_t nnz_total_dev_host_ptr, intptr_t p_buffer)
cpdef scsr2gebsr(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim, int col_block_dim, intptr_t p_buffer)
cpdef dcsr2gebsr(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim, int col_block_dim, intptr_t p_buffer)
cpdef ccsr2gebsr(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim, int col_block_dim, intptr_t p_buffer)
cpdef zcsr2gebsr(intptr_t handle, int dir_a, int m, int n, intptr_t descr_a, intptr_t csr_sorted_val_a, intptr_t csr_sorted_row_ptr_a, intptr_t csr_sorted_col_ind_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim, int col_block_dim, intptr_t p_buffer)
cpdef int sgebsr2gebsr_buffer_size(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? -1
cpdef int dgebsr2gebsr_buffer_size(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? -1
cpdef int cgebsr2gebsr_buffer_size(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? -1
cpdef int zgebsr2gebsr_buffer_size(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? -1
cpdef size_t sgebsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? 0
cpdef size_t dgebsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? 0
cpdef size_t cgebsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? 0
cpdef size_t zgebsr2gebsr_buffer_size_ext(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, int row_block_dim_c, int col_block_dim_c) except? 0
cpdef xgebsr2gebsr_nnz(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, intptr_t descr_c, intptr_t bsr_sorted_row_ptr_c, int row_block_dim_c, int col_block_dim_c, intptr_t nnz_total_dev_host_ptr, intptr_t p_buffer)
cpdef sgebsr2gebsr(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim_c, int col_block_dim_c, intptr_t p_buffer)
cpdef dgebsr2gebsr(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim_c, int col_block_dim_c, intptr_t p_buffer)
cpdef cgebsr2gebsr(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim_c, int col_block_dim_c, intptr_t p_buffer)
cpdef zgebsr2gebsr(intptr_t handle, int dir_a, int mb, int nb, int nnzb, intptr_t descr_a, intptr_t bsr_sorted_val_a, intptr_t bsr_sorted_row_ptr_a, intptr_t bsr_sorted_col_ind_a, int row_block_dim_a, int col_block_dim_a, intptr_t descr_c, intptr_t bsr_sorted_val_c, intptr_t bsr_sorted_row_ptr_c, intptr_t bsr_sorted_col_ind_c, int row_block_dim_c, int col_block_dim_c, intptr_t p_buffer)
cpdef size_t xcoosort_buffer_size_ext(intptr_t handle, int m, int n, int nnz, intptr_t coo_rows_a, intptr_t coo_cols_a) except? 0
cpdef xcoosort_by_row(intptr_t handle, int m, int n, int nnz, intptr_t coo_rows_a, intptr_t coo_cols_a, intptr_t p, intptr_t p_buffer)
cpdef xcoosort_by_column(intptr_t handle, int m, int n, int nnz, intptr_t coo_rows_a, intptr_t coo_cols_a, intptr_t p, intptr_t p_buffer)
cpdef size_t xcsrsort_buffer_size_ext(intptr_t handle, int m, int n, int nnz, intptr_t csr_row_ptr_a, intptr_t csr_col_ind_a) except? 0
cpdef xcsrsort(intptr_t handle, int m, int n, int nnz, intptr_t descr_a, intptr_t csr_row_ptr_a, intptr_t csr_col_ind_a, intptr_t p, intptr_t p_buffer)
cpdef size_t xcscsort_buffer_size_ext(intptr_t handle, int m, int n, int nnz, intptr_t csc_col_ptr_a, intptr_t csc_row_ind_a) except? 0
cpdef xcscsort(intptr_t handle, int m, int n, int nnz, intptr_t descr_a, intptr_t csc_col_ptr_a, intptr_t csc_row_ind_a, intptr_t p, intptr_t p_buffer)
cpdef csr2csc_ex2(intptr_t handle, int m, int n, int nnz, intptr_t csr_val, intptr_t csr_row_ptr, intptr_t csr_col_ind, intptr_t csc_val, intptr_t csc_col_ptr, intptr_t csc_row_ind, int val_type, int copy_values, int idx_base, int alg, intptr_t buffer)
cpdef size_t csr2csc_ex2_buffer_size(intptr_t handle, int m, int n, int nnz, intptr_t csr_val, intptr_t csr_row_ptr, intptr_t csr_col_ind, intptr_t csc_val, intptr_t csc_col_ptr, intptr_t csc_row_ind, int val_type, int copy_values, int idx_base, int alg) except? 0
cpdef intptr_t create_sp_vec(int64_t size, int64_t nnz, intptr_t indices, intptr_t values, int idx_type, int idx_base, int value_type) except? 0
cpdef destroy_sp_vec(intptr_t sp_vec_descr)
cpdef tuple sp_vec_get(intptr_t sp_vec_descr)
cpdef int sp_vec_get_index_base(intptr_t sp_vec_descr) except? -1
cpdef intptr_t sp_vec_get_values(intptr_t sp_vec_descr) except? -1
cpdef sp_vec_set_values(intptr_t sp_vec_descr, intptr_t values)
cpdef intptr_t create_dn_vec(int64_t size, intptr_t values, int value_type) except? 0
cpdef destroy_dn_vec(intptr_t dn_vec_descr)
cpdef tuple dn_vec_get(intptr_t dn_vec_descr)
cpdef intptr_t dn_vec_get_values(intptr_t dn_vec_descr) except? -1
cpdef dn_vec_set_values(intptr_t dn_vec_descr, intptr_t values)
cpdef destroy_sp_mat(intptr_t sp_mat_descr)
cpdef int sp_mat_get_format(intptr_t sp_mat_descr) except? -1
cpdef int sp_mat_get_index_base(intptr_t sp_mat_descr) except? -1
cpdef intptr_t sp_mat_get_values(intptr_t sp_mat_descr) except? -1
cpdef sp_mat_set_values(intptr_t sp_mat_descr, intptr_t values)
cpdef tuple sp_mat_get_size(intptr_t sp_mat_descr)
cpdef int sp_mat_get_strided_batch(intptr_t sp_mat_descr) except? -1
cpdef coo_set_strided_batch(intptr_t sp_mat_descr, int batch_count, int64_t batch_stride)
cpdef csr_set_strided_batch(intptr_t sp_mat_descr, int batch_count, int64_t offsets_batch_stride, int64_t columns_values_batch_stride)
cpdef intptr_t create_csr(int64_t rows, int64_t cols, int64_t nnz, intptr_t csr_row_offsets, intptr_t csr_col_ind, intptr_t csr_values, int csr_row_offsets_type, int csr_col_ind_type, int idx_base, int value_type) except? 0
cpdef tuple csr_get(intptr_t sp_mat_descr)
cpdef csr_set_pointers(intptr_t sp_mat_descr, intptr_t csr_row_offsets, intptr_t csr_col_ind, intptr_t csr_values)
cpdef intptr_t create_coo(int64_t rows, int64_t cols, int64_t nnz, intptr_t coo_row_ind, intptr_t coo_col_ind, intptr_t coo_values, int coo_idx_type, int idx_base, int value_type) except? 0
cpdef tuple coo_get(intptr_t sp_mat_descr)
cpdef intptr_t create_dn_mat(int64_t rows, int64_t cols, int64_t ld, intptr_t values, int value_type, int order) except? 0
cpdef destroy_dn_mat(intptr_t dn_mat_descr)
cpdef tuple dn_mat_get(intptr_t dn_mat_descr)
cpdef intptr_t dn_mat_get_values(intptr_t dn_mat_descr) except? -1
cpdef dn_mat_set_values(intptr_t dn_mat_descr, intptr_t values)
cpdef dn_mat_set_strided_batch(intptr_t dn_mat_descr, int batch_count, int64_t batch_stride)
cpdef tuple dn_mat_get_strided_batch(intptr_t dn_mat_descr)
cpdef axpby(intptr_t handle, intptr_t alpha, intptr_t vec_x, intptr_t beta, intptr_t vec_y)
cpdef gather(intptr_t handle, intptr_t vec_y, intptr_t vec_x)
cpdef scatter(intptr_t handle, intptr_t vec_x, intptr_t vec_y)
cpdef size_t sp_vv_buffer_size(intptr_t handle, int op_x, intptr_t vec_x, intptr_t vec_y, intptr_t result, int compute_type) except? 0
cpdef sp_vv(intptr_t handle, int op_x, intptr_t vec_x, intptr_t vec_y, intptr_t result, int compute_type, intptr_t external_buffer)
cpdef sp_mv(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t beta, intptr_t vec_y, int compute_type, int alg, intptr_t external_buffer)
cpdef size_t sp_mv_buffer_size(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t beta, intptr_t vec_y, int compute_type, int alg) except? 0
cpdef sp_mm(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t external_buffer)
cpdef size_t sp_mm_buffer_size(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg) except? 0
cpdef intptr_t sp_gemm_create_descr() except? 0
cpdef sp_gemm_destroy_descr(intptr_t descr)
cpdef size_t sp_gemm_work_estimation(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t spgemm_descr, intptr_t external_buffer1) except? 0
cpdef sp_gemm_compute(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t spgemm_descr, intptr_t buffer_size2, intptr_t external_buffer2)
cpdef sp_gemm_copy(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t spgemm_descr)
cpdef intptr_t create_csc(int64_t rows, int64_t cols, int64_t nnz, intptr_t csc_col_offsets, intptr_t csc_row_ind, intptr_t csc_values, int csc_col_offsets_type, int csc_row_ind_type, int idx_base, int value_type) except? 0
cpdef csc_set_pointers(intptr_t sp_mat_descr, intptr_t csc_col_offsets, intptr_t csc_row_ind, intptr_t csc_values)
cpdef coo_set_pointers(intptr_t sp_mat_descr, intptr_t coo_rows, intptr_t coo_columns, intptr_t coo_values)
cpdef size_t sparse_to_dense_buffer_size(intptr_t handle, intptr_t mat_a, intptr_t mat_b, int alg) except? 0
cpdef sparse_to_dense(intptr_t handle, intptr_t mat_a, intptr_t mat_b, int alg, intptr_t external_buffer)
cpdef size_t dense_to_sparse_buffer_size(intptr_t handle, intptr_t mat_a, intptr_t mat_b, int alg) except? 0
cpdef dense_to_sparse_analysis(intptr_t handle, intptr_t mat_a, intptr_t mat_b, int alg, intptr_t external_buffer)
cpdef dense_to_sparse_convert(intptr_t handle, intptr_t mat_a, intptr_t mat_b, int alg, intptr_t external_buffer)
cpdef intptr_t create_blocked_ell(int64_t rows, int64_t cols, int64_t ell_block_size, int64_t ell_cols, intptr_t ell_col_ind, intptr_t ell_value, int ell_idx_type, int idx_base, int value_type) except? 0
cpdef tuple blocked_ell_get(intptr_t sp_mat_descr)
cpdef sp_mm_preprocess(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t external_buffer)
cpdef sddmm_buffer_size(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t buffer_size)
cpdef sddmm_preprocess(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t external_buffer)
cpdef sddmm(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t external_buffer)
cpdef get_sp_mat_attribute_dtype(int attr)
cpdef sp_mat_get_attribute(intptr_t sp_mat_descr, int attribute, intptr_t data, size_t data_size)
cpdef sp_mat_set_attribute(intptr_t sp_mat_descr, int attribute, intptr_t data, size_t data_size)
cpdef intptr_t sp_sv_create_descr() except? 0
cpdef sp_sv_destroy_descr(intptr_t descr)
cpdef size_t sp_sv_buffer_size(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t vec_y, int compute_type, int alg, intptr_t spsv_descr) except? 0
cpdef sp_sv_analysis(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t vec_y, int compute_type, int alg, intptr_t spsv_descr, intptr_t external_buffer)
cpdef sp_sv_solve(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t vec_y, int compute_type, int alg, intptr_t spsv_descr)
cpdef intptr_t sp_sm_create_descr() except? 0
cpdef sp_sm_destroy_descr(intptr_t descr)
cpdef size_t sp_sm_buffer_size(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int compute_type, int alg, intptr_t spsm_descr) except? 0
cpdef sp_sm_analysis(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int compute_type, int alg, intptr_t spsm_descr, intptr_t external_buffer)
cpdef sp_sm_solve(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int compute_type, int alg, intptr_t spsm_descr)
cpdef size_t sp_gemm_reuse_work_estimation(intptr_t handle, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int alg, intptr_t spgemm_descr, intptr_t external_buffer1) except? 0
cpdef tuple sp_gemm_reuse_nnz(intptr_t handle, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int alg, intptr_t spgemm_descr, intptr_t external_buffer2, intptr_t external_buffer3, intptr_t external_buffer4)
cpdef size_t sp_gemm_reuse_copy(intptr_t handle, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int alg, intptr_t spgemm_descr, intptr_t external_buffer5) except? 0
cpdef sp_gemm_reuse_compute(intptr_t handle, int op_a, int op_b, intptr_t alpha, intptr_t mat_a, intptr_t mat_b, intptr_t beta, intptr_t mat_c, int compute_type, int alg, intptr_t spgemm_descr)
cpdef logger_open_file(log_file)
cpdef logger_set_level(int level)
cpdef logger_set_mask(int mask)
cpdef logger_force_disable()
cpdef tuple sp_mm_op_create_plan(intptr_t handle, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, int compute_type, int alg, intptr_t add_operation_nvvm_buffer, size_t add_operation_buffer_size, intptr_t mul_operation_nvvm_buffer, size_t mul_operation_buffer_size, intptr_t epilogue_nvvm_buffer, size_t epilogue_buffer_size)
cpdef sp_mm_op(intptr_t plan, intptr_t external_buffer)
cpdef sp_mm_op_destroy_plan(intptr_t plan)
cpdef tuple csc_get(intptr_t sp_mat_descr)
cpdef intptr_t create_const_sp_vec(int64_t size, int64_t nnz, intptr_t indices, intptr_t values, int idx_type, int idx_base, int value_type) except? 0
cpdef tuple const_sp_vec_get(intptr_t sp_vec_descr)
cpdef intptr_t const_sp_vec_get_values(intptr_t sp_vec_descr) except? -1
cpdef intptr_t create_const_dn_vec(int64_t size, intptr_t values, int value_type) except? 0
cpdef tuple const_dn_vec_get(intptr_t dn_vec_descr)
cpdef intptr_t const_dn_vec_get_values(intptr_t dn_vec_descr) except? -1
cpdef intptr_t const_sp_mat_get_values(intptr_t sp_mat_descr) except? -1
cpdef intptr_t create_const_csr(int64_t rows, int64_t cols, int64_t nnz, intptr_t csr_row_offsets, intptr_t csr_col_ind, intptr_t csr_values, int csr_row_offsets_type, int csr_col_ind_type, int idx_base, int value_type) except? 0
cpdef intptr_t create_const_csc(int64_t rows, int64_t cols, int64_t nnz, intptr_t csc_col_offsets, intptr_t csc_row_ind, intptr_t csc_values, int csc_col_offsets_type, int csc_row_ind_type, int idx_base, int value_type) except? 0
cpdef tuple const_csr_get(intptr_t sp_mat_descr)
cpdef tuple const_csc_get(intptr_t sp_mat_descr)
cpdef intptr_t create_const_coo(int64_t rows, int64_t cols, int64_t nnz, intptr_t coo_row_ind, intptr_t coo_col_ind, intptr_t coo_values, int coo_idx_type, int idx_base, int value_type) except? 0
cpdef tuple const_coo_get(intptr_t sp_mat_descr)
cpdef intptr_t create_const_blocked_ell(int64_t rows, int64_t cols, int64_t ell_block_size, int64_t ell_cols, intptr_t ell_col_ind, intptr_t ell_value, int ell_idx_type, int idx_base, int value_type) except? 0
cpdef tuple const_blocked_ell_get(intptr_t sp_mat_descr)
cpdef intptr_t create_const_dn_mat(int64_t rows, int64_t cols, int64_t ld, intptr_t values, int value_type, int order) except? 0
cpdef tuple const_dn_mat_get(intptr_t dn_mat_descr)
cpdef intptr_t const_dn_mat_get_values(intptr_t dn_mat_descr) except? -1
cpdef int64_t sp_gemm_get_num_products(intptr_t spgemm_descr) except? -1
cpdef bsr_set_strided_batch(intptr_t sp_mat_descr, int batch_count, int64_t offsets_batch_stride, int64_t columns_batch_stride, int64_t values_batch_stride)
cpdef intptr_t create_bsr(int64_t brows, int64_t bcols, int64_t bnnz, int64_t row_block_size, int64_t col_block_size, intptr_t bsr_row_offsets, intptr_t bsr_col_ind, intptr_t bsr_values, int bsr_row_offsets_type, int bsr_col_ind_type, int idx_base, int value_type, int order) except? 0
cpdef intptr_t create_const_bsr(int64_t brows, int64_t bcols, int64_t bnnz, int64_t row_block_dim, int64_t col_block_dim, intptr_t bsr_row_offsets, intptr_t bsr_col_ind, intptr_t bsr_values, int bsr_row_offsets_type, int bsr_col_ind_type, int idx_base, int value_type, int order) except? 0
cpdef intptr_t create_sliced_ell(int64_t rows, int64_t cols, int64_t nnz, int64_t sell_values_size, int64_t slice_size, intptr_t sell_slice_offsets, intptr_t sell_col_ind, intptr_t sell_values, int sell_slice_offsets_type, int sell_col_ind_type, int idx_base, int value_type) except? 0
cpdef intptr_t create_const_sliced_ell(int64_t rows, int64_t cols, int64_t nnz, int64_t sell_values_size, int64_t slice_size, intptr_t sell_slice_offsets, intptr_t sell_col_ind, intptr_t sell_values, int sell_slice_offsets_type, int sell_col_ind_type, int idx_base, int value_type) except? 0
cpdef sp_sv_update_matrix(intptr_t handle, intptr_t spsv_descr, intptr_t new_values, int update_part)
cpdef sp_mv_preprocess(intptr_t handle, int op_a, intptr_t alpha, intptr_t mat_a, intptr_t vec_x, intptr_t beta, intptr_t vec_y, int compute_type, int alg, intptr_t external_buffer)
cpdef sp_sm_update_matrix(intptr_t handle, intptr_t spsm_descr, intptr_t new_values, int update_part)
