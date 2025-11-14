# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.4.1. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int64_t, int32_t

###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum CBLAS_ORDER "CBLAS_ORDER":
    CblasRowMajor "CblasRowMajor" = 101
    CblasColMajor "CblasColMajor" = 102

ctypedef enum CBLAS_TRANSPOSE "CBLAS_TRANSPOSE":
    CblasNoTrans "CblasNoTrans" = 111
    CblasTrans "CblasTrans" = 112
    CblasConjTrans "CblasConjTrans" = 113

ctypedef enum CBLAS_UPLO "CBLAS_UPLO":
    CblasUpper "CblasUpper" = 121
    CblasLower "CblasLower" = 122

ctypedef enum CBLAS_DIAG "CBLAS_DIAG":
    CblasNonUnit "CblasNonUnit" = 131
    CblasUnit "CblasUnit" = 132

ctypedef enum CBLAS_SIDE "CBLAS_SIDE":
    CblasLeft "CblasLeft" = 141
    CblasRight "CblasRight" = 142


# types
ctypedef int64_t nvpl_int64_t 'nvpl_int64_t'
ctypedef int32_t nvpl_int32_t 'nvpl_int32_t'
ctypedef struct nvpl_scomplex_t 'nvpl_scomplex_t':
    float real
    float imag
ctypedef struct nvpl_dcomplex_t 'nvpl_dcomplex_t':
    double real
    double imag
ctypedef nvpl_int64_t nvpl_int_t 'nvpl_int_t'


###############################################################################
# Functions
###############################################################################

cdef int MKL_mkl_set_num_threads_local(int nth) except?-42 nogil
cdef void MKL_mkl_set_num_threads(int nth) except* nogil
cdef void openblas_openblas_set_num_threads(int num_threads) except* nogil
cdef int openblas_openblas_set_num_threads_local(int num_threads) except?-42 nogil
cdef int nvpl_blas_get_version() except?-42 nogil
cdef int nvpl_blas_get_max_threads() except?-42 nogil
cdef void nvpl_blas_set_num_threads(int nthr) except* nogil
cdef int nvpl_blas_set_num_threads_local(int nthr_local) except?-42 nogil
cdef void cblas_sgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const float alpha, const float* A, const nvpl_int_t lda, const float* X, const nvpl_int_t incX, const float beta, float* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_sgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t KL, const nvpl_int_t KU, const float alpha, const float* A, const nvpl_int_t lda, const float* X, const nvpl_int_t incX, const float beta, float* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_strmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const float* A, const nvpl_int_t lda, float* X, const nvpl_int_t incX) except* nogil
cdef void cblas_stbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const float* A, const nvpl_int_t lda, float* X, const nvpl_int_t incX) except* nogil
cdef void cblas_stpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const float* Ap, float* X, const nvpl_int_t incX) except* nogil
cdef void cblas_strsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const float* A, const nvpl_int_t lda, float* X, const nvpl_int_t incX) except* nogil
cdef void cblas_stbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const float* A, const nvpl_int_t lda, float* X, const nvpl_int_t incX) except* nogil
cdef void cblas_stpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const float* Ap, float* X, const nvpl_int_t incX) except* nogil
cdef void cblas_dgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const double alpha, const double* A, const nvpl_int_t lda, const double* X, const nvpl_int_t incX, const double beta, double* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_dgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t KL, const nvpl_int_t KU, const double alpha, const double* A, const nvpl_int_t lda, const double* X, const nvpl_int_t incX, const double beta, double* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_dtrmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const double* A, const nvpl_int_t lda, double* X, const nvpl_int_t incX) except* nogil
cdef void cblas_dtbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const double* A, const nvpl_int_t lda, double* X, const nvpl_int_t incX) except* nogil
cdef void cblas_dtpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const double* Ap, double* X, const nvpl_int_t incX) except* nogil
cdef void cblas_dtrsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const double* A, const nvpl_int_t lda, double* X, const nvpl_int_t incX) except* nogil
cdef void cblas_dtbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const double* A, const nvpl_int_t lda, double* X, const nvpl_int_t incX) except* nogil
cdef void cblas_dtpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const double* Ap, double* X, const nvpl_int_t incX) except* nogil
cdef void cblas_cgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_cgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t KL, const nvpl_int_t KU, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_ctrmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ctbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ctpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* Ap, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ctrsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ctbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ctpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* Ap, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_zgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_zgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t KL, const nvpl_int_t KU, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_ztrmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ztbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ztpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* Ap, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ztrsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ztbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ztpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* Ap, void* X, const nvpl_int_t incX) except* nogil
cdef void cblas_ssymv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* A, const nvpl_int_t lda, const float* X, const nvpl_int_t incX, const float beta, float* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_ssbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const float* A, const nvpl_int_t lda, const float* X, const nvpl_int_t incX, const float beta, float* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_sspmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* Ap, const float* X, const nvpl_int_t incX, const float beta, float* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_sger(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const float alpha, const float* X, const nvpl_int_t incX, const float* Y, const nvpl_int_t incY, float* A, const nvpl_int_t lda) except* nogil
cdef void cblas_ssyr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* X, const nvpl_int_t incX, float* A, const nvpl_int_t lda) except* nogil
cdef void cblas_sspr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* X, const nvpl_int_t incX, float* Ap) except* nogil
cdef void cblas_ssyr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* X, const nvpl_int_t incX, const float* Y, const nvpl_int_t incY, float* A, const nvpl_int_t lda) except* nogil
cdef void cblas_sspr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* X, const nvpl_int_t incX, const float* Y, const nvpl_int_t incY, float* A) except* nogil
cdef void cblas_dsymv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* A, const nvpl_int_t lda, const double* X, const nvpl_int_t incX, const double beta, double* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_dsbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const double* A, const nvpl_int_t lda, const double* X, const nvpl_int_t incX, const double beta, double* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_dspmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* Ap, const double* X, const nvpl_int_t incX, const double beta, double* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_dger(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const double alpha, const double* X, const nvpl_int_t incX, const double* Y, const nvpl_int_t incY, double* A, const nvpl_int_t lda) except* nogil
cdef void cblas_dsyr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* X, const nvpl_int_t incX, double* A, const nvpl_int_t lda) except* nogil
cdef void cblas_dspr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* X, const nvpl_int_t incX, double* Ap) except* nogil
cdef void cblas_dsyr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* X, const nvpl_int_t incX, const double* Y, const nvpl_int_t incY, double* A, const nvpl_int_t lda) except* nogil
cdef void cblas_dspr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* X, const nvpl_int_t incX, const double* Y, const nvpl_int_t incY, double* A) except* nogil
cdef void cblas_chemv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_chbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_chpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* Ap, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_cgeru(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil
cdef void cblas_cgerc(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil
cdef void cblas_cher(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const void* X, const nvpl_int_t incX, void* A, const nvpl_int_t lda) except* nogil
cdef void cblas_chpr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const void* X, const nvpl_int_t incX, void* A) except* nogil
cdef void cblas_cher2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil
cdef void cblas_chpr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* Ap) except* nogil
cdef void cblas_zhemv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_zhbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_zhpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* Ap, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil
cdef void cblas_zgeru(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil
cdef void cblas_zgerc(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil
cdef void cblas_zher(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const void* X, const nvpl_int_t incX, void* A, const nvpl_int_t lda) except* nogil
cdef void cblas_zhpr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const void* X, const nvpl_int_t incX, void* A) except* nogil
cdef void cblas_zher2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil
cdef void cblas_zhpr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* Ap) except* nogil
cdef void cblas_sgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const float* A, const nvpl_int_t lda, const float* B, const nvpl_int_t ldb, const float beta, float* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_ssymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const float alpha, const float* A, const nvpl_int_t lda, const float* B, const nvpl_int_t ldb, const float beta, float* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_ssyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const float* A, const nvpl_int_t lda, const float beta, float* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_ssyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const float* A, const nvpl_int_t lda, const float* B, const nvpl_int_t ldb, const float beta, float* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_strmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const float alpha, const float* A, const nvpl_int_t lda, float* B, const nvpl_int_t ldb) except* nogil
cdef void cblas_strsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const float alpha, const float* A, const nvpl_int_t lda, float* B, const nvpl_int_t ldb) except* nogil
cdef void cblas_dgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const double* A, const nvpl_int_t lda, const double* B, const nvpl_int_t ldb, const double beta, double* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_dsymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const double alpha, const double* A, const nvpl_int_t lda, const double* B, const nvpl_int_t ldb, const double beta, double* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_dsyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const double* A, const nvpl_int_t lda, const double beta, double* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_dsyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const double* A, const nvpl_int_t lda, const double* B, const nvpl_int_t ldb, const double beta, double* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_dtrmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const double alpha, const double* A, const nvpl_int_t lda, double* B, const nvpl_int_t ldb) except* nogil
cdef void cblas_dtrsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const double alpha, const double* A, const nvpl_int_t lda, double* B, const nvpl_int_t ldb) except* nogil
cdef void cblas_cgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_csymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_csyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_csyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_ctrmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, void* B, const nvpl_int_t ldb) except* nogil
cdef void cblas_ctrsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, void* B, const nvpl_int_t ldb) except* nogil
cdef void cblas_zgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_zsymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_zsyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_zsyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_ztrmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, void* B, const nvpl_int_t ldb) except* nogil
cdef void cblas_ztrsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, void* B, const nvpl_int_t ldb) except* nogil
cdef void cblas_chemm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_cherk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const void* A, const nvpl_int_t lda, const float beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_cher2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const float beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_zhemm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_zherk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const void* A, const nvpl_int_t lda, const double beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_zher2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const double beta, void* C, const nvpl_int_t ldc) except* nogil
cdef void cblas_sgemm_batch(CBLAS_ORDER Order, CBLAS_TRANSPOSE* TransA_array, CBLAS_TRANSPOSE* TransB_array, nvpl_int_t* M_array, nvpl_int_t* N_array, nvpl_int_t* K_array, const float* alpha_array, const float** A_array, nvpl_int_t* lda_array, const float** B_array, nvpl_int_t* ldb_array, const float* beta_array, float** C_array, nvpl_int_t* ldc_array, nvpl_int_t group_count, nvpl_int_t* group_size) except* nogil
cdef void cblas_dgemm_batch(CBLAS_ORDER Order, CBLAS_TRANSPOSE* TransA_array, CBLAS_TRANSPOSE* TransB_array, nvpl_int_t* M_array, nvpl_int_t* N_array, nvpl_int_t* K_array, const double* alpha_array, const double** A_array, nvpl_int_t* lda_array, const double** B_array, nvpl_int_t* ldb_array, const double* beta_array, double** C_array, nvpl_int_t* ldc_array, nvpl_int_t group_count, nvpl_int_t* group_size) except* nogil
cdef void cblas_cgemm_batch(CBLAS_ORDER Order, CBLAS_TRANSPOSE* TransA_array, CBLAS_TRANSPOSE* TransB_array, nvpl_int_t* M_array, nvpl_int_t* N_array, nvpl_int_t* K_array, const void* alpha_array, const void** A_array, nvpl_int_t* lda_array, const void** B_array, nvpl_int_t* ldb_array, const void* beta_array, void** C_array, nvpl_int_t* ldc_array, nvpl_int_t group_count, nvpl_int_t* group_size) except* nogil
cdef void cblas_zgemm_batch(CBLAS_ORDER Order, CBLAS_TRANSPOSE* TransA_array, CBLAS_TRANSPOSE* TransB_array, nvpl_int_t* M_array, nvpl_int_t* N_array, nvpl_int_t* K_array, const void* alpha_array, const void** A_array, nvpl_int_t* lda_array, const void** B_array, nvpl_int_t* ldb_array, const void* beta_array, void** C_array, nvpl_int_t* ldc_array, nvpl_int_t group_count, nvpl_int_t* group_size) except* nogil
cdef void cblas_sgemm_batch_strided(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const float* A, const nvpl_int_t lda, const nvpl_int_t stridea, const float* B, const nvpl_int_t ldb, const nvpl_int_t strideb, const float beta, float* C, const nvpl_int_t ldc, const nvpl_int_t stridec, const nvpl_int_t batch_size) except* nogil
cdef void cblas_dgemm_batch_strided(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const double* A, const nvpl_int_t lda, const nvpl_int_t stridea, const double* B, const nvpl_int_t ldb, const nvpl_int_t strideb, const double beta, double* C, const nvpl_int_t ldc, const nvpl_int_t stridec, const nvpl_int_t batch_size) except* nogil
cdef void cblas_cgemm_batch_strided(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const nvpl_int_t stridea, const void* B, const nvpl_int_t ldb, const nvpl_int_t strideb, const void* beta, void* C, const nvpl_int_t ldc, const nvpl_int_t stridec, const nvpl_int_t batch_size) except* nogil
cdef void cblas_zgemm_batch_strided(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const nvpl_int_t stridea, const void* B, const nvpl_int_t ldb, const nvpl_int_t strideb, const void* beta, void* C, const nvpl_int_t ldc, const nvpl_int_t stridec, const nvpl_int_t batch_size) except* nogil
