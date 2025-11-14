# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.4.1. Do not modify it directly.

cimport cython

from ._internal cimport blas as _nvpl_blas


###############################################################################
# Wrapper functions
###############################################################################

cdef int MKL_mkl_set_num_threads_local(int nth) except?-42 nogil:
    return _nvpl_blas._MKL_mkl_set_num_threads_local(nth)


@cython.show_performance_hints(False)
cdef void MKL_mkl_set_num_threads(int nth) except* nogil:
    _nvpl_blas._MKL_mkl_set_num_threads(nth)


@cython.show_performance_hints(False)
cdef void openblas_openblas_set_num_threads(int num_threads) except* nogil:
    _nvpl_blas._openblas_openblas_set_num_threads(num_threads)


cdef int openblas_openblas_set_num_threads_local(int num_threads) except?-42 nogil:
    return _nvpl_blas._openblas_openblas_set_num_threads_local(num_threads)


cdef int nvpl_blas_get_version() except?-42 nogil:
    return _nvpl_blas._nvpl_blas_get_version()


cdef int nvpl_blas_get_max_threads() except?-42 nogil:
    return _nvpl_blas._nvpl_blas_get_max_threads()


@cython.show_performance_hints(False)
cdef void nvpl_blas_set_num_threads(int nthr) except* nogil:
    _nvpl_blas._nvpl_blas_set_num_threads(nthr)


cdef int nvpl_blas_set_num_threads_local(int nthr_local) except?-42 nogil:
    return _nvpl_blas._nvpl_blas_set_num_threads_local(nthr_local)


@cython.show_performance_hints(False)
cdef void cblas_sgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const float alpha, const float* A, const nvpl_int_t lda, const float* X, const nvpl_int_t incX, const float beta, float* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_sgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t KL, const nvpl_int_t KU, const float alpha, const float* A, const nvpl_int_t lda, const float* X, const nvpl_int_t incX, const float beta, float* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_sgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_strmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const float* A, const nvpl_int_t lda, float* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_strmv(order, Uplo, TransA, Diag, N, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_stbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const float* A, const nvpl_int_t lda, float* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_stbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_stpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const float* Ap, float* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_stpmv(order, Uplo, TransA, Diag, N, Ap, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_strsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const float* A, const nvpl_int_t lda, float* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_strsv(order, Uplo, TransA, Diag, N, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_stbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const float* A, const nvpl_int_t lda, float* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_stbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_stpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const float* Ap, float* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_stpsv(order, Uplo, TransA, Diag, N, Ap, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_dgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const double alpha, const double* A, const nvpl_int_t lda, const double* X, const nvpl_int_t incX, const double beta, double* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_dgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_dgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t KL, const nvpl_int_t KU, const double alpha, const double* A, const nvpl_int_t lda, const double* X, const nvpl_int_t incX, const double beta, double* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_dgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_dtrmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const double* A, const nvpl_int_t lda, double* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_dtrmv(order, Uplo, TransA, Diag, N, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_dtbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const double* A, const nvpl_int_t lda, double* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_dtbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_dtpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const double* Ap, double* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_dtpmv(order, Uplo, TransA, Diag, N, Ap, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_dtrsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const double* A, const nvpl_int_t lda, double* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_dtrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_dtbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const double* A, const nvpl_int_t lda, double* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_dtbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_dtpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const double* Ap, double* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_dtpsv(order, Uplo, TransA, Diag, N, Ap, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_cgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_cgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_cgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t KL, const nvpl_int_t KU, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_cgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_ctrmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ctrmv(order, Uplo, TransA, Diag, N, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ctbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ctbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ctpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* Ap, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ctpmv(order, Uplo, TransA, Diag, N, Ap, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ctrsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ctrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ctbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ctbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ctpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* Ap, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ctpsv(order, Uplo, TransA, Diag, N, Ap, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_zgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_zgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_zgbmv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t KL, const nvpl_int_t KU, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_zgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_ztrmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ztrmv(order, Uplo, TransA, Diag, N, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ztbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ztbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ztpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* Ap, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ztpmv(order, Uplo, TransA, Diag, N, Ap, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ztrsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ztrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ztbsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const nvpl_int_t K, const void* A, const nvpl_int_t lda, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ztbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ztpsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t N, const void* Ap, void* X, const nvpl_int_t incX) except* nogil:
    _nvpl_blas._cblas_ztpsv(order, Uplo, TransA, Diag, N, Ap, X, incX)


@cython.show_performance_hints(False)
cdef void cblas_ssymv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* A, const nvpl_int_t lda, const float* X, const nvpl_int_t incX, const float beta, float* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_ssymv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_ssbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const float* A, const nvpl_int_t lda, const float* X, const nvpl_int_t incX, const float beta, float* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_ssbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_sspmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* Ap, const float* X, const nvpl_int_t incX, const float beta, float* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_sspmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_sger(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const float alpha, const float* X, const nvpl_int_t incX, const float* Y, const nvpl_int_t incY, float* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_sger(order, M, N, alpha, X, incX, Y, incY, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_ssyr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* X, const nvpl_int_t incX, float* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_ssyr(order, Uplo, N, alpha, X, incX, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_sspr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* X, const nvpl_int_t incX, float* Ap) except* nogil:
    _nvpl_blas._cblas_sspr(order, Uplo, N, alpha, X, incX, Ap)


@cython.show_performance_hints(False)
cdef void cblas_ssyr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* X, const nvpl_int_t incX, const float* Y, const nvpl_int_t incY, float* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_ssyr2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_sspr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const float* X, const nvpl_int_t incX, const float* Y, const nvpl_int_t incY, float* A) except* nogil:
    _nvpl_blas._cblas_sspr2(order, Uplo, N, alpha, X, incX, Y, incY, A)


@cython.show_performance_hints(False)
cdef void cblas_dsymv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* A, const nvpl_int_t lda, const double* X, const nvpl_int_t incX, const double beta, double* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_dsymv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_dsbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const double* A, const nvpl_int_t lda, const double* X, const nvpl_int_t incX, const double beta, double* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_dsbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_dspmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* Ap, const double* X, const nvpl_int_t incX, const double beta, double* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_dspmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_dger(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const double alpha, const double* X, const nvpl_int_t incX, const double* Y, const nvpl_int_t incY, double* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_dger(order, M, N, alpha, X, incX, Y, incY, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_dsyr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* X, const nvpl_int_t incX, double* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_dsyr(order, Uplo, N, alpha, X, incX, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_dspr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* X, const nvpl_int_t incX, double* Ap) except* nogil:
    _nvpl_blas._cblas_dspr(order, Uplo, N, alpha, X, incX, Ap)


@cython.show_performance_hints(False)
cdef void cblas_dsyr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* X, const nvpl_int_t incX, const double* Y, const nvpl_int_t incY, double* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_dsyr2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_dspr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const double* X, const nvpl_int_t incX, const double* Y, const nvpl_int_t incY, double* A) except* nogil:
    _nvpl_blas._cblas_dspr2(order, Uplo, N, alpha, X, incX, Y, incY, A)


@cython.show_performance_hints(False)
cdef void cblas_chemv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_chemv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_chbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_chbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_chpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* Ap, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_chpmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_cgeru(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_cgeru(order, M, N, alpha, X, incX, Y, incY, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_cgerc(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_cgerc(order, M, N, alpha, X, incX, Y, incY, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_cher(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const void* X, const nvpl_int_t incX, void* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_cher(order, Uplo, N, alpha, X, incX, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_chpr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const float alpha, const void* X, const nvpl_int_t incX, void* A) except* nogil:
    _nvpl_blas._cblas_chpr(order, Uplo, N, alpha, X, incX, A)


@cython.show_performance_hints(False)
cdef void cblas_cher2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_cher2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_chpr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* Ap) except* nogil:
    _nvpl_blas._cblas_chpr2(order, Uplo, N, alpha, X, incX, Y, incY, Ap)


@cython.show_performance_hints(False)
cdef void cblas_zhemv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_zhemv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_zhbmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_zhbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_zhpmv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* Ap, const void* X, const nvpl_int_t incX, const void* beta, void* Y, const nvpl_int_t incY) except* nogil:
    _nvpl_blas._cblas_zhpmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY)


@cython.show_performance_hints(False)
cdef void cblas_zgeru(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_zgeru(order, M, N, alpha, X, incX, Y, incY, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_zgerc(const CBLAS_ORDER order, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_zgerc(order, M, N, alpha, X, incX, Y, incY, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_zher(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const void* X, const nvpl_int_t incX, void* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_zher(order, Uplo, N, alpha, X, incX, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_zhpr(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const double alpha, const void* X, const nvpl_int_t incX, void* A) except* nogil:
    _nvpl_blas._cblas_zhpr(order, Uplo, N, alpha, X, incX, A)


@cython.show_performance_hints(False)
cdef void cblas_zher2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* A, const nvpl_int_t lda) except* nogil:
    _nvpl_blas._cblas_zher2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda)


@cython.show_performance_hints(False)
cdef void cblas_zhpr2(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const nvpl_int_t N, const void* alpha, const void* X, const nvpl_int_t incX, const void* Y, const nvpl_int_t incY, void* Ap) except* nogil:
    _nvpl_blas._cblas_zhpr2(order, Uplo, N, alpha, X, incX, Y, incY, Ap)


@cython.show_performance_hints(False)
cdef void cblas_sgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const float* A, const nvpl_int_t lda, const float* B, const nvpl_int_t ldb, const float beta, float* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_ssymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const float alpha, const float* A, const nvpl_int_t lda, const float* B, const nvpl_int_t ldb, const float beta, float* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_ssymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_ssyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const float* A, const nvpl_int_t lda, const float beta, float* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_ssyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_ssyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const float* A, const nvpl_int_t lda, const float* B, const nvpl_int_t ldb, const float beta, float* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_ssyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_strmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const float alpha, const float* A, const nvpl_int_t lda, float* B, const nvpl_int_t ldb) except* nogil:
    _nvpl_blas._cblas_strmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)


@cython.show_performance_hints(False)
cdef void cblas_strsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const float alpha, const float* A, const nvpl_int_t lda, float* B, const nvpl_int_t ldb) except* nogil:
    _nvpl_blas._cblas_strsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)


@cython.show_performance_hints(False)
cdef void cblas_dgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const double* A, const nvpl_int_t lda, const double* B, const nvpl_int_t ldb, const double beta, double* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_dsymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const double alpha, const double* A, const nvpl_int_t lda, const double* B, const nvpl_int_t ldb, const double beta, double* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_dsymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_dsyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const double* A, const nvpl_int_t lda, const double beta, double* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_dsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_dsyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const double* A, const nvpl_int_t lda, const double* B, const nvpl_int_t ldb, const double beta, double* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_dsyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_dtrmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const double alpha, const double* A, const nvpl_int_t lda, double* B, const nvpl_int_t ldb) except* nogil:
    _nvpl_blas._cblas_dtrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)


@cython.show_performance_hints(False)
cdef void cblas_dtrsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const double alpha, const double* A, const nvpl_int_t lda, double* B, const nvpl_int_t ldb) except* nogil:
    _nvpl_blas._cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)


@cython.show_performance_hints(False)
cdef void cblas_cgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_cgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_csymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_csymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_csyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_csyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_csyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_csyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_ctrmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, void* B, const nvpl_int_t ldb) except* nogil:
    _nvpl_blas._cblas_ctrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)


@cython.show_performance_hints(False)
cdef void cblas_ctrsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, void* B, const nvpl_int_t ldb) except* nogil:
    _nvpl_blas._cblas_ctrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)


@cython.show_performance_hints(False)
cdef void cblas_zgemm(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_zgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_zsymm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_zsymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_zsyrk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_zsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_zsyr2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_zsyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_ztrmm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, void* B, const nvpl_int_t ldb) except* nogil:
    _nvpl_blas._cblas_ztrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)


@cython.show_performance_hints(False)
cdef void cblas_ztrsm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, void* B, const nvpl_int_t ldb) except* nogil:
    _nvpl_blas._cblas_ztrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)


@cython.show_performance_hints(False)
cdef void cblas_chemm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_chemm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_cherk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const void* A, const nvpl_int_t lda, const float beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_cherk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_cher2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const float beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_cher2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_zhemm(const CBLAS_ORDER Order, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const nvpl_int_t M, const nvpl_int_t N, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const void* beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_zhemm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_zherk(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const void* A, const nvpl_int_t lda, const double beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_zherk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_zher2k(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const void* B, const nvpl_int_t ldb, const double beta, void* C, const nvpl_int_t ldc) except* nogil:
    _nvpl_blas._cblas_zher2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)


@cython.show_performance_hints(False)
cdef void cblas_sgemm_batch(CBLAS_ORDER Order, CBLAS_TRANSPOSE* TransA_array, CBLAS_TRANSPOSE* TransB_array, nvpl_int_t* M_array, nvpl_int_t* N_array, nvpl_int_t* K_array, const float* alpha_array, const float** A_array, nvpl_int_t* lda_array, const float** B_array, nvpl_int_t* ldb_array, const float* beta_array, float** C_array, nvpl_int_t* ldc_array, nvpl_int_t group_count, nvpl_int_t* group_size) except* nogil:
    _nvpl_blas._cblas_sgemm_batch(Order, TransA_array, TransB_array, M_array, N_array, K_array, alpha_array, A_array, lda_array, B_array, ldb_array, beta_array, C_array, ldc_array, group_count, group_size)


@cython.show_performance_hints(False)
cdef void cblas_dgemm_batch(CBLAS_ORDER Order, CBLAS_TRANSPOSE* TransA_array, CBLAS_TRANSPOSE* TransB_array, nvpl_int_t* M_array, nvpl_int_t* N_array, nvpl_int_t* K_array, const double* alpha_array, const double** A_array, nvpl_int_t* lda_array, const double** B_array, nvpl_int_t* ldb_array, const double* beta_array, double** C_array, nvpl_int_t* ldc_array, nvpl_int_t group_count, nvpl_int_t* group_size) except* nogil:
    _nvpl_blas._cblas_dgemm_batch(Order, TransA_array, TransB_array, M_array, N_array, K_array, alpha_array, A_array, lda_array, B_array, ldb_array, beta_array, C_array, ldc_array, group_count, group_size)


@cython.show_performance_hints(False)
cdef void cblas_cgemm_batch(CBLAS_ORDER Order, CBLAS_TRANSPOSE* TransA_array, CBLAS_TRANSPOSE* TransB_array, nvpl_int_t* M_array, nvpl_int_t* N_array, nvpl_int_t* K_array, const void* alpha_array, const void** A_array, nvpl_int_t* lda_array, const void** B_array, nvpl_int_t* ldb_array, const void* beta_array, void** C_array, nvpl_int_t* ldc_array, nvpl_int_t group_count, nvpl_int_t* group_size) except* nogil:
    _nvpl_blas._cblas_cgemm_batch(Order, TransA_array, TransB_array, M_array, N_array, K_array, alpha_array, A_array, lda_array, B_array, ldb_array, beta_array, C_array, ldc_array, group_count, group_size)


@cython.show_performance_hints(False)
cdef void cblas_zgemm_batch(CBLAS_ORDER Order, CBLAS_TRANSPOSE* TransA_array, CBLAS_TRANSPOSE* TransB_array, nvpl_int_t* M_array, nvpl_int_t* N_array, nvpl_int_t* K_array, const void* alpha_array, const void** A_array, nvpl_int_t* lda_array, const void** B_array, nvpl_int_t* ldb_array, const void* beta_array, void** C_array, nvpl_int_t* ldc_array, nvpl_int_t group_count, nvpl_int_t* group_size) except* nogil:
    _nvpl_blas._cblas_zgemm_batch(Order, TransA_array, TransB_array, M_array, N_array, K_array, alpha_array, A_array, lda_array, B_array, ldb_array, beta_array, C_array, ldc_array, group_count, group_size)


@cython.show_performance_hints(False)
cdef void cblas_sgemm_batch_strided(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const float alpha, const float* A, const nvpl_int_t lda, const nvpl_int_t stridea, const float* B, const nvpl_int_t ldb, const nvpl_int_t strideb, const float beta, float* C, const nvpl_int_t ldc, const nvpl_int_t stridec, const nvpl_int_t batch_size) except* nogil:
    _nvpl_blas._cblas_sgemm_batch_strided(Order, TransA, TransB, M, N, K, alpha, A, lda, stridea, B, ldb, strideb, beta, C, ldc, stridec, batch_size)


@cython.show_performance_hints(False)
cdef void cblas_dgemm_batch_strided(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const double alpha, const double* A, const nvpl_int_t lda, const nvpl_int_t stridea, const double* B, const nvpl_int_t ldb, const nvpl_int_t strideb, const double beta, double* C, const nvpl_int_t ldc, const nvpl_int_t stridec, const nvpl_int_t batch_size) except* nogil:
    _nvpl_blas._cblas_dgemm_batch_strided(Order, TransA, TransB, M, N, K, alpha, A, lda, stridea, B, ldb, strideb, beta, C, ldc, stridec, batch_size)


@cython.show_performance_hints(False)
cdef void cblas_cgemm_batch_strided(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const nvpl_int_t stridea, const void* B, const nvpl_int_t ldb, const nvpl_int_t strideb, const void* beta, void* C, const nvpl_int_t ldc, const nvpl_int_t stridec, const nvpl_int_t batch_size) except* nogil:
    _nvpl_blas._cblas_cgemm_batch_strided(Order, TransA, TransB, M, N, K, alpha, A, lda, stridea, B, ldb, strideb, beta, C, ldc, stridec, batch_size)


@cython.show_performance_hints(False)
cdef void cblas_zgemm_batch_strided(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const nvpl_int_t M, const nvpl_int_t N, const nvpl_int_t K, const void* alpha, const void* A, const nvpl_int_t lda, const nvpl_int_t stridea, const void* B, const nvpl_int_t ldb, const nvpl_int_t strideb, const void* beta, void* C, const nvpl_int_t ldc, const nvpl_int_t stridec, const nvpl_int_t batch_size) except* nogil:
    _nvpl_blas._cblas_zgemm_batch_strided(Order, TransA, TransB, M, N, K, alpha, A, lda, stridea, B, ldb, strideb, beta, C, ldc, stridec, batch_size)
