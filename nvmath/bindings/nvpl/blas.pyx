# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.4.1. Do not modify it directly.

cimport cython  # NOQA

from .._internal.utils cimport get_resource_ptr, nullable_unique_ptr

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################

class ORDER(_IntEnum):
    """See `CBLAS_ORDER`."""
    RowMajor = CblasRowMajor
    ColMajor = CblasColMajor

class TRANSPOSE(_IntEnum):
    """See `CBLAS_TRANSPOSE`."""
    NoTrans = CblasNoTrans
    Trans = CblasTrans
    ConjTrans = CblasConjTrans

class UPLO(_IntEnum):
    """See `CBLAS_UPLO`."""
    Upper = CblasUpper
    Lower = CblasLower

class DIAG(_IntEnum):
    """See `CBLAS_DIAG`."""
    NonUnit = CblasNonUnit
    Unit = CblasUnit

class SIDE(_IntEnum):
    """See `CBLAS_SIDE`."""
    Left = CblasLeft
    Right = CblasRight


###############################################################################
# Types
###############################################################################


###############################################################################
# Error handling
###############################################################################


###############################################################################
# Convenience wrappers/adapters
###############################################################################


###############################################################################
# Wrapper functions
###############################################################################

cpdef int mkl_set_num_threads_local(int nth) except? -1:
    """See `MKL_mkl_set_num_threads_local`."""
    return MKL_mkl_set_num_threads_local(nth)


cpdef void mkl_set_num_threads(int nth) except*:
    """See `MKL_mkl_set_num_threads`."""
    MKL_mkl_set_num_threads(nth)


cpdef void openblas_set_num_threads(int num_threads) except*:
    """See `openblas_openblas_set_num_threads`."""
    openblas_openblas_set_num_threads(num_threads)


cpdef int openblas_set_num_threads_local(int num_threads) except? -1:
    """See `openblas_openblas_set_num_threads_local`."""
    return openblas_openblas_set_num_threads_local(num_threads)


cpdef int get_version() except? -1:
    """See `nvpl_blas_get_version`."""
    return nvpl_blas_get_version()


cpdef int get_max_threads() except? -1:
    """See `nvpl_blas_get_max_threads`."""
    return nvpl_blas_get_max_threads()


cpdef void set_num_threads(int nthr) except*:
    """See `nvpl_blas_set_num_threads`."""
    nvpl_blas_set_num_threads(nthr)


cpdef int set_num_threads_local(int nthr_local) except? -1:
    """See `nvpl_blas_set_num_threads_local`."""
    return nvpl_blas_set_num_threads_local(nthr_local)


cpdef void sgemm(int order, int trans_a, int trans_b, nvpl_int32_t m, nvpl_int32_t n, nvpl_int32_t k, float alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, float beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_sgemm`."""
    cblas_sgemm(<const _ORDER>order, <const _TRANSPOSE>trans_a, <const _TRANSPOSE>trans_b, <const nvpl_int_t>m, <const nvpl_int_t>n, <const nvpl_int_t>k, <const float>alpha, <const float*>a, <const nvpl_int_t>lda, <const float*>b, <const nvpl_int_t>ldb, <const float>beta, <float*>c, <const nvpl_int_t>ldc)


cpdef void ssymm(int order, int side, int uplo, nvpl_int32_t m, nvpl_int32_t n, float alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, float beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_ssymm`."""
    cblas_ssymm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const nvpl_int_t>m, <const nvpl_int_t>n, <const float>alpha, <const float*>a, <const nvpl_int_t>lda, <const float*>b, <const nvpl_int_t>ldb, <const float>beta, <float*>c, <const nvpl_int_t>ldc)


cpdef void ssyrk(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, float alpha, intptr_t a, nvpl_int32_t lda, float beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_ssyrk`."""
    cblas_ssyrk(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const float>alpha, <const float*>a, <const nvpl_int_t>lda, <const float>beta, <float*>c, <const nvpl_int_t>ldc)


cpdef void ssyr2k(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, float alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, float beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_ssyr2k`."""
    cblas_ssyr2k(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const float>alpha, <const float*>a, <const nvpl_int_t>lda, <const float*>b, <const nvpl_int_t>ldb, <const float>beta, <float*>c, <const nvpl_int_t>ldc)


cpdef void strmm(int order, int side, int uplo, int trans_a, int diag, nvpl_int32_t m, nvpl_int32_t n, float alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb) except*:
    """See `cblas_strmm`."""
    cblas_strmm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const _TRANSPOSE>trans_a, <const _DIAG>diag, <const nvpl_int_t>m, <const nvpl_int_t>n, <const float>alpha, <const float*>a, <const nvpl_int_t>lda, <float*>b, <const nvpl_int_t>ldb)


cpdef void strsm(int order, int side, int uplo, int trans_a, int diag, nvpl_int32_t m, nvpl_int32_t n, float alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb) except*:
    """See `cblas_strsm`."""
    cblas_strsm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const _TRANSPOSE>trans_a, <const _DIAG>diag, <const nvpl_int_t>m, <const nvpl_int_t>n, <const float>alpha, <const float*>a, <const nvpl_int_t>lda, <float*>b, <const nvpl_int_t>ldb)


cpdef void dgemm(int order, int trans_a, int trans_b, nvpl_int32_t m, nvpl_int32_t n, nvpl_int32_t k, double alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, double beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_dgemm`."""
    cblas_dgemm(<const _ORDER>order, <const _TRANSPOSE>trans_a, <const _TRANSPOSE>trans_b, <const nvpl_int_t>m, <const nvpl_int_t>n, <const nvpl_int_t>k, <const double>alpha, <const double*>a, <const nvpl_int_t>lda, <const double*>b, <const nvpl_int_t>ldb, <const double>beta, <double*>c, <const nvpl_int_t>ldc)


cpdef void dsymm(int order, int side, int uplo, nvpl_int32_t m, nvpl_int32_t n, double alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, double beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_dsymm`."""
    cblas_dsymm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const nvpl_int_t>m, <const nvpl_int_t>n, <const double>alpha, <const double*>a, <const nvpl_int_t>lda, <const double*>b, <const nvpl_int_t>ldb, <const double>beta, <double*>c, <const nvpl_int_t>ldc)


cpdef void dsyrk(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, double alpha, intptr_t a, nvpl_int32_t lda, double beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_dsyrk`."""
    cblas_dsyrk(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const double>alpha, <const double*>a, <const nvpl_int_t>lda, <const double>beta, <double*>c, <const nvpl_int_t>ldc)


cpdef void dsyr2k(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, double alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, double beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_dsyr2k`."""
    cblas_dsyr2k(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const double>alpha, <const double*>a, <const nvpl_int_t>lda, <const double*>b, <const nvpl_int_t>ldb, <const double>beta, <double*>c, <const nvpl_int_t>ldc)


cpdef void dtrmm(int order, int side, int uplo, int trans_a, int diag, nvpl_int32_t m, nvpl_int32_t n, double alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb) except*:
    """See `cblas_dtrmm`."""
    cblas_dtrmm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const _TRANSPOSE>trans_a, <const _DIAG>diag, <const nvpl_int_t>m, <const nvpl_int_t>n, <const double>alpha, <const double*>a, <const nvpl_int_t>lda, <double*>b, <const nvpl_int_t>ldb)


cpdef void dtrsm(int order, int side, int uplo, int trans_a, int diag, nvpl_int32_t m, nvpl_int32_t n, double alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb) except*:
    """See `cblas_dtrsm`."""
    cblas_dtrsm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const _TRANSPOSE>trans_a, <const _DIAG>diag, <const nvpl_int_t>m, <const nvpl_int_t>n, <const double>alpha, <const double*>a, <const nvpl_int_t>lda, <double*>b, <const nvpl_int_t>ldb)


cpdef void cgemm(int order, int trans_a, int trans_b, nvpl_int32_t m, nvpl_int32_t n, nvpl_int32_t k, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, intptr_t beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_cgemm`."""
    cblas_cgemm(<const _ORDER>order, <const _TRANSPOSE>trans_a, <const _TRANSPOSE>trans_b, <const nvpl_int_t>m, <const nvpl_int_t>n, <const nvpl_int_t>k, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>b, <const nvpl_int_t>ldb, <const void*>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void csymm(int order, int side, int uplo, nvpl_int32_t m, nvpl_int32_t n, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, intptr_t beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_csymm`."""
    cblas_csymm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const nvpl_int_t>m, <const nvpl_int_t>n, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>b, <const nvpl_int_t>ldb, <const void*>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void csyrk(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_csyrk`."""
    cblas_csyrk(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void csyr2k(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, intptr_t beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_csyr2k`."""
    cblas_csyr2k(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>b, <const nvpl_int_t>ldb, <const void*>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void ctrmm(int order, int side, int uplo, int trans_a, int diag, nvpl_int32_t m, nvpl_int32_t n, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb) except*:
    """See `cblas_ctrmm`."""
    cblas_ctrmm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const _TRANSPOSE>trans_a, <const _DIAG>diag, <const nvpl_int_t>m, <const nvpl_int_t>n, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <void*>b, <const nvpl_int_t>ldb)


cpdef void ctrsm(int order, int side, int uplo, int trans_a, int diag, nvpl_int32_t m, nvpl_int32_t n, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb) except*:
    """See `cblas_ctrsm`."""
    cblas_ctrsm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const _TRANSPOSE>trans_a, <const _DIAG>diag, <const nvpl_int_t>m, <const nvpl_int_t>n, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <void*>b, <const nvpl_int_t>ldb)


cpdef void zgemm(int order, int trans_a, int trans_b, nvpl_int32_t m, nvpl_int32_t n, nvpl_int32_t k, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, intptr_t beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_zgemm`."""
    cblas_zgemm(<const _ORDER>order, <const _TRANSPOSE>trans_a, <const _TRANSPOSE>trans_b, <const nvpl_int_t>m, <const nvpl_int_t>n, <const nvpl_int_t>k, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>b, <const nvpl_int_t>ldb, <const void*>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void zsymm(int order, int side, int uplo, nvpl_int32_t m, nvpl_int32_t n, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, intptr_t beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_zsymm`."""
    cblas_zsymm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const nvpl_int_t>m, <const nvpl_int_t>n, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>b, <const nvpl_int_t>ldb, <const void*>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void zsyrk(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_zsyrk`."""
    cblas_zsyrk(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void zsyr2k(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, intptr_t beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_zsyr2k`."""
    cblas_zsyr2k(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>b, <const nvpl_int_t>ldb, <const void*>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void ztrmm(int order, int side, int uplo, int trans_a, int diag, nvpl_int32_t m, nvpl_int32_t n, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb) except*:
    """See `cblas_ztrmm`."""
    cblas_ztrmm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const _TRANSPOSE>trans_a, <const _DIAG>diag, <const nvpl_int_t>m, <const nvpl_int_t>n, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <void*>b, <const nvpl_int_t>ldb)


cpdef void ztrsm(int order, int side, int uplo, int trans_a, int diag, nvpl_int32_t m, nvpl_int32_t n, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb) except*:
    """See `cblas_ztrsm`."""
    cblas_ztrsm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const _TRANSPOSE>trans_a, <const _DIAG>diag, <const nvpl_int_t>m, <const nvpl_int_t>n, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <void*>b, <const nvpl_int_t>ldb)


cpdef void chemm(int order, int side, int uplo, nvpl_int32_t m, nvpl_int32_t n, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, intptr_t beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_chemm`."""
    cblas_chemm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const nvpl_int_t>m, <const nvpl_int_t>n, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>b, <const nvpl_int_t>ldb, <const void*>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void cherk(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, float alpha, intptr_t a, nvpl_int32_t lda, float beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_cherk`."""
    cblas_cherk(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const float>alpha, <const void*>a, <const nvpl_int_t>lda, <const float>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void cher2k(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, float beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_cher2k`."""
    cblas_cher2k(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>b, <const nvpl_int_t>ldb, <const float>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void zhemm(int order, int side, int uplo, nvpl_int32_t m, nvpl_int32_t n, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, intptr_t beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_zhemm`."""
    cblas_zhemm(<const _ORDER>order, <const _SIDE>side, <const _UPLO>uplo, <const nvpl_int_t>m, <const nvpl_int_t>n, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>b, <const nvpl_int_t>ldb, <const void*>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void zherk(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, double alpha, intptr_t a, nvpl_int32_t lda, double beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_zherk`."""
    cblas_zherk(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const double>alpha, <const void*>a, <const nvpl_int_t>lda, <const double>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void zher2k(int order, int uplo, int trans, nvpl_int32_t n, nvpl_int32_t k, intptr_t alpha, intptr_t a, nvpl_int32_t lda, intptr_t b, nvpl_int32_t ldb, double beta, intptr_t c, nvpl_int32_t ldc) except*:
    """See `cblas_zher2k`."""
    cblas_zher2k(<const _ORDER>order, <const _UPLO>uplo, <const _TRANSPOSE>trans, <const nvpl_int_t>n, <const nvpl_int_t>k, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const void*>b, <const nvpl_int_t>ldb, <const double>beta, <void*>c, <const nvpl_int_t>ldc)


cpdef void sgemm_batch(int order, intptr_t trans_a_array, intptr_t trans_b_array, intptr_t m_array, intptr_t n_array, intptr_t k_array, intptr_t alpha_array, intptr_t a_array, intptr_t lda_array, intptr_t b_array, intptr_t ldb_array, intptr_t beta_array, intptr_t c_array, intptr_t ldc_array, nvpl_int32_t group_count, intptr_t group_size) except*:
    """See `cblas_sgemm_batch`."""
    cblas_sgemm_batch(<_ORDER>order, <_TRANSPOSE*>trans_a_array, <_TRANSPOSE*>trans_b_array, <nvpl_int_t*>m_array, <nvpl_int_t*>n_array, <nvpl_int_t*>k_array, <const float*>alpha_array, <const float**>a_array, <nvpl_int_t*>lda_array, <const float**>b_array, <nvpl_int_t*>ldb_array, <const float*>beta_array, <float**>c_array, <nvpl_int_t*>ldc_array, <nvpl_int_t>group_count, <nvpl_int_t*>group_size)


cpdef void dgemm_batch(int order, intptr_t trans_a_array, intptr_t trans_b_array, intptr_t m_array, intptr_t n_array, intptr_t k_array, intptr_t alpha_array, intptr_t a_array, intptr_t lda_array, intptr_t b_array, intptr_t ldb_array, intptr_t beta_array, intptr_t c_array, intptr_t ldc_array, nvpl_int32_t group_count, intptr_t group_size) except*:
    """See `cblas_dgemm_batch`."""
    cblas_dgemm_batch(<_ORDER>order, <_TRANSPOSE*>trans_a_array, <_TRANSPOSE*>trans_b_array, <nvpl_int_t*>m_array, <nvpl_int_t*>n_array, <nvpl_int_t*>k_array, <const double*>alpha_array, <const double**>a_array, <nvpl_int_t*>lda_array, <const double**>b_array, <nvpl_int_t*>ldb_array, <const double*>beta_array, <double**>c_array, <nvpl_int_t*>ldc_array, <nvpl_int_t>group_count, <nvpl_int_t*>group_size)


cpdef void cgemm_batch(int order, intptr_t trans_a_array, intptr_t trans_b_array, intptr_t m_array, intptr_t n_array, intptr_t k_array, intptr_t alpha_array, intptr_t a_array, intptr_t lda_array, intptr_t b_array, intptr_t ldb_array, intptr_t beta_array, intptr_t c_array, intptr_t ldc_array, nvpl_int32_t group_count, intptr_t group_size) except*:
    """See `cblas_cgemm_batch`."""
    cblas_cgemm_batch(<_ORDER>order, <_TRANSPOSE*>trans_a_array, <_TRANSPOSE*>trans_b_array, <nvpl_int_t*>m_array, <nvpl_int_t*>n_array, <nvpl_int_t*>k_array, <const void*>alpha_array, <const void**>a_array, <nvpl_int_t*>lda_array, <const void**>b_array, <nvpl_int_t*>ldb_array, <const void*>beta_array, <void**>c_array, <nvpl_int_t*>ldc_array, <nvpl_int_t>group_count, <nvpl_int_t*>group_size)


cpdef void zgemm_batch(int order, intptr_t trans_a_array, intptr_t trans_b_array, intptr_t m_array, intptr_t n_array, intptr_t k_array, intptr_t alpha_array, intptr_t a_array, intptr_t lda_array, intptr_t b_array, intptr_t ldb_array, intptr_t beta_array, intptr_t c_array, intptr_t ldc_array, nvpl_int32_t group_count, intptr_t group_size) except*:
    """See `cblas_zgemm_batch`."""
    cblas_zgemm_batch(<_ORDER>order, <_TRANSPOSE*>trans_a_array, <_TRANSPOSE*>trans_b_array, <nvpl_int_t*>m_array, <nvpl_int_t*>n_array, <nvpl_int_t*>k_array, <const void*>alpha_array, <const void**>a_array, <nvpl_int_t*>lda_array, <const void**>b_array, <nvpl_int_t*>ldb_array, <const void*>beta_array, <void**>c_array, <nvpl_int_t*>ldc_array, <nvpl_int_t>group_count, <nvpl_int_t*>group_size)


cpdef void sgemm_batch_strided(int order, int trans_a, int trans_b, nvpl_int32_t m, nvpl_int32_t n, nvpl_int32_t k, float alpha, intptr_t a, nvpl_int32_t lda, nvpl_int32_t stridea, intptr_t b, nvpl_int32_t ldb, nvpl_int32_t strideb, float beta, intptr_t c, nvpl_int32_t ldc, nvpl_int32_t stridec, nvpl_int32_t batch_size) except*:
    """See `cblas_sgemm_batch_strided`."""
    cblas_sgemm_batch_strided(<const _ORDER>order, <const _TRANSPOSE>trans_a, <const _TRANSPOSE>trans_b, <const nvpl_int_t>m, <const nvpl_int_t>n, <const nvpl_int_t>k, <const float>alpha, <const float*>a, <const nvpl_int_t>lda, <const nvpl_int_t>stridea, <const float*>b, <const nvpl_int_t>ldb, <const nvpl_int_t>strideb, <const float>beta, <float*>c, <const nvpl_int_t>ldc, <const nvpl_int_t>stridec, <const nvpl_int_t>batch_size)


cpdef void dgemm_batch_strided(int order, int trans_a, int trans_b, nvpl_int32_t m, nvpl_int32_t n, nvpl_int32_t k, double alpha, intptr_t a, nvpl_int32_t lda, nvpl_int32_t stridea, intptr_t b, nvpl_int32_t ldb, nvpl_int32_t strideb, double beta, intptr_t c, nvpl_int32_t ldc, nvpl_int32_t stridec, nvpl_int32_t batch_size) except*:
    """See `cblas_dgemm_batch_strided`."""
    cblas_dgemm_batch_strided(<const _ORDER>order, <const _TRANSPOSE>trans_a, <const _TRANSPOSE>trans_b, <const nvpl_int_t>m, <const nvpl_int_t>n, <const nvpl_int_t>k, <const double>alpha, <const double*>a, <const nvpl_int_t>lda, <const nvpl_int_t>stridea, <const double*>b, <const nvpl_int_t>ldb, <const nvpl_int_t>strideb, <const double>beta, <double*>c, <const nvpl_int_t>ldc, <const nvpl_int_t>stridec, <const nvpl_int_t>batch_size)


cpdef void cgemm_batch_strided(int order, int trans_a, int trans_b, nvpl_int32_t m, nvpl_int32_t n, nvpl_int32_t k, intptr_t alpha, intptr_t a, nvpl_int32_t lda, nvpl_int32_t stridea, intptr_t b, nvpl_int32_t ldb, nvpl_int32_t strideb, intptr_t beta, intptr_t c, nvpl_int32_t ldc, nvpl_int32_t stridec, nvpl_int32_t batch_size) except*:
    """See `cblas_cgemm_batch_strided`."""
    cblas_cgemm_batch_strided(<const _ORDER>order, <const _TRANSPOSE>trans_a, <const _TRANSPOSE>trans_b, <const nvpl_int_t>m, <const nvpl_int_t>n, <const nvpl_int_t>k, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const nvpl_int_t>stridea, <const void*>b, <const nvpl_int_t>ldb, <const nvpl_int_t>strideb, <const void*>beta, <void*>c, <const nvpl_int_t>ldc, <const nvpl_int_t>stridec, <const nvpl_int_t>batch_size)


cpdef void zgemm_batch_strided(int order, int trans_a, int trans_b, nvpl_int32_t m, nvpl_int32_t n, nvpl_int32_t k, intptr_t alpha, intptr_t a, nvpl_int32_t lda, nvpl_int32_t stridea, intptr_t b, nvpl_int32_t ldb, nvpl_int32_t strideb, intptr_t beta, intptr_t c, nvpl_int32_t ldc, nvpl_int32_t stridec, nvpl_int32_t batch_size) except*:
    """See `cblas_zgemm_batch_strided`."""
    cblas_zgemm_batch_strided(<const _ORDER>order, <const _TRANSPOSE>trans_a, <const _TRANSPOSE>trans_b, <const nvpl_int_t>m, <const nvpl_int_t>n, <const nvpl_int_t>k, <const void*>alpha, <const void*>a, <const nvpl_int_t>lda, <const nvpl_int_t>stridea, <const void*>b, <const nvpl_int_t>ldb, <const nvpl_int_t>strideb, <const void*>beta, <void*>c, <const nvpl_int_t>ldc, <const nvpl_int_t>stridec, <const nvpl_int_t>batch_size)
