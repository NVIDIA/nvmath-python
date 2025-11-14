# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.4.1. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cyblas cimport *


###############################################################################
# Types
###############################################################################

ctypedef nvpl_scomplex_t scomplex
ctypedef nvpl_dcomplex_t dcomplex


###############################################################################
# Enum
###############################################################################

ctypedef CBLAS_ORDER _ORDER
ctypedef CBLAS_TRANSPOSE _TRANSPOSE
ctypedef CBLAS_UPLO _UPLO
ctypedef CBLAS_DIAG _DIAG
ctypedef CBLAS_SIDE _SIDE


###############################################################################
# Convenience wrappers/adapters
###############################################################################


###############################################################################
# Functions
###############################################################################

cpdef int mkl_set_num_threads_local(int nth) except? -1
cpdef void mkl_set_num_threads(int nth) except*
cpdef void openblas_set_num_threads(int num_threads) except*
cpdef int openblas_set_num_threads_local(int num_threads) except? -1
cpdef int get_version() except? -1
cpdef int get_max_threads() except? -1
cpdef void set_num_threads(int nthr) except*
cpdef int set_num_threads_local(int nthr_local) except? -1
cpdef void sgemm(int order, int trans_a, int trans_b, nvpl_int64_t m, nvpl_int64_t n, nvpl_int64_t k, float alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, float beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void ssymm(int order, int side, int uplo, nvpl_int64_t m, nvpl_int64_t n, float alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, float beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void ssyrk(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, float alpha, intptr_t a, nvpl_int64_t lda, float beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void ssyr2k(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, float alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, float beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void strmm(int order, int side, int uplo, int trans_a, int diag, nvpl_int64_t m, nvpl_int64_t n, float alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb) except*
cpdef void strsm(int order, int side, int uplo, int trans_a, int diag, nvpl_int64_t m, nvpl_int64_t n, float alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb) except*
cpdef void dgemm(int order, int trans_a, int trans_b, nvpl_int64_t m, nvpl_int64_t n, nvpl_int64_t k, double alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, double beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void dsymm(int order, int side, int uplo, nvpl_int64_t m, nvpl_int64_t n, double alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, double beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void dsyrk(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, double alpha, intptr_t a, nvpl_int64_t lda, double beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void dsyr2k(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, double alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, double beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void dtrmm(int order, int side, int uplo, int trans_a, int diag, nvpl_int64_t m, nvpl_int64_t n, double alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb) except*
cpdef void dtrsm(int order, int side, int uplo, int trans_a, int diag, nvpl_int64_t m, nvpl_int64_t n, double alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb) except*
cpdef void cgemm(int order, int trans_a, int trans_b, nvpl_int64_t m, nvpl_int64_t n, nvpl_int64_t k, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, intptr_t beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void csymm(int order, int side, int uplo, nvpl_int64_t m, nvpl_int64_t n, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, intptr_t beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void csyrk(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void csyr2k(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, intptr_t beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void ctrmm(int order, int side, int uplo, int trans_a, int diag, nvpl_int64_t m, nvpl_int64_t n, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb) except*
cpdef void ctrsm(int order, int side, int uplo, int trans_a, int diag, nvpl_int64_t m, nvpl_int64_t n, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb) except*
cpdef void zgemm(int order, int trans_a, int trans_b, nvpl_int64_t m, nvpl_int64_t n, nvpl_int64_t k, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, intptr_t beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void zsymm(int order, int side, int uplo, nvpl_int64_t m, nvpl_int64_t n, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, intptr_t beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void zsyrk(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void zsyr2k(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, intptr_t beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void ztrmm(int order, int side, int uplo, int trans_a, int diag, nvpl_int64_t m, nvpl_int64_t n, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb) except*
cpdef void ztrsm(int order, int side, int uplo, int trans_a, int diag, nvpl_int64_t m, nvpl_int64_t n, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb) except*
cpdef void chemm(int order, int side, int uplo, nvpl_int64_t m, nvpl_int64_t n, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, intptr_t beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void cherk(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, float alpha, intptr_t a, nvpl_int64_t lda, float beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void cher2k(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, float beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void zhemm(int order, int side, int uplo, nvpl_int64_t m, nvpl_int64_t n, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, intptr_t beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void zherk(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, double alpha, intptr_t a, nvpl_int64_t lda, double beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void zher2k(int order, int uplo, int trans, nvpl_int64_t n, nvpl_int64_t k, intptr_t alpha, intptr_t a, nvpl_int64_t lda, intptr_t b, nvpl_int64_t ldb, double beta, intptr_t c, nvpl_int64_t ldc) except*
cpdef void sgemm_batch(int order, intptr_t trans_a_array, intptr_t trans_b_array, intptr_t m_array, intptr_t n_array, intptr_t k_array, intptr_t alpha_array, intptr_t a_array, intptr_t lda_array, intptr_t b_array, intptr_t ldb_array, intptr_t beta_array, intptr_t c_array, intptr_t ldc_array, nvpl_int64_t group_count, intptr_t group_size) except*
cpdef void dgemm_batch(int order, intptr_t trans_a_array, intptr_t trans_b_array, intptr_t m_array, intptr_t n_array, intptr_t k_array, intptr_t alpha_array, intptr_t a_array, intptr_t lda_array, intptr_t b_array, intptr_t ldb_array, intptr_t beta_array, intptr_t c_array, intptr_t ldc_array, nvpl_int64_t group_count, intptr_t group_size) except*
cpdef void cgemm_batch(int order, intptr_t trans_a_array, intptr_t trans_b_array, intptr_t m_array, intptr_t n_array, intptr_t k_array, intptr_t alpha_array, intptr_t a_array, intptr_t lda_array, intptr_t b_array, intptr_t ldb_array, intptr_t beta_array, intptr_t c_array, intptr_t ldc_array, nvpl_int64_t group_count, intptr_t group_size) except*
cpdef void zgemm_batch(int order, intptr_t trans_a_array, intptr_t trans_b_array, intptr_t m_array, intptr_t n_array, intptr_t k_array, intptr_t alpha_array, intptr_t a_array, intptr_t lda_array, intptr_t b_array, intptr_t ldb_array, intptr_t beta_array, intptr_t c_array, intptr_t ldc_array, nvpl_int64_t group_count, intptr_t group_size) except*
cpdef void sgemm_batch_strided(int order, int trans_a, int trans_b, nvpl_int64_t m, nvpl_int64_t n, nvpl_int64_t k, float alpha, intptr_t a, nvpl_int64_t lda, nvpl_int64_t stridea, intptr_t b, nvpl_int64_t ldb, nvpl_int64_t strideb, float beta, intptr_t c, nvpl_int64_t ldc, nvpl_int64_t stridec, nvpl_int64_t batch_size) except*
cpdef void dgemm_batch_strided(int order, int trans_a, int trans_b, nvpl_int64_t m, nvpl_int64_t n, nvpl_int64_t k, double alpha, intptr_t a, nvpl_int64_t lda, nvpl_int64_t stridea, intptr_t b, nvpl_int64_t ldb, nvpl_int64_t strideb, double beta, intptr_t c, nvpl_int64_t ldc, nvpl_int64_t stridec, nvpl_int64_t batch_size) except*
cpdef void cgemm_batch_strided(int order, int trans_a, int trans_b, nvpl_int64_t m, nvpl_int64_t n, nvpl_int64_t k, intptr_t alpha, intptr_t a, nvpl_int64_t lda, nvpl_int64_t stridea, intptr_t b, nvpl_int64_t ldb, nvpl_int64_t strideb, intptr_t beta, intptr_t c, nvpl_int64_t ldc, nvpl_int64_t stridec, nvpl_int64_t batch_size) except*
cpdef void zgemm_batch_strided(int order, int trans_a, int trans_b, nvpl_int64_t m, nvpl_int64_t n, nvpl_int64_t k, intptr_t alpha, intptr_t a, nvpl_int64_t lda, nvpl_int64_t stridea, intptr_t b, nvpl_int64_t ldb, nvpl_int64_t strideb, intptr_t beta, intptr_t c, nvpl_int64_t ldc, nvpl_int64_t stridec, nvpl_int64_t batch_size) except*
