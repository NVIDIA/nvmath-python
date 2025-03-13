# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycusolverDn cimport *


###############################################################################
# Types
###############################################################################

ctypedef cusolverDnHandle_t Handle
ctypedef syevjInfo_t syevjInfo
ctypedef gesvdjInfo_t gesvdjInfo
ctypedef cusolverDnIRSParams_t IRSParams
ctypedef cusolverDnIRSInfos_t IRSInfos
ctypedef cusolverDnParams_t Params
ctypedef cusolverDnLoggerCallback_t LoggerCallback

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef cusolverDnFunction_t _Function


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef set_stream(intptr_t handle, intptr_t stream_id)
cpdef intptr_t get_stream(intptr_t handle) except? 0
cpdef intptr_t irs_params_create() except? 0
cpdef irs_params_destroy(intptr_t params)
cpdef irs_params_set_refinement_solver(intptr_t params, int refinement_solver)
cpdef irs_params_set_solver_main_precision(intptr_t params, int solver_main_precision)
cpdef irs_params_set_solver_lowest_precision(intptr_t params, int solver_lowest_precision)
cpdef irs_params_set_solver_precisions(intptr_t params, int solver_main_precision, int solver_lowest_precision)
cpdef irs_params_set_tol(intptr_t params, double val)
cpdef irs_params_set_tol_inner(intptr_t params, double val)
cpdef irs_params_set_max_iters(intptr_t params, int maxiters)
cpdef irs_params_set_max_iters_inner(intptr_t params, int maxiters_inner)
cpdef int irs_params_get_max_iters(intptr_t params) except? -1
cpdef irs_params_enable_fallback(intptr_t params)
cpdef irs_params_disable_fallback(intptr_t params)
cpdef irs_infos_destroy(intptr_t infos)
cpdef intptr_t irs_infos_create() except? 0
cpdef int irs_infos_get_niters(intptr_t infos) except? -1
cpdef int irs_infos_get_outer_niters(intptr_t infos) except? -1
cpdef irs_infos_request_residual(intptr_t infos)
cpdef irs_infos_get_residual_history(intptr_t infos, intptr_t residual_history)
cpdef int irs_infos_get_max_iters(intptr_t infos) except? -1
cpdef int zz_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int zc_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int zk_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int ze_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int zy_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int cc_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int ce_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int ck_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int cy_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int dd_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int ds_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int dh_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int db_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int dx_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int ss_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int sh_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int sb_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int sx_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef size_t zz_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t zc_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t zk_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t ze_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t zy_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t cc_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t ck_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t ce_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t cy_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t dd_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t ds_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t dh_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t db_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t dx_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t ss_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t sh_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t sb_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t sx_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef int zz_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int zc_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int zk_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int ze_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int zy_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int cc_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int ck_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int ce_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int cy_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int dd_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int ds_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int dh_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int db_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int dx_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int ss_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int sh_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int sb_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef int sx_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef size_t zz_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t zc_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t zk_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t ze_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t zy_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t cc_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t ck_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t ce_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t cy_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t dd_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t ds_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t dh_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t db_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t dx_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t ss_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t sh_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t sb_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef size_t sx_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0
cpdef int irs_xgesv(intptr_t handle, intptr_t gesv_irs_params, intptr_t gesv_irs_infos, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef size_t irs_xgesv_buffer_size(intptr_t handle, intptr_t params, int n, int nrhs) except? 0
cpdef int irs_xgels(intptr_t handle, intptr_t gels_irs_params, intptr_t gels_irs_infos, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1
cpdef size_t irs_xgels_buffer_size(intptr_t handle, intptr_t params, int m, int n, int nrhs) except? 0
cpdef int spotrf_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef int dpotrf_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef int cpotrf_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef int zpotrf_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef spotrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t workspace, int lwork, intptr_t dev_info)
cpdef dpotrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t workspace, int lwork, intptr_t dev_info)
cpdef cpotrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t workspace, int lwork, intptr_t dev_info)
cpdef zpotrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t workspace, int lwork, intptr_t dev_info)
cpdef spotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t a, int lda, intptr_t b, int ldb, intptr_t dev_info)
cpdef dpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t a, int lda, intptr_t b, int ldb, intptr_t dev_info)
cpdef cpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t a, int lda, intptr_t b, int ldb, intptr_t dev_info)
cpdef zpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t a, int lda, intptr_t b, int ldb, intptr_t dev_info)
cpdef spotrf_batched(intptr_t handle, int uplo, int n, aarray, int lda, intptr_t info_array, int batch_size)
cpdef dpotrf_batched(intptr_t handle, int uplo, int n, aarray, int lda, intptr_t info_array, int batch_size)
cpdef cpotrf_batched(intptr_t handle, int uplo, int n, aarray, int lda, intptr_t info_array, int batch_size)
cpdef zpotrf_batched(intptr_t handle, int uplo, int n, aarray, int lda, intptr_t info_array, int batch_size)
cpdef spotrs_batched(intptr_t handle, int uplo, int n, int nrhs, a, int lda, b, int ldb, intptr_t d_info, int batch_size)
cpdef dpotrs_batched(intptr_t handle, int uplo, int n, int nrhs, a, int lda, b, int ldb, intptr_t d_info, int batch_size)
cpdef cpotrs_batched(intptr_t handle, int uplo, int n, int nrhs, a, int lda, b, int ldb, intptr_t d_info, int batch_size)
cpdef zpotrs_batched(intptr_t handle, int uplo, int n, int nrhs, a, int lda, b, int ldb, intptr_t d_info, int batch_size)
cpdef int spotri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef int dpotri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef int cpotri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef int zpotri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef spotri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info)
cpdef dpotri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info)
cpdef cpotri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info)
cpdef zpotri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info)
cpdef int slauum_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef int dlauum_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef int clauum_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef int zlauum_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1
cpdef slauum(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info)
cpdef dlauum(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info)
cpdef clauum(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info)
cpdef zlauum(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info)
cpdef int sgetrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1
cpdef int dgetrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1
cpdef int cgetrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1
cpdef int zgetrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1
cpdef sgetrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t workspace, intptr_t dev_ipiv, intptr_t dev_info)
cpdef dgetrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t workspace, intptr_t dev_ipiv, intptr_t dev_info)
cpdef cgetrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t workspace, intptr_t dev_ipiv, intptr_t dev_info)
cpdef zgetrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t workspace, intptr_t dev_ipiv, intptr_t dev_info)
cpdef slaswp(intptr_t handle, int n, intptr_t a, int lda, int k1, int k2, intptr_t dev_ipiv, int incx)
cpdef dlaswp(intptr_t handle, int n, intptr_t a, int lda, int k1, int k2, intptr_t dev_ipiv, int incx)
cpdef claswp(intptr_t handle, int n, intptr_t a, int lda, int k1, int k2, intptr_t dev_ipiv, int incx)
cpdef zlaswp(intptr_t handle, int n, intptr_t a, int lda, int k1, int k2, intptr_t dev_ipiv, int incx)
cpdef sgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t a, int lda, intptr_t dev_ipiv, intptr_t b, int ldb, intptr_t dev_info)
cpdef dgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t a, int lda, intptr_t dev_ipiv, intptr_t b, int ldb, intptr_t dev_info)
cpdef cgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t a, int lda, intptr_t dev_ipiv, intptr_t b, int ldb, intptr_t dev_info)
cpdef zgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t a, int lda, intptr_t dev_ipiv, intptr_t b, int ldb, intptr_t dev_info)
cpdef int sgeqrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1
cpdef int dgeqrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1
cpdef int cgeqrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1
cpdef int zgeqrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1
cpdef sgeqrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t workspace, int lwork, intptr_t dev_info)
cpdef dgeqrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t workspace, int lwork, intptr_t dev_info)
cpdef cgeqrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t workspace, int lwork, intptr_t dev_info)
cpdef zgeqrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t workspace, int lwork, intptr_t dev_info)
cpdef int sorgqr_buffer_size(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1
cpdef int dorgqr_buffer_size(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1
cpdef int cungqr_buffer_size(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1
cpdef int zungqr_buffer_size(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1
cpdef sorgqr(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef dorgqr(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef cungqr(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef zungqr(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef int sormqr_buffer_size(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1
cpdef int dormqr_buffer_size(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1
cpdef int cunmqr_buffer_size(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1
cpdef int zunmqr_buffer_size(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1
cpdef sormqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t dev_info)
cpdef dormqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t dev_info)
cpdef cunmqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t dev_info)
cpdef zunmqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t dev_info)
cpdef int ssytrf_buffer_size(intptr_t handle, int n, intptr_t a, int lda) except? -1
cpdef int dsytrf_buffer_size(intptr_t handle, int n, intptr_t a, int lda) except? -1
cpdef int csytrf_buffer_size(intptr_t handle, int n, intptr_t a, int lda) except? -1
cpdef int zsytrf_buffer_size(intptr_t handle, int n, intptr_t a, int lda) except? -1
cpdef ssytrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef dsytrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef csytrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef zsytrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef int ssytri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv) except? -1
cpdef int dsytri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv) except? -1
cpdef int csytri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv) except? -1
cpdef int zsytri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv) except? -1
cpdef ssytri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef dsytri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef csytri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef zsytri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info)
cpdef int sgebrd_buffer_size(intptr_t handle, int m, int n) except? -1
cpdef int dgebrd_buffer_size(intptr_t handle, int m, int n) except? -1
cpdef int cgebrd_buffer_size(intptr_t handle, int m, int n) except? -1
cpdef int zgebrd_buffer_size(intptr_t handle, int m, int n) except? -1
cpdef sgebrd(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tauq, intptr_t taup, intptr_t work, int lwork, intptr_t dev_info)
cpdef dgebrd(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tauq, intptr_t taup, intptr_t work, int lwork, intptr_t dev_info)
cpdef cgebrd(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tauq, intptr_t taup, intptr_t work, int lwork, intptr_t dev_info)
cpdef zgebrd(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tauq, intptr_t taup, intptr_t work, int lwork, intptr_t dev_info)
cpdef int sorgbr_buffer_size(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1
cpdef int dorgbr_buffer_size(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1
cpdef int cungbr_buffer_size(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1
cpdef int zungbr_buffer_size(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1
cpdef sorgbr(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef dorgbr(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef cungbr(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef zungbr(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef int ssytrd_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau) except? -1
cpdef int dsytrd_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau) except? -1
cpdef int chetrd_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau) except? -1
cpdef int zhetrd_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau) except? -1
cpdef ssytrd(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef dsytrd(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef chetrd(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef zhetrd(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef int sorgtr_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau) except? -1
cpdef int dorgtr_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau) except? -1
cpdef int cungtr_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau) except? -1
cpdef int zungtr_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau) except? -1
cpdef sorgtr(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef dorgtr(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef cungtr(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef zungtr(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info)
cpdef int sormtr_buffer_size(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1
cpdef int dormtr_buffer_size(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1
cpdef int cunmtr_buffer_size(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1
cpdef int zunmtr_buffer_size(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1
cpdef sormtr(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t info)
cpdef dormtr(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t info)
cpdef cunmtr(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t info)
cpdef zunmtr(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t info)
cpdef int sgesvd_buffer_size(intptr_t handle, int m, int n) except? -1
cpdef int dgesvd_buffer_size(intptr_t handle, int m, int n) except? -1
cpdef int cgesvd_buffer_size(intptr_t handle, int m, int n) except? -1
cpdef int zgesvd_buffer_size(intptr_t handle, int m, int n) except? -1
cpdef sgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t vt, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info)
cpdef dgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t vt, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info)
cpdef cgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t vt, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info)
cpdef zgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t vt, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info)
cpdef int ssyevd_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w) except? -1
cpdef int dsyevd_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w) except? -1
cpdef int cheevd_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w) except? -1
cpdef int zheevd_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w) except? -1
cpdef ssyevd(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef dsyevd(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef cheevd(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef zheevd(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef int ssyevdx_buffer_size(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w) except? -1
cpdef int dsyevdx_buffer_size(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w) except? -1
cpdef int cheevdx_buffer_size(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w) except? -1
cpdef int zheevdx_buffer_size(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w) except? -1
cpdef int ssyevdx(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, float vl, float vu, int il, int iu, intptr_t w, intptr_t work, int lwork, intptr_t info) except? -1
cpdef int dsyevdx(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, double vl, double vu, int il, int iu, intptr_t w, intptr_t work, int lwork, intptr_t info) except? -1
cpdef int cheevdx(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, float vl, float vu, int il, int iu, intptr_t w, intptr_t work, int lwork, intptr_t info) except? -1
cpdef int zheevdx(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, double vl, double vu, int il, int iu, intptr_t w, intptr_t work, int lwork, intptr_t info) except? -1
cpdef int ssygvdx_buffer_size(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w) except? -1
cpdef int dsygvdx_buffer_size(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w) except? -1
cpdef int chegvdx_buffer_size(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w) except? -1
cpdef int zhegvdx_buffer_size(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w) except? -1
cpdef ssygvdx(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef dsygvdx(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef chegvdx(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef zhegvdx(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef int ssygvd_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w) except? -1
cpdef int dsygvd_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w) except? -1
cpdef int chegvd_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w) except? -1
cpdef int zhegvd_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w) except? -1
cpdef ssygvd(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef dsygvd(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef chegvd(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef zhegvd(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info)
cpdef intptr_t create_syevj_info() except? 0
cpdef destroy_syevj_info(intptr_t info)
cpdef xsyevj_set_tolerance(intptr_t info, double tolerance)
cpdef xsyevj_set_max_sweeps(intptr_t info, int max_sweeps)
cpdef xsyevj_set_sort_eig(intptr_t info, int sort_eig)
cpdef double xsyevj_get_residual(intptr_t handle, intptr_t info) except? 0
cpdef int xsyevj_get_sweeps(intptr_t handle, intptr_t info) except? -1
cpdef int ssyevj_batched_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params, int batch_size) except? -1
cpdef int dsyevj_batched_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params, int batch_size) except? -1
cpdef int cheevj_batched_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params, int batch_size) except? -1
cpdef int zheevj_batched_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params, int batch_size) except? -1
cpdef ssyevj_batched(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size)
cpdef dsyevj_batched(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size)
cpdef cheevj_batched(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size)
cpdef zheevj_batched(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size)
cpdef int ssyevj_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params) except? -1
cpdef int dsyevj_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params) except? -1
cpdef int cheevj_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params) except? -1
cpdef int zheevj_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params) except? -1
cpdef ssyevj(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef dsyevj(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef cheevj(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef zheevj(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef int ssygvj_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t params) except? -1
cpdef int dsygvj_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t params) except? -1
cpdef int chegvj_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t params) except? -1
cpdef int zhegvj_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t params) except? -1
cpdef ssygvj(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef dsygvj(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef chegvj(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef zhegvj(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef intptr_t create_gesvdj_info() except? 0
cpdef destroy_gesvdj_info(intptr_t info)
cpdef xgesvdj_set_tolerance(intptr_t info, double tolerance)
cpdef xgesvdj_set_max_sweeps(intptr_t info, int max_sweeps)
cpdef xgesvdj_set_sort_eig(intptr_t info, int sort_svd)
cpdef double xgesvdj_get_residual(intptr_t handle, intptr_t info) except? 0
cpdef int xgesvdj_get_sweeps(intptr_t handle, intptr_t info) except? -1
cpdef int sgesvdj_batched_buffer_size(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params, int batch_size) except? -1
cpdef int dgesvdj_batched_buffer_size(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params, int batch_size) except? -1
cpdef int cgesvdj_batched_buffer_size(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params, int batch_size) except? -1
cpdef int zgesvdj_batched_buffer_size(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params, int batch_size) except? -1
cpdef sgesvdj_batched(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size)
cpdef dgesvdj_batched(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size)
cpdef cgesvdj_batched(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size)
cpdef zgesvdj_batched(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size)
cpdef int sgesvdj_buffer_size(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params) except? -1
cpdef int dgesvdj_buffer_size(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params) except? -1
cpdef int cgesvdj_buffer_size(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params) except? -1
cpdef int zgesvdj_buffer_size(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params) except? -1
cpdef sgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef dgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef cgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef zgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params)
cpdef int sgesvda_strided_batched_buffer_size(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, int batch_size) except? -1
cpdef int dgesvda_strided_batched_buffer_size(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, int batch_size) except? -1
cpdef int cgesvda_strided_batched_buffer_size(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, int batch_size) except? -1
cpdef int zgesvda_strided_batched_buffer_size(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, int batch_size) except? -1
cpdef sgesvda_strided_batched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_r_nrm_f, int batch_size)
cpdef dgesvda_strided_batched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_r_nrm_f, int batch_size)
cpdef cgesvda_strided_batched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_r_nrm_f, int batch_size)
cpdef zgesvda_strided_batched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_r_nrm_f, int batch_size)
cpdef intptr_t create_params() except? 0
cpdef destroy_params(intptr_t params)
cpdef set_adv_options(intptr_t params, int function, int algo)
cpdef tuple xpotrf_buffer_size(intptr_t handle, intptr_t params, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int compute_type)
cpdef xpotrf(intptr_t handle, intptr_t params, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info)
cpdef xpotrs(intptr_t handle, intptr_t params, int uplo, int64_t n, int64_t nrhs, int data_type_a, intptr_t a, int64_t lda, int data_type_b, intptr_t b, int64_t ldb, intptr_t info)
cpdef tuple xgeqrf_buffer_size(intptr_t handle, intptr_t params, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_tau, intptr_t tau, int compute_type)
cpdef xgeqrf(intptr_t handle, intptr_t params, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_tau, intptr_t tau, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info)
cpdef tuple xgetrf_buffer_size(intptr_t handle, intptr_t params, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int compute_type)
cpdef xgetrf(intptr_t handle, intptr_t params, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, intptr_t ipiv, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info)
cpdef xgetrs(intptr_t handle, intptr_t params, int trans, int64_t n, int64_t nrhs, int data_type_a, intptr_t a, int64_t lda, intptr_t ipiv, int data_type_b, intptr_t b, int64_t ldb, intptr_t info)
cpdef tuple xsyevd_buffer_size(intptr_t handle, intptr_t params, int jobz, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int compute_type)
cpdef xsyevd(intptr_t handle, intptr_t params, int jobz, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info)
cpdef tuple xsyevdx_buffer_size(intptr_t handle, intptr_t params, int jobz, int range, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, intptr_t vl, intptr_t vu, int64_t il, int64_t iu, intptr_t h_meig, int data_type_w, intptr_t w, int compute_type)
cpdef int64_t xsyevdx(intptr_t handle, intptr_t params, int jobz, int range, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, intptr_t vl, intptr_t vu, int64_t il, int64_t iu, int data_type_w, intptr_t w, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info) except? -1
cpdef tuple xgesvd_buffer_size(intptr_t handle, intptr_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_s, intptr_t s, int data_type_u, intptr_t u, int64_t ldu, int data_type_vt, intptr_t vt, int64_t ldvt, int compute_type)
cpdef xgesvd(intptr_t handle, intptr_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_s, intptr_t s, int data_type_u, intptr_t u, int64_t ldu, int data_type_vt, intptr_t vt, int64_t ldvt, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info)
cpdef tuple xgesvdp_buffer_size(intptr_t handle, intptr_t params, int jobz, int econ, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_s, intptr_t s, int data_type_u, intptr_t u, int64_t ldu, int data_type_v, intptr_t v, int64_t ldv, int compute_type)
cpdef double xgesvdp(intptr_t handle, intptr_t params, int jobz, int econ, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_s, intptr_t s, int data_type_u, intptr_t u, int64_t ldu, int data_type_v, intptr_t v, int64_t ldv, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t d_info) except? 0
cpdef tuple xgesvdr_buffer_size(intptr_t handle, intptr_t params, signed char jobu, signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, int data_type_a, intptr_t a, int64_t lda, int data_type_srand, intptr_t srand, int data_type_urand, intptr_t urand, int64_t ld_urand, int data_type_vrand, intptr_t vrand, int64_t ld_vrand, int compute_type)
cpdef xgesvdr(intptr_t handle, intptr_t params, signed char jobu, signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, int data_type_a, intptr_t a, int64_t lda, int data_type_srand, intptr_t srand, int data_type_urand, intptr_t urand, int64_t ld_urand, int data_type_vrand, intptr_t vrand, int64_t ld_vrand, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t d_info)
cpdef tuple xsytrs_buffer_size(intptr_t handle, int uplo, int64_t n, int64_t nrhs, int data_type_a, intptr_t a, int64_t lda, intptr_t ipiv, int data_type_b, intptr_t b, int64_t ldb)
cpdef xsytrs(intptr_t handle, int uplo, int64_t n, int64_t nrhs, int data_type_a, intptr_t a, int64_t lda, intptr_t ipiv, int data_type_b, intptr_t b, int64_t ldb, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info)
cpdef tuple xtrtri_buffer_size(intptr_t handle, int uplo, int diag, int64_t n, int data_type_a, intptr_t a, int64_t lda)
cpdef xtrtri(intptr_t handle, int uplo, int diag, int64_t n, int data_type_a, intptr_t a, int64_t lda, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t dev_info)
cpdef logger_open_file(log_file)
cpdef logger_set_level(int level)
cpdef logger_set_mask(int mask)
cpdef logger_force_disable()
cpdef set_deterministic_mode(intptr_t handle, int mode)
cpdef int get_deterministic_mode(intptr_t handle) except *
cpdef tuple xlarft_buffer_size(intptr_t handle, intptr_t params, cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n, int64_t k, int data_type_v, intptr_t v, int64_t ldv, int data_type_tau, intptr_t tau, int data_type_t, intptr_t t, int64_t ldt, int compute_type)
cpdef xlarft(intptr_t handle, intptr_t params, cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n, int64_t k, int data_type_v, intptr_t v, int64_t ldv, int data_type_tau, intptr_t tau, int data_type_t, intptr_t t, int64_t ldt, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host)
cpdef tuple xsyev_batched_buffer_size(intptr_t handle, intptr_t params, int jobz, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int compute_type, int64_t batch_size)
cpdef xsyev_batched(intptr_t handle, intptr_t params, int jobz, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info, int64_t batch_size)
cpdef tuple xgeev_buffer_size(intptr_t handle, intptr_t params, int jobvl, int jobvr, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int data_type_vl, intptr_t vl, int64_t ldvl, int data_type_vr, intptr_t vr, int64_t ldvr, int compute_type)
cpdef xgeev(intptr_t handle, intptr_t params, int jobvl, int jobvr, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int data_type_vl, intptr_t vl, int64_t ldvl, int data_type_vr, intptr_t vr, int64_t ldvr, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info)
