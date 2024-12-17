# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from libcpp.vector cimport vector

from .cusolver cimport check_status
from ._internal.utils cimport get_resource_ptrs, nullable_unique_ptr

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################

class Function(_IntEnum):
    """See `cusolverDnFunction_t`."""
    GETRF = CUSOLVERDN_GETRF
    POTRF = CUSOLVERDN_POTRF


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create() except? 0:
    """See `cusolverDnCreate`."""
    cdef Handle handle
    with nogil:
        status = cusolverDnCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """See `cusolverDnDestroy`."""
    with nogil:
        status = cusolverDnDestroy(<Handle>handle)
    check_status(status)


cpdef set_stream(intptr_t handle, intptr_t stream_id):
    """See `cusolverDnSetStream`."""
    with nogil:
        status = cusolverDnSetStream(<Handle>handle, <Stream>stream_id)
    check_status(status)


cpdef intptr_t get_stream(intptr_t handle) except? 0:
    """See `cusolverDnGetStream`."""
    cdef Stream stream_id
    with nogil:
        status = cusolverDnGetStream(<Handle>handle, &stream_id)
    check_status(status)
    return <intptr_t>stream_id


cpdef intptr_t irs_params_create() except? 0:
    """See `cusolverDnIRSParamsCreate`."""
    cdef IRSParams params_ptr
    with nogil:
        status = cusolverDnIRSParamsCreate(&params_ptr)
    check_status(status)
    return <intptr_t>params_ptr


cpdef irs_params_destroy(intptr_t params):
    """See `cusolverDnIRSParamsDestroy`."""
    with nogil:
        status = cusolverDnIRSParamsDestroy(<IRSParams>params)
    check_status(status)


cpdef irs_params_set_refinement_solver(intptr_t params, int refinement_solver):
    """See `cusolverDnIRSParamsSetRefinementSolver`."""
    with nogil:
        status = cusolverDnIRSParamsSetRefinementSolver(<IRSParams>params, <cusolverIRSRefinement_t>refinement_solver)
    check_status(status)


cpdef irs_params_set_solver_main_precision(intptr_t params, int solver_main_precision):
    """See `cusolverDnIRSParamsSetSolverMainPrecision`."""
    with nogil:
        status = cusolverDnIRSParamsSetSolverMainPrecision(<IRSParams>params, <cusolverPrecType_t>solver_main_precision)
    check_status(status)


cpdef irs_params_set_solver_lowest_precision(intptr_t params, int solver_lowest_precision):
    """See `cusolverDnIRSParamsSetSolverLowestPrecision`."""
    with nogil:
        status = cusolverDnIRSParamsSetSolverLowestPrecision(<IRSParams>params, <cusolverPrecType_t>solver_lowest_precision)
    check_status(status)


cpdef irs_params_set_solver_precisions(intptr_t params, int solver_main_precision, int solver_lowest_precision):
    """See `cusolverDnIRSParamsSetSolverPrecisions`."""
    with nogil:
        status = cusolverDnIRSParamsSetSolverPrecisions(<IRSParams>params, <cusolverPrecType_t>solver_main_precision, <cusolverPrecType_t>solver_lowest_precision)
    check_status(status)


cpdef irs_params_set_tol(intptr_t params, double val):
    """See `cusolverDnIRSParamsSetTol`."""
    with nogil:
        status = cusolverDnIRSParamsSetTol(<IRSParams>params, val)
    check_status(status)


cpdef irs_params_set_tol_inner(intptr_t params, double val):
    """See `cusolverDnIRSParamsSetTolInner`."""
    with nogil:
        status = cusolverDnIRSParamsSetTolInner(<IRSParams>params, val)
    check_status(status)


cpdef irs_params_set_max_iters(intptr_t params, int maxiters):
    """See `cusolverDnIRSParamsSetMaxIters`."""
    with nogil:
        status = cusolverDnIRSParamsSetMaxIters(<IRSParams>params, <cusolver_int_t>maxiters)
    check_status(status)


cpdef irs_params_set_max_iters_inner(intptr_t params, int maxiters_inner):
    """See `cusolverDnIRSParamsSetMaxItersInner`."""
    with nogil:
        status = cusolverDnIRSParamsSetMaxItersInner(<IRSParams>params, <cusolver_int_t>maxiters_inner)
    check_status(status)


cpdef int irs_params_get_max_iters(intptr_t params) except? -1:
    """See `cusolverDnIRSParamsGetMaxIters`."""
    cdef cusolver_int_t maxiters
    with nogil:
        status = cusolverDnIRSParamsGetMaxIters(<IRSParams>params, &maxiters)
    check_status(status)
    return <int>maxiters


cpdef irs_params_enable_fallback(intptr_t params):
    """See `cusolverDnIRSParamsEnableFallback`."""
    with nogil:
        status = cusolverDnIRSParamsEnableFallback(<IRSParams>params)
    check_status(status)


cpdef irs_params_disable_fallback(intptr_t params):
    """See `cusolverDnIRSParamsDisableFallback`."""
    with nogil:
        status = cusolverDnIRSParamsDisableFallback(<IRSParams>params)
    check_status(status)


cpdef irs_infos_destroy(intptr_t infos):
    """See `cusolverDnIRSInfosDestroy`."""
    with nogil:
        status = cusolverDnIRSInfosDestroy(<IRSInfos>infos)
    check_status(status)


cpdef intptr_t irs_infos_create() except? 0:
    """See `cusolverDnIRSInfosCreate`."""
    cdef IRSInfos infos_ptr
    with nogil:
        status = cusolverDnIRSInfosCreate(&infos_ptr)
    check_status(status)
    return <intptr_t>infos_ptr


cpdef int irs_infos_get_niters(intptr_t infos) except? -1:
    """See `cusolverDnIRSInfosGetNiters`."""
    cdef cusolver_int_t niters
    with nogil:
        status = cusolverDnIRSInfosGetNiters(<IRSInfos>infos, &niters)
    check_status(status)
    return <int>niters


cpdef int irs_infos_get_outer_niters(intptr_t infos) except? -1:
    """See `cusolverDnIRSInfosGetOuterNiters`."""
    cdef cusolver_int_t outer_niters
    with nogil:
        status = cusolverDnIRSInfosGetOuterNiters(<IRSInfos>infos, &outer_niters)
    check_status(status)
    return <int>outer_niters


cpdef irs_infos_request_residual(intptr_t infos):
    """See `cusolverDnIRSInfosRequestResidual`."""
    with nogil:
        status = cusolverDnIRSInfosRequestResidual(<IRSInfos>infos)
    check_status(status)


cpdef irs_infos_get_residual_history(intptr_t infos, intptr_t residual_history):
    """See `cusolverDnIRSInfosGetResidualHistory`."""
    with nogil:
        status = cusolverDnIRSInfosGetResidualHistory(<IRSInfos>infos, <void**>residual_history)
    check_status(status)


cpdef int irs_infos_get_max_iters(intptr_t infos) except? -1:
    """See `cusolverDnIRSInfosGetMaxIters`."""
    cdef cusolver_int_t maxiters
    with nogil:
        status = cusolverDnIRSInfosGetMaxIters(<IRSInfos>infos, &maxiters)
    check_status(status)
    return <int>maxiters


cpdef int zz_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnZZgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnZZgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int zc_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnZCgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnZCgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int zk_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnZKgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnZKgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int ze_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnZEgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnZEgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int zy_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnZYgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnZYgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int cc_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnCCgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnCCgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int ce_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnCEgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnCEgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int ck_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnCKgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnCKgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int cy_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnCYgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnCYgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int dd_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnDDgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnDDgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int ds_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnDSgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnDSgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int dh_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnDHgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnDHgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int db_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnDBgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnDBgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int dx_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnDXgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnDXgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int ss_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnSSgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnSSgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int sh_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnSHgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnSHgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int sb_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnSBgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnSBgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int sx_gesv(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnSXgesv`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnSXgesv(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef size_t zz_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnZZgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnZZgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t zc_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnZCgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnZCgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t zk_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnZKgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnZKgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t ze_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnZEgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnZEgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t zy_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnZYgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnZYgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t cc_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnCCgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnCCgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t ck_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnCKgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnCKgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t ce_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnCEgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnCEgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t cy_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnCYgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnCYgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t dd_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnDDgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnDDgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t ds_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnDSgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnDSgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t dh_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnDHgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnDHgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t db_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnDBgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnDBgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t dx_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnDXgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnDXgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t ss_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnSSgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnSSgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t sh_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnSHgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnSHgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t sb_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnSBgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnSBgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t sx_gesv_buffer_size(intptr_t handle, int n, int nrhs, intptr_t d_a, int ldda, intptr_t dipiv, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnSXgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnSXgesv_bufferSize(<Handle>handle, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <cusolver_int_t*>dipiv, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef int zz_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnZZgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnZZgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int zc_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnZCgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnZCgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int zk_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnZKgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnZKgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int ze_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnZEgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnZEgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int zy_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnZYgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnZYgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int cc_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnCCgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnCCgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int ck_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnCKgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnCKgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int ce_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnCEgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnCEgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int cy_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnCYgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnCYgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int dd_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnDDgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnDDgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int ds_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnDSgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnDSgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int dh_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnDHgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnDHgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int db_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnDBgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnDBgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int dx_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnDXgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnDXgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int ss_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnSSgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnSSgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int sh_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnSHgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnSHgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int sb_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnSBgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnSBgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef int sx_gels(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnSXgels`."""
    cdef cusolver_int_t iter
    with nogil:
        status = cusolverDnSXgels(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &iter, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>iter


cpdef size_t zz_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnZZgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnZZgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t zc_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnZCgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnZCgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t zk_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnZKgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnZKgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t ze_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnZEgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnZEgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t zy_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnZYgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnZYgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuDoubleComplex*>d_a, <cusolver_int_t>ldda, <cuDoubleComplex*>d_b, <cusolver_int_t>lddb, <cuDoubleComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t cc_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnCCgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnCCgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t ck_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnCKgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnCKgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t ce_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnCEgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnCEgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t cy_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnCYgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnCYgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <cuComplex*>d_a, <cusolver_int_t>ldda, <cuComplex*>d_b, <cusolver_int_t>lddb, <cuComplex*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t dd_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnDDgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnDDgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t ds_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnDSgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnDSgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t dh_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnDHgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnDHgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t db_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnDBgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnDBgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t dx_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnDXgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnDXgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <double*>d_a, <cusolver_int_t>ldda, <double*>d_b, <cusolver_int_t>lddb, <double*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t ss_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnSSgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnSSgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t sh_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnSHgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnSHgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t sb_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnSBgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnSBgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef size_t sx_gels_buffer_size(intptr_t handle, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace) except? 0:
    """See `cusolverDnSXgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnSXgels_bufferSize(<Handle>handle, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <float*>d_a, <cusolver_int_t>ldda, <float*>d_b, <cusolver_int_t>lddb, <float*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef int irs_xgesv(intptr_t handle, intptr_t gesv_irs_params, intptr_t gesv_irs_infos, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnIRSXgesv`."""
    cdef cusolver_int_t niters
    with nogil:
        status = cusolverDnIRSXgesv(<Handle>handle, <IRSParams>gesv_irs_params, <IRSInfos>gesv_irs_infos, <cusolver_int_t>n, <cusolver_int_t>nrhs, <void*>d_a, <cusolver_int_t>ldda, <void*>d_b, <cusolver_int_t>lddb, <void*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &niters, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>niters


cpdef size_t irs_xgesv_buffer_size(intptr_t handle, intptr_t params, int n, int nrhs) except? 0:
    """See `cusolverDnIRSXgesv_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnIRSXgesv_bufferSize(<Handle>handle, <IRSParams>params, <cusolver_int_t>n, <cusolver_int_t>nrhs, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef int irs_xgels(intptr_t handle, intptr_t gels_irs_params, intptr_t gels_irs_infos, int m, int n, int nrhs, intptr_t d_a, int ldda, intptr_t d_b, int lddb, intptr_t d_x, int lddx, intptr_t d_workspace, size_t lwork_bytes, intptr_t d_info) except? -1:
    """See `cusolverDnIRSXgels`."""
    cdef cusolver_int_t niters
    with nogil:
        status = cusolverDnIRSXgels(<Handle>handle, <IRSParams>gels_irs_params, <IRSInfos>gels_irs_infos, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, <void*>d_a, <cusolver_int_t>ldda, <void*>d_b, <cusolver_int_t>lddb, <void*>d_x, <cusolver_int_t>lddx, <void*>d_workspace, lwork_bytes, &niters, <cusolver_int_t*>d_info)
    check_status(status)
    return <int>niters


cpdef size_t irs_xgels_buffer_size(intptr_t handle, intptr_t params, int m, int n, int nrhs) except? 0:
    """See `cusolverDnIRSXgels_bufferSize`."""
    cdef size_t lwork_bytes
    with nogil:
        status = cusolverDnIRSXgels_bufferSize(<Handle>handle, <IRSParams>params, <cusolver_int_t>m, <cusolver_int_t>n, <cusolver_int_t>nrhs, &lwork_bytes)
    check_status(status)
    return lwork_bytes


cpdef int spotrf_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnSpotrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSpotrf_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int dpotrf_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnDpotrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDpotrf_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int cpotrf_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnCpotrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCpotrf_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int zpotrf_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnZpotrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZpotrf_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef spotrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t workspace, int lwork, intptr_t dev_info):
    """See `cusolverDnSpotrf`."""
    with nogil:
        status = cusolverDnSpotrf(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, <float*>workspace, lwork, <int*>dev_info)
    check_status(status)


cpdef dpotrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t workspace, int lwork, intptr_t dev_info):
    """See `cusolverDnDpotrf`."""
    with nogil:
        status = cusolverDnDpotrf(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, <double*>workspace, lwork, <int*>dev_info)
    check_status(status)


cpdef cpotrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t workspace, int lwork, intptr_t dev_info):
    """See `cusolverDnCpotrf`."""
    with nogil:
        status = cusolverDnCpotrf(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <cuComplex*>workspace, lwork, <int*>dev_info)
    check_status(status)


cpdef zpotrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t workspace, int lwork, intptr_t dev_info):
    """See `cusolverDnZpotrf`."""
    with nogil:
        status = cusolverDnZpotrf(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <cuDoubleComplex*>workspace, lwork, <int*>dev_info)
    check_status(status)


cpdef spotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t a, int lda, intptr_t b, int ldb, intptr_t dev_info):
    """See `cusolverDnSpotrs`."""
    with nogil:
        status = cusolverDnSpotrs(<Handle>handle, <cublasFillMode_t>uplo, n, nrhs, <const float*>a, lda, <float*>b, ldb, <int*>dev_info)
    check_status(status)


cpdef dpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t a, int lda, intptr_t b, int ldb, intptr_t dev_info):
    """See `cusolverDnDpotrs`."""
    with nogil:
        status = cusolverDnDpotrs(<Handle>handle, <cublasFillMode_t>uplo, n, nrhs, <const double*>a, lda, <double*>b, ldb, <int*>dev_info)
    check_status(status)


cpdef cpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t a, int lda, intptr_t b, int ldb, intptr_t dev_info):
    """See `cusolverDnCpotrs`."""
    with nogil:
        status = cusolverDnCpotrs(<Handle>handle, <cublasFillMode_t>uplo, n, nrhs, <const cuComplex*>a, lda, <cuComplex*>b, ldb, <int*>dev_info)
    check_status(status)


cpdef zpotrs(intptr_t handle, int uplo, int n, int nrhs, intptr_t a, int lda, intptr_t b, int ldb, intptr_t dev_info):
    """See `cusolverDnZpotrs`."""
    with nogil:
        status = cusolverDnZpotrs(<Handle>handle, <cublasFillMode_t>uplo, n, nrhs, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>b, ldb, <int*>dev_info)
    check_status(status)


cpdef spotrf_batched(intptr_t handle, int uplo, int n, aarray, int lda, intptr_t info_array, int batch_size):
    """See `cusolverDnSpotrfBatched`."""
    cdef nullable_unique_ptr[ vector[float*] ] _aarray_
    get_resource_ptrs[float](_aarray_, aarray, <float*>NULL)
    with nogil:
        status = cusolverDnSpotrfBatched(<Handle>handle, <cublasFillMode_t>uplo, n, <float**>(_aarray_.data()), lda, <int*>info_array, batch_size)
    check_status(status)


cpdef dpotrf_batched(intptr_t handle, int uplo, int n, aarray, int lda, intptr_t info_array, int batch_size):
    """See `cusolverDnDpotrfBatched`."""
    cdef nullable_unique_ptr[ vector[double*] ] _aarray_
    get_resource_ptrs[double](_aarray_, aarray, <double*>NULL)
    with nogil:
        status = cusolverDnDpotrfBatched(<Handle>handle, <cublasFillMode_t>uplo, n, <double**>(_aarray_.data()), lda, <int*>info_array, batch_size)
    check_status(status)


cpdef cpotrf_batched(intptr_t handle, int uplo, int n, aarray, int lda, intptr_t info_array, int batch_size):
    """See `cusolverDnCpotrfBatched`."""
    cdef nullable_unique_ptr[ vector[cuComplex*] ] _aarray_
    get_resource_ptrs[cuComplex](_aarray_, aarray, <cuComplex*>NULL)
    with nogil:
        status = cusolverDnCpotrfBatched(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex**>(_aarray_.data()), lda, <int*>info_array, batch_size)
    check_status(status)


cpdef zpotrf_batched(intptr_t handle, int uplo, int n, aarray, int lda, intptr_t info_array, int batch_size):
    """See `cusolverDnZpotrfBatched`."""
    cdef nullable_unique_ptr[ vector[cuDoubleComplex*] ] _aarray_
    get_resource_ptrs[cuDoubleComplex](_aarray_, aarray, <cuDoubleComplex*>NULL)
    with nogil:
        status = cusolverDnZpotrfBatched(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex**>(_aarray_.data()), lda, <int*>info_array, batch_size)
    check_status(status)


cpdef spotrs_batched(intptr_t handle, int uplo, int n, int nrhs, a, int lda, b, int ldb, intptr_t d_info, int batch_size):
    """See `cusolverDnSpotrsBatched`."""
    cdef nullable_unique_ptr[ vector[float*] ] _a_
    get_resource_ptrs[float](_a_, a, <float*>NULL)
    cdef nullable_unique_ptr[ vector[float*] ] _b_
    get_resource_ptrs[float](_b_, b, <float*>NULL)
    with nogil:
        status = cusolverDnSpotrsBatched(<Handle>handle, <cublasFillMode_t>uplo, n, nrhs, <float**>(_a_.data()), lda, <float**>(_b_.data()), ldb, <int*>d_info, batch_size)
    check_status(status)


cpdef dpotrs_batched(intptr_t handle, int uplo, int n, int nrhs, a, int lda, b, int ldb, intptr_t d_info, int batch_size):
    """See `cusolverDnDpotrsBatched`."""
    cdef nullable_unique_ptr[ vector[double*] ] _a_
    get_resource_ptrs[double](_a_, a, <double*>NULL)
    cdef nullable_unique_ptr[ vector[double*] ] _b_
    get_resource_ptrs[double](_b_, b, <double*>NULL)
    with nogil:
        status = cusolverDnDpotrsBatched(<Handle>handle, <cublasFillMode_t>uplo, n, nrhs, <double**>(_a_.data()), lda, <double**>(_b_.data()), ldb, <int*>d_info, batch_size)
    check_status(status)


cpdef cpotrs_batched(intptr_t handle, int uplo, int n, int nrhs, a, int lda, b, int ldb, intptr_t d_info, int batch_size):
    """See `cusolverDnCpotrsBatched`."""
    cdef nullable_unique_ptr[ vector[cuComplex*] ] _a_
    get_resource_ptrs[cuComplex](_a_, a, <cuComplex*>NULL)
    cdef nullable_unique_ptr[ vector[cuComplex*] ] _b_
    get_resource_ptrs[cuComplex](_b_, b, <cuComplex*>NULL)
    with nogil:
        status = cusolverDnCpotrsBatched(<Handle>handle, <cublasFillMode_t>uplo, n, nrhs, <cuComplex**>(_a_.data()), lda, <cuComplex**>(_b_.data()), ldb, <int*>d_info, batch_size)
    check_status(status)


cpdef zpotrs_batched(intptr_t handle, int uplo, int n, int nrhs, a, int lda, b, int ldb, intptr_t d_info, int batch_size):
    """See `cusolverDnZpotrsBatched`."""
    cdef nullable_unique_ptr[ vector[cuDoubleComplex*] ] _a_
    get_resource_ptrs[cuDoubleComplex](_a_, a, <cuDoubleComplex*>NULL)
    cdef nullable_unique_ptr[ vector[cuDoubleComplex*] ] _b_
    get_resource_ptrs[cuDoubleComplex](_b_, b, <cuDoubleComplex*>NULL)
    with nogil:
        status = cusolverDnZpotrsBatched(<Handle>handle, <cublasFillMode_t>uplo, n, nrhs, <cuDoubleComplex**>(_a_.data()), lda, <cuDoubleComplex**>(_b_.data()), ldb, <int*>d_info, batch_size)
    check_status(status)


cpdef int spotri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnSpotri_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSpotri_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int dpotri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnDpotri_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDpotri_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int cpotri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnCpotri_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCpotri_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int zpotri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnZpotri_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZpotri_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef spotri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnSpotri`."""
    with nogil:
        status = cusolverDnSpotri(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, <float*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef dpotri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnDpotri`."""
    with nogil:
        status = cusolverDnDpotri(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, <double*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef cpotri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnCpotri`."""
    with nogil:
        status = cusolverDnCpotri(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <cuComplex*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef zpotri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnZpotri`."""
    with nogil:
        status = cusolverDnZpotri(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <cuDoubleComplex*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef int slauum_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnSlauum_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSlauum_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int dlauum_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnDlauum_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDlauum_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int clauum_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnClauum_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnClauum_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int zlauum_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnZlauum_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZlauum_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef slauum(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnSlauum`."""
    with nogil:
        status = cusolverDnSlauum(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, <float*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef dlauum(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnDlauum`."""
    with nogil:
        status = cusolverDnDlauum(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, <double*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef clauum(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnClauum`."""
    with nogil:
        status = cusolverDnClauum(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <cuComplex*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef zlauum(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnZlauum`."""
    with nogil:
        status = cusolverDnZlauum(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <cuDoubleComplex*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef int sgetrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnSgetrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSgetrf_bufferSize(<Handle>handle, m, n, <float*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int dgetrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnDgetrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDgetrf_bufferSize(<Handle>handle, m, n, <double*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int cgetrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnCgetrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCgetrf_bufferSize(<Handle>handle, m, n, <cuComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int zgetrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnZgetrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZgetrf_bufferSize(<Handle>handle, m, n, <cuDoubleComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef sgetrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t workspace, intptr_t dev_ipiv, intptr_t dev_info):
    """See `cusolverDnSgetrf`."""
    with nogil:
        status = cusolverDnSgetrf(<Handle>handle, m, n, <float*>a, lda, <float*>workspace, <int*>dev_ipiv, <int*>dev_info)
    check_status(status)


cpdef dgetrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t workspace, intptr_t dev_ipiv, intptr_t dev_info):
    """See `cusolverDnDgetrf`."""
    with nogil:
        status = cusolverDnDgetrf(<Handle>handle, m, n, <double*>a, lda, <double*>workspace, <int*>dev_ipiv, <int*>dev_info)
    check_status(status)


cpdef cgetrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t workspace, intptr_t dev_ipiv, intptr_t dev_info):
    """See `cusolverDnCgetrf`."""
    with nogil:
        status = cusolverDnCgetrf(<Handle>handle, m, n, <cuComplex*>a, lda, <cuComplex*>workspace, <int*>dev_ipiv, <int*>dev_info)
    check_status(status)


cpdef zgetrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t workspace, intptr_t dev_ipiv, intptr_t dev_info):
    """See `cusolverDnZgetrf`."""
    with nogil:
        status = cusolverDnZgetrf(<Handle>handle, m, n, <cuDoubleComplex*>a, lda, <cuDoubleComplex*>workspace, <int*>dev_ipiv, <int*>dev_info)
    check_status(status)


cpdef slaswp(intptr_t handle, int n, intptr_t a, int lda, int k1, int k2, intptr_t dev_ipiv, int incx):
    """See `cusolverDnSlaswp`."""
    with nogil:
        status = cusolverDnSlaswp(<Handle>handle, n, <float*>a, lda, k1, k2, <const int*>dev_ipiv, incx)
    check_status(status)


cpdef dlaswp(intptr_t handle, int n, intptr_t a, int lda, int k1, int k2, intptr_t dev_ipiv, int incx):
    """See `cusolverDnDlaswp`."""
    with nogil:
        status = cusolverDnDlaswp(<Handle>handle, n, <double*>a, lda, k1, k2, <const int*>dev_ipiv, incx)
    check_status(status)


cpdef claswp(intptr_t handle, int n, intptr_t a, int lda, int k1, int k2, intptr_t dev_ipiv, int incx):
    """See `cusolverDnClaswp`."""
    with nogil:
        status = cusolverDnClaswp(<Handle>handle, n, <cuComplex*>a, lda, k1, k2, <const int*>dev_ipiv, incx)
    check_status(status)


cpdef zlaswp(intptr_t handle, int n, intptr_t a, int lda, int k1, int k2, intptr_t dev_ipiv, int incx):
    """See `cusolverDnZlaswp`."""
    with nogil:
        status = cusolverDnZlaswp(<Handle>handle, n, <cuDoubleComplex*>a, lda, k1, k2, <const int*>dev_ipiv, incx)
    check_status(status)


cpdef sgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t a, int lda, intptr_t dev_ipiv, intptr_t b, int ldb, intptr_t dev_info):
    """See `cusolverDnSgetrs`."""
    with nogil:
        status = cusolverDnSgetrs(<Handle>handle, <cublasOperation_t>trans, n, nrhs, <const float*>a, lda, <const int*>dev_ipiv, <float*>b, ldb, <int*>dev_info)
    check_status(status)


cpdef dgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t a, int lda, intptr_t dev_ipiv, intptr_t b, int ldb, intptr_t dev_info):
    """See `cusolverDnDgetrs`."""
    with nogil:
        status = cusolverDnDgetrs(<Handle>handle, <cublasOperation_t>trans, n, nrhs, <const double*>a, lda, <const int*>dev_ipiv, <double*>b, ldb, <int*>dev_info)
    check_status(status)


cpdef cgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t a, int lda, intptr_t dev_ipiv, intptr_t b, int ldb, intptr_t dev_info):
    """See `cusolverDnCgetrs`."""
    with nogil:
        status = cusolverDnCgetrs(<Handle>handle, <cublasOperation_t>trans, n, nrhs, <const cuComplex*>a, lda, <const int*>dev_ipiv, <cuComplex*>b, ldb, <int*>dev_info)
    check_status(status)


cpdef zgetrs(intptr_t handle, int trans, int n, int nrhs, intptr_t a, int lda, intptr_t dev_ipiv, intptr_t b, int ldb, intptr_t dev_info):
    """See `cusolverDnZgetrs`."""
    with nogil:
        status = cusolverDnZgetrs(<Handle>handle, <cublasOperation_t>trans, n, nrhs, <const cuDoubleComplex*>a, lda, <const int*>dev_ipiv, <cuDoubleComplex*>b, ldb, <int*>dev_info)
    check_status(status)


cpdef int sgeqrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnSgeqrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSgeqrf_bufferSize(<Handle>handle, m, n, <float*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int dgeqrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnDgeqrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDgeqrf_bufferSize(<Handle>handle, m, n, <double*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int cgeqrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnCgeqrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCgeqrf_bufferSize(<Handle>handle, m, n, <cuComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int zgeqrf_buffer_size(intptr_t handle, int m, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnZgeqrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZgeqrf_bufferSize(<Handle>handle, m, n, <cuDoubleComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef sgeqrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t workspace, int lwork, intptr_t dev_info):
    """See `cusolverDnSgeqrf`."""
    with nogil:
        status = cusolverDnSgeqrf(<Handle>handle, m, n, <float*>a, lda, <float*>tau, <float*>workspace, lwork, <int*>dev_info)
    check_status(status)


cpdef dgeqrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t workspace, int lwork, intptr_t dev_info):
    """See `cusolverDnDgeqrf`."""
    with nogil:
        status = cusolverDnDgeqrf(<Handle>handle, m, n, <double*>a, lda, <double*>tau, <double*>workspace, lwork, <int*>dev_info)
    check_status(status)


cpdef cgeqrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t workspace, int lwork, intptr_t dev_info):
    """See `cusolverDnCgeqrf`."""
    with nogil:
        status = cusolverDnCgeqrf(<Handle>handle, m, n, <cuComplex*>a, lda, <cuComplex*>tau, <cuComplex*>workspace, lwork, <int*>dev_info)
    check_status(status)


cpdef zgeqrf(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t workspace, int lwork, intptr_t dev_info):
    """See `cusolverDnZgeqrf`."""
    with nogil:
        status = cusolverDnZgeqrf(<Handle>handle, m, n, <cuDoubleComplex*>a, lda, <cuDoubleComplex*>tau, <cuDoubleComplex*>workspace, lwork, <int*>dev_info)
    check_status(status)


cpdef int sorgqr_buffer_size(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnSorgqr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSorgqr_bufferSize(<Handle>handle, m, n, k, <const float*>a, lda, <const float*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int dorgqr_buffer_size(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnDorgqr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDorgqr_bufferSize(<Handle>handle, m, n, k, <const double*>a, lda, <const double*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int cungqr_buffer_size(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnCungqr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCungqr_bufferSize(<Handle>handle, m, n, k, <const cuComplex*>a, lda, <const cuComplex*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int zungqr_buffer_size(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnZungqr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZungqr_bufferSize(<Handle>handle, m, n, k, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>tau, &lwork)
    check_status(status)
    return lwork


cpdef sorgqr(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnSorgqr`."""
    with nogil:
        status = cusolverDnSorgqr(<Handle>handle, m, n, k, <float*>a, lda, <const float*>tau, <float*>work, lwork, <int*>info)
    check_status(status)


cpdef dorgqr(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnDorgqr`."""
    with nogil:
        status = cusolverDnDorgqr(<Handle>handle, m, n, k, <double*>a, lda, <const double*>tau, <double*>work, lwork, <int*>info)
    check_status(status)


cpdef cungqr(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnCungqr`."""
    with nogil:
        status = cusolverDnCungqr(<Handle>handle, m, n, k, <cuComplex*>a, lda, <const cuComplex*>tau, <cuComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef zungqr(intptr_t handle, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnZungqr`."""
    with nogil:
        status = cusolverDnZungqr(<Handle>handle, m, n, k, <cuDoubleComplex*>a, lda, <const cuDoubleComplex*>tau, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef int sormqr_buffer_size(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1:
    """See `cusolverDnSormqr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSormqr_bufferSize(<Handle>handle, <cublasSideMode_t>side, <cublasOperation_t>trans, m, n, k, <const float*>a, lda, <const float*>tau, <const float*>c, ldc, &lwork)
    check_status(status)
    return lwork


cpdef int dormqr_buffer_size(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1:
    """See `cusolverDnDormqr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDormqr_bufferSize(<Handle>handle, <cublasSideMode_t>side, <cublasOperation_t>trans, m, n, k, <const double*>a, lda, <const double*>tau, <const double*>c, ldc, &lwork)
    check_status(status)
    return lwork


cpdef int cunmqr_buffer_size(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1:
    """See `cusolverDnCunmqr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCunmqr_bufferSize(<Handle>handle, <cublasSideMode_t>side, <cublasOperation_t>trans, m, n, k, <const cuComplex*>a, lda, <const cuComplex*>tau, <const cuComplex*>c, ldc, &lwork)
    check_status(status)
    return lwork


cpdef int zunmqr_buffer_size(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1:
    """See `cusolverDnZunmqr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZunmqr_bufferSize(<Handle>handle, <cublasSideMode_t>side, <cublasOperation_t>trans, m, n, k, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>tau, <const cuDoubleComplex*>c, ldc, &lwork)
    check_status(status)
    return lwork


cpdef sormqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnSormqr`."""
    with nogil:
        status = cusolverDnSormqr(<Handle>handle, <cublasSideMode_t>side, <cublasOperation_t>trans, m, n, k, <const float*>a, lda, <const float*>tau, <float*>c, ldc, <float*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef dormqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnDormqr`."""
    with nogil:
        status = cusolverDnDormqr(<Handle>handle, <cublasSideMode_t>side, <cublasOperation_t>trans, m, n, k, <const double*>a, lda, <const double*>tau, <double*>c, ldc, <double*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef cunmqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnCunmqr`."""
    with nogil:
        status = cusolverDnCunmqr(<Handle>handle, <cublasSideMode_t>side, <cublasOperation_t>trans, m, n, k, <const cuComplex*>a, lda, <const cuComplex*>tau, <cuComplex*>c, ldc, <cuComplex*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef zunmqr(intptr_t handle, int side, int trans, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnZunmqr`."""
    with nogil:
        status = cusolverDnZunmqr(<Handle>handle, <cublasSideMode_t>side, <cublasOperation_t>trans, m, n, k, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>tau, <cuDoubleComplex*>c, ldc, <cuDoubleComplex*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef int ssytrf_buffer_size(intptr_t handle, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnSsytrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSsytrf_bufferSize(<Handle>handle, n, <float*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int dsytrf_buffer_size(intptr_t handle, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnDsytrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDsytrf_bufferSize(<Handle>handle, n, <double*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int csytrf_buffer_size(intptr_t handle, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnCsytrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCsytrf_bufferSize(<Handle>handle, n, <cuComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef int zsytrf_buffer_size(intptr_t handle, int n, intptr_t a, int lda) except? -1:
    """See `cusolverDnZsytrf_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZsytrf_bufferSize(<Handle>handle, n, <cuDoubleComplex*>a, lda, &lwork)
    check_status(status)
    return lwork


cpdef ssytrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnSsytrf`."""
    with nogil:
        status = cusolverDnSsytrf(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, <int*>ipiv, <float*>work, lwork, <int*>info)
    check_status(status)


cpdef dsytrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnDsytrf`."""
    with nogil:
        status = cusolverDnDsytrf(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, <int*>ipiv, <double*>work, lwork, <int*>info)
    check_status(status)


cpdef csytrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnCsytrf`."""
    with nogil:
        status = cusolverDnCsytrf(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <int*>ipiv, <cuComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef zsytrf(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnZsytrf`."""
    with nogil:
        status = cusolverDnZsytrf(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <int*>ipiv, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef int ssytri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv) except? -1:
    """See `cusolverDnSsytri_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSsytri_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, <const int*>ipiv, &lwork)
    check_status(status)
    return lwork


cpdef int dsytri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv) except? -1:
    """See `cusolverDnDsytri_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDsytri_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, <const int*>ipiv, &lwork)
    check_status(status)
    return lwork


cpdef int csytri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv) except? -1:
    """See `cusolverDnCsytri_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCsytri_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <const int*>ipiv, &lwork)
    check_status(status)
    return lwork


cpdef int zsytri_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv) except? -1:
    """See `cusolverDnZsytri_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZsytri_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <const int*>ipiv, &lwork)
    check_status(status)
    return lwork


cpdef ssytri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnSsytri`."""
    with nogil:
        status = cusolverDnSsytri(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, <const int*>ipiv, <float*>work, lwork, <int*>info)
    check_status(status)


cpdef dsytri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnDsytri`."""
    with nogil:
        status = cusolverDnDsytri(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, <const int*>ipiv, <double*>work, lwork, <int*>info)
    check_status(status)


cpdef csytri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnCsytri`."""
    with nogil:
        status = cusolverDnCsytri(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <const int*>ipiv, <cuComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef zsytri(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ipiv, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnZsytri`."""
    with nogil:
        status = cusolverDnZsytri(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <const int*>ipiv, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef int sgebrd_buffer_size(intptr_t handle, int m, int n) except? -1:
    """See `cusolverDnSgebrd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork


cpdef int dgebrd_buffer_size(intptr_t handle, int m, int n) except? -1:
    """See `cusolverDnDgebrd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork


cpdef int cgebrd_buffer_size(intptr_t handle, int m, int n) except? -1:
    """See `cusolverDnCgebrd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork


cpdef int zgebrd_buffer_size(intptr_t handle, int m, int n) except? -1:
    """See `cusolverDnZgebrd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork


cpdef sgebrd(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tauq, intptr_t taup, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnSgebrd`."""
    with nogil:
        status = cusolverDnSgebrd(<Handle>handle, m, n, <float*>a, lda, <float*>d, <float*>e, <float*>tauq, <float*>taup, <float*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef dgebrd(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tauq, intptr_t taup, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnDgebrd`."""
    with nogil:
        status = cusolverDnDgebrd(<Handle>handle, m, n, <double*>a, lda, <double*>d, <double*>e, <double*>tauq, <double*>taup, <double*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef cgebrd(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tauq, intptr_t taup, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnCgebrd`."""
    with nogil:
        status = cusolverDnCgebrd(<Handle>handle, m, n, <cuComplex*>a, lda, <float*>d, <float*>e, <cuComplex*>tauq, <cuComplex*>taup, <cuComplex*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef zgebrd(intptr_t handle, int m, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tauq, intptr_t taup, intptr_t work, int lwork, intptr_t dev_info):
    """See `cusolverDnZgebrd`."""
    with nogil:
        status = cusolverDnZgebrd(<Handle>handle, m, n, <cuDoubleComplex*>a, lda, <double*>d, <double*>e, <cuDoubleComplex*>tauq, <cuDoubleComplex*>taup, <cuDoubleComplex*>work, lwork, <int*>dev_info)
    check_status(status)


cpdef int sorgbr_buffer_size(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnSorgbr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSorgbr_bufferSize(<Handle>handle, <cublasSideMode_t>side, m, n, k, <const float*>a, lda, <const float*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int dorgbr_buffer_size(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnDorgbr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDorgbr_bufferSize(<Handle>handle, <cublasSideMode_t>side, m, n, k, <const double*>a, lda, <const double*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int cungbr_buffer_size(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnCungbr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCungbr_bufferSize(<Handle>handle, <cublasSideMode_t>side, m, n, k, <const cuComplex*>a, lda, <const cuComplex*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int zungbr_buffer_size(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnZungbr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZungbr_bufferSize(<Handle>handle, <cublasSideMode_t>side, m, n, k, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>tau, &lwork)
    check_status(status)
    return lwork


cpdef sorgbr(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnSorgbr`."""
    with nogil:
        status = cusolverDnSorgbr(<Handle>handle, <cublasSideMode_t>side, m, n, k, <float*>a, lda, <const float*>tau, <float*>work, lwork, <int*>info)
    check_status(status)


cpdef dorgbr(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnDorgbr`."""
    with nogil:
        status = cusolverDnDorgbr(<Handle>handle, <cublasSideMode_t>side, m, n, k, <double*>a, lda, <const double*>tau, <double*>work, lwork, <int*>info)
    check_status(status)


cpdef cungbr(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnCungbr`."""
    with nogil:
        status = cusolverDnCungbr(<Handle>handle, <cublasSideMode_t>side, m, n, k, <cuComplex*>a, lda, <const cuComplex*>tau, <cuComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef zungbr(intptr_t handle, int side, int m, int n, int k, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnZungbr`."""
    with nogil:
        status = cusolverDnZungbr(<Handle>handle, <cublasSideMode_t>side, m, n, k, <cuDoubleComplex*>a, lda, <const cuDoubleComplex*>tau, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef int ssytrd_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau) except? -1:
    """See `cusolverDnSsytrd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSsytrd_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <const float*>a, lda, <const float*>d, <const float*>e, <const float*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int dsytrd_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau) except? -1:
    """See `cusolverDnDsytrd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDsytrd_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <const double*>a, lda, <const double*>d, <const double*>e, <const double*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int chetrd_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau) except? -1:
    """See `cusolverDnChetrd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnChetrd_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <const cuComplex*>a, lda, <const float*>d, <const float*>e, <const cuComplex*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int zhetrd_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau) except? -1:
    """See `cusolverDnZhetrd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZhetrd_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <const cuDoubleComplex*>a, lda, <const double*>d, <const double*>e, <const cuDoubleComplex*>tau, &lwork)
    check_status(status)
    return lwork


cpdef ssytrd(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnSsytrd`."""
    with nogil:
        status = cusolverDnSsytrd(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, <float*>d, <float*>e, <float*>tau, <float*>work, lwork, <int*>info)
    check_status(status)


cpdef dsytrd(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnDsytrd`."""
    with nogil:
        status = cusolverDnDsytrd(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, <double*>d, <double*>e, <double*>tau, <double*>work, lwork, <int*>info)
    check_status(status)


cpdef chetrd(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnChetrd`."""
    with nogil:
        status = cusolverDnChetrd(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <float*>d, <float*>e, <cuComplex*>tau, <cuComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef zhetrd(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t d, intptr_t e, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnZhetrd`."""
    with nogil:
        status = cusolverDnZhetrd(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <double*>d, <double*>e, <cuDoubleComplex*>tau, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef int sorgtr_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnSorgtr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSorgtr_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <const float*>a, lda, <const float*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int dorgtr_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnDorgtr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDorgtr_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <const double*>a, lda, <const double*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int cungtr_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnCungtr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCungtr_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <const cuComplex*>a, lda, <const cuComplex*>tau, &lwork)
    check_status(status)
    return lwork


cpdef int zungtr_buffer_size(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau) except? -1:
    """See `cusolverDnZungtr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZungtr_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>tau, &lwork)
    check_status(status)
    return lwork


cpdef sorgtr(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnSorgtr`."""
    with nogil:
        status = cusolverDnSorgtr(<Handle>handle, <cublasFillMode_t>uplo, n, <float*>a, lda, <const float*>tau, <float*>work, lwork, <int*>info)
    check_status(status)


cpdef dorgtr(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnDorgtr`."""
    with nogil:
        status = cusolverDnDorgtr(<Handle>handle, <cublasFillMode_t>uplo, n, <double*>a, lda, <const double*>tau, <double*>work, lwork, <int*>info)
    check_status(status)


cpdef cungtr(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnCungtr`."""
    with nogil:
        status = cusolverDnCungtr(<Handle>handle, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <const cuComplex*>tau, <cuComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef zungtr(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t tau, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnZungtr`."""
    with nogil:
        status = cusolverDnZungtr(<Handle>handle, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <const cuDoubleComplex*>tau, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef int sormtr_buffer_size(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1:
    """See `cusolverDnSormtr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSormtr_bufferSize(<Handle>handle, <cublasSideMode_t>side, <cublasFillMode_t>uplo, <cublasOperation_t>trans, m, n, <const float*>a, lda, <const float*>tau, <const float*>c, ldc, &lwork)
    check_status(status)
    return lwork


cpdef int dormtr_buffer_size(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1:
    """See `cusolverDnDormtr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDormtr_bufferSize(<Handle>handle, <cublasSideMode_t>side, <cublasFillMode_t>uplo, <cublasOperation_t>trans, m, n, <const double*>a, lda, <const double*>tau, <const double*>c, ldc, &lwork)
    check_status(status)
    return lwork


cpdef int cunmtr_buffer_size(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1:
    """See `cusolverDnCunmtr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCunmtr_bufferSize(<Handle>handle, <cublasSideMode_t>side, <cublasFillMode_t>uplo, <cublasOperation_t>trans, m, n, <const cuComplex*>a, lda, <const cuComplex*>tau, <const cuComplex*>c, ldc, &lwork)
    check_status(status)
    return lwork


cpdef int zunmtr_buffer_size(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc) except? -1:
    """See `cusolverDnZunmtr_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZunmtr_bufferSize(<Handle>handle, <cublasSideMode_t>side, <cublasFillMode_t>uplo, <cublasOperation_t>trans, m, n, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>tau, <const cuDoubleComplex*>c, ldc, &lwork)
    check_status(status)
    return lwork


cpdef sormtr(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnSormtr`."""
    with nogil:
        status = cusolverDnSormtr(<Handle>handle, <cublasSideMode_t>side, <cublasFillMode_t>uplo, <cublasOperation_t>trans, m, n, <float*>a, lda, <float*>tau, <float*>c, ldc, <float*>work, lwork, <int*>info)
    check_status(status)


cpdef dormtr(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnDormtr`."""
    with nogil:
        status = cusolverDnDormtr(<Handle>handle, <cublasSideMode_t>side, <cublasFillMode_t>uplo, <cublasOperation_t>trans, m, n, <double*>a, lda, <double*>tau, <double*>c, ldc, <double*>work, lwork, <int*>info)
    check_status(status)


cpdef cunmtr(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnCunmtr`."""
    with nogil:
        status = cusolverDnCunmtr(<Handle>handle, <cublasSideMode_t>side, <cublasFillMode_t>uplo, <cublasOperation_t>trans, m, n, <cuComplex*>a, lda, <cuComplex*>tau, <cuComplex*>c, ldc, <cuComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef zunmtr(intptr_t handle, int side, int uplo, int trans, int m, int n, intptr_t a, int lda, intptr_t tau, intptr_t c, int ldc, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnZunmtr`."""
    with nogil:
        status = cusolverDnZunmtr(<Handle>handle, <cublasSideMode_t>side, <cublasFillMode_t>uplo, <cublasOperation_t>trans, m, n, <cuDoubleComplex*>a, lda, <cuDoubleComplex*>tau, <cuDoubleComplex*>c, ldc, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef int sgesvd_buffer_size(intptr_t handle, int m, int n) except? -1:
    """See `cusolverDnSgesvd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork


cpdef int dgesvd_buffer_size(intptr_t handle, int m, int n) except? -1:
    """See `cusolverDnDgesvd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork


cpdef int cgesvd_buffer_size(intptr_t handle, int m, int n) except? -1:
    """See `cusolverDnCgesvd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork


cpdef int zgesvd_buffer_size(intptr_t handle, int m, int n) except? -1:
    """See `cusolverDnZgesvd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork


cpdef sgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t vt, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info):
    """See `cusolverDnSgesvd`."""
    with nogil:
        status = cusolverDnSgesvd(<Handle>handle, jobu, jobvt, m, n, <float*>a, lda, <float*>s, <float*>u, ldu, <float*>vt, ldvt, <float*>work, lwork, <float*>rwork, <int*>info)
    check_status(status)


cpdef dgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t vt, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info):
    """See `cusolverDnDgesvd`."""
    with nogil:
        status = cusolverDnDgesvd(<Handle>handle, jobu, jobvt, m, n, <double*>a, lda, <double*>s, <double*>u, ldu, <double*>vt, ldvt, <double*>work, lwork, <double*>rwork, <int*>info)
    check_status(status)


cpdef cgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t vt, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info):
    """See `cusolverDnCgesvd`."""
    with nogil:
        status = cusolverDnCgesvd(<Handle>handle, jobu, jobvt, m, n, <cuComplex*>a, lda, <float*>s, <cuComplex*>u, ldu, <cuComplex*>vt, ldvt, <cuComplex*>work, lwork, <float*>rwork, <int*>info)
    check_status(status)


cpdef zgesvd(intptr_t handle, signed char jobu, signed char jobvt, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t vt, int ldvt, intptr_t work, int lwork, intptr_t rwork, intptr_t info):
    """See `cusolverDnZgesvd`."""
    with nogil:
        status = cusolverDnZgesvd(<Handle>handle, jobu, jobvt, m, n, <cuDoubleComplex*>a, lda, <double*>s, <cuDoubleComplex*>u, ldu, <cuDoubleComplex*>vt, ldvt, <cuDoubleComplex*>work, lwork, <double*>rwork, <int*>info)
    check_status(status)


cpdef int ssyevd_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w) except? -1:
    """See `cusolverDnSsyevd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSsyevd_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const float*>a, lda, <const float*>w, &lwork)
    check_status(status)
    return lwork


cpdef int dsyevd_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w) except? -1:
    """See `cusolverDnDsyevd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDsyevd_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const double*>a, lda, <const double*>w, &lwork)
    check_status(status)
    return lwork


cpdef int cheevd_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w) except? -1:
    """See `cusolverDnCheevd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCheevd_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const cuComplex*>a, lda, <const float*>w, &lwork)
    check_status(status)
    return lwork


cpdef int zheevd_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w) except? -1:
    """See `cusolverDnZheevd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZheevd_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const cuDoubleComplex*>a, lda, <const double*>w, &lwork)
    check_status(status)
    return lwork


cpdef ssyevd(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnSsyevd`."""
    with nogil:
        status = cusolverDnSsyevd(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <float*>a, lda, <float*>w, <float*>work, lwork, <int*>info)
    check_status(status)


cpdef dsyevd(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnDsyevd`."""
    with nogil:
        status = cusolverDnDsyevd(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <double*>a, lda, <double*>w, <double*>work, lwork, <int*>info)
    check_status(status)


cpdef cheevd(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnCheevd`."""
    with nogil:
        status = cusolverDnCheevd(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <float*>w, <cuComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef zheevd(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnZheevd`."""
    with nogil:
        status = cusolverDnZheevd(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <double*>w, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef int ssyevdx_buffer_size(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w) except? -1:
    """See `cusolverDnSsyevdx_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSsyevdx_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <const float*>a, lda, vl, vu, il, iu, <int*>meig, <const float*>w, &lwork)
    check_status(status)
    return lwork


cpdef int dsyevdx_buffer_size(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w) except? -1:
    """See `cusolverDnDsyevdx_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDsyevdx_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <const double*>a, lda, vl, vu, il, iu, <int*>meig, <const double*>w, &lwork)
    check_status(status)
    return lwork


cpdef int cheevdx_buffer_size(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w) except? -1:
    """See `cusolverDnCheevdx_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCheevdx_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <const cuComplex*>a, lda, vl, vu, il, iu, <int*>meig, <const float*>w, &lwork)
    check_status(status)
    return lwork


cpdef int zheevdx_buffer_size(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w) except? -1:
    """See `cusolverDnZheevdx_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZheevdx_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <const cuDoubleComplex*>a, lda, vl, vu, il, iu, <int*>meig, <const double*>w, &lwork)
    check_status(status)
    return lwork


cpdef int ssyevdx(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, float vl, float vu, int il, int iu, intptr_t w, intptr_t work, int lwork, intptr_t info) except? -1:
    """See `cusolverDnSsyevdx`."""
    cdef int meig
    with nogil:
        status = cusolverDnSsyevdx(<Handle>handle, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <float*>a, lda, vl, vu, il, iu, &meig, <float*>w, <float*>work, lwork, <int*>info)
    check_status(status)
    return meig


cpdef int dsyevdx(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, double vl, double vu, int il, int iu, intptr_t w, intptr_t work, int lwork, intptr_t info) except? -1:
    """See `cusolverDnDsyevdx`."""
    cdef int meig
    with nogil:
        status = cusolverDnDsyevdx(<Handle>handle, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <double*>a, lda, vl, vu, il, iu, &meig, <double*>w, <double*>work, lwork, <int*>info)
    check_status(status)
    return meig


cpdef int cheevdx(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, float vl, float vu, int il, int iu, intptr_t w, intptr_t work, int lwork, intptr_t info) except? -1:
    """See `cusolverDnCheevdx`."""
    cdef int meig
    with nogil:
        status = cusolverDnCheevdx(<Handle>handle, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, vl, vu, il, iu, &meig, <float*>w, <cuComplex*>work, lwork, <int*>info)
    check_status(status)
    return meig


cpdef int zheevdx(intptr_t handle, int jobz, int range, int uplo, int n, intptr_t a, int lda, double vl, double vu, int il, int iu, intptr_t w, intptr_t work, int lwork, intptr_t info) except? -1:
    """See `cusolverDnZheevdx`."""
    cdef int meig
    with nogil:
        status = cusolverDnZheevdx(<Handle>handle, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, vl, vu, il, iu, &meig, <double*>w, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)
    return meig


cpdef int ssygvdx_buffer_size(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w) except? -1:
    """See `cusolverDnSsygvdx_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSsygvdx_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <const float*>a, lda, <const float*>b, ldb, vl, vu, il, iu, <int*>meig, <const float*>w, &lwork)
    check_status(status)
    return lwork


cpdef int dsygvdx_buffer_size(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w) except? -1:
    """See `cusolverDnDsygvdx_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDsygvdx_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <const double*>a, lda, <const double*>b, ldb, vl, vu, il, iu, <int*>meig, <const double*>w, &lwork)
    check_status(status)
    return lwork


cpdef int chegvdx_buffer_size(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w) except? -1:
    """See `cusolverDnChegvdx_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnChegvdx_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, vl, vu, il, iu, <int*>meig, <const float*>w, &lwork)
    check_status(status)
    return lwork


cpdef int zhegvdx_buffer_size(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w) except? -1:
    """See `cusolverDnZhegvdx_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZhegvdx_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, vl, vu, il, iu, <int*>meig, <const double*>w, &lwork)
    check_status(status)
    return lwork


cpdef ssygvdx(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnSsygvdx`."""
    with nogil:
        status = cusolverDnSsygvdx(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <float*>a, lda, <float*>b, ldb, vl, vu, il, iu, <int*>meig, <float*>w, <float*>work, lwork, <int*>info)
    check_status(status)


cpdef dsygvdx(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnDsygvdx`."""
    with nogil:
        status = cusolverDnDsygvdx(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <double*>a, lda, <double*>b, ldb, vl, vu, il, iu, <int*>meig, <double*>w, <double*>work, lwork, <int*>info)
    check_status(status)


cpdef chegvdx(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, float vl, float vu, int il, int iu, intptr_t meig, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnChegvdx`."""
    with nogil:
        status = cusolverDnChegvdx(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <cuComplex*>b, ldb, vl, vu, il, iu, <int*>meig, <float*>w, <cuComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef zhegvdx(intptr_t handle, int itype, int jobz, int range, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, double vl, double vu, int il, int iu, intptr_t meig, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnZhegvdx`."""
    with nogil:
        status = cusolverDnZhegvdx(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <cuDoubleComplex*>b, ldb, vl, vu, il, iu, <int*>meig, <double*>w, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef int ssygvd_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w) except? -1:
    """See `cusolverDnSsygvd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSsygvd_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const float*>a, lda, <const float*>b, ldb, <const float*>w, &lwork)
    check_status(status)
    return lwork


cpdef int dsygvd_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w) except? -1:
    """See `cusolverDnDsygvd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDsygvd_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const double*>a, lda, <const double*>b, ldb, <const double*>w, &lwork)
    check_status(status)
    return lwork


cpdef int chegvd_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w) except? -1:
    """See `cusolverDnChegvd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnChegvd_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const float*>w, &lwork)
    check_status(status)
    return lwork


cpdef int zhegvd_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w) except? -1:
    """See `cusolverDnZhegvd_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZhegvd_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const double*>w, &lwork)
    check_status(status)
    return lwork


cpdef ssygvd(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnSsygvd`."""
    with nogil:
        status = cusolverDnSsygvd(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <float*>a, lda, <float*>b, ldb, <float*>w, <float*>work, lwork, <int*>info)
    check_status(status)


cpdef dsygvd(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnDsygvd`."""
    with nogil:
        status = cusolverDnDsygvd(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <double*>a, lda, <double*>b, ldb, <double*>w, <double*>work, lwork, <int*>info)
    check_status(status)


cpdef chegvd(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnChegvd`."""
    with nogil:
        status = cusolverDnChegvd(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <cuComplex*>b, ldb, <float*>w, <cuComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef zhegvd(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info):
    """See `cusolverDnZhegvd`."""
    with nogil:
        status = cusolverDnZhegvd(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <cuDoubleComplex*>b, ldb, <double*>w, <cuDoubleComplex*>work, lwork, <int*>info)
    check_status(status)


cpdef intptr_t create_syevj_info() except? 0:
    """See `cusolverDnCreateSyevjInfo`."""
    cdef syevjInfo info
    with nogil:
        status = cusolverDnCreateSyevjInfo(&info)
    check_status(status)
    return <intptr_t>info


cpdef destroy_syevj_info(intptr_t info):
    """See `cusolverDnDestroySyevjInfo`."""
    with nogil:
        status = cusolverDnDestroySyevjInfo(<syevjInfo>info)
    check_status(status)


cpdef xsyevj_set_tolerance(intptr_t info, double tolerance):
    """See `cusolverDnXsyevjSetTolerance`."""
    with nogil:
        status = cusolverDnXsyevjSetTolerance(<syevjInfo>info, tolerance)
    check_status(status)


cpdef xsyevj_set_max_sweeps(intptr_t info, int max_sweeps):
    """See `cusolverDnXsyevjSetMaxSweeps`."""
    with nogil:
        status = cusolverDnXsyevjSetMaxSweeps(<syevjInfo>info, max_sweeps)
    check_status(status)


cpdef xsyevj_set_sort_eig(intptr_t info, int sort_eig):
    """See `cusolverDnXsyevjSetSortEig`."""
    with nogil:
        status = cusolverDnXsyevjSetSortEig(<syevjInfo>info, sort_eig)
    check_status(status)


cpdef double xsyevj_get_residual(intptr_t handle, intptr_t info) except? 0:
    """See `cusolverDnXsyevjGetResidual`."""
    cdef double residual
    with nogil:
        status = cusolverDnXsyevjGetResidual(<Handle>handle, <syevjInfo>info, &residual)
    check_status(status)
    return residual


cpdef int xsyevj_get_sweeps(intptr_t handle, intptr_t info) except? -1:
    """See `cusolverDnXsyevjGetSweeps`."""
    cdef int executed_sweeps
    with nogil:
        status = cusolverDnXsyevjGetSweeps(<Handle>handle, <syevjInfo>info, &executed_sweeps)
    check_status(status)
    return executed_sweeps


cpdef int ssyevj_batched_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params, int batch_size) except? -1:
    """See `cusolverDnSsyevjBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSsyevjBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const float*>a, lda, <const float*>w, &lwork, <syevjInfo>params, batch_size)
    check_status(status)
    return lwork


cpdef int dsyevj_batched_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params, int batch_size) except? -1:
    """See `cusolverDnDsyevjBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDsyevjBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const double*>a, lda, <const double*>w, &lwork, <syevjInfo>params, batch_size)
    check_status(status)
    return lwork


cpdef int cheevj_batched_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params, int batch_size) except? -1:
    """See `cusolverDnCheevjBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCheevjBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const cuComplex*>a, lda, <const float*>w, &lwork, <syevjInfo>params, batch_size)
    check_status(status)
    return lwork


cpdef int zheevj_batched_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params, int batch_size) except? -1:
    """See `cusolverDnZheevjBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZheevjBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const cuDoubleComplex*>a, lda, <const double*>w, &lwork, <syevjInfo>params, batch_size)
    check_status(status)
    return lwork


cpdef ssyevj_batched(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size):
    """See `cusolverDnSsyevjBatched`."""
    with nogil:
        status = cusolverDnSsyevjBatched(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <float*>a, lda, <float*>w, <float*>work, lwork, <int*>info, <syevjInfo>params, batch_size)
    check_status(status)


cpdef dsyevj_batched(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size):
    """See `cusolverDnDsyevjBatched`."""
    with nogil:
        status = cusolverDnDsyevjBatched(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <double*>a, lda, <double*>w, <double*>work, lwork, <int*>info, <syevjInfo>params, batch_size)
    check_status(status)


cpdef cheevj_batched(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size):
    """See `cusolverDnCheevjBatched`."""
    with nogil:
        status = cusolverDnCheevjBatched(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <float*>w, <cuComplex*>work, lwork, <int*>info, <syevjInfo>params, batch_size)
    check_status(status)


cpdef zheevj_batched(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size):
    """See `cusolverDnZheevjBatched`."""
    with nogil:
        status = cusolverDnZheevjBatched(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <double*>w, <cuDoubleComplex*>work, lwork, <int*>info, <syevjInfo>params, batch_size)
    check_status(status)


cpdef int ssyevj_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params) except? -1:
    """See `cusolverDnSsyevj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSsyevj_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const float*>a, lda, <const float*>w, &lwork, <syevjInfo>params)
    check_status(status)
    return lwork


cpdef int dsyevj_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params) except? -1:
    """See `cusolverDnDsyevj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDsyevj_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const double*>a, lda, <const double*>w, &lwork, <syevjInfo>params)
    check_status(status)
    return lwork


cpdef int cheevj_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params) except? -1:
    """See `cusolverDnCheevj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCheevj_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const cuComplex*>a, lda, <const float*>w, &lwork, <syevjInfo>params)
    check_status(status)
    return lwork


cpdef int zheevj_buffer_size(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t params) except? -1:
    """See `cusolverDnZheevj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZheevj_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const cuDoubleComplex*>a, lda, <const double*>w, &lwork, <syevjInfo>params)
    check_status(status)
    return lwork


cpdef ssyevj(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnSsyevj`."""
    with nogil:
        status = cusolverDnSsyevj(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <float*>a, lda, <float*>w, <float*>work, lwork, <int*>info, <syevjInfo>params)
    check_status(status)


cpdef dsyevj(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnDsyevj`."""
    with nogil:
        status = cusolverDnDsyevj(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <double*>a, lda, <double*>w, <double*>work, lwork, <int*>info, <syevjInfo>params)
    check_status(status)


cpdef cheevj(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnCheevj`."""
    with nogil:
        status = cusolverDnCheevj(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <float*>w, <cuComplex*>work, lwork, <int*>info, <syevjInfo>params)
    check_status(status)


cpdef zheevj(intptr_t handle, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnZheevj`."""
    with nogil:
        status = cusolverDnZheevj(<Handle>handle, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <double*>w, <cuDoubleComplex*>work, lwork, <int*>info, <syevjInfo>params)
    check_status(status)


cpdef int ssygvj_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t params) except? -1:
    """See `cusolverDnSsygvj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSsygvj_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const float*>a, lda, <const float*>b, ldb, <const float*>w, &lwork, <syevjInfo>params)
    check_status(status)
    return lwork


cpdef int dsygvj_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t params) except? -1:
    """See `cusolverDnDsygvj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDsygvj_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const double*>a, lda, <const double*>b, ldb, <const double*>w, &lwork, <syevjInfo>params)
    check_status(status)
    return lwork


cpdef int chegvj_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t params) except? -1:
    """See `cusolverDnChegvj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnChegvj_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const float*>w, &lwork, <syevjInfo>params)
    check_status(status)
    return lwork


cpdef int zhegvj_buffer_size(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t params) except? -1:
    """See `cusolverDnZhegvj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZhegvj_bufferSize(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const double*>w, &lwork, <syevjInfo>params)
    check_status(status)
    return lwork


cpdef ssygvj(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnSsygvj`."""
    with nogil:
        status = cusolverDnSsygvj(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <float*>a, lda, <float*>b, ldb, <float*>w, <float*>work, lwork, <int*>info, <syevjInfo>params)
    check_status(status)


cpdef dsygvj(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnDsygvj`."""
    with nogil:
        status = cusolverDnDsygvj(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <double*>a, lda, <double*>b, ldb, <double*>w, <double*>work, lwork, <int*>info, <syevjInfo>params)
    check_status(status)


cpdef chegvj(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnChegvj`."""
    with nogil:
        status = cusolverDnChegvj(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <cuComplex*>a, lda, <cuComplex*>b, ldb, <float*>w, <cuComplex*>work, lwork, <int*>info, <syevjInfo>params)
    check_status(status)


cpdef zhegvj(intptr_t handle, int itype, int jobz, int uplo, int n, intptr_t a, int lda, intptr_t b, int ldb, intptr_t w, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnZhegvj`."""
    with nogil:
        status = cusolverDnZhegvj(<Handle>handle, <cusolverEigType_t>itype, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <cuDoubleComplex*>a, lda, <cuDoubleComplex*>b, ldb, <double*>w, <cuDoubleComplex*>work, lwork, <int*>info, <syevjInfo>params)
    check_status(status)


cpdef intptr_t create_gesvdj_info() except? 0:
    """See `cusolverDnCreateGesvdjInfo`."""
    cdef gesvdjInfo info
    with nogil:
        status = cusolverDnCreateGesvdjInfo(&info)
    check_status(status)
    return <intptr_t>info


cpdef destroy_gesvdj_info(intptr_t info):
    """See `cusolverDnDestroyGesvdjInfo`."""
    with nogil:
        status = cusolverDnDestroyGesvdjInfo(<gesvdjInfo>info)
    check_status(status)


cpdef xgesvdj_set_tolerance(intptr_t info, double tolerance):
    """See `cusolverDnXgesvdjSetTolerance`."""
    with nogil:
        status = cusolverDnXgesvdjSetTolerance(<gesvdjInfo>info, tolerance)
    check_status(status)


cpdef xgesvdj_set_max_sweeps(intptr_t info, int max_sweeps):
    """See `cusolverDnXgesvdjSetMaxSweeps`."""
    with nogil:
        status = cusolverDnXgesvdjSetMaxSweeps(<gesvdjInfo>info, max_sweeps)
    check_status(status)


cpdef xgesvdj_set_sort_eig(intptr_t info, int sort_svd):
    """See `cusolverDnXgesvdjSetSortEig`."""
    with nogil:
        status = cusolverDnXgesvdjSetSortEig(<gesvdjInfo>info, sort_svd)
    check_status(status)


cpdef double xgesvdj_get_residual(intptr_t handle, intptr_t info) except? 0:
    """See `cusolverDnXgesvdjGetResidual`."""
    cdef double residual
    with nogil:
        status = cusolverDnXgesvdjGetResidual(<Handle>handle, <gesvdjInfo>info, &residual)
    check_status(status)
    return residual


cpdef int xgesvdj_get_sweeps(intptr_t handle, intptr_t info) except? -1:
    """See `cusolverDnXgesvdjGetSweeps`."""
    cdef int executed_sweeps
    with nogil:
        status = cusolverDnXgesvdjGetSweeps(<Handle>handle, <gesvdjInfo>info, &executed_sweeps)
    check_status(status)
    return executed_sweeps


cpdef int sgesvdj_batched_buffer_size(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params, int batch_size) except? -1:
    """See `cusolverDnSgesvdjBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSgesvdjBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, m, n, <const float*>a, lda, <const float*>s, <const float*>u, ldu, <const float*>v, ldv, &lwork, <gesvdjInfo>params, batch_size)
    check_status(status)
    return lwork


cpdef int dgesvdj_batched_buffer_size(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params, int batch_size) except? -1:
    """See `cusolverDnDgesvdjBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDgesvdjBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, m, n, <const double*>a, lda, <const double*>s, <const double*>u, ldu, <const double*>v, ldv, &lwork, <gesvdjInfo>params, batch_size)
    check_status(status)
    return lwork


cpdef int cgesvdj_batched_buffer_size(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params, int batch_size) except? -1:
    """See `cusolverDnCgesvdjBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCgesvdjBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, m, n, <const cuComplex*>a, lda, <const float*>s, <const cuComplex*>u, ldu, <const cuComplex*>v, ldv, &lwork, <gesvdjInfo>params, batch_size)
    check_status(status)
    return lwork


cpdef int zgesvdj_batched_buffer_size(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params, int batch_size) except? -1:
    """See `cusolverDnZgesvdjBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZgesvdjBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, m, n, <const cuDoubleComplex*>a, lda, <const double*>s, <const cuDoubleComplex*>u, ldu, <const cuDoubleComplex*>v, ldv, &lwork, <gesvdjInfo>params, batch_size)
    check_status(status)
    return lwork


cpdef sgesvdj_batched(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size):
    """See `cusolverDnSgesvdjBatched`."""
    with nogil:
        status = cusolverDnSgesvdjBatched(<Handle>handle, <cusolverEigMode_t>jobz, m, n, <float*>a, lda, <float*>s, <float*>u, ldu, <float*>v, ldv, <float*>work, lwork, <int*>info, <gesvdjInfo>params, batch_size)
    check_status(status)


cpdef dgesvdj_batched(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size):
    """See `cusolverDnDgesvdjBatched`."""
    with nogil:
        status = cusolverDnDgesvdjBatched(<Handle>handle, <cusolverEigMode_t>jobz, m, n, <double*>a, lda, <double*>s, <double*>u, ldu, <double*>v, ldv, <double*>work, lwork, <int*>info, <gesvdjInfo>params, batch_size)
    check_status(status)


cpdef cgesvdj_batched(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size):
    """See `cusolverDnCgesvdjBatched`."""
    with nogil:
        status = cusolverDnCgesvdjBatched(<Handle>handle, <cusolverEigMode_t>jobz, m, n, <cuComplex*>a, lda, <float*>s, <cuComplex*>u, ldu, <cuComplex*>v, ldv, <cuComplex*>work, lwork, <int*>info, <gesvdjInfo>params, batch_size)
    check_status(status)


cpdef zgesvdj_batched(intptr_t handle, int jobz, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params, int batch_size):
    """See `cusolverDnZgesvdjBatched`."""
    with nogil:
        status = cusolverDnZgesvdjBatched(<Handle>handle, <cusolverEigMode_t>jobz, m, n, <cuDoubleComplex*>a, lda, <double*>s, <cuDoubleComplex*>u, ldu, <cuDoubleComplex*>v, ldv, <cuDoubleComplex*>work, lwork, <int*>info, <gesvdjInfo>params, batch_size)
    check_status(status)


cpdef int sgesvdj_buffer_size(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params) except? -1:
    """See `cusolverDnSgesvdj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSgesvdj_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, econ, m, n, <const float*>a, lda, <const float*>s, <const float*>u, ldu, <const float*>v, ldv, &lwork, <gesvdjInfo>params)
    check_status(status)
    return lwork


cpdef int dgesvdj_buffer_size(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params) except? -1:
    """See `cusolverDnDgesvdj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDgesvdj_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, econ, m, n, <const double*>a, lda, <const double*>s, <const double*>u, ldu, <const double*>v, ldv, &lwork, <gesvdjInfo>params)
    check_status(status)
    return lwork


cpdef int cgesvdj_buffer_size(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params) except? -1:
    """See `cusolverDnCgesvdj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCgesvdj_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, econ, m, n, <const cuComplex*>a, lda, <const float*>s, <const cuComplex*>u, ldu, <const cuComplex*>v, ldv, &lwork, <gesvdjInfo>params)
    check_status(status)
    return lwork


cpdef int zgesvdj_buffer_size(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t params) except? -1:
    """See `cusolverDnZgesvdj_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZgesvdj_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, econ, m, n, <const cuDoubleComplex*>a, lda, <const double*>s, <const cuDoubleComplex*>u, ldu, <const cuDoubleComplex*>v, ldv, &lwork, <gesvdjInfo>params)
    check_status(status)
    return lwork


cpdef sgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnSgesvdj`."""
    with nogil:
        status = cusolverDnSgesvdj(<Handle>handle, <cusolverEigMode_t>jobz, econ, m, n, <float*>a, lda, <float*>s, <float*>u, ldu, <float*>v, ldv, <float*>work, lwork, <int*>info, <gesvdjInfo>params)
    check_status(status)


cpdef dgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnDgesvdj`."""
    with nogil:
        status = cusolverDnDgesvdj(<Handle>handle, <cusolverEigMode_t>jobz, econ, m, n, <double*>a, lda, <double*>s, <double*>u, ldu, <double*>v, ldv, <double*>work, lwork, <int*>info, <gesvdjInfo>params)
    check_status(status)


cpdef cgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnCgesvdj`."""
    with nogil:
        status = cusolverDnCgesvdj(<Handle>handle, <cusolverEigMode_t>jobz, econ, m, n, <cuComplex*>a, lda, <float*>s, <cuComplex*>u, ldu, <cuComplex*>v, ldv, <cuComplex*>work, lwork, <int*>info, <gesvdjInfo>params)
    check_status(status)


cpdef zgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t a, int lda, intptr_t s, intptr_t u, int ldu, intptr_t v, int ldv, intptr_t work, int lwork, intptr_t info, intptr_t params):
    """See `cusolverDnZgesvdj`."""
    with nogil:
        status = cusolverDnZgesvdj(<Handle>handle, <cusolverEigMode_t>jobz, econ, m, n, <cuDoubleComplex*>a, lda, <double*>s, <cuDoubleComplex*>u, ldu, <cuDoubleComplex*>v, ldv, <cuDoubleComplex*>work, lwork, <int*>info, <gesvdjInfo>params)
    check_status(status)


cpdef int sgesvda_strided_batched_buffer_size(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, int batch_size) except? -1:
    """See `cusolverDnSgesvdaStridedBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnSgesvdaStridedBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, rank, m, n, <const float*>d_a, lda, stride_a, <const float*>d_s, stride_s, <const float*>d_u, ldu, stride_u, <const float*>d_v, ldv, stride_v, &lwork, batch_size)
    check_status(status)
    return lwork


cpdef int dgesvda_strided_batched_buffer_size(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, int batch_size) except? -1:
    """See `cusolverDnDgesvdaStridedBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnDgesvdaStridedBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, rank, m, n, <const double*>d_a, lda, stride_a, <const double*>d_s, stride_s, <const double*>d_u, ldu, stride_u, <const double*>d_v, ldv, stride_v, &lwork, batch_size)
    check_status(status)
    return lwork


cpdef int cgesvda_strided_batched_buffer_size(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, int batch_size) except? -1:
    """See `cusolverDnCgesvdaStridedBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnCgesvdaStridedBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, rank, m, n, <const cuComplex*>d_a, lda, stride_a, <const float*>d_s, stride_s, <const cuComplex*>d_u, ldu, stride_u, <const cuComplex*>d_v, ldv, stride_v, &lwork, batch_size)
    check_status(status)
    return lwork


cpdef int zgesvda_strided_batched_buffer_size(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, int batch_size) except? -1:
    """See `cusolverDnZgesvdaStridedBatched_bufferSize`."""
    cdef int lwork
    with nogil:
        status = cusolverDnZgesvdaStridedBatched_bufferSize(<Handle>handle, <cusolverEigMode_t>jobz, rank, m, n, <const cuDoubleComplex*>d_a, lda, stride_a, <const double*>d_s, stride_s, <const cuDoubleComplex*>d_u, ldu, stride_u, <const cuDoubleComplex*>d_v, ldv, stride_v, &lwork, batch_size)
    check_status(status)
    return lwork


cpdef sgesvda_strided_batched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_r_nrm_f, int batch_size):
    """See `cusolverDnSgesvdaStridedBatched`."""
    with nogil:
        status = cusolverDnSgesvdaStridedBatched(<Handle>handle, <cusolverEigMode_t>jobz, rank, m, n, <const float*>d_a, lda, stride_a, <float*>d_s, stride_s, <float*>d_u, ldu, stride_u, <float*>d_v, ldv, stride_v, <float*>d_work, lwork, <int*>d_info, <double*>h_r_nrm_f, batch_size)
    check_status(status)


cpdef dgesvda_strided_batched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_r_nrm_f, int batch_size):
    """See `cusolverDnDgesvdaStridedBatched`."""
    with nogil:
        status = cusolverDnDgesvdaStridedBatched(<Handle>handle, <cusolverEigMode_t>jobz, rank, m, n, <const double*>d_a, lda, stride_a, <double*>d_s, stride_s, <double*>d_u, ldu, stride_u, <double*>d_v, ldv, stride_v, <double*>d_work, lwork, <int*>d_info, <double*>h_r_nrm_f, batch_size)
    check_status(status)


cpdef cgesvda_strided_batched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_r_nrm_f, int batch_size):
    """See `cusolverDnCgesvdaStridedBatched`."""
    with nogil:
        status = cusolverDnCgesvdaStridedBatched(<Handle>handle, <cusolverEigMode_t>jobz, rank, m, n, <const cuComplex*>d_a, lda, stride_a, <float*>d_s, stride_s, <cuComplex*>d_u, ldu, stride_u, <cuComplex*>d_v, ldv, stride_v, <cuComplex*>d_work, lwork, <int*>d_info, <double*>h_r_nrm_f, batch_size)
    check_status(status)


cpdef zgesvda_strided_batched(intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_a, int lda, long long int stride_a, intptr_t d_s, long long int stride_s, intptr_t d_u, int ldu, long long int stride_u, intptr_t d_v, int ldv, long long int stride_v, intptr_t d_work, int lwork, intptr_t d_info, intptr_t h_r_nrm_f, int batch_size):
    """See `cusolverDnZgesvdaStridedBatched`."""
    with nogil:
        status = cusolverDnZgesvdaStridedBatched(<Handle>handle, <cusolverEigMode_t>jobz, rank, m, n, <const cuDoubleComplex*>d_a, lda, stride_a, <double*>d_s, stride_s, <cuDoubleComplex*>d_u, ldu, stride_u, <cuDoubleComplex*>d_v, ldv, stride_v, <cuDoubleComplex*>d_work, lwork, <int*>d_info, <double*>h_r_nrm_f, batch_size)
    check_status(status)


cpdef intptr_t create_params() except? 0:
    """See `cusolverDnCreateParams`."""
    cdef Params params
    with nogil:
        status = cusolverDnCreateParams(&params)
    check_status(status)
    return <intptr_t>params


cpdef destroy_params(intptr_t params):
    """See `cusolverDnDestroyParams`."""
    with nogil:
        status = cusolverDnDestroyParams(<Params>params)
    check_status(status)


cpdef set_adv_options(intptr_t params, int function, int algo):
    """See `cusolverDnSetAdvOptions`."""
    with nogil:
        status = cusolverDnSetAdvOptions(<Params>params, <_Function>function, <cusolverAlgMode_t>algo)
    check_status(status)


cpdef tuple xpotrf_buffer_size(intptr_t handle, intptr_t params, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int compute_type):
    """See `cusolverDnXpotrf_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXpotrf_bufferSize(<Handle>handle, <Params>params, <cublasFillMode_t>uplo, n, <DataType>data_type_a, <const void*>a, lda, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xpotrf(intptr_t handle, intptr_t params, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info):
    """See `cusolverDnXpotrf`."""
    with nogil:
        status = cusolverDnXpotrf(<Handle>handle, <Params>params, <cublasFillMode_t>uplo, n, <DataType>data_type_a, <void*>a, lda, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>info)
    check_status(status)


cpdef xpotrs(intptr_t handle, intptr_t params, int uplo, int64_t n, int64_t nrhs, int data_type_a, intptr_t a, int64_t lda, int data_type_b, intptr_t b, int64_t ldb, intptr_t info):
    """See `cusolverDnXpotrs`."""
    with nogil:
        status = cusolverDnXpotrs(<Handle>handle, <Params>params, <cublasFillMode_t>uplo, n, nrhs, <DataType>data_type_a, <const void*>a, lda, <DataType>data_type_b, <void*>b, ldb, <int*>info)
    check_status(status)


cpdef tuple xgeqrf_buffer_size(intptr_t handle, intptr_t params, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_tau, intptr_t tau, int compute_type):
    """See `cusolverDnXgeqrf_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXgeqrf_bufferSize(<Handle>handle, <Params>params, m, n, <DataType>data_type_a, <const void*>a, lda, <DataType>data_type_tau, <const void*>tau, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xgeqrf(intptr_t handle, intptr_t params, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_tau, intptr_t tau, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info):
    """See `cusolverDnXgeqrf`."""
    with nogil:
        status = cusolverDnXgeqrf(<Handle>handle, <Params>params, m, n, <DataType>data_type_a, <void*>a, lda, <DataType>data_type_tau, <void*>tau, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>info)
    check_status(status)


cpdef tuple xgetrf_buffer_size(intptr_t handle, intptr_t params, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int compute_type):
    """See `cusolverDnXgetrf_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXgetrf_bufferSize(<Handle>handle, <Params>params, m, n, <DataType>data_type_a, <const void*>a, lda, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xgetrf(intptr_t handle, intptr_t params, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, intptr_t ipiv, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info):
    """See `cusolverDnXgetrf`."""
    with nogil:
        status = cusolverDnXgetrf(<Handle>handle, <Params>params, m, n, <DataType>data_type_a, <void*>a, lda, <int64_t*>ipiv, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>info)
    check_status(status)


cpdef xgetrs(intptr_t handle, intptr_t params, int trans, int64_t n, int64_t nrhs, int data_type_a, intptr_t a, int64_t lda, intptr_t ipiv, int data_type_b, intptr_t b, int64_t ldb, intptr_t info):
    """See `cusolverDnXgetrs`."""
    with nogil:
        status = cusolverDnXgetrs(<Handle>handle, <Params>params, <cublasOperation_t>trans, n, nrhs, <DataType>data_type_a, <const void*>a, lda, <const int64_t*>ipiv, <DataType>data_type_b, <void*>b, ldb, <int*>info)
    check_status(status)


cpdef tuple xsyevd_buffer_size(intptr_t handle, intptr_t params, int jobz, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int compute_type):
    """See `cusolverDnXsyevd_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXsyevd_bufferSize(<Handle>handle, <Params>params, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <DataType>data_type_a, <const void*>a, lda, <DataType>data_type_w, <const void*>w, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xsyevd(intptr_t handle, intptr_t params, int jobz, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info):
    """See `cusolverDnXsyevd`."""
    with nogil:
        status = cusolverDnXsyevd(<Handle>handle, <Params>params, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <DataType>data_type_a, <void*>a, lda, <DataType>data_type_w, <void*>w, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>info)
    check_status(status)


cpdef tuple xsyevdx_buffer_size(intptr_t handle, intptr_t params, int jobz, int range, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, intptr_t vl, intptr_t vu, int64_t il, int64_t iu, intptr_t h_meig, int data_type_w, intptr_t w, int compute_type):
    """See `cusolverDnXsyevdx_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXsyevdx_bufferSize(<Handle>handle, <Params>params, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <DataType>data_type_a, <const void*>a, lda, <void*>vl, <void*>vu, il, iu, <int64_t*>h_meig, <DataType>data_type_w, <const void*>w, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef int64_t xsyevdx(intptr_t handle, intptr_t params, int jobz, int range, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, intptr_t vl, intptr_t vu, int64_t il, int64_t iu, int data_type_w, intptr_t w, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info) except? -1:
    """See `cusolverDnXsyevdx`."""
    cdef int64_t meig64
    with nogil:
        status = cusolverDnXsyevdx(<Handle>handle, <Params>params, <cusolverEigMode_t>jobz, <cusolverEigRange_t>range, <cublasFillMode_t>uplo, n, <DataType>data_type_a, <void*>a, lda, <void*>vl, <void*>vu, il, iu, &meig64, <DataType>data_type_w, <void*>w, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>info)
    check_status(status)
    return meig64


cpdef tuple xgesvd_buffer_size(intptr_t handle, intptr_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_s, intptr_t s, int data_type_u, intptr_t u, int64_t ldu, int data_type_vt, intptr_t vt, int64_t ldvt, int compute_type):
    """See `cusolverDnXgesvd_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXgesvd_bufferSize(<Handle>handle, <Params>params, jobu, jobvt, m, n, <DataType>data_type_a, <const void*>a, lda, <DataType>data_type_s, <const void*>s, <DataType>data_type_u, <const void*>u, ldu, <DataType>data_type_vt, <const void*>vt, ldvt, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xgesvd(intptr_t handle, intptr_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_s, intptr_t s, int data_type_u, intptr_t u, int64_t ldu, int data_type_vt, intptr_t vt, int64_t ldvt, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info):
    """See `cusolverDnXgesvd`."""
    with nogil:
        status = cusolverDnXgesvd(<Handle>handle, <Params>params, jobu, jobvt, m, n, <DataType>data_type_a, <void*>a, lda, <DataType>data_type_s, <void*>s, <DataType>data_type_u, <void*>u, ldu, <DataType>data_type_vt, <void*>vt, ldvt, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>info)
    check_status(status)


cpdef tuple xgesvdp_buffer_size(intptr_t handle, intptr_t params, int jobz, int econ, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_s, intptr_t s, int data_type_u, intptr_t u, int64_t ldu, int data_type_v, intptr_t v, int64_t ldv, int compute_type):
    """See `cusolverDnXgesvdp_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXgesvdp_bufferSize(<Handle>handle, <Params>params, <cusolverEigMode_t>jobz, econ, m, n, <DataType>data_type_a, <const void*>a, lda, <DataType>data_type_s, <const void*>s, <DataType>data_type_u, <const void*>u, ldu, <DataType>data_type_v, <const void*>v, ldv, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef double xgesvdp(intptr_t handle, intptr_t params, int jobz, int econ, int64_t m, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_s, intptr_t s, int data_type_u, intptr_t u, int64_t ldu, int data_type_v, intptr_t v, int64_t ldv, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t d_info) except? 0:
    """See `cusolverDnXgesvdp`."""
    cdef double h_err_sigma
    with nogil:
        status = cusolverDnXgesvdp(<Handle>handle, <Params>params, <cusolverEigMode_t>jobz, econ, m, n, <DataType>data_type_a, <void*>a, lda, <DataType>data_type_s, <void*>s, <DataType>data_type_u, <void*>u, ldu, <DataType>data_type_v, <void*>v, ldv, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>d_info, &h_err_sigma)
    check_status(status)
    return h_err_sigma


cpdef tuple xgesvdr_buffer_size(intptr_t handle, intptr_t params, signed char jobu, signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, int data_type_a, intptr_t a, int64_t lda, int data_type_srand, intptr_t srand, int data_type_urand, intptr_t urand, int64_t ld_urand, int data_type_vrand, intptr_t vrand, int64_t ld_vrand, int compute_type):
    """See `cusolverDnXgesvdr_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXgesvdr_bufferSize(<Handle>handle, <Params>params, jobu, jobv, m, n, k, p, niters, <DataType>data_type_a, <const void*>a, lda, <DataType>data_type_srand, <const void*>srand, <DataType>data_type_urand, <const void*>urand, ld_urand, <DataType>data_type_vrand, <const void*>vrand, ld_vrand, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xgesvdr(intptr_t handle, intptr_t params, signed char jobu, signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, int data_type_a, intptr_t a, int64_t lda, int data_type_srand, intptr_t srand, int data_type_urand, intptr_t urand, int64_t ld_urand, int data_type_vrand, intptr_t vrand, int64_t ld_vrand, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t d_info):
    """See `cusolverDnXgesvdr`."""
    with nogil:
        status = cusolverDnXgesvdr(<Handle>handle, <Params>params, jobu, jobv, m, n, k, p, niters, <DataType>data_type_a, <void*>a, lda, <DataType>data_type_srand, <void*>srand, <DataType>data_type_urand, <void*>urand, ld_urand, <DataType>data_type_vrand, <void*>vrand, ld_vrand, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>d_info)
    check_status(status)


cpdef tuple xsytrs_buffer_size(intptr_t handle, int uplo, int64_t n, int64_t nrhs, int data_type_a, intptr_t a, int64_t lda, intptr_t ipiv, int data_type_b, intptr_t b, int64_t ldb):
    """See `cusolverDnXsytrs_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXsytrs_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, n, nrhs, <DataType>data_type_a, <const void*>a, lda, <const int64_t*>ipiv, <DataType>data_type_b, <void*>b, ldb, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xsytrs(intptr_t handle, int uplo, int64_t n, int64_t nrhs, int data_type_a, intptr_t a, int64_t lda, intptr_t ipiv, int data_type_b, intptr_t b, int64_t ldb, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info):
    """See `cusolverDnXsytrs`."""
    with nogil:
        status = cusolverDnXsytrs(<Handle>handle, <cublasFillMode_t>uplo, n, nrhs, <DataType>data_type_a, <const void*>a, lda, <const int64_t*>ipiv, <DataType>data_type_b, <void*>b, ldb, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>info)
    check_status(status)


cpdef tuple xtrtri_buffer_size(intptr_t handle, int uplo, int diag, int64_t n, int data_type_a, intptr_t a, int64_t lda):
    """See `cusolverDnXtrtri_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXtrtri_bufferSize(<Handle>handle, <cublasFillMode_t>uplo, <cublasDiagType_t>diag, n, <DataType>data_type_a, <void*>a, lda, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xtrtri(intptr_t handle, int uplo, int diag, int64_t n, int data_type_a, intptr_t a, int64_t lda, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t dev_info):
    """See `cusolverDnXtrtri`."""
    with nogil:
        status = cusolverDnXtrtri(<Handle>handle, <cublasFillMode_t>uplo, <cublasDiagType_t>diag, n, <DataType>data_type_a, <void*>a, lda, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>dev_info)
    check_status(status)


cpdef logger_open_file(log_file):
    """See `cusolverDnLoggerOpenFile`."""
    if not isinstance(log_file, str):
        raise TypeError("log_file must be a Python str")
    cdef bytes _temp_log_file_ = (<str>log_file).encode()
    cdef char* _log_file_ = _temp_log_file_
    with nogil:
        status = cusolverDnLoggerOpenFile(<const char*>_log_file_)
    check_status(status)


cpdef logger_set_level(int level):
    """See `cusolverDnLoggerSetLevel`."""
    with nogil:
        status = cusolverDnLoggerSetLevel(level)
    check_status(status)


cpdef logger_set_mask(int mask):
    """See `cusolverDnLoggerSetMask`."""
    with nogil:
        status = cusolverDnLoggerSetMask(mask)
    check_status(status)


cpdef logger_force_disable():
    """See `cusolverDnLoggerForceDisable`."""
    with nogil:
        status = cusolverDnLoggerForceDisable()
    check_status(status)


cpdef set_deterministic_mode(intptr_t handle, int mode):
    """See `cusolverDnSetDeterministicMode`."""
    with nogil:
        status = cusolverDnSetDeterministicMode(<Handle>handle, <cusolverDeterministicMode_t>mode)
    check_status(status)


cpdef int get_deterministic_mode(intptr_t handle) except *:
    """See `cusolverDnGetDeterministicMode`."""
    cdef cusolverDeterministicMode_t mode
    with nogil:
        status = cusolverDnGetDeterministicMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


cpdef tuple xlarft_buffer_size(intptr_t handle, intptr_t params, cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n, int64_t k, int data_type_v, intptr_t v, int64_t ldv, int data_type_tau, intptr_t tau, int data_type_t, intptr_t t, int64_t ldt, int compute_type):
    """See `cusolverDnXlarft_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXlarft_bufferSize(<Handle>handle, <Params>params, direct, storev, n, k, <DataType>data_type_v, <const void*>v, ldv, <DataType>data_type_tau, <const void*>tau, <DataType>data_type_t, <void*>t, ldt, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xlarft(intptr_t handle, intptr_t params, cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n, int64_t k, int data_type_v, intptr_t v, int64_t ldv, int data_type_tau, intptr_t tau, int data_type_t, intptr_t t, int64_t ldt, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host):
    """See `cusolverDnXlarft`."""
    with nogil:
        status = cusolverDnXlarft(<Handle>handle, <Params>params, direct, storev, n, k, <DataType>data_type_v, <const void*>v, ldv, <DataType>data_type_tau, <const void*>tau, <DataType>data_type_t, <void*>t, ldt, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host)
    check_status(status)


cpdef tuple xsyev_batched_buffer_size(intptr_t handle, intptr_t params, int jobz, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int compute_type, int64_t batch_size):
    """See `cusolverDnXsyevBatched_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXsyevBatched_bufferSize(<Handle>handle, <Params>params, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <DataType>data_type_a, <const void*>a, lda, <DataType>data_type_w, <const void*>w, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host, batch_size)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xsyev_batched(intptr_t handle, intptr_t params, int jobz, int uplo, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info, int64_t batch_size):
    """See `cusolverDnXsyevBatched`."""
    with nogil:
        status = cusolverDnXsyevBatched(<Handle>handle, <Params>params, <cusolverEigMode_t>jobz, <cublasFillMode_t>uplo, n, <DataType>data_type_a, <void*>a, lda, <DataType>data_type_w, <void*>w, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>info, batch_size)
    check_status(status)


cpdef tuple xgeev_buffer_size(intptr_t handle, intptr_t params, int jobvl, int jobvr, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int data_type_vl, intptr_t vl, int64_t ldvl, int data_type_vr, intptr_t vr, int64_t ldvr, int compute_type):
    """See `cusolverDnXgeev_bufferSize`."""
    cdef size_t workspace_in_bytes_on_device
    cdef size_t workspace_in_bytes_on_host
    with nogil:
        status = cusolverDnXgeev_bufferSize(<Handle>handle, <Params>params, <cusolverEigMode_t>jobvl, <cusolverEigMode_t>jobvr, n, <DataType>data_type_a, <const void*>a, lda, <DataType>data_type_w, <const void*>w, <DataType>data_type_vl, <const void*>vl, ldvl, <DataType>data_type_vr, <const void*>vr, ldvr, <DataType>compute_type, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host)
    check_status(status)
    return (workspace_in_bytes_on_device, workspace_in_bytes_on_host)


cpdef xgeev(intptr_t handle, intptr_t params, int jobvl, int jobvr, int64_t n, int data_type_a, intptr_t a, int64_t lda, int data_type_w, intptr_t w, int data_type_vl, intptr_t vl, int64_t ldvl, int data_type_vr, intptr_t vr, int64_t ldvr, int compute_type, intptr_t buffer_on_device, size_t workspace_in_bytes_on_device, intptr_t buffer_on_host, size_t workspace_in_bytes_on_host, intptr_t info):
    """See `cusolverDnXgeev`."""
    with nogil:
        status = cusolverDnXgeev(<Handle>handle, <Params>params, <cusolverEigMode_t>jobvl, <cusolverEigMode_t>jobvr, n, <DataType>data_type_a, <void*>a, lda, <DataType>data_type_w, <void*>w, <DataType>data_type_vl, <void*>vl, ldvl, <DataType>data_type_vr, <void*>vr, ldvr, <DataType>compute_type, <void*>buffer_on_device, workspace_in_bytes_on_device, <void*>buffer_on_host, workspace_in_bytes_on_host, <int*>info)
    check_status(status)
