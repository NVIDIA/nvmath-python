# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from ._internal cimport cusolverDn as _cusolverDn


###############################################################################
# Wrapper functions
###############################################################################

cdef cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) except* nogil:
    return _cusolverDn._cusolverDnCreate(handle)


cdef cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) except* nogil:
    return _cusolverDn._cusolverDnDestroy(handle)


cdef cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId) except* nogil:
    return _cusolverDn._cusolverDnSetStream(handle, streamId)


cdef cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t* streamId) except* nogil:
    return _cusolverDn._cusolverDnGetStream(handle, streamId)


cdef cusolverStatus_t cusolverDnIRSParamsCreate(cusolverDnIRSParams_t* params_ptr) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsCreate(params_ptr)


cdef cusolverStatus_t cusolverDnIRSParamsDestroy(cusolverDnIRSParams_t params) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsDestroy(params)


cdef cusolverStatus_t cusolverDnIRSParamsSetRefinementSolver(cusolverDnIRSParams_t params, cusolverIRSRefinement_t refinement_solver) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsSetRefinementSolver(params, refinement_solver)


cdef cusolverStatus_t cusolverDnIRSParamsSetSolverMainPrecision(cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsSetSolverMainPrecision(params, solver_main_precision)


cdef cusolverStatus_t cusolverDnIRSParamsSetSolverLowestPrecision(cusolverDnIRSParams_t params, cusolverPrecType_t solver_lowest_precision) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsSetSolverLowestPrecision(params, solver_lowest_precision)


cdef cusolverStatus_t cusolverDnIRSParamsSetSolverPrecisions(cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision, cusolverPrecType_t solver_lowest_precision) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsSetSolverPrecisions(params, solver_main_precision, solver_lowest_precision)


cdef cusolverStatus_t cusolverDnIRSParamsSetTol(cusolverDnIRSParams_t params, double val) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsSetTol(params, val)


cdef cusolverStatus_t cusolverDnIRSParamsSetTolInner(cusolverDnIRSParams_t params, double val) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsSetTolInner(params, val)


cdef cusolverStatus_t cusolverDnIRSParamsSetMaxIters(cusolverDnIRSParams_t params, cusolver_int_t maxiters) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsSetMaxIters(params, maxiters)


cdef cusolverStatus_t cusolverDnIRSParamsSetMaxItersInner(cusolverDnIRSParams_t params, cusolver_int_t maxiters_inner) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsSetMaxItersInner(params, maxiters_inner)


cdef cusolverStatus_t cusolverDnIRSParamsGetMaxIters(cusolverDnIRSParams_t params, cusolver_int_t* maxiters) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsGetMaxIters(params, maxiters)


cdef cusolverStatus_t cusolverDnIRSParamsEnableFallback(cusolverDnIRSParams_t params) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsEnableFallback(params)


cdef cusolverStatus_t cusolverDnIRSParamsDisableFallback(cusolverDnIRSParams_t params) except* nogil:
    return _cusolverDn._cusolverDnIRSParamsDisableFallback(params)


cdef cusolverStatus_t cusolverDnIRSInfosDestroy(cusolverDnIRSInfos_t infos) except* nogil:
    return _cusolverDn._cusolverDnIRSInfosDestroy(infos)


cdef cusolverStatus_t cusolverDnIRSInfosCreate(cusolverDnIRSInfos_t* infos_ptr) except* nogil:
    return _cusolverDn._cusolverDnIRSInfosCreate(infos_ptr)


cdef cusolverStatus_t cusolverDnIRSInfosGetNiters(cusolverDnIRSInfos_t infos, cusolver_int_t* niters) except* nogil:
    return _cusolverDn._cusolverDnIRSInfosGetNiters(infos, niters)


cdef cusolverStatus_t cusolverDnIRSInfosGetOuterNiters(cusolverDnIRSInfos_t infos, cusolver_int_t* outer_niters) except* nogil:
    return _cusolverDn._cusolverDnIRSInfosGetOuterNiters(infos, outer_niters)


cdef cusolverStatus_t cusolverDnIRSInfosRequestResidual(cusolverDnIRSInfos_t infos) except* nogil:
    return _cusolverDn._cusolverDnIRSInfosRequestResidual(infos)


cdef cusolverStatus_t cusolverDnIRSInfosGetResidualHistory(cusolverDnIRSInfos_t infos, void** residual_history) except* nogil:
    return _cusolverDn._cusolverDnIRSInfosGetResidualHistory(infos, residual_history)


cdef cusolverStatus_t cusolverDnIRSInfosGetMaxIters(cusolverDnIRSInfos_t infos, cusolver_int_t* maxiters) except* nogil:
    return _cusolverDn._cusolverDnIRSInfosGetMaxIters(infos, maxiters)


cdef cusolverStatus_t cusolverDnZZgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnZZgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnZCgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnZCgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnZKgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnZKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnZEgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnZEgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnZYgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnZYgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnCCgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnCCgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnCEgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnCEgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnCKgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnCKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnCYgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnCYgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnDDgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnDDgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnDSgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnDSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnDHgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnDHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnDBgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnDBgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnDXgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnDXgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnSSgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnSSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnSHgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnSHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnSBgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnSBgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnSXgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnSXgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnZZgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnZZgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnZCgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnZCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnZKgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnZKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnZEgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnZEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnZYgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnZYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnCCgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnCCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnCKgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnCKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnCEgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnCEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnCYgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnCYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnDDgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnDDgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnDSgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnDSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnDHgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnDHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnDBgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnDBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnDXgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnDXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnSSgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnSSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnSHgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnSHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnSBgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnSBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnSXgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnSXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnZZgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnZZgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnZCgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnZCgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnZKgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnZKgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnZEgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnZEgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnZYgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnZYgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnCCgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnCCgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnCKgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnCKgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnCEgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnCEgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnCYgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnCYgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnDDgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnDDgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnDSgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnDSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnDHgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnDHgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnDBgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnDBgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnDXgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnDXgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnSSgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnSSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnSHgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnSHgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnSBgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnSBgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnSXgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnSXgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t cusolverDnZZgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnZZgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnZCgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnZCgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnZKgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnZKgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnZEgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnZEgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnZYgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnZYgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnCCgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnCCgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnCKgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnCKgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnCEgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnCEgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnCYgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnCYgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnDDgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnDDgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnDSgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnDSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnDHgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnDHgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnDBgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnDBgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnDXgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnDXgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnSSgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnSSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnSHgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnSHgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnSBgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnSBgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnSXgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnSXgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t cusolverDnIRSXgesv(cusolverDnHandle_t handle, cusolverDnIRSParams_t gesv_irs_params, cusolverDnIRSInfos_t gesv_irs_infos, cusolver_int_t n, cusolver_int_t nrhs, void* dA, cusolver_int_t ldda, void* dB, cusolver_int_t lddb, void* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* niters, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnIRSXgesv(handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info)


cdef cusolverStatus_t cusolverDnIRSXgesv_bufferSize(cusolverDnHandle_t handle, cusolverDnIRSParams_t params, cusolver_int_t n, cusolver_int_t nrhs, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnIRSXgesv_bufferSize(handle, params, n, nrhs, lwork_bytes)


cdef cusolverStatus_t cusolverDnIRSXgels(cusolverDnHandle_t handle, cusolverDnIRSParams_t gels_irs_params, cusolverDnIRSInfos_t gels_irs_infos, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, void* dA, cusolver_int_t ldda, void* dB, cusolver_int_t lddb, void* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* niters, cusolver_int_t* d_info) except* nogil:
    return _cusolverDn._cusolverDnIRSXgels(handle, gels_irs_params, gels_irs_infos, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info)


cdef cusolverStatus_t cusolverDnIRSXgels_bufferSize(cusolverDnHandle_t handle, cusolverDnIRSParams_t params, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, size_t* lwork_bytes) except* nogil:
    return _cusolverDn._cusolverDnIRSXgels_bufferSize(handle, params, m, n, nrhs, lwork_bytes)


cdef cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnCpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnZpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* Workspace, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* Workspace, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnCpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* Workspace, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnZpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* Workspace, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnZpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float* A, int lda, float* B, int ldb, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t cusolverDnCpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuComplex* A, int lda, cuComplex* B, int ldb, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t cusolverDnZpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t cusolverDnSpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* Aarray[], int lda, int* infoArray, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnSpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t cusolverDnDpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* Aarray[], int lda, int* infoArray, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnDpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t cusolverDnCpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* Aarray[], int lda, int* infoArray, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnCpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t cusolverDnZpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* Aarray[], int lda, int* infoArray, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnZpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t cusolverDnSpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, float* A[], int lda, float* B[], int ldb, int* d_info, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t cusolverDnDpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, double* A[], int lda, double* B[], int ldb, int* d_info, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t cusolverDnCpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, cuComplex* A[], int lda, cuComplex* B[], int ldb, int* d_info, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t cusolverDnZpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, cuDoubleComplex* A[], int lda, cuDoubleComplex* B[], int ldb, int* d_info, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t cusolverDnSpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSpotri_bufferSize(handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnDpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDpotri_bufferSize(handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnCpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCpotri_bufferSize(handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnZpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZpotri_bufferSize(handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnSpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnSpotri(handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnDpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnDpotri(handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnCpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnCpotri(handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnZpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnZpotri(handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnSlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSlauum_bufferSize(handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnDlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDlauum_bufferSize(handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnClauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnClauum_bufferSize(handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnZlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZlauum_bufferSize(handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnSlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnSlauum(handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnDlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnDlauum(handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnClauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnClauum(handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnZlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnZlauum(handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* Workspace, int* devIpiv, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* Workspace, int* devIpiv, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t cusolverDnCgetrf(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, cuComplex* Workspace, int* devIpiv, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t cusolverDnZgetrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* Workspace, int* devIpiv, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t cusolverDnSlaswp(cusolverDnHandle_t handle, int n, float* A, int lda, int k1, int k2, const int* devIpiv, int incx) except* nogil:
    return _cusolverDn._cusolverDnSlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t cusolverDnDlaswp(cusolverDnHandle_t handle, int n, double* A, int lda, int k1, int k2, const int* devIpiv, int incx) except* nogil:
    return _cusolverDn._cusolverDnDlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t cusolverDnClaswp(cusolverDnHandle_t handle, int n, cuComplex* A, int lda, int k1, int k2, const int* devIpiv, int incx) except* nogil:
    return _cusolverDn._cusolverDnClaswp(handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t cusolverDnZlaswp(cusolverDnHandle_t handle, int n, cuDoubleComplex* A, int lda, int k1, int k2, const int* devIpiv, int incx) except* nogil:
    return _cusolverDn._cusolverDnZlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t cusolverDnCgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* A, int lda, const int* devIpiv, cuComplex* B, int ldb, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t cusolverDnZgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* A, int lda, const int* devIpiv, cuDoubleComplex* B, int ldb, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnCgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCgeqrf_bufferSize(handle, m, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnZgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZgeqrf_bufferSize(handle, m, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* TAU, float* Workspace, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* TAU, double* Workspace, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnCgeqrf(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, cuComplex* TAU, cuComplex* Workspace, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnCgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnZgeqrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* TAU, cuDoubleComplex* Workspace, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnZgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const float* A, int lda, const float* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnDorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const double* A, int lda, const double* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnCungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnZungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnSorgqr(cusolverDnHandle_t handle, int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnDorgqr(cusolverDnHandle_t handle, int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnCungqr(cusolverDnHandle_t handle, int m, int n, int k, cuComplex* A, int lda, const cuComplex* tau, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnCungqr(handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnZungqr(cusolverDnHandle_t handle, int m, int n, int k, cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZungqr(handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float* A, int lda, const float* tau, const float* C, int ldc, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double* A, int lda, const double* tau, const double* C, int ldc, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t cusolverDnCunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, const cuComplex* C, int ldc, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t cusolverDnZunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, const cuDoubleComplex* C, int ldc, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t cusolverDnSormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, float* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnDormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, double* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnCunmqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, cuComplex* C, int ldc, cuComplex* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnZunmqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* C, int ldc, cuDoubleComplex* work, int lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t cusolverDnSsytrf_bufferSize(cusolverDnHandle_t handle, int n, float* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSsytrf_bufferSize(handle, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnDsytrf_bufferSize(cusolverDnHandle_t handle, int n, double* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDsytrf_bufferSize(handle, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnCsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuComplex* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCsytrf_bufferSize(handle, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnZsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuDoubleComplex* A, int lda, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZsytrf_bufferSize(handle, n, A, lda, lwork)


cdef cusolverStatus_t cusolverDnSsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* ipiv, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t cusolverDnDsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* ipiv, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t cusolverDnCsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* ipiv, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t cusolverDnZsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* ipiv, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t cusolverDnSsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, const int* ipiv, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t cusolverDnDsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, const int* ipiv, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t cusolverDnCsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, const int* ipiv, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t cusolverDnZsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, const int* ipiv, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t cusolverDnSsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, const int* ipiv, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t cusolverDnDsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, const int* ipiv, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t cusolverDnCsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, const int* ipiv, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnCsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t cusolverDnZsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, const int* ipiv, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t cusolverDnSgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnSgebrd_bufferSize(handle, m, n, Lwork)


cdef cusolverStatus_t cusolverDnDgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnDgebrd_bufferSize(handle, m, n, Lwork)


cdef cusolverStatus_t cusolverDnCgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnCgebrd_bufferSize(handle, m, n, Lwork)


cdef cusolverStatus_t cusolverDnZgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except* nogil:
    return _cusolverDn._cusolverDnZgebrd_bufferSize(handle, m, n, Lwork)


cdef cusolverStatus_t cusolverDnSgebrd(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* D, float* E, float* TAUQ, float* TAUP, float* Work, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnSgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnDgebrd(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* D, double* E, double* TAUQ, double* TAUP, double* Work, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnDgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnCgebrd(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, float* D, float* E, cuComplex* TAUQ, cuComplex* TAUP, cuComplex* Work, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnCgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnZgebrd(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, double* D, double* E, cuDoubleComplex* TAUQ, cuDoubleComplex* TAUP, cuDoubleComplex* Work, int Lwork, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnZgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t cusolverDnSorgbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const float* A, int lda, const float* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnDorgbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const double* A, int lda, const double* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnCungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnZungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnSorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnDorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnCungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, cuComplex* A, int lda, const cuComplex* tau, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnZungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnSsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, const float* d, const float* e, const float* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t cusolverDnDsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, const double* d, const double* e, const double* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t cusolverDnChetrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* d, const float* e, const cuComplex* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnChetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t cusolverDnZhetrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* d, const double* e, const cuDoubleComplex* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZhetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t cusolverDnSsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* d, float* e, float* tau, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnDsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* d, double* e, double* tau, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnChetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* d, float* e, cuComplex* tau, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnChetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnZhetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* d, double* e, cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZhetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnSorgtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, const float* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnDorgtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, const double* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnCungtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnZungtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t cusolverDnSorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, const float* tau, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnDorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, const double* tau, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnCungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, const cuComplex* tau, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnCungtr(handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnZungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZungtr(handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t cusolverDnSormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const float* A, int lda, const float* tau, const float* C, int ldc, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t cusolverDnDormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const double* A, int lda, const double* tau, const double* C, int ldc, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t cusolverDnCunmtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const cuComplex* A, int lda, const cuComplex* tau, const cuComplex* C, int ldc, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t cusolverDnZunmtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, const cuDoubleComplex* C, int ldc, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t cusolverDnSormtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, float* A, int lda, float* tau, float* C, int ldc, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t cusolverDnDormtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, double* A, int lda, double* tau, double* C, int ldc, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t cusolverDnCunmtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, cuComplex* A, int lda, cuComplex* tau, cuComplex* C, int ldc, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t cusolverDnZunmtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* tau, cuDoubleComplex* C, int ldc, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSgesvd_bufferSize(handle, m, n, lwork)


cdef cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDgesvd_bufferSize(handle, m, n, lwork)


cdef cusolverStatus_t cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCgesvd_bufferSize(handle, m, n, lwork)


cdef cusolverStatus_t cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZgesvd_bufferSize(handle, m, n, lwork)


cdef cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* VT, int ldvt, float* work, int lwork, float* rwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt, double* work, int lwork, double* rwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* VT, int ldvt, cuComplex* work, int lwork, float* rwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnCgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* VT, int ldvt, cuDoubleComplex* work, int lwork, double* rwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnCheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnSsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float* A, int lda, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t cusolverDnDsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double* A, int lda, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t cusolverDnCheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t cusolverDnZheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t cusolverDnSsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, float* A, int lda, float vl, float vu, int il, int iu, int* meig, float* W, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnDsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, double* A, int lda, double vl, double vu, int il, int iu, int* meig, double* W, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnCheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float vl, float vu, int il, int iu, int* meig, float* W, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnZheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double vl, double vu, int il, int iu, int* meig, double* W, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnSsygvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t cusolverDnDsygvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t cusolverDnChegvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* B, int ldb, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t cusolverDnZhegvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t cusolverDnSsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float vl, float vu, int il, int iu, int* meig, float* W, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnDsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double vl, double vu, int il, int iu, int* meig, double* W, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnChegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb, float vl, float vu, int il, int iu, int* meig, float* W, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnZhegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, double vl, double vu, int il, int iu, int* meig, double* W, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnSsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, const float* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t cusolverDnDsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, const double* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t cusolverDnChegvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t cusolverDnZhegvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* W, int* lwork) except* nogil:
    return _cusolverDn._cusolverDnZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t cusolverDnSsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float* W, float* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnDsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double* W, double* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnChegvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb, float* W, cuComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnZhegvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, double* W, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    return _cusolverDn._cusolverDnZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* info) except* nogil:
    return _cusolverDn._cusolverDnCreateSyevjInfo(info)


cdef cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info) except* nogil:
    return _cusolverDn._cusolverDnDestroySyevjInfo(info)


cdef cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance) except* nogil:
    return _cusolverDn._cusolverDnXsyevjSetTolerance(info, tolerance)


cdef cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps) except* nogil:
    return _cusolverDn._cusolverDnXsyevjSetMaxSweeps(info, max_sweeps)


cdef cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig) except* nogil:
    return _cusolverDn._cusolverDnXsyevjSetSortEig(info, sort_eig)


cdef cusolverStatus_t cusolverDnXsyevjGetResidual(cusolverDnHandle_t handle, syevjInfo_t info, double* residual) except* nogil:
    return _cusolverDn._cusolverDnXsyevjGetResidual(handle, info, residual)


cdef cusolverStatus_t cusolverDnXsyevjGetSweeps(cusolverDnHandle_t handle, syevjInfo_t info, int* executed_sweeps) except* nogil:
    return _cusolverDn._cusolverDnXsyevjGetSweeps(handle, info, executed_sweeps)


cdef cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork, syevjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork, syevjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t cusolverDnCheevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork, syevjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t cusolverDnZheevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork, syevjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t cusolverDnSsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info, syevjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info, syevjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t cusolverDnCheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t cusolverDnZheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t cusolverDnSsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t cusolverDnDsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t cusolverDnCheevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t cusolverDnZheevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t cusolverDnSsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnDsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnCheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnZheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnSsygvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, const float* W, int* lwork, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t cusolverDnDsygvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, const double* W, int* lwork, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t cusolverDnChegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* W, int* lwork, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t cusolverDnZhegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* W, int* lwork, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t cusolverDnSsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float* W, float* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnDsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double* W, double* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnChegvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnZhegvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info) except* nogil:
    return _cusolverDn._cusolverDnCreateGesvdjInfo(info)


cdef cusolverStatus_t cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info) except* nogil:
    return _cusolverDn._cusolverDnDestroyGesvdjInfo(info)


cdef cusolverStatus_t cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance) except* nogil:
    return _cusolverDn._cusolverDnXgesvdjSetTolerance(info, tolerance)


cdef cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int max_sweeps) except* nogil:
    return _cusolverDn._cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps)


cdef cusolverStatus_t cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_svd) except* nogil:
    return _cusolverDn._cusolverDnXgesvdjSetSortEig(info, sort_svd)


cdef cusolverStatus_t cusolverDnXgesvdjGetResidual(cusolverDnHandle_t handle, gesvdjInfo_t info, double* residual) except* nogil:
    return _cusolverDn._cusolverDnXgesvdjGetResidual(handle, info, residual)


cdef cusolverStatus_t cusolverDnXgesvdjGetSweeps(cusolverDnHandle_t handle, gesvdjInfo_t info, int* executed_sweeps) except* nogil:
    return _cusolverDn._cusolverDnXgesvdjGetSweeps(handle, info, executed_sweeps)


cdef cusolverStatus_t cusolverDnSgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const float* A, int lda, const float* S, const float* U, int ldu, const float* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t cusolverDnDgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const double* A, int lda, const double* S, const double* U, int ldu, const double* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t cusolverDnCgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuComplex* A, int lda, const float* S, const cuComplex* U, int ldu, const cuComplex* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t cusolverDnZgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuDoubleComplex* A, int lda, const double* S, const cuDoubleComplex* U, int ldu, const cuDoubleComplex* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t cusolverDnSgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t cusolverDnDgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t cusolverDnCgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* V, int ldv, cuComplex* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t cusolverDnZgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* V, int ldv, cuDoubleComplex* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t cusolverDnSgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const float* A, int lda, const float* S, const float* U, int ldu, const float* V, int ldv, int* lwork, gesvdjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t cusolverDnDgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const double* A, int lda, const double* S, const double* U, int ldu, const double* V, int ldv, int* lwork, gesvdjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t cusolverDnCgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuComplex* A, int lda, const float* S, const cuComplex* U, int ldu, const cuComplex* V, int ldv, int* lwork, gesvdjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t cusolverDnZgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuDoubleComplex* A, int lda, const double* S, const cuDoubleComplex* U, int ldu, const cuDoubleComplex* V, int ldv, int* lwork, gesvdjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t cusolverDnSgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, int* info, gesvdjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnDgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, int* info, gesvdjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnCgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* V, int ldv, cuComplex* work, int lwork, int* info, gesvdjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnZgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* V, int ldv, cuDoubleComplex* work, int lwork, int* info, gesvdjInfo_t params) except* nogil:
    return _cusolverDn._cusolverDnZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t cusolverDnSgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float* d_A, int lda, long long int strideA, const float* d_S, long long int strideS, const float* d_U, int ldu, long long int strideU, const float* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnSgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t cusolverDnDgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double* d_A, int lda, long long int strideA, const double* d_S, long long int strideS, const double* d_U, int ldu, long long int strideU, const double* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnDgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t cusolverDnCgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex* d_A, int lda, long long int strideA, const float* d_S, long long int strideS, const cuComplex* d_U, int ldu, long long int strideU, const cuComplex* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnCgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t cusolverDnZgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex* d_A, int lda, long long int strideA, const double* d_S, long long int strideS, const cuDoubleComplex* d_U, int ldu, long long int strideU, const cuDoubleComplex* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnZgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t cusolverDnSgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float* d_A, int lda, long long int strideA, float* d_S, long long int strideS, float* d_U, int ldu, long long int strideU, float* d_V, int ldv, long long int strideV, float* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnSgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t cusolverDnDgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double* d_A, int lda, long long int strideA, double* d_S, long long int strideS, double* d_U, int ldu, long long int strideU, double* d_V, int ldv, long long int strideV, double* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnDgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t cusolverDnCgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex* d_A, int lda, long long int strideA, float* d_S, long long int strideS, cuComplex* d_U, int ldu, long long int strideU, cuComplex* d_V, int ldv, long long int strideV, cuComplex* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnCgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t cusolverDnZgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex* d_A, int lda, long long int strideA, double* d_S, long long int strideS, cuDoubleComplex* d_U, int ldu, long long int strideU, cuDoubleComplex* d_V, int ldv, long long int strideV, cuDoubleComplex* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except* nogil:
    return _cusolverDn._cusolverDnZgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params) except* nogil:
    return _cusolverDn._cusolverDnCreateParams(params)


cdef cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params) except* nogil:
    return _cusolverDn._cusolverDnDestroyParams(params)


cdef cusolverStatus_t cusolverDnSetAdvOptions(cusolverDnParams_t params, cusolverDnFunction_t function, cusolverAlgMode_t algo) except* nogil:
    return _cusolverDn._cusolverDnSetAdvOptions(params, function, algo)


cdef cusolverStatus_t cusolverDnXpotrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXpotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXpotrf(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    return _cusolverDn._cusolverDnXpotrf(handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t cusolverDnXpotrs(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeB, void* B, int64_t ldb, int* info) except* nogil:
    return _cusolverDn._cusolverDnXpotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info)


cdef cusolverStatus_t cusolverDnXgeqrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeTau, const void* tau, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXgeqrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXgeqrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeTau, void* tau, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    return _cusolverDn._cusolverDnXgeqrf(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t cusolverDnXgetrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXgetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXgetrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, int64_t* ipiv, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    return _cusolverDn._cusolverDnXgetrf(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasOperation_t trans, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, const int64_t* ipiv, cudaDataType dataTypeB, void* B, int64_t ldb, int* info) except* nogil:
    return _cusolverDn._cusolverDnXgetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info)


cdef cusolverStatus_t cusolverDnXsyevd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeW, const void* W, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXsyevd_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    return _cusolverDn._cusolverDnXsyevd(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t cusolverDnXsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, void* vl, void* vu, int64_t il, int64_t iu, int64_t* h_meig, cudaDataType dataTypeW, const void* W, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXsyevdx_bufferSize(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXsyevdx(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, void* vl, void* vu, int64_t il, int64_t iu, int64_t* meig64, cudaDataType dataTypeW, void* W, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    return _cusolverDn._cusolverDnXsyevdx(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t cusolverDnXgesvd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeS, const void* S, cudaDataType dataTypeU, const void* U, int64_t ldu, cudaDataType dataTypeVT, const void* VT, int64_t ldvt, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXgesvd_bufferSize(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXgesvd(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeS, void* S, cudaDataType dataTypeU, void* U, int64_t ldu, cudaDataType dataTypeVT, void* VT, int64_t ldvt, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    return _cusolverDn._cusolverDnXgesvd(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t cusolverDnXgesvdp_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, int econ, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeS, const void* S, cudaDataType dataTypeU, const void* U, int64_t ldu, cudaDataType dataTypeV, const void* V, int64_t ldv, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXgesvdp_bufferSize(handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXgesvdp(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, int econ, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeS, void* S, cudaDataType dataTypeU, void* U, int64_t ldu, cudaDataType dataTypeV, void* V, int64_t ldv, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* d_info, double* h_err_sigma) except* nogil:
    return _cusolverDn._cusolverDnXgesvdp(handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info, h_err_sigma)


cdef cusolverStatus_t cusolverDnXgesvdr_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeSrand, const void* Srand, cudaDataType dataTypeUrand, const void* Urand, int64_t ldUrand, cudaDataType dataTypeVrand, const void* Vrand, int64_t ldVrand, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXgesvdr_bufferSize(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXgesvdr(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeSrand, void* Srand, cudaDataType dataTypeUrand, void* Urand, int64_t ldUrand, cudaDataType dataTypeVrand, void* Vrand, int64_t ldVrand, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* d_info) except* nogil:
    return _cusolverDn._cusolverDnXgesvdr(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)


cdef cusolverStatus_t cusolverDnXsytrs_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, const int64_t* ipiv, cudaDataType dataTypeB, void* B, int64_t ldb, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXsytrs_bufferSize(handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXsytrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, const int64_t* ipiv, cudaDataType dataTypeB, void* B, int64_t ldb, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    return _cusolverDn._cusolverDnXsytrs(handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t cusolverDnXtrtri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXtrtri_bufferSize(handle, uplo, diag, n, dataTypeA, A, lda, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXtrtri(cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* devInfo) except* nogil:
    return _cusolverDn._cusolverDnXtrtri(handle, uplo, diag, n, dataTypeA, A, lda, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, devInfo)


cdef cusolverStatus_t cusolverDnLoggerSetCallback(cusolverDnLoggerCallback_t callback) except* nogil:
    return _cusolverDn._cusolverDnLoggerSetCallback(callback)


cdef cusolverStatus_t cusolverDnLoggerSetFile(FILE* file) except* nogil:
    return _cusolverDn._cusolverDnLoggerSetFile(file)


cdef cusolverStatus_t cusolverDnLoggerOpenFile(const char* logFile) except* nogil:
    return _cusolverDn._cusolverDnLoggerOpenFile(logFile)


cdef cusolverStatus_t cusolverDnLoggerSetLevel(int level) except* nogil:
    return _cusolverDn._cusolverDnLoggerSetLevel(level)


cdef cusolverStatus_t cusolverDnLoggerSetMask(int mask) except* nogil:
    return _cusolverDn._cusolverDnLoggerSetMask(mask)


cdef cusolverStatus_t cusolverDnLoggerForceDisable() except* nogil:
    return _cusolverDn._cusolverDnLoggerForceDisable()


cdef cusolverStatus_t cusolverDnSetDeterministicMode(cusolverDnHandle_t handle, cusolverDeterministicMode_t mode) except* nogil:
    return _cusolverDn._cusolverDnSetDeterministicMode(handle, mode)


cdef cusolverStatus_t cusolverDnGetDeterministicMode(cusolverDnHandle_t handle, cusolverDeterministicMode_t* mode) except* nogil:
    return _cusolverDn._cusolverDnGetDeterministicMode(handle, mode)


cdef cusolverStatus_t cusolverDnXlarft_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n, int64_t k, cudaDataType dataTypeV, const void* V, int64_t ldv, cudaDataType dataTypeTau, const void* tau, cudaDataType dataTypeT, void* T, int64_t ldt, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXlarft_bufferSize(handle, params, direct, storev, n, k, dataTypeV, V, ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXlarft(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n, int64_t k, cudaDataType dataTypeV, const void* V, int64_t ldv, cudaDataType dataTypeTau, const void* tau, cudaDataType dataTypeT, void* T, int64_t ldt, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXlarft(handle, params, direct, storev, n, k, dataTypeV, V, ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXsyevBatched_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeW, const void* W, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost, int64_t batchSize) except* nogil:
    return _cusolverDn._cusolverDnXsyevBatched_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost, batchSize)


cdef cusolverStatus_t cusolverDnXsyevBatched(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info, int64_t batchSize) except* nogil:
    return _cusolverDn._cusolverDnXsyevBatched(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info, batchSize)


cdef cusolverStatus_t cusolverDnXgeev_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobvl, cusolverEigMode_t jobvr, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeW, const void* W, cudaDataType dataTypeVL, const void* VL, int64_t ldvl, cudaDataType dataTypeVR, const void* VR, int64_t ldvr, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    return _cusolverDn._cusolverDnXgeev_bufferSize(handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t cusolverDnXgeev(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobvl, cusolverEigMode_t jobvr, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType dataTypeVL, void* VL, int64_t ldvl, cudaDataType dataTypeVR, void* VR, int64_t ldvr, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    return _cusolverDn._cusolverDnXgeev(handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
