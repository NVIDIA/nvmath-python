# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_cusolver_dso_version_suffix

from .utils import FunctionNotFoundError, NotSupportedError


###############################################################################
# Extern
###############################################################################

cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()
    void* dlsym(void*, const char*)
    int dlclose(void*)

    enum:
        RTLD_LAZY
        RTLD_NOW
        RTLD_GLOBAL
        RTLD_LOCAL

    const void* RTLD_DEFAULT 'RTLD_DEFAULT'


###############################################################################
# Wrapper init
###############################################################################

cdef bint __py_cusolverDn_init = False
cdef void* __cuDriverGetVersion = NULL

cdef void* __cusolverDnCreate = NULL
cdef void* __cusolverDnDestroy = NULL
cdef void* __cusolverDnSetStream = NULL
cdef void* __cusolverDnGetStream = NULL
cdef void* __cusolverDnIRSParamsCreate = NULL
cdef void* __cusolverDnIRSParamsDestroy = NULL
cdef void* __cusolverDnIRSParamsSetRefinementSolver = NULL
cdef void* __cusolverDnIRSParamsSetSolverMainPrecision = NULL
cdef void* __cusolverDnIRSParamsSetSolverLowestPrecision = NULL
cdef void* __cusolverDnIRSParamsSetSolverPrecisions = NULL
cdef void* __cusolverDnIRSParamsSetTol = NULL
cdef void* __cusolverDnIRSParamsSetTolInner = NULL
cdef void* __cusolverDnIRSParamsSetMaxIters = NULL
cdef void* __cusolverDnIRSParamsSetMaxItersInner = NULL
cdef void* __cusolverDnIRSParamsGetMaxIters = NULL
cdef void* __cusolverDnIRSParamsEnableFallback = NULL
cdef void* __cusolverDnIRSParamsDisableFallback = NULL
cdef void* __cusolverDnIRSInfosDestroy = NULL
cdef void* __cusolverDnIRSInfosCreate = NULL
cdef void* __cusolverDnIRSInfosGetNiters = NULL
cdef void* __cusolverDnIRSInfosGetOuterNiters = NULL
cdef void* __cusolverDnIRSInfosRequestResidual = NULL
cdef void* __cusolverDnIRSInfosGetResidualHistory = NULL
cdef void* __cusolverDnIRSInfosGetMaxIters = NULL
cdef void* __cusolverDnZZgesv = NULL
cdef void* __cusolverDnZCgesv = NULL
cdef void* __cusolverDnZKgesv = NULL
cdef void* __cusolverDnZEgesv = NULL
cdef void* __cusolverDnZYgesv = NULL
cdef void* __cusolverDnCCgesv = NULL
cdef void* __cusolverDnCEgesv = NULL
cdef void* __cusolverDnCKgesv = NULL
cdef void* __cusolverDnCYgesv = NULL
cdef void* __cusolverDnDDgesv = NULL
cdef void* __cusolverDnDSgesv = NULL
cdef void* __cusolverDnDHgesv = NULL
cdef void* __cusolverDnDBgesv = NULL
cdef void* __cusolverDnDXgesv = NULL
cdef void* __cusolverDnSSgesv = NULL
cdef void* __cusolverDnSHgesv = NULL
cdef void* __cusolverDnSBgesv = NULL
cdef void* __cusolverDnSXgesv = NULL
cdef void* __cusolverDnZZgesv_bufferSize = NULL
cdef void* __cusolverDnZCgesv_bufferSize = NULL
cdef void* __cusolverDnZKgesv_bufferSize = NULL
cdef void* __cusolverDnZEgesv_bufferSize = NULL
cdef void* __cusolverDnZYgesv_bufferSize = NULL
cdef void* __cusolverDnCCgesv_bufferSize = NULL
cdef void* __cusolverDnCKgesv_bufferSize = NULL
cdef void* __cusolverDnCEgesv_bufferSize = NULL
cdef void* __cusolverDnCYgesv_bufferSize = NULL
cdef void* __cusolverDnDDgesv_bufferSize = NULL
cdef void* __cusolverDnDSgesv_bufferSize = NULL
cdef void* __cusolverDnDHgesv_bufferSize = NULL
cdef void* __cusolverDnDBgesv_bufferSize = NULL
cdef void* __cusolverDnDXgesv_bufferSize = NULL
cdef void* __cusolverDnSSgesv_bufferSize = NULL
cdef void* __cusolverDnSHgesv_bufferSize = NULL
cdef void* __cusolverDnSBgesv_bufferSize = NULL
cdef void* __cusolverDnSXgesv_bufferSize = NULL
cdef void* __cusolverDnZZgels = NULL
cdef void* __cusolverDnZCgels = NULL
cdef void* __cusolverDnZKgels = NULL
cdef void* __cusolverDnZEgels = NULL
cdef void* __cusolverDnZYgels = NULL
cdef void* __cusolverDnCCgels = NULL
cdef void* __cusolverDnCKgels = NULL
cdef void* __cusolverDnCEgels = NULL
cdef void* __cusolverDnCYgels = NULL
cdef void* __cusolverDnDDgels = NULL
cdef void* __cusolverDnDSgels = NULL
cdef void* __cusolverDnDHgels = NULL
cdef void* __cusolverDnDBgels = NULL
cdef void* __cusolverDnDXgels = NULL
cdef void* __cusolverDnSSgels = NULL
cdef void* __cusolverDnSHgels = NULL
cdef void* __cusolverDnSBgels = NULL
cdef void* __cusolverDnSXgels = NULL
cdef void* __cusolverDnZZgels_bufferSize = NULL
cdef void* __cusolverDnZCgels_bufferSize = NULL
cdef void* __cusolverDnZKgels_bufferSize = NULL
cdef void* __cusolverDnZEgels_bufferSize = NULL
cdef void* __cusolverDnZYgels_bufferSize = NULL
cdef void* __cusolverDnCCgels_bufferSize = NULL
cdef void* __cusolverDnCKgels_bufferSize = NULL
cdef void* __cusolverDnCEgels_bufferSize = NULL
cdef void* __cusolverDnCYgels_bufferSize = NULL
cdef void* __cusolverDnDDgels_bufferSize = NULL
cdef void* __cusolverDnDSgels_bufferSize = NULL
cdef void* __cusolverDnDHgels_bufferSize = NULL
cdef void* __cusolverDnDBgels_bufferSize = NULL
cdef void* __cusolverDnDXgels_bufferSize = NULL
cdef void* __cusolverDnSSgels_bufferSize = NULL
cdef void* __cusolverDnSHgels_bufferSize = NULL
cdef void* __cusolverDnSBgels_bufferSize = NULL
cdef void* __cusolverDnSXgels_bufferSize = NULL
cdef void* __cusolverDnIRSXgesv = NULL
cdef void* __cusolverDnIRSXgesv_bufferSize = NULL
cdef void* __cusolverDnIRSXgels = NULL
cdef void* __cusolverDnIRSXgels_bufferSize = NULL
cdef void* __cusolverDnSpotrf_bufferSize = NULL
cdef void* __cusolverDnDpotrf_bufferSize = NULL
cdef void* __cusolverDnCpotrf_bufferSize = NULL
cdef void* __cusolverDnZpotrf_bufferSize = NULL
cdef void* __cusolverDnSpotrf = NULL
cdef void* __cusolverDnDpotrf = NULL
cdef void* __cusolverDnCpotrf = NULL
cdef void* __cusolverDnZpotrf = NULL
cdef void* __cusolverDnSpotrs = NULL
cdef void* __cusolverDnDpotrs = NULL
cdef void* __cusolverDnCpotrs = NULL
cdef void* __cusolverDnZpotrs = NULL
cdef void* __cusolverDnSpotrfBatched = NULL
cdef void* __cusolverDnDpotrfBatched = NULL
cdef void* __cusolverDnCpotrfBatched = NULL
cdef void* __cusolverDnZpotrfBatched = NULL
cdef void* __cusolverDnSpotrsBatched = NULL
cdef void* __cusolverDnDpotrsBatched = NULL
cdef void* __cusolverDnCpotrsBatched = NULL
cdef void* __cusolverDnZpotrsBatched = NULL
cdef void* __cusolverDnSpotri_bufferSize = NULL
cdef void* __cusolverDnDpotri_bufferSize = NULL
cdef void* __cusolverDnCpotri_bufferSize = NULL
cdef void* __cusolverDnZpotri_bufferSize = NULL
cdef void* __cusolverDnSpotri = NULL
cdef void* __cusolverDnDpotri = NULL
cdef void* __cusolverDnCpotri = NULL
cdef void* __cusolverDnZpotri = NULL
cdef void* __cusolverDnSlauum_bufferSize = NULL
cdef void* __cusolverDnDlauum_bufferSize = NULL
cdef void* __cusolverDnClauum_bufferSize = NULL
cdef void* __cusolverDnZlauum_bufferSize = NULL
cdef void* __cusolverDnSlauum = NULL
cdef void* __cusolverDnDlauum = NULL
cdef void* __cusolverDnClauum = NULL
cdef void* __cusolverDnZlauum = NULL
cdef void* __cusolverDnSgetrf_bufferSize = NULL
cdef void* __cusolverDnDgetrf_bufferSize = NULL
cdef void* __cusolverDnCgetrf_bufferSize = NULL
cdef void* __cusolverDnZgetrf_bufferSize = NULL
cdef void* __cusolverDnSgetrf = NULL
cdef void* __cusolverDnDgetrf = NULL
cdef void* __cusolverDnCgetrf = NULL
cdef void* __cusolverDnZgetrf = NULL
cdef void* __cusolverDnSlaswp = NULL
cdef void* __cusolverDnDlaswp = NULL
cdef void* __cusolverDnClaswp = NULL
cdef void* __cusolverDnZlaswp = NULL
cdef void* __cusolverDnSgetrs = NULL
cdef void* __cusolverDnDgetrs = NULL
cdef void* __cusolverDnCgetrs = NULL
cdef void* __cusolverDnZgetrs = NULL
cdef void* __cusolverDnSgeqrf_bufferSize = NULL
cdef void* __cusolverDnDgeqrf_bufferSize = NULL
cdef void* __cusolverDnCgeqrf_bufferSize = NULL
cdef void* __cusolverDnZgeqrf_bufferSize = NULL
cdef void* __cusolverDnSgeqrf = NULL
cdef void* __cusolverDnDgeqrf = NULL
cdef void* __cusolverDnCgeqrf = NULL
cdef void* __cusolverDnZgeqrf = NULL
cdef void* __cusolverDnSorgqr_bufferSize = NULL
cdef void* __cusolverDnDorgqr_bufferSize = NULL
cdef void* __cusolverDnCungqr_bufferSize = NULL
cdef void* __cusolverDnZungqr_bufferSize = NULL
cdef void* __cusolverDnSorgqr = NULL
cdef void* __cusolverDnDorgqr = NULL
cdef void* __cusolverDnCungqr = NULL
cdef void* __cusolverDnZungqr = NULL
cdef void* __cusolverDnSormqr_bufferSize = NULL
cdef void* __cusolverDnDormqr_bufferSize = NULL
cdef void* __cusolverDnCunmqr_bufferSize = NULL
cdef void* __cusolverDnZunmqr_bufferSize = NULL
cdef void* __cusolverDnSormqr = NULL
cdef void* __cusolverDnDormqr = NULL
cdef void* __cusolverDnCunmqr = NULL
cdef void* __cusolverDnZunmqr = NULL
cdef void* __cusolverDnSsytrf_bufferSize = NULL
cdef void* __cusolverDnDsytrf_bufferSize = NULL
cdef void* __cusolverDnCsytrf_bufferSize = NULL
cdef void* __cusolverDnZsytrf_bufferSize = NULL
cdef void* __cusolverDnSsytrf = NULL
cdef void* __cusolverDnDsytrf = NULL
cdef void* __cusolverDnCsytrf = NULL
cdef void* __cusolverDnZsytrf = NULL
cdef void* __cusolverDnSsytri_bufferSize = NULL
cdef void* __cusolverDnDsytri_bufferSize = NULL
cdef void* __cusolverDnCsytri_bufferSize = NULL
cdef void* __cusolverDnZsytri_bufferSize = NULL
cdef void* __cusolverDnSsytri = NULL
cdef void* __cusolverDnDsytri = NULL
cdef void* __cusolverDnCsytri = NULL
cdef void* __cusolverDnZsytri = NULL
cdef void* __cusolverDnSgebrd_bufferSize = NULL
cdef void* __cusolverDnDgebrd_bufferSize = NULL
cdef void* __cusolverDnCgebrd_bufferSize = NULL
cdef void* __cusolverDnZgebrd_bufferSize = NULL
cdef void* __cusolverDnSgebrd = NULL
cdef void* __cusolverDnDgebrd = NULL
cdef void* __cusolverDnCgebrd = NULL
cdef void* __cusolverDnZgebrd = NULL
cdef void* __cusolverDnSorgbr_bufferSize = NULL
cdef void* __cusolverDnDorgbr_bufferSize = NULL
cdef void* __cusolverDnCungbr_bufferSize = NULL
cdef void* __cusolverDnZungbr_bufferSize = NULL
cdef void* __cusolverDnSorgbr = NULL
cdef void* __cusolverDnDorgbr = NULL
cdef void* __cusolverDnCungbr = NULL
cdef void* __cusolverDnZungbr = NULL
cdef void* __cusolverDnSsytrd_bufferSize = NULL
cdef void* __cusolverDnDsytrd_bufferSize = NULL
cdef void* __cusolverDnChetrd_bufferSize = NULL
cdef void* __cusolverDnZhetrd_bufferSize = NULL
cdef void* __cusolverDnSsytrd = NULL
cdef void* __cusolverDnDsytrd = NULL
cdef void* __cusolverDnChetrd = NULL
cdef void* __cusolverDnZhetrd = NULL
cdef void* __cusolverDnSorgtr_bufferSize = NULL
cdef void* __cusolverDnDorgtr_bufferSize = NULL
cdef void* __cusolverDnCungtr_bufferSize = NULL
cdef void* __cusolverDnZungtr_bufferSize = NULL
cdef void* __cusolverDnSorgtr = NULL
cdef void* __cusolverDnDorgtr = NULL
cdef void* __cusolverDnCungtr = NULL
cdef void* __cusolverDnZungtr = NULL
cdef void* __cusolverDnSormtr_bufferSize = NULL
cdef void* __cusolverDnDormtr_bufferSize = NULL
cdef void* __cusolverDnCunmtr_bufferSize = NULL
cdef void* __cusolverDnZunmtr_bufferSize = NULL
cdef void* __cusolverDnSormtr = NULL
cdef void* __cusolverDnDormtr = NULL
cdef void* __cusolverDnCunmtr = NULL
cdef void* __cusolverDnZunmtr = NULL
cdef void* __cusolverDnSgesvd_bufferSize = NULL
cdef void* __cusolverDnDgesvd_bufferSize = NULL
cdef void* __cusolverDnCgesvd_bufferSize = NULL
cdef void* __cusolverDnZgesvd_bufferSize = NULL
cdef void* __cusolverDnSgesvd = NULL
cdef void* __cusolverDnDgesvd = NULL
cdef void* __cusolverDnCgesvd = NULL
cdef void* __cusolverDnZgesvd = NULL
cdef void* __cusolverDnSsyevd_bufferSize = NULL
cdef void* __cusolverDnDsyevd_bufferSize = NULL
cdef void* __cusolverDnCheevd_bufferSize = NULL
cdef void* __cusolverDnZheevd_bufferSize = NULL
cdef void* __cusolverDnSsyevd = NULL
cdef void* __cusolverDnDsyevd = NULL
cdef void* __cusolverDnCheevd = NULL
cdef void* __cusolverDnZheevd = NULL
cdef void* __cusolverDnSsyevdx_bufferSize = NULL
cdef void* __cusolverDnDsyevdx_bufferSize = NULL
cdef void* __cusolverDnCheevdx_bufferSize = NULL
cdef void* __cusolverDnZheevdx_bufferSize = NULL
cdef void* __cusolverDnSsyevdx = NULL
cdef void* __cusolverDnDsyevdx = NULL
cdef void* __cusolverDnCheevdx = NULL
cdef void* __cusolverDnZheevdx = NULL
cdef void* __cusolverDnSsygvdx_bufferSize = NULL
cdef void* __cusolverDnDsygvdx_bufferSize = NULL
cdef void* __cusolverDnChegvdx_bufferSize = NULL
cdef void* __cusolverDnZhegvdx_bufferSize = NULL
cdef void* __cusolverDnSsygvdx = NULL
cdef void* __cusolverDnDsygvdx = NULL
cdef void* __cusolverDnChegvdx = NULL
cdef void* __cusolverDnZhegvdx = NULL
cdef void* __cusolverDnSsygvd_bufferSize = NULL
cdef void* __cusolverDnDsygvd_bufferSize = NULL
cdef void* __cusolverDnChegvd_bufferSize = NULL
cdef void* __cusolverDnZhegvd_bufferSize = NULL
cdef void* __cusolverDnSsygvd = NULL
cdef void* __cusolverDnDsygvd = NULL
cdef void* __cusolverDnChegvd = NULL
cdef void* __cusolverDnZhegvd = NULL
cdef void* __cusolverDnCreateSyevjInfo = NULL
cdef void* __cusolverDnDestroySyevjInfo = NULL
cdef void* __cusolverDnXsyevjSetTolerance = NULL
cdef void* __cusolverDnXsyevjSetMaxSweeps = NULL
cdef void* __cusolverDnXsyevjSetSortEig = NULL
cdef void* __cusolverDnXsyevjGetResidual = NULL
cdef void* __cusolverDnXsyevjGetSweeps = NULL
cdef void* __cusolverDnSsyevjBatched_bufferSize = NULL
cdef void* __cusolverDnDsyevjBatched_bufferSize = NULL
cdef void* __cusolverDnCheevjBatched_bufferSize = NULL
cdef void* __cusolverDnZheevjBatched_bufferSize = NULL
cdef void* __cusolverDnSsyevjBatched = NULL
cdef void* __cusolverDnDsyevjBatched = NULL
cdef void* __cusolverDnCheevjBatched = NULL
cdef void* __cusolverDnZheevjBatched = NULL
cdef void* __cusolverDnSsyevj_bufferSize = NULL
cdef void* __cusolverDnDsyevj_bufferSize = NULL
cdef void* __cusolverDnCheevj_bufferSize = NULL
cdef void* __cusolverDnZheevj_bufferSize = NULL
cdef void* __cusolverDnSsyevj = NULL
cdef void* __cusolverDnDsyevj = NULL
cdef void* __cusolverDnCheevj = NULL
cdef void* __cusolverDnZheevj = NULL
cdef void* __cusolverDnSsygvj_bufferSize = NULL
cdef void* __cusolverDnDsygvj_bufferSize = NULL
cdef void* __cusolverDnChegvj_bufferSize = NULL
cdef void* __cusolverDnZhegvj_bufferSize = NULL
cdef void* __cusolverDnSsygvj = NULL
cdef void* __cusolverDnDsygvj = NULL
cdef void* __cusolverDnChegvj = NULL
cdef void* __cusolverDnZhegvj = NULL
cdef void* __cusolverDnCreateGesvdjInfo = NULL
cdef void* __cusolverDnDestroyGesvdjInfo = NULL
cdef void* __cusolverDnXgesvdjSetTolerance = NULL
cdef void* __cusolverDnXgesvdjSetMaxSweeps = NULL
cdef void* __cusolverDnXgesvdjSetSortEig = NULL
cdef void* __cusolverDnXgesvdjGetResidual = NULL
cdef void* __cusolverDnXgesvdjGetSweeps = NULL
cdef void* __cusolverDnSgesvdjBatched_bufferSize = NULL
cdef void* __cusolverDnDgesvdjBatched_bufferSize = NULL
cdef void* __cusolverDnCgesvdjBatched_bufferSize = NULL
cdef void* __cusolverDnZgesvdjBatched_bufferSize = NULL
cdef void* __cusolverDnSgesvdjBatched = NULL
cdef void* __cusolverDnDgesvdjBatched = NULL
cdef void* __cusolverDnCgesvdjBatched = NULL
cdef void* __cusolverDnZgesvdjBatched = NULL
cdef void* __cusolverDnSgesvdj_bufferSize = NULL
cdef void* __cusolverDnDgesvdj_bufferSize = NULL
cdef void* __cusolverDnCgesvdj_bufferSize = NULL
cdef void* __cusolverDnZgesvdj_bufferSize = NULL
cdef void* __cusolverDnSgesvdj = NULL
cdef void* __cusolverDnDgesvdj = NULL
cdef void* __cusolverDnCgesvdj = NULL
cdef void* __cusolverDnZgesvdj = NULL
cdef void* __cusolverDnSgesvdaStridedBatched_bufferSize = NULL
cdef void* __cusolverDnDgesvdaStridedBatched_bufferSize = NULL
cdef void* __cusolverDnCgesvdaStridedBatched_bufferSize = NULL
cdef void* __cusolverDnZgesvdaStridedBatched_bufferSize = NULL
cdef void* __cusolverDnSgesvdaStridedBatched = NULL
cdef void* __cusolverDnDgesvdaStridedBatched = NULL
cdef void* __cusolverDnCgesvdaStridedBatched = NULL
cdef void* __cusolverDnZgesvdaStridedBatched = NULL
cdef void* __cusolverDnCreateParams = NULL
cdef void* __cusolverDnDestroyParams = NULL
cdef void* __cusolverDnSetAdvOptions = NULL
cdef void* __cusolverDnXpotrf_bufferSize = NULL
cdef void* __cusolverDnXpotrf = NULL
cdef void* __cusolverDnXpotrs = NULL
cdef void* __cusolverDnXgeqrf_bufferSize = NULL
cdef void* __cusolverDnXgeqrf = NULL
cdef void* __cusolverDnXgetrf_bufferSize = NULL
cdef void* __cusolverDnXgetrf = NULL
cdef void* __cusolverDnXgetrs = NULL
cdef void* __cusolverDnXsyevd_bufferSize = NULL
cdef void* __cusolverDnXsyevd = NULL
cdef void* __cusolverDnXsyevdx_bufferSize = NULL
cdef void* __cusolverDnXsyevdx = NULL
cdef void* __cusolverDnXgesvd_bufferSize = NULL
cdef void* __cusolverDnXgesvd = NULL
cdef void* __cusolverDnXgesvdp_bufferSize = NULL
cdef void* __cusolverDnXgesvdp = NULL
cdef void* __cusolverDnXgesvdr_bufferSize = NULL
cdef void* __cusolverDnXgesvdr = NULL
cdef void* __cusolverDnXsytrs_bufferSize = NULL
cdef void* __cusolverDnXsytrs = NULL
cdef void* __cusolverDnXtrtri_bufferSize = NULL
cdef void* __cusolverDnXtrtri = NULL
cdef void* __cusolverDnLoggerSetCallback = NULL
cdef void* __cusolverDnLoggerSetFile = NULL
cdef void* __cusolverDnLoggerOpenFile = NULL
cdef void* __cusolverDnLoggerSetLevel = NULL
cdef void* __cusolverDnLoggerSetMask = NULL
cdef void* __cusolverDnLoggerForceDisable = NULL
cdef void* __cusolverDnSetDeterministicMode = NULL
cdef void* __cusolverDnGetDeterministicMode = NULL
cdef void* __cusolverDnXlarft_bufferSize = NULL
cdef void* __cusolverDnXlarft = NULL
cdef void* __cusolverDnXsyevBatched_bufferSize = NULL
cdef void* __cusolverDnXsyevBatched = NULL
cdef void* __cusolverDnXgeev_bufferSize = NULL
cdef void* __cusolverDnXgeev = NULL


cdef void* load_library(const int driver_ver) except* with gil:
    cdef void* handle
    for suffix in get_cusolver_dso_version_suffix(driver_ver):
        so_name = "libcusolver.so" + (f".{suffix}" if suffix else suffix)
        handle = dlopen(so_name.encode(), RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            break
    else:
        err_msg = dlerror()
        raise RuntimeError(f'Failed to dlopen libcusolverDn ({err_msg.decode()})')
    return handle


cdef int _check_or_init_cusolverDn() except -1 nogil:
    global __py_cusolverDn_init
    if __py_cusolverDn_init:
        return 0

    # Load driver to check version
    cdef void* handle = NULL
    handle = dlopen('libcuda.so.1', RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise NotSupportedError(f'CUDA driver is not found ({err_msg.decode()})')
    global __cuDriverGetVersion
    if __cuDriverGetVersion == NULL:
        __cuDriverGetVersion = dlsym(handle, "cuDriverGetVersion")
    if __cuDriverGetVersion == NULL:
        with gil:
            raise RuntimeError('something went wrong')
    cdef int err, driver_ver
    err = (<int (*)(int*) noexcept nogil>__cuDriverGetVersion)(&driver_ver)
    if err != 0:
        with gil:
            raise RuntimeError('something went wrong')
    #dlclose(handle)
    handle = NULL

    # Load function
    global __cusolverDnCreate
    __cusolverDnCreate = dlsym(RTLD_DEFAULT, 'cusolverDnCreate')
    if __cusolverDnCreate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCreate = dlsym(handle, 'cusolverDnCreate')

    global __cusolverDnDestroy
    __cusolverDnDestroy = dlsym(RTLD_DEFAULT, 'cusolverDnDestroy')
    if __cusolverDnDestroy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDestroy = dlsym(handle, 'cusolverDnDestroy')

    global __cusolverDnSetStream
    __cusolverDnSetStream = dlsym(RTLD_DEFAULT, 'cusolverDnSetStream')
    if __cusolverDnSetStream == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSetStream = dlsym(handle, 'cusolverDnSetStream')

    global __cusolverDnGetStream
    __cusolverDnGetStream = dlsym(RTLD_DEFAULT, 'cusolverDnGetStream')
    if __cusolverDnGetStream == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnGetStream = dlsym(handle, 'cusolverDnGetStream')

    global __cusolverDnIRSParamsCreate
    __cusolverDnIRSParamsCreate = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsCreate')
    if __cusolverDnIRSParamsCreate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsCreate = dlsym(handle, 'cusolverDnIRSParamsCreate')

    global __cusolverDnIRSParamsDestroy
    __cusolverDnIRSParamsDestroy = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsDestroy')
    if __cusolverDnIRSParamsDestroy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsDestroy = dlsym(handle, 'cusolverDnIRSParamsDestroy')

    global __cusolverDnIRSParamsSetRefinementSolver
    __cusolverDnIRSParamsSetRefinementSolver = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsSetRefinementSolver')
    if __cusolverDnIRSParamsSetRefinementSolver == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsSetRefinementSolver = dlsym(handle, 'cusolverDnIRSParamsSetRefinementSolver')

    global __cusolverDnIRSParamsSetSolverMainPrecision
    __cusolverDnIRSParamsSetSolverMainPrecision = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsSetSolverMainPrecision')
    if __cusolverDnIRSParamsSetSolverMainPrecision == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsSetSolverMainPrecision = dlsym(handle, 'cusolverDnIRSParamsSetSolverMainPrecision')

    global __cusolverDnIRSParamsSetSolverLowestPrecision
    __cusolverDnIRSParamsSetSolverLowestPrecision = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsSetSolverLowestPrecision')
    if __cusolverDnIRSParamsSetSolverLowestPrecision == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsSetSolverLowestPrecision = dlsym(handle, 'cusolverDnIRSParamsSetSolverLowestPrecision')

    global __cusolverDnIRSParamsSetSolverPrecisions
    __cusolverDnIRSParamsSetSolverPrecisions = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsSetSolverPrecisions')
    if __cusolverDnIRSParamsSetSolverPrecisions == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsSetSolverPrecisions = dlsym(handle, 'cusolverDnIRSParamsSetSolverPrecisions')

    global __cusolverDnIRSParamsSetTol
    __cusolverDnIRSParamsSetTol = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsSetTol')
    if __cusolverDnIRSParamsSetTol == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsSetTol = dlsym(handle, 'cusolverDnIRSParamsSetTol')

    global __cusolverDnIRSParamsSetTolInner
    __cusolverDnIRSParamsSetTolInner = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsSetTolInner')
    if __cusolverDnIRSParamsSetTolInner == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsSetTolInner = dlsym(handle, 'cusolverDnIRSParamsSetTolInner')

    global __cusolverDnIRSParamsSetMaxIters
    __cusolverDnIRSParamsSetMaxIters = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsSetMaxIters')
    if __cusolverDnIRSParamsSetMaxIters == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsSetMaxIters = dlsym(handle, 'cusolverDnIRSParamsSetMaxIters')

    global __cusolverDnIRSParamsSetMaxItersInner
    __cusolverDnIRSParamsSetMaxItersInner = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsSetMaxItersInner')
    if __cusolverDnIRSParamsSetMaxItersInner == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsSetMaxItersInner = dlsym(handle, 'cusolverDnIRSParamsSetMaxItersInner')

    global __cusolverDnIRSParamsGetMaxIters
    __cusolverDnIRSParamsGetMaxIters = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsGetMaxIters')
    if __cusolverDnIRSParamsGetMaxIters == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsGetMaxIters = dlsym(handle, 'cusolverDnIRSParamsGetMaxIters')

    global __cusolverDnIRSParamsEnableFallback
    __cusolverDnIRSParamsEnableFallback = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsEnableFallback')
    if __cusolverDnIRSParamsEnableFallback == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsEnableFallback = dlsym(handle, 'cusolverDnIRSParamsEnableFallback')

    global __cusolverDnIRSParamsDisableFallback
    __cusolverDnIRSParamsDisableFallback = dlsym(RTLD_DEFAULT, 'cusolverDnIRSParamsDisableFallback')
    if __cusolverDnIRSParamsDisableFallback == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSParamsDisableFallback = dlsym(handle, 'cusolverDnIRSParamsDisableFallback')

    global __cusolverDnIRSInfosDestroy
    __cusolverDnIRSInfosDestroy = dlsym(RTLD_DEFAULT, 'cusolverDnIRSInfosDestroy')
    if __cusolverDnIRSInfosDestroy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSInfosDestroy = dlsym(handle, 'cusolverDnIRSInfosDestroy')

    global __cusolverDnIRSInfosCreate
    __cusolverDnIRSInfosCreate = dlsym(RTLD_DEFAULT, 'cusolverDnIRSInfosCreate')
    if __cusolverDnIRSInfosCreate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSInfosCreate = dlsym(handle, 'cusolverDnIRSInfosCreate')

    global __cusolverDnIRSInfosGetNiters
    __cusolverDnIRSInfosGetNiters = dlsym(RTLD_DEFAULT, 'cusolverDnIRSInfosGetNiters')
    if __cusolverDnIRSInfosGetNiters == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSInfosGetNiters = dlsym(handle, 'cusolverDnIRSInfosGetNiters')

    global __cusolverDnIRSInfosGetOuterNiters
    __cusolverDnIRSInfosGetOuterNiters = dlsym(RTLD_DEFAULT, 'cusolverDnIRSInfosGetOuterNiters')
    if __cusolverDnIRSInfosGetOuterNiters == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSInfosGetOuterNiters = dlsym(handle, 'cusolverDnIRSInfosGetOuterNiters')

    global __cusolverDnIRSInfosRequestResidual
    __cusolverDnIRSInfosRequestResidual = dlsym(RTLD_DEFAULT, 'cusolverDnIRSInfosRequestResidual')
    if __cusolverDnIRSInfosRequestResidual == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSInfosRequestResidual = dlsym(handle, 'cusolverDnIRSInfosRequestResidual')

    global __cusolverDnIRSInfosGetResidualHistory
    __cusolverDnIRSInfosGetResidualHistory = dlsym(RTLD_DEFAULT, 'cusolverDnIRSInfosGetResidualHistory')
    if __cusolverDnIRSInfosGetResidualHistory == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSInfosGetResidualHistory = dlsym(handle, 'cusolverDnIRSInfosGetResidualHistory')

    global __cusolverDnIRSInfosGetMaxIters
    __cusolverDnIRSInfosGetMaxIters = dlsym(RTLD_DEFAULT, 'cusolverDnIRSInfosGetMaxIters')
    if __cusolverDnIRSInfosGetMaxIters == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSInfosGetMaxIters = dlsym(handle, 'cusolverDnIRSInfosGetMaxIters')

    global __cusolverDnZZgesv
    __cusolverDnZZgesv = dlsym(RTLD_DEFAULT, 'cusolverDnZZgesv')
    if __cusolverDnZZgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZZgesv = dlsym(handle, 'cusolverDnZZgesv')

    global __cusolverDnZCgesv
    __cusolverDnZCgesv = dlsym(RTLD_DEFAULT, 'cusolverDnZCgesv')
    if __cusolverDnZCgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZCgesv = dlsym(handle, 'cusolverDnZCgesv')

    global __cusolverDnZKgesv
    __cusolverDnZKgesv = dlsym(RTLD_DEFAULT, 'cusolverDnZKgesv')
    if __cusolverDnZKgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZKgesv = dlsym(handle, 'cusolverDnZKgesv')

    global __cusolverDnZEgesv
    __cusolverDnZEgesv = dlsym(RTLD_DEFAULT, 'cusolverDnZEgesv')
    if __cusolverDnZEgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZEgesv = dlsym(handle, 'cusolverDnZEgesv')

    global __cusolverDnZYgesv
    __cusolverDnZYgesv = dlsym(RTLD_DEFAULT, 'cusolverDnZYgesv')
    if __cusolverDnZYgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZYgesv = dlsym(handle, 'cusolverDnZYgesv')

    global __cusolverDnCCgesv
    __cusolverDnCCgesv = dlsym(RTLD_DEFAULT, 'cusolverDnCCgesv')
    if __cusolverDnCCgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCCgesv = dlsym(handle, 'cusolverDnCCgesv')

    global __cusolverDnCEgesv
    __cusolverDnCEgesv = dlsym(RTLD_DEFAULT, 'cusolverDnCEgesv')
    if __cusolverDnCEgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCEgesv = dlsym(handle, 'cusolverDnCEgesv')

    global __cusolverDnCKgesv
    __cusolverDnCKgesv = dlsym(RTLD_DEFAULT, 'cusolverDnCKgesv')
    if __cusolverDnCKgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCKgesv = dlsym(handle, 'cusolverDnCKgesv')

    global __cusolverDnCYgesv
    __cusolverDnCYgesv = dlsym(RTLD_DEFAULT, 'cusolverDnCYgesv')
    if __cusolverDnCYgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCYgesv = dlsym(handle, 'cusolverDnCYgesv')

    global __cusolverDnDDgesv
    __cusolverDnDDgesv = dlsym(RTLD_DEFAULT, 'cusolverDnDDgesv')
    if __cusolverDnDDgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDDgesv = dlsym(handle, 'cusolverDnDDgesv')

    global __cusolverDnDSgesv
    __cusolverDnDSgesv = dlsym(RTLD_DEFAULT, 'cusolverDnDSgesv')
    if __cusolverDnDSgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDSgesv = dlsym(handle, 'cusolverDnDSgesv')

    global __cusolverDnDHgesv
    __cusolverDnDHgesv = dlsym(RTLD_DEFAULT, 'cusolverDnDHgesv')
    if __cusolverDnDHgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDHgesv = dlsym(handle, 'cusolverDnDHgesv')

    global __cusolverDnDBgesv
    __cusolverDnDBgesv = dlsym(RTLD_DEFAULT, 'cusolverDnDBgesv')
    if __cusolverDnDBgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDBgesv = dlsym(handle, 'cusolverDnDBgesv')

    global __cusolverDnDXgesv
    __cusolverDnDXgesv = dlsym(RTLD_DEFAULT, 'cusolverDnDXgesv')
    if __cusolverDnDXgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDXgesv = dlsym(handle, 'cusolverDnDXgesv')

    global __cusolverDnSSgesv
    __cusolverDnSSgesv = dlsym(RTLD_DEFAULT, 'cusolverDnSSgesv')
    if __cusolverDnSSgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSSgesv = dlsym(handle, 'cusolverDnSSgesv')

    global __cusolverDnSHgesv
    __cusolverDnSHgesv = dlsym(RTLD_DEFAULT, 'cusolverDnSHgesv')
    if __cusolverDnSHgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSHgesv = dlsym(handle, 'cusolverDnSHgesv')

    global __cusolverDnSBgesv
    __cusolverDnSBgesv = dlsym(RTLD_DEFAULT, 'cusolverDnSBgesv')
    if __cusolverDnSBgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSBgesv = dlsym(handle, 'cusolverDnSBgesv')

    global __cusolverDnSXgesv
    __cusolverDnSXgesv = dlsym(RTLD_DEFAULT, 'cusolverDnSXgesv')
    if __cusolverDnSXgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSXgesv = dlsym(handle, 'cusolverDnSXgesv')

    global __cusolverDnZZgesv_bufferSize
    __cusolverDnZZgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZZgesv_bufferSize')
    if __cusolverDnZZgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZZgesv_bufferSize = dlsym(handle, 'cusolverDnZZgesv_bufferSize')

    global __cusolverDnZCgesv_bufferSize
    __cusolverDnZCgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZCgesv_bufferSize')
    if __cusolverDnZCgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZCgesv_bufferSize = dlsym(handle, 'cusolverDnZCgesv_bufferSize')

    global __cusolverDnZKgesv_bufferSize
    __cusolverDnZKgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZKgesv_bufferSize')
    if __cusolverDnZKgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZKgesv_bufferSize = dlsym(handle, 'cusolverDnZKgesv_bufferSize')

    global __cusolverDnZEgesv_bufferSize
    __cusolverDnZEgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZEgesv_bufferSize')
    if __cusolverDnZEgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZEgesv_bufferSize = dlsym(handle, 'cusolverDnZEgesv_bufferSize')

    global __cusolverDnZYgesv_bufferSize
    __cusolverDnZYgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZYgesv_bufferSize')
    if __cusolverDnZYgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZYgesv_bufferSize = dlsym(handle, 'cusolverDnZYgesv_bufferSize')

    global __cusolverDnCCgesv_bufferSize
    __cusolverDnCCgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCCgesv_bufferSize')
    if __cusolverDnCCgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCCgesv_bufferSize = dlsym(handle, 'cusolverDnCCgesv_bufferSize')

    global __cusolverDnCKgesv_bufferSize
    __cusolverDnCKgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCKgesv_bufferSize')
    if __cusolverDnCKgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCKgesv_bufferSize = dlsym(handle, 'cusolverDnCKgesv_bufferSize')

    global __cusolverDnCEgesv_bufferSize
    __cusolverDnCEgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCEgesv_bufferSize')
    if __cusolverDnCEgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCEgesv_bufferSize = dlsym(handle, 'cusolverDnCEgesv_bufferSize')

    global __cusolverDnCYgesv_bufferSize
    __cusolverDnCYgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCYgesv_bufferSize')
    if __cusolverDnCYgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCYgesv_bufferSize = dlsym(handle, 'cusolverDnCYgesv_bufferSize')

    global __cusolverDnDDgesv_bufferSize
    __cusolverDnDDgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDDgesv_bufferSize')
    if __cusolverDnDDgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDDgesv_bufferSize = dlsym(handle, 'cusolverDnDDgesv_bufferSize')

    global __cusolverDnDSgesv_bufferSize
    __cusolverDnDSgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDSgesv_bufferSize')
    if __cusolverDnDSgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDSgesv_bufferSize = dlsym(handle, 'cusolverDnDSgesv_bufferSize')

    global __cusolverDnDHgesv_bufferSize
    __cusolverDnDHgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDHgesv_bufferSize')
    if __cusolverDnDHgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDHgesv_bufferSize = dlsym(handle, 'cusolverDnDHgesv_bufferSize')

    global __cusolverDnDBgesv_bufferSize
    __cusolverDnDBgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDBgesv_bufferSize')
    if __cusolverDnDBgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDBgesv_bufferSize = dlsym(handle, 'cusolverDnDBgesv_bufferSize')

    global __cusolverDnDXgesv_bufferSize
    __cusolverDnDXgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDXgesv_bufferSize')
    if __cusolverDnDXgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDXgesv_bufferSize = dlsym(handle, 'cusolverDnDXgesv_bufferSize')

    global __cusolverDnSSgesv_bufferSize
    __cusolverDnSSgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSSgesv_bufferSize')
    if __cusolverDnSSgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSSgesv_bufferSize = dlsym(handle, 'cusolverDnSSgesv_bufferSize')

    global __cusolverDnSHgesv_bufferSize
    __cusolverDnSHgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSHgesv_bufferSize')
    if __cusolverDnSHgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSHgesv_bufferSize = dlsym(handle, 'cusolverDnSHgesv_bufferSize')

    global __cusolverDnSBgesv_bufferSize
    __cusolverDnSBgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSBgesv_bufferSize')
    if __cusolverDnSBgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSBgesv_bufferSize = dlsym(handle, 'cusolverDnSBgesv_bufferSize')

    global __cusolverDnSXgesv_bufferSize
    __cusolverDnSXgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSXgesv_bufferSize')
    if __cusolverDnSXgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSXgesv_bufferSize = dlsym(handle, 'cusolverDnSXgesv_bufferSize')

    global __cusolverDnZZgels
    __cusolverDnZZgels = dlsym(RTLD_DEFAULT, 'cusolverDnZZgels')
    if __cusolverDnZZgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZZgels = dlsym(handle, 'cusolverDnZZgels')

    global __cusolverDnZCgels
    __cusolverDnZCgels = dlsym(RTLD_DEFAULT, 'cusolverDnZCgels')
    if __cusolverDnZCgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZCgels = dlsym(handle, 'cusolverDnZCgels')

    global __cusolverDnZKgels
    __cusolverDnZKgels = dlsym(RTLD_DEFAULT, 'cusolverDnZKgels')
    if __cusolverDnZKgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZKgels = dlsym(handle, 'cusolverDnZKgels')

    global __cusolverDnZEgels
    __cusolverDnZEgels = dlsym(RTLD_DEFAULT, 'cusolverDnZEgels')
    if __cusolverDnZEgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZEgels = dlsym(handle, 'cusolverDnZEgels')

    global __cusolverDnZYgels
    __cusolverDnZYgels = dlsym(RTLD_DEFAULT, 'cusolverDnZYgels')
    if __cusolverDnZYgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZYgels = dlsym(handle, 'cusolverDnZYgels')

    global __cusolverDnCCgels
    __cusolverDnCCgels = dlsym(RTLD_DEFAULT, 'cusolverDnCCgels')
    if __cusolverDnCCgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCCgels = dlsym(handle, 'cusolverDnCCgels')

    global __cusolverDnCKgels
    __cusolverDnCKgels = dlsym(RTLD_DEFAULT, 'cusolverDnCKgels')
    if __cusolverDnCKgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCKgels = dlsym(handle, 'cusolverDnCKgels')

    global __cusolverDnCEgels
    __cusolverDnCEgels = dlsym(RTLD_DEFAULT, 'cusolverDnCEgels')
    if __cusolverDnCEgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCEgels = dlsym(handle, 'cusolverDnCEgels')

    global __cusolverDnCYgels
    __cusolverDnCYgels = dlsym(RTLD_DEFAULT, 'cusolverDnCYgels')
    if __cusolverDnCYgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCYgels = dlsym(handle, 'cusolverDnCYgels')

    global __cusolverDnDDgels
    __cusolverDnDDgels = dlsym(RTLD_DEFAULT, 'cusolverDnDDgels')
    if __cusolverDnDDgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDDgels = dlsym(handle, 'cusolverDnDDgels')

    global __cusolverDnDSgels
    __cusolverDnDSgels = dlsym(RTLD_DEFAULT, 'cusolverDnDSgels')
    if __cusolverDnDSgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDSgels = dlsym(handle, 'cusolverDnDSgels')

    global __cusolverDnDHgels
    __cusolverDnDHgels = dlsym(RTLD_DEFAULT, 'cusolverDnDHgels')
    if __cusolverDnDHgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDHgels = dlsym(handle, 'cusolverDnDHgels')

    global __cusolverDnDBgels
    __cusolverDnDBgels = dlsym(RTLD_DEFAULT, 'cusolverDnDBgels')
    if __cusolverDnDBgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDBgels = dlsym(handle, 'cusolverDnDBgels')

    global __cusolverDnDXgels
    __cusolverDnDXgels = dlsym(RTLD_DEFAULT, 'cusolverDnDXgels')
    if __cusolverDnDXgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDXgels = dlsym(handle, 'cusolverDnDXgels')

    global __cusolverDnSSgels
    __cusolverDnSSgels = dlsym(RTLD_DEFAULT, 'cusolverDnSSgels')
    if __cusolverDnSSgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSSgels = dlsym(handle, 'cusolverDnSSgels')

    global __cusolverDnSHgels
    __cusolverDnSHgels = dlsym(RTLD_DEFAULT, 'cusolverDnSHgels')
    if __cusolverDnSHgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSHgels = dlsym(handle, 'cusolverDnSHgels')

    global __cusolverDnSBgels
    __cusolverDnSBgels = dlsym(RTLD_DEFAULT, 'cusolverDnSBgels')
    if __cusolverDnSBgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSBgels = dlsym(handle, 'cusolverDnSBgels')

    global __cusolverDnSXgels
    __cusolverDnSXgels = dlsym(RTLD_DEFAULT, 'cusolverDnSXgels')
    if __cusolverDnSXgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSXgels = dlsym(handle, 'cusolverDnSXgels')

    global __cusolverDnZZgels_bufferSize
    __cusolverDnZZgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZZgels_bufferSize')
    if __cusolverDnZZgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZZgels_bufferSize = dlsym(handle, 'cusolverDnZZgels_bufferSize')

    global __cusolverDnZCgels_bufferSize
    __cusolverDnZCgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZCgels_bufferSize')
    if __cusolverDnZCgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZCgels_bufferSize = dlsym(handle, 'cusolverDnZCgels_bufferSize')

    global __cusolverDnZKgels_bufferSize
    __cusolverDnZKgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZKgels_bufferSize')
    if __cusolverDnZKgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZKgels_bufferSize = dlsym(handle, 'cusolverDnZKgels_bufferSize')

    global __cusolverDnZEgels_bufferSize
    __cusolverDnZEgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZEgels_bufferSize')
    if __cusolverDnZEgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZEgels_bufferSize = dlsym(handle, 'cusolverDnZEgels_bufferSize')

    global __cusolverDnZYgels_bufferSize
    __cusolverDnZYgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZYgels_bufferSize')
    if __cusolverDnZYgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZYgels_bufferSize = dlsym(handle, 'cusolverDnZYgels_bufferSize')

    global __cusolverDnCCgels_bufferSize
    __cusolverDnCCgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCCgels_bufferSize')
    if __cusolverDnCCgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCCgels_bufferSize = dlsym(handle, 'cusolverDnCCgels_bufferSize')

    global __cusolverDnCKgels_bufferSize
    __cusolverDnCKgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCKgels_bufferSize')
    if __cusolverDnCKgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCKgels_bufferSize = dlsym(handle, 'cusolverDnCKgels_bufferSize')

    global __cusolverDnCEgels_bufferSize
    __cusolverDnCEgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCEgels_bufferSize')
    if __cusolverDnCEgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCEgels_bufferSize = dlsym(handle, 'cusolverDnCEgels_bufferSize')

    global __cusolverDnCYgels_bufferSize
    __cusolverDnCYgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCYgels_bufferSize')
    if __cusolverDnCYgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCYgels_bufferSize = dlsym(handle, 'cusolverDnCYgels_bufferSize')

    global __cusolverDnDDgels_bufferSize
    __cusolverDnDDgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDDgels_bufferSize')
    if __cusolverDnDDgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDDgels_bufferSize = dlsym(handle, 'cusolverDnDDgels_bufferSize')

    global __cusolverDnDSgels_bufferSize
    __cusolverDnDSgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDSgels_bufferSize')
    if __cusolverDnDSgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDSgels_bufferSize = dlsym(handle, 'cusolverDnDSgels_bufferSize')

    global __cusolverDnDHgels_bufferSize
    __cusolverDnDHgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDHgels_bufferSize')
    if __cusolverDnDHgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDHgels_bufferSize = dlsym(handle, 'cusolverDnDHgels_bufferSize')

    global __cusolverDnDBgels_bufferSize
    __cusolverDnDBgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDBgels_bufferSize')
    if __cusolverDnDBgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDBgels_bufferSize = dlsym(handle, 'cusolverDnDBgels_bufferSize')

    global __cusolverDnDXgels_bufferSize
    __cusolverDnDXgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDXgels_bufferSize')
    if __cusolverDnDXgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDXgels_bufferSize = dlsym(handle, 'cusolverDnDXgels_bufferSize')

    global __cusolverDnSSgels_bufferSize
    __cusolverDnSSgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSSgels_bufferSize')
    if __cusolverDnSSgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSSgels_bufferSize = dlsym(handle, 'cusolverDnSSgels_bufferSize')

    global __cusolverDnSHgels_bufferSize
    __cusolverDnSHgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSHgels_bufferSize')
    if __cusolverDnSHgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSHgels_bufferSize = dlsym(handle, 'cusolverDnSHgels_bufferSize')

    global __cusolverDnSBgels_bufferSize
    __cusolverDnSBgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSBgels_bufferSize')
    if __cusolverDnSBgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSBgels_bufferSize = dlsym(handle, 'cusolverDnSBgels_bufferSize')

    global __cusolverDnSXgels_bufferSize
    __cusolverDnSXgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSXgels_bufferSize')
    if __cusolverDnSXgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSXgels_bufferSize = dlsym(handle, 'cusolverDnSXgels_bufferSize')

    global __cusolverDnIRSXgesv
    __cusolverDnIRSXgesv = dlsym(RTLD_DEFAULT, 'cusolverDnIRSXgesv')
    if __cusolverDnIRSXgesv == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSXgesv = dlsym(handle, 'cusolverDnIRSXgesv')

    global __cusolverDnIRSXgesv_bufferSize
    __cusolverDnIRSXgesv_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnIRSXgesv_bufferSize')
    if __cusolverDnIRSXgesv_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSXgesv_bufferSize = dlsym(handle, 'cusolverDnIRSXgesv_bufferSize')

    global __cusolverDnIRSXgels
    __cusolverDnIRSXgels = dlsym(RTLD_DEFAULT, 'cusolverDnIRSXgels')
    if __cusolverDnIRSXgels == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSXgels = dlsym(handle, 'cusolverDnIRSXgels')

    global __cusolverDnIRSXgels_bufferSize
    __cusolverDnIRSXgels_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnIRSXgels_bufferSize')
    if __cusolverDnIRSXgels_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnIRSXgels_bufferSize = dlsym(handle, 'cusolverDnIRSXgels_bufferSize')

    global __cusolverDnSpotrf_bufferSize
    __cusolverDnSpotrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSpotrf_bufferSize')
    if __cusolverDnSpotrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSpotrf_bufferSize = dlsym(handle, 'cusolverDnSpotrf_bufferSize')

    global __cusolverDnDpotrf_bufferSize
    __cusolverDnDpotrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDpotrf_bufferSize')
    if __cusolverDnDpotrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDpotrf_bufferSize = dlsym(handle, 'cusolverDnDpotrf_bufferSize')

    global __cusolverDnCpotrf_bufferSize
    __cusolverDnCpotrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCpotrf_bufferSize')
    if __cusolverDnCpotrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCpotrf_bufferSize = dlsym(handle, 'cusolverDnCpotrf_bufferSize')

    global __cusolverDnZpotrf_bufferSize
    __cusolverDnZpotrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZpotrf_bufferSize')
    if __cusolverDnZpotrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZpotrf_bufferSize = dlsym(handle, 'cusolverDnZpotrf_bufferSize')

    global __cusolverDnSpotrf
    __cusolverDnSpotrf = dlsym(RTLD_DEFAULT, 'cusolverDnSpotrf')
    if __cusolverDnSpotrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSpotrf = dlsym(handle, 'cusolverDnSpotrf')

    global __cusolverDnDpotrf
    __cusolverDnDpotrf = dlsym(RTLD_DEFAULT, 'cusolverDnDpotrf')
    if __cusolverDnDpotrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDpotrf = dlsym(handle, 'cusolverDnDpotrf')

    global __cusolverDnCpotrf
    __cusolverDnCpotrf = dlsym(RTLD_DEFAULT, 'cusolverDnCpotrf')
    if __cusolverDnCpotrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCpotrf = dlsym(handle, 'cusolverDnCpotrf')

    global __cusolverDnZpotrf
    __cusolverDnZpotrf = dlsym(RTLD_DEFAULT, 'cusolverDnZpotrf')
    if __cusolverDnZpotrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZpotrf = dlsym(handle, 'cusolverDnZpotrf')

    global __cusolverDnSpotrs
    __cusolverDnSpotrs = dlsym(RTLD_DEFAULT, 'cusolverDnSpotrs')
    if __cusolverDnSpotrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSpotrs = dlsym(handle, 'cusolverDnSpotrs')

    global __cusolverDnDpotrs
    __cusolverDnDpotrs = dlsym(RTLD_DEFAULT, 'cusolverDnDpotrs')
    if __cusolverDnDpotrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDpotrs = dlsym(handle, 'cusolverDnDpotrs')

    global __cusolverDnCpotrs
    __cusolverDnCpotrs = dlsym(RTLD_DEFAULT, 'cusolverDnCpotrs')
    if __cusolverDnCpotrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCpotrs = dlsym(handle, 'cusolverDnCpotrs')

    global __cusolverDnZpotrs
    __cusolverDnZpotrs = dlsym(RTLD_DEFAULT, 'cusolverDnZpotrs')
    if __cusolverDnZpotrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZpotrs = dlsym(handle, 'cusolverDnZpotrs')

    global __cusolverDnSpotrfBatched
    __cusolverDnSpotrfBatched = dlsym(RTLD_DEFAULT, 'cusolverDnSpotrfBatched')
    if __cusolverDnSpotrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSpotrfBatched = dlsym(handle, 'cusolverDnSpotrfBatched')

    global __cusolverDnDpotrfBatched
    __cusolverDnDpotrfBatched = dlsym(RTLD_DEFAULT, 'cusolverDnDpotrfBatched')
    if __cusolverDnDpotrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDpotrfBatched = dlsym(handle, 'cusolverDnDpotrfBatched')

    global __cusolverDnCpotrfBatched
    __cusolverDnCpotrfBatched = dlsym(RTLD_DEFAULT, 'cusolverDnCpotrfBatched')
    if __cusolverDnCpotrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCpotrfBatched = dlsym(handle, 'cusolverDnCpotrfBatched')

    global __cusolverDnZpotrfBatched
    __cusolverDnZpotrfBatched = dlsym(RTLD_DEFAULT, 'cusolverDnZpotrfBatched')
    if __cusolverDnZpotrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZpotrfBatched = dlsym(handle, 'cusolverDnZpotrfBatched')

    global __cusolverDnSpotrsBatched
    __cusolverDnSpotrsBatched = dlsym(RTLD_DEFAULT, 'cusolverDnSpotrsBatched')
    if __cusolverDnSpotrsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSpotrsBatched = dlsym(handle, 'cusolverDnSpotrsBatched')

    global __cusolverDnDpotrsBatched
    __cusolverDnDpotrsBatched = dlsym(RTLD_DEFAULT, 'cusolverDnDpotrsBatched')
    if __cusolverDnDpotrsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDpotrsBatched = dlsym(handle, 'cusolverDnDpotrsBatched')

    global __cusolverDnCpotrsBatched
    __cusolverDnCpotrsBatched = dlsym(RTLD_DEFAULT, 'cusolverDnCpotrsBatched')
    if __cusolverDnCpotrsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCpotrsBatched = dlsym(handle, 'cusolverDnCpotrsBatched')

    global __cusolverDnZpotrsBatched
    __cusolverDnZpotrsBatched = dlsym(RTLD_DEFAULT, 'cusolverDnZpotrsBatched')
    if __cusolverDnZpotrsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZpotrsBatched = dlsym(handle, 'cusolverDnZpotrsBatched')

    global __cusolverDnSpotri_bufferSize
    __cusolverDnSpotri_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSpotri_bufferSize')
    if __cusolverDnSpotri_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSpotri_bufferSize = dlsym(handle, 'cusolverDnSpotri_bufferSize')

    global __cusolverDnDpotri_bufferSize
    __cusolverDnDpotri_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDpotri_bufferSize')
    if __cusolverDnDpotri_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDpotri_bufferSize = dlsym(handle, 'cusolverDnDpotri_bufferSize')

    global __cusolverDnCpotri_bufferSize
    __cusolverDnCpotri_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCpotri_bufferSize')
    if __cusolverDnCpotri_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCpotri_bufferSize = dlsym(handle, 'cusolverDnCpotri_bufferSize')

    global __cusolverDnZpotri_bufferSize
    __cusolverDnZpotri_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZpotri_bufferSize')
    if __cusolverDnZpotri_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZpotri_bufferSize = dlsym(handle, 'cusolverDnZpotri_bufferSize')

    global __cusolverDnSpotri
    __cusolverDnSpotri = dlsym(RTLD_DEFAULT, 'cusolverDnSpotri')
    if __cusolverDnSpotri == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSpotri = dlsym(handle, 'cusolverDnSpotri')

    global __cusolverDnDpotri
    __cusolverDnDpotri = dlsym(RTLD_DEFAULT, 'cusolverDnDpotri')
    if __cusolverDnDpotri == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDpotri = dlsym(handle, 'cusolverDnDpotri')

    global __cusolverDnCpotri
    __cusolverDnCpotri = dlsym(RTLD_DEFAULT, 'cusolverDnCpotri')
    if __cusolverDnCpotri == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCpotri = dlsym(handle, 'cusolverDnCpotri')

    global __cusolverDnZpotri
    __cusolverDnZpotri = dlsym(RTLD_DEFAULT, 'cusolverDnZpotri')
    if __cusolverDnZpotri == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZpotri = dlsym(handle, 'cusolverDnZpotri')

    global __cusolverDnSlauum_bufferSize
    __cusolverDnSlauum_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSlauum_bufferSize')
    if __cusolverDnSlauum_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSlauum_bufferSize = dlsym(handle, 'cusolverDnSlauum_bufferSize')

    global __cusolverDnDlauum_bufferSize
    __cusolverDnDlauum_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDlauum_bufferSize')
    if __cusolverDnDlauum_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDlauum_bufferSize = dlsym(handle, 'cusolverDnDlauum_bufferSize')

    global __cusolverDnClauum_bufferSize
    __cusolverDnClauum_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnClauum_bufferSize')
    if __cusolverDnClauum_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnClauum_bufferSize = dlsym(handle, 'cusolverDnClauum_bufferSize')

    global __cusolverDnZlauum_bufferSize
    __cusolverDnZlauum_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZlauum_bufferSize')
    if __cusolverDnZlauum_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZlauum_bufferSize = dlsym(handle, 'cusolverDnZlauum_bufferSize')

    global __cusolverDnSlauum
    __cusolverDnSlauum = dlsym(RTLD_DEFAULT, 'cusolverDnSlauum')
    if __cusolverDnSlauum == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSlauum = dlsym(handle, 'cusolverDnSlauum')

    global __cusolverDnDlauum
    __cusolverDnDlauum = dlsym(RTLD_DEFAULT, 'cusolverDnDlauum')
    if __cusolverDnDlauum == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDlauum = dlsym(handle, 'cusolverDnDlauum')

    global __cusolverDnClauum
    __cusolverDnClauum = dlsym(RTLD_DEFAULT, 'cusolverDnClauum')
    if __cusolverDnClauum == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnClauum = dlsym(handle, 'cusolverDnClauum')

    global __cusolverDnZlauum
    __cusolverDnZlauum = dlsym(RTLD_DEFAULT, 'cusolverDnZlauum')
    if __cusolverDnZlauum == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZlauum = dlsym(handle, 'cusolverDnZlauum')

    global __cusolverDnSgetrf_bufferSize
    __cusolverDnSgetrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSgetrf_bufferSize')
    if __cusolverDnSgetrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgetrf_bufferSize = dlsym(handle, 'cusolverDnSgetrf_bufferSize')

    global __cusolverDnDgetrf_bufferSize
    __cusolverDnDgetrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDgetrf_bufferSize')
    if __cusolverDnDgetrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgetrf_bufferSize = dlsym(handle, 'cusolverDnDgetrf_bufferSize')

    global __cusolverDnCgetrf_bufferSize
    __cusolverDnCgetrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCgetrf_bufferSize')
    if __cusolverDnCgetrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgetrf_bufferSize = dlsym(handle, 'cusolverDnCgetrf_bufferSize')

    global __cusolverDnZgetrf_bufferSize
    __cusolverDnZgetrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZgetrf_bufferSize')
    if __cusolverDnZgetrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgetrf_bufferSize = dlsym(handle, 'cusolverDnZgetrf_bufferSize')

    global __cusolverDnSgetrf
    __cusolverDnSgetrf = dlsym(RTLD_DEFAULT, 'cusolverDnSgetrf')
    if __cusolverDnSgetrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgetrf = dlsym(handle, 'cusolverDnSgetrf')

    global __cusolverDnDgetrf
    __cusolverDnDgetrf = dlsym(RTLD_DEFAULT, 'cusolverDnDgetrf')
    if __cusolverDnDgetrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgetrf = dlsym(handle, 'cusolverDnDgetrf')

    global __cusolverDnCgetrf
    __cusolverDnCgetrf = dlsym(RTLD_DEFAULT, 'cusolverDnCgetrf')
    if __cusolverDnCgetrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgetrf = dlsym(handle, 'cusolverDnCgetrf')

    global __cusolverDnZgetrf
    __cusolverDnZgetrf = dlsym(RTLD_DEFAULT, 'cusolverDnZgetrf')
    if __cusolverDnZgetrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgetrf = dlsym(handle, 'cusolverDnZgetrf')

    global __cusolverDnSlaswp
    __cusolverDnSlaswp = dlsym(RTLD_DEFAULT, 'cusolverDnSlaswp')
    if __cusolverDnSlaswp == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSlaswp = dlsym(handle, 'cusolverDnSlaswp')

    global __cusolverDnDlaswp
    __cusolverDnDlaswp = dlsym(RTLD_DEFAULT, 'cusolverDnDlaswp')
    if __cusolverDnDlaswp == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDlaswp = dlsym(handle, 'cusolverDnDlaswp')

    global __cusolverDnClaswp
    __cusolverDnClaswp = dlsym(RTLD_DEFAULT, 'cusolverDnClaswp')
    if __cusolverDnClaswp == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnClaswp = dlsym(handle, 'cusolverDnClaswp')

    global __cusolverDnZlaswp
    __cusolverDnZlaswp = dlsym(RTLD_DEFAULT, 'cusolverDnZlaswp')
    if __cusolverDnZlaswp == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZlaswp = dlsym(handle, 'cusolverDnZlaswp')

    global __cusolverDnSgetrs
    __cusolverDnSgetrs = dlsym(RTLD_DEFAULT, 'cusolverDnSgetrs')
    if __cusolverDnSgetrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgetrs = dlsym(handle, 'cusolverDnSgetrs')

    global __cusolverDnDgetrs
    __cusolverDnDgetrs = dlsym(RTLD_DEFAULT, 'cusolverDnDgetrs')
    if __cusolverDnDgetrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgetrs = dlsym(handle, 'cusolverDnDgetrs')

    global __cusolverDnCgetrs
    __cusolverDnCgetrs = dlsym(RTLD_DEFAULT, 'cusolverDnCgetrs')
    if __cusolverDnCgetrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgetrs = dlsym(handle, 'cusolverDnCgetrs')

    global __cusolverDnZgetrs
    __cusolverDnZgetrs = dlsym(RTLD_DEFAULT, 'cusolverDnZgetrs')
    if __cusolverDnZgetrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgetrs = dlsym(handle, 'cusolverDnZgetrs')

    global __cusolverDnSgeqrf_bufferSize
    __cusolverDnSgeqrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSgeqrf_bufferSize')
    if __cusolverDnSgeqrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgeqrf_bufferSize = dlsym(handle, 'cusolverDnSgeqrf_bufferSize')

    global __cusolverDnDgeqrf_bufferSize
    __cusolverDnDgeqrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDgeqrf_bufferSize')
    if __cusolverDnDgeqrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgeqrf_bufferSize = dlsym(handle, 'cusolverDnDgeqrf_bufferSize')

    global __cusolverDnCgeqrf_bufferSize
    __cusolverDnCgeqrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCgeqrf_bufferSize')
    if __cusolverDnCgeqrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgeqrf_bufferSize = dlsym(handle, 'cusolverDnCgeqrf_bufferSize')

    global __cusolverDnZgeqrf_bufferSize
    __cusolverDnZgeqrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZgeqrf_bufferSize')
    if __cusolverDnZgeqrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgeqrf_bufferSize = dlsym(handle, 'cusolverDnZgeqrf_bufferSize')

    global __cusolverDnSgeqrf
    __cusolverDnSgeqrf = dlsym(RTLD_DEFAULT, 'cusolverDnSgeqrf')
    if __cusolverDnSgeqrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgeqrf = dlsym(handle, 'cusolverDnSgeqrf')

    global __cusolverDnDgeqrf
    __cusolverDnDgeqrf = dlsym(RTLD_DEFAULT, 'cusolverDnDgeqrf')
    if __cusolverDnDgeqrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgeqrf = dlsym(handle, 'cusolverDnDgeqrf')

    global __cusolverDnCgeqrf
    __cusolverDnCgeqrf = dlsym(RTLD_DEFAULT, 'cusolverDnCgeqrf')
    if __cusolverDnCgeqrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgeqrf = dlsym(handle, 'cusolverDnCgeqrf')

    global __cusolverDnZgeqrf
    __cusolverDnZgeqrf = dlsym(RTLD_DEFAULT, 'cusolverDnZgeqrf')
    if __cusolverDnZgeqrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgeqrf = dlsym(handle, 'cusolverDnZgeqrf')

    global __cusolverDnSorgqr_bufferSize
    __cusolverDnSorgqr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSorgqr_bufferSize')
    if __cusolverDnSorgqr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSorgqr_bufferSize = dlsym(handle, 'cusolverDnSorgqr_bufferSize')

    global __cusolverDnDorgqr_bufferSize
    __cusolverDnDorgqr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDorgqr_bufferSize')
    if __cusolverDnDorgqr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDorgqr_bufferSize = dlsym(handle, 'cusolverDnDorgqr_bufferSize')

    global __cusolverDnCungqr_bufferSize
    __cusolverDnCungqr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCungqr_bufferSize')
    if __cusolverDnCungqr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCungqr_bufferSize = dlsym(handle, 'cusolverDnCungqr_bufferSize')

    global __cusolverDnZungqr_bufferSize
    __cusolverDnZungqr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZungqr_bufferSize')
    if __cusolverDnZungqr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZungqr_bufferSize = dlsym(handle, 'cusolverDnZungqr_bufferSize')

    global __cusolverDnSorgqr
    __cusolverDnSorgqr = dlsym(RTLD_DEFAULT, 'cusolverDnSorgqr')
    if __cusolverDnSorgqr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSorgqr = dlsym(handle, 'cusolverDnSorgqr')

    global __cusolverDnDorgqr
    __cusolverDnDorgqr = dlsym(RTLD_DEFAULT, 'cusolverDnDorgqr')
    if __cusolverDnDorgqr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDorgqr = dlsym(handle, 'cusolverDnDorgqr')

    global __cusolverDnCungqr
    __cusolverDnCungqr = dlsym(RTLD_DEFAULT, 'cusolverDnCungqr')
    if __cusolverDnCungqr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCungqr = dlsym(handle, 'cusolverDnCungqr')

    global __cusolverDnZungqr
    __cusolverDnZungqr = dlsym(RTLD_DEFAULT, 'cusolverDnZungqr')
    if __cusolverDnZungqr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZungqr = dlsym(handle, 'cusolverDnZungqr')

    global __cusolverDnSormqr_bufferSize
    __cusolverDnSormqr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSormqr_bufferSize')
    if __cusolverDnSormqr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSormqr_bufferSize = dlsym(handle, 'cusolverDnSormqr_bufferSize')

    global __cusolverDnDormqr_bufferSize
    __cusolverDnDormqr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDormqr_bufferSize')
    if __cusolverDnDormqr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDormqr_bufferSize = dlsym(handle, 'cusolverDnDormqr_bufferSize')

    global __cusolverDnCunmqr_bufferSize
    __cusolverDnCunmqr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCunmqr_bufferSize')
    if __cusolverDnCunmqr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCunmqr_bufferSize = dlsym(handle, 'cusolverDnCunmqr_bufferSize')

    global __cusolverDnZunmqr_bufferSize
    __cusolverDnZunmqr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZunmqr_bufferSize')
    if __cusolverDnZunmqr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZunmqr_bufferSize = dlsym(handle, 'cusolverDnZunmqr_bufferSize')

    global __cusolverDnSormqr
    __cusolverDnSormqr = dlsym(RTLD_DEFAULT, 'cusolverDnSormqr')
    if __cusolverDnSormqr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSormqr = dlsym(handle, 'cusolverDnSormqr')

    global __cusolverDnDormqr
    __cusolverDnDormqr = dlsym(RTLD_DEFAULT, 'cusolverDnDormqr')
    if __cusolverDnDormqr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDormqr = dlsym(handle, 'cusolverDnDormqr')

    global __cusolverDnCunmqr
    __cusolverDnCunmqr = dlsym(RTLD_DEFAULT, 'cusolverDnCunmqr')
    if __cusolverDnCunmqr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCunmqr = dlsym(handle, 'cusolverDnCunmqr')

    global __cusolverDnZunmqr
    __cusolverDnZunmqr = dlsym(RTLD_DEFAULT, 'cusolverDnZunmqr')
    if __cusolverDnZunmqr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZunmqr = dlsym(handle, 'cusolverDnZunmqr')

    global __cusolverDnSsytrf_bufferSize
    __cusolverDnSsytrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSsytrf_bufferSize')
    if __cusolverDnSsytrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsytrf_bufferSize = dlsym(handle, 'cusolverDnSsytrf_bufferSize')

    global __cusolverDnDsytrf_bufferSize
    __cusolverDnDsytrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDsytrf_bufferSize')
    if __cusolverDnDsytrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsytrf_bufferSize = dlsym(handle, 'cusolverDnDsytrf_bufferSize')

    global __cusolverDnCsytrf_bufferSize
    __cusolverDnCsytrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCsytrf_bufferSize')
    if __cusolverDnCsytrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCsytrf_bufferSize = dlsym(handle, 'cusolverDnCsytrf_bufferSize')

    global __cusolverDnZsytrf_bufferSize
    __cusolverDnZsytrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZsytrf_bufferSize')
    if __cusolverDnZsytrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZsytrf_bufferSize = dlsym(handle, 'cusolverDnZsytrf_bufferSize')

    global __cusolverDnSsytrf
    __cusolverDnSsytrf = dlsym(RTLD_DEFAULT, 'cusolverDnSsytrf')
    if __cusolverDnSsytrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsytrf = dlsym(handle, 'cusolverDnSsytrf')

    global __cusolverDnDsytrf
    __cusolverDnDsytrf = dlsym(RTLD_DEFAULT, 'cusolverDnDsytrf')
    if __cusolverDnDsytrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsytrf = dlsym(handle, 'cusolverDnDsytrf')

    global __cusolverDnCsytrf
    __cusolverDnCsytrf = dlsym(RTLD_DEFAULT, 'cusolverDnCsytrf')
    if __cusolverDnCsytrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCsytrf = dlsym(handle, 'cusolverDnCsytrf')

    global __cusolverDnZsytrf
    __cusolverDnZsytrf = dlsym(RTLD_DEFAULT, 'cusolverDnZsytrf')
    if __cusolverDnZsytrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZsytrf = dlsym(handle, 'cusolverDnZsytrf')

    global __cusolverDnSsytri_bufferSize
    __cusolverDnSsytri_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSsytri_bufferSize')
    if __cusolverDnSsytri_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsytri_bufferSize = dlsym(handle, 'cusolverDnSsytri_bufferSize')

    global __cusolverDnDsytri_bufferSize
    __cusolverDnDsytri_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDsytri_bufferSize')
    if __cusolverDnDsytri_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsytri_bufferSize = dlsym(handle, 'cusolverDnDsytri_bufferSize')

    global __cusolverDnCsytri_bufferSize
    __cusolverDnCsytri_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCsytri_bufferSize')
    if __cusolverDnCsytri_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCsytri_bufferSize = dlsym(handle, 'cusolverDnCsytri_bufferSize')

    global __cusolverDnZsytri_bufferSize
    __cusolverDnZsytri_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZsytri_bufferSize')
    if __cusolverDnZsytri_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZsytri_bufferSize = dlsym(handle, 'cusolverDnZsytri_bufferSize')

    global __cusolverDnSsytri
    __cusolverDnSsytri = dlsym(RTLD_DEFAULT, 'cusolverDnSsytri')
    if __cusolverDnSsytri == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsytri = dlsym(handle, 'cusolverDnSsytri')

    global __cusolverDnDsytri
    __cusolverDnDsytri = dlsym(RTLD_DEFAULT, 'cusolverDnDsytri')
    if __cusolverDnDsytri == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsytri = dlsym(handle, 'cusolverDnDsytri')

    global __cusolverDnCsytri
    __cusolverDnCsytri = dlsym(RTLD_DEFAULT, 'cusolverDnCsytri')
    if __cusolverDnCsytri == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCsytri = dlsym(handle, 'cusolverDnCsytri')

    global __cusolverDnZsytri
    __cusolverDnZsytri = dlsym(RTLD_DEFAULT, 'cusolverDnZsytri')
    if __cusolverDnZsytri == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZsytri = dlsym(handle, 'cusolverDnZsytri')

    global __cusolverDnSgebrd_bufferSize
    __cusolverDnSgebrd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSgebrd_bufferSize')
    if __cusolverDnSgebrd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgebrd_bufferSize = dlsym(handle, 'cusolverDnSgebrd_bufferSize')

    global __cusolverDnDgebrd_bufferSize
    __cusolverDnDgebrd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDgebrd_bufferSize')
    if __cusolverDnDgebrd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgebrd_bufferSize = dlsym(handle, 'cusolverDnDgebrd_bufferSize')

    global __cusolverDnCgebrd_bufferSize
    __cusolverDnCgebrd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCgebrd_bufferSize')
    if __cusolverDnCgebrd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgebrd_bufferSize = dlsym(handle, 'cusolverDnCgebrd_bufferSize')

    global __cusolverDnZgebrd_bufferSize
    __cusolverDnZgebrd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZgebrd_bufferSize')
    if __cusolverDnZgebrd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgebrd_bufferSize = dlsym(handle, 'cusolverDnZgebrd_bufferSize')

    global __cusolverDnSgebrd
    __cusolverDnSgebrd = dlsym(RTLD_DEFAULT, 'cusolverDnSgebrd')
    if __cusolverDnSgebrd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgebrd = dlsym(handle, 'cusolverDnSgebrd')

    global __cusolverDnDgebrd
    __cusolverDnDgebrd = dlsym(RTLD_DEFAULT, 'cusolverDnDgebrd')
    if __cusolverDnDgebrd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgebrd = dlsym(handle, 'cusolverDnDgebrd')

    global __cusolverDnCgebrd
    __cusolverDnCgebrd = dlsym(RTLD_DEFAULT, 'cusolverDnCgebrd')
    if __cusolverDnCgebrd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgebrd = dlsym(handle, 'cusolverDnCgebrd')

    global __cusolverDnZgebrd
    __cusolverDnZgebrd = dlsym(RTLD_DEFAULT, 'cusolverDnZgebrd')
    if __cusolverDnZgebrd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgebrd = dlsym(handle, 'cusolverDnZgebrd')

    global __cusolverDnSorgbr_bufferSize
    __cusolverDnSorgbr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSorgbr_bufferSize')
    if __cusolverDnSorgbr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSorgbr_bufferSize = dlsym(handle, 'cusolverDnSorgbr_bufferSize')

    global __cusolverDnDorgbr_bufferSize
    __cusolverDnDorgbr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDorgbr_bufferSize')
    if __cusolverDnDorgbr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDorgbr_bufferSize = dlsym(handle, 'cusolverDnDorgbr_bufferSize')

    global __cusolverDnCungbr_bufferSize
    __cusolverDnCungbr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCungbr_bufferSize')
    if __cusolverDnCungbr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCungbr_bufferSize = dlsym(handle, 'cusolverDnCungbr_bufferSize')

    global __cusolverDnZungbr_bufferSize
    __cusolverDnZungbr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZungbr_bufferSize')
    if __cusolverDnZungbr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZungbr_bufferSize = dlsym(handle, 'cusolverDnZungbr_bufferSize')

    global __cusolverDnSorgbr
    __cusolverDnSorgbr = dlsym(RTLD_DEFAULT, 'cusolverDnSorgbr')
    if __cusolverDnSorgbr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSorgbr = dlsym(handle, 'cusolverDnSorgbr')

    global __cusolverDnDorgbr
    __cusolverDnDorgbr = dlsym(RTLD_DEFAULT, 'cusolverDnDorgbr')
    if __cusolverDnDorgbr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDorgbr = dlsym(handle, 'cusolverDnDorgbr')

    global __cusolverDnCungbr
    __cusolverDnCungbr = dlsym(RTLD_DEFAULT, 'cusolverDnCungbr')
    if __cusolverDnCungbr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCungbr = dlsym(handle, 'cusolverDnCungbr')

    global __cusolverDnZungbr
    __cusolverDnZungbr = dlsym(RTLD_DEFAULT, 'cusolverDnZungbr')
    if __cusolverDnZungbr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZungbr = dlsym(handle, 'cusolverDnZungbr')

    global __cusolverDnSsytrd_bufferSize
    __cusolverDnSsytrd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSsytrd_bufferSize')
    if __cusolverDnSsytrd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsytrd_bufferSize = dlsym(handle, 'cusolverDnSsytrd_bufferSize')

    global __cusolverDnDsytrd_bufferSize
    __cusolverDnDsytrd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDsytrd_bufferSize')
    if __cusolverDnDsytrd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsytrd_bufferSize = dlsym(handle, 'cusolverDnDsytrd_bufferSize')

    global __cusolverDnChetrd_bufferSize
    __cusolverDnChetrd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnChetrd_bufferSize')
    if __cusolverDnChetrd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnChetrd_bufferSize = dlsym(handle, 'cusolverDnChetrd_bufferSize')

    global __cusolverDnZhetrd_bufferSize
    __cusolverDnZhetrd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZhetrd_bufferSize')
    if __cusolverDnZhetrd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZhetrd_bufferSize = dlsym(handle, 'cusolverDnZhetrd_bufferSize')

    global __cusolverDnSsytrd
    __cusolverDnSsytrd = dlsym(RTLD_DEFAULT, 'cusolverDnSsytrd')
    if __cusolverDnSsytrd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsytrd = dlsym(handle, 'cusolverDnSsytrd')

    global __cusolverDnDsytrd
    __cusolverDnDsytrd = dlsym(RTLD_DEFAULT, 'cusolverDnDsytrd')
    if __cusolverDnDsytrd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsytrd = dlsym(handle, 'cusolverDnDsytrd')

    global __cusolverDnChetrd
    __cusolverDnChetrd = dlsym(RTLD_DEFAULT, 'cusolverDnChetrd')
    if __cusolverDnChetrd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnChetrd = dlsym(handle, 'cusolverDnChetrd')

    global __cusolverDnZhetrd
    __cusolverDnZhetrd = dlsym(RTLD_DEFAULT, 'cusolverDnZhetrd')
    if __cusolverDnZhetrd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZhetrd = dlsym(handle, 'cusolverDnZhetrd')

    global __cusolverDnSorgtr_bufferSize
    __cusolverDnSorgtr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSorgtr_bufferSize')
    if __cusolverDnSorgtr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSorgtr_bufferSize = dlsym(handle, 'cusolverDnSorgtr_bufferSize')

    global __cusolverDnDorgtr_bufferSize
    __cusolverDnDorgtr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDorgtr_bufferSize')
    if __cusolverDnDorgtr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDorgtr_bufferSize = dlsym(handle, 'cusolverDnDorgtr_bufferSize')

    global __cusolverDnCungtr_bufferSize
    __cusolverDnCungtr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCungtr_bufferSize')
    if __cusolverDnCungtr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCungtr_bufferSize = dlsym(handle, 'cusolverDnCungtr_bufferSize')

    global __cusolverDnZungtr_bufferSize
    __cusolverDnZungtr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZungtr_bufferSize')
    if __cusolverDnZungtr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZungtr_bufferSize = dlsym(handle, 'cusolverDnZungtr_bufferSize')

    global __cusolverDnSorgtr
    __cusolverDnSorgtr = dlsym(RTLD_DEFAULT, 'cusolverDnSorgtr')
    if __cusolverDnSorgtr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSorgtr = dlsym(handle, 'cusolverDnSorgtr')

    global __cusolverDnDorgtr
    __cusolverDnDorgtr = dlsym(RTLD_DEFAULT, 'cusolverDnDorgtr')
    if __cusolverDnDorgtr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDorgtr = dlsym(handle, 'cusolverDnDorgtr')

    global __cusolverDnCungtr
    __cusolverDnCungtr = dlsym(RTLD_DEFAULT, 'cusolverDnCungtr')
    if __cusolverDnCungtr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCungtr = dlsym(handle, 'cusolverDnCungtr')

    global __cusolverDnZungtr
    __cusolverDnZungtr = dlsym(RTLD_DEFAULT, 'cusolverDnZungtr')
    if __cusolverDnZungtr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZungtr = dlsym(handle, 'cusolverDnZungtr')

    global __cusolverDnSormtr_bufferSize
    __cusolverDnSormtr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSormtr_bufferSize')
    if __cusolverDnSormtr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSormtr_bufferSize = dlsym(handle, 'cusolverDnSormtr_bufferSize')

    global __cusolverDnDormtr_bufferSize
    __cusolverDnDormtr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDormtr_bufferSize')
    if __cusolverDnDormtr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDormtr_bufferSize = dlsym(handle, 'cusolverDnDormtr_bufferSize')

    global __cusolverDnCunmtr_bufferSize
    __cusolverDnCunmtr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCunmtr_bufferSize')
    if __cusolverDnCunmtr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCunmtr_bufferSize = dlsym(handle, 'cusolverDnCunmtr_bufferSize')

    global __cusolverDnZunmtr_bufferSize
    __cusolverDnZunmtr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZunmtr_bufferSize')
    if __cusolverDnZunmtr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZunmtr_bufferSize = dlsym(handle, 'cusolverDnZunmtr_bufferSize')

    global __cusolverDnSormtr
    __cusolverDnSormtr = dlsym(RTLD_DEFAULT, 'cusolverDnSormtr')
    if __cusolverDnSormtr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSormtr = dlsym(handle, 'cusolverDnSormtr')

    global __cusolverDnDormtr
    __cusolverDnDormtr = dlsym(RTLD_DEFAULT, 'cusolverDnDormtr')
    if __cusolverDnDormtr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDormtr = dlsym(handle, 'cusolverDnDormtr')

    global __cusolverDnCunmtr
    __cusolverDnCunmtr = dlsym(RTLD_DEFAULT, 'cusolverDnCunmtr')
    if __cusolverDnCunmtr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCunmtr = dlsym(handle, 'cusolverDnCunmtr')

    global __cusolverDnZunmtr
    __cusolverDnZunmtr = dlsym(RTLD_DEFAULT, 'cusolverDnZunmtr')
    if __cusolverDnZunmtr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZunmtr = dlsym(handle, 'cusolverDnZunmtr')

    global __cusolverDnSgesvd_bufferSize
    __cusolverDnSgesvd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSgesvd_bufferSize')
    if __cusolverDnSgesvd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgesvd_bufferSize = dlsym(handle, 'cusolverDnSgesvd_bufferSize')

    global __cusolverDnDgesvd_bufferSize
    __cusolverDnDgesvd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDgesvd_bufferSize')
    if __cusolverDnDgesvd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgesvd_bufferSize = dlsym(handle, 'cusolverDnDgesvd_bufferSize')

    global __cusolverDnCgesvd_bufferSize
    __cusolverDnCgesvd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCgesvd_bufferSize')
    if __cusolverDnCgesvd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgesvd_bufferSize = dlsym(handle, 'cusolverDnCgesvd_bufferSize')

    global __cusolverDnZgesvd_bufferSize
    __cusolverDnZgesvd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZgesvd_bufferSize')
    if __cusolverDnZgesvd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgesvd_bufferSize = dlsym(handle, 'cusolverDnZgesvd_bufferSize')

    global __cusolverDnSgesvd
    __cusolverDnSgesvd = dlsym(RTLD_DEFAULT, 'cusolverDnSgesvd')
    if __cusolverDnSgesvd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgesvd = dlsym(handle, 'cusolverDnSgesvd')

    global __cusolverDnDgesvd
    __cusolverDnDgesvd = dlsym(RTLD_DEFAULT, 'cusolverDnDgesvd')
    if __cusolverDnDgesvd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgesvd = dlsym(handle, 'cusolverDnDgesvd')

    global __cusolverDnCgesvd
    __cusolverDnCgesvd = dlsym(RTLD_DEFAULT, 'cusolverDnCgesvd')
    if __cusolverDnCgesvd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgesvd = dlsym(handle, 'cusolverDnCgesvd')

    global __cusolverDnZgesvd
    __cusolverDnZgesvd = dlsym(RTLD_DEFAULT, 'cusolverDnZgesvd')
    if __cusolverDnZgesvd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgesvd = dlsym(handle, 'cusolverDnZgesvd')

    global __cusolverDnSsyevd_bufferSize
    __cusolverDnSsyevd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSsyevd_bufferSize')
    if __cusolverDnSsyevd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsyevd_bufferSize = dlsym(handle, 'cusolverDnSsyevd_bufferSize')

    global __cusolverDnDsyevd_bufferSize
    __cusolverDnDsyevd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDsyevd_bufferSize')
    if __cusolverDnDsyevd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsyevd_bufferSize = dlsym(handle, 'cusolverDnDsyevd_bufferSize')

    global __cusolverDnCheevd_bufferSize
    __cusolverDnCheevd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCheevd_bufferSize')
    if __cusolverDnCheevd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCheevd_bufferSize = dlsym(handle, 'cusolverDnCheevd_bufferSize')

    global __cusolverDnZheevd_bufferSize
    __cusolverDnZheevd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZheevd_bufferSize')
    if __cusolverDnZheevd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZheevd_bufferSize = dlsym(handle, 'cusolverDnZheevd_bufferSize')

    global __cusolverDnSsyevd
    __cusolverDnSsyevd = dlsym(RTLD_DEFAULT, 'cusolverDnSsyevd')
    if __cusolverDnSsyevd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsyevd = dlsym(handle, 'cusolverDnSsyevd')

    global __cusolverDnDsyevd
    __cusolverDnDsyevd = dlsym(RTLD_DEFAULT, 'cusolverDnDsyevd')
    if __cusolverDnDsyevd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsyevd = dlsym(handle, 'cusolverDnDsyevd')

    global __cusolverDnCheevd
    __cusolverDnCheevd = dlsym(RTLD_DEFAULT, 'cusolverDnCheevd')
    if __cusolverDnCheevd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCheevd = dlsym(handle, 'cusolverDnCheevd')

    global __cusolverDnZheevd
    __cusolverDnZheevd = dlsym(RTLD_DEFAULT, 'cusolverDnZheevd')
    if __cusolverDnZheevd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZheevd = dlsym(handle, 'cusolverDnZheevd')

    global __cusolverDnSsyevdx_bufferSize
    __cusolverDnSsyevdx_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSsyevdx_bufferSize')
    if __cusolverDnSsyevdx_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsyevdx_bufferSize = dlsym(handle, 'cusolverDnSsyevdx_bufferSize')

    global __cusolverDnDsyevdx_bufferSize
    __cusolverDnDsyevdx_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDsyevdx_bufferSize')
    if __cusolverDnDsyevdx_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsyevdx_bufferSize = dlsym(handle, 'cusolverDnDsyevdx_bufferSize')

    global __cusolverDnCheevdx_bufferSize
    __cusolverDnCheevdx_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCheevdx_bufferSize')
    if __cusolverDnCheevdx_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCheevdx_bufferSize = dlsym(handle, 'cusolverDnCheevdx_bufferSize')

    global __cusolverDnZheevdx_bufferSize
    __cusolverDnZheevdx_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZheevdx_bufferSize')
    if __cusolverDnZheevdx_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZheevdx_bufferSize = dlsym(handle, 'cusolverDnZheevdx_bufferSize')

    global __cusolverDnSsyevdx
    __cusolverDnSsyevdx = dlsym(RTLD_DEFAULT, 'cusolverDnSsyevdx')
    if __cusolverDnSsyevdx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsyevdx = dlsym(handle, 'cusolverDnSsyevdx')

    global __cusolverDnDsyevdx
    __cusolverDnDsyevdx = dlsym(RTLD_DEFAULT, 'cusolverDnDsyevdx')
    if __cusolverDnDsyevdx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsyevdx = dlsym(handle, 'cusolverDnDsyevdx')

    global __cusolverDnCheevdx
    __cusolverDnCheevdx = dlsym(RTLD_DEFAULT, 'cusolverDnCheevdx')
    if __cusolverDnCheevdx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCheevdx = dlsym(handle, 'cusolverDnCheevdx')

    global __cusolverDnZheevdx
    __cusolverDnZheevdx = dlsym(RTLD_DEFAULT, 'cusolverDnZheevdx')
    if __cusolverDnZheevdx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZheevdx = dlsym(handle, 'cusolverDnZheevdx')

    global __cusolverDnSsygvdx_bufferSize
    __cusolverDnSsygvdx_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSsygvdx_bufferSize')
    if __cusolverDnSsygvdx_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsygvdx_bufferSize = dlsym(handle, 'cusolverDnSsygvdx_bufferSize')

    global __cusolverDnDsygvdx_bufferSize
    __cusolverDnDsygvdx_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDsygvdx_bufferSize')
    if __cusolverDnDsygvdx_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsygvdx_bufferSize = dlsym(handle, 'cusolverDnDsygvdx_bufferSize')

    global __cusolverDnChegvdx_bufferSize
    __cusolverDnChegvdx_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnChegvdx_bufferSize')
    if __cusolverDnChegvdx_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnChegvdx_bufferSize = dlsym(handle, 'cusolverDnChegvdx_bufferSize')

    global __cusolverDnZhegvdx_bufferSize
    __cusolverDnZhegvdx_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZhegvdx_bufferSize')
    if __cusolverDnZhegvdx_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZhegvdx_bufferSize = dlsym(handle, 'cusolverDnZhegvdx_bufferSize')

    global __cusolverDnSsygvdx
    __cusolverDnSsygvdx = dlsym(RTLD_DEFAULT, 'cusolverDnSsygvdx')
    if __cusolverDnSsygvdx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsygvdx = dlsym(handle, 'cusolverDnSsygvdx')

    global __cusolverDnDsygvdx
    __cusolverDnDsygvdx = dlsym(RTLD_DEFAULT, 'cusolverDnDsygvdx')
    if __cusolverDnDsygvdx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsygvdx = dlsym(handle, 'cusolverDnDsygvdx')

    global __cusolverDnChegvdx
    __cusolverDnChegvdx = dlsym(RTLD_DEFAULT, 'cusolverDnChegvdx')
    if __cusolverDnChegvdx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnChegvdx = dlsym(handle, 'cusolverDnChegvdx')

    global __cusolverDnZhegvdx
    __cusolverDnZhegvdx = dlsym(RTLD_DEFAULT, 'cusolverDnZhegvdx')
    if __cusolverDnZhegvdx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZhegvdx = dlsym(handle, 'cusolverDnZhegvdx')

    global __cusolverDnSsygvd_bufferSize
    __cusolverDnSsygvd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSsygvd_bufferSize')
    if __cusolverDnSsygvd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsygvd_bufferSize = dlsym(handle, 'cusolverDnSsygvd_bufferSize')

    global __cusolverDnDsygvd_bufferSize
    __cusolverDnDsygvd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDsygvd_bufferSize')
    if __cusolverDnDsygvd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsygvd_bufferSize = dlsym(handle, 'cusolverDnDsygvd_bufferSize')

    global __cusolverDnChegvd_bufferSize
    __cusolverDnChegvd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnChegvd_bufferSize')
    if __cusolverDnChegvd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnChegvd_bufferSize = dlsym(handle, 'cusolverDnChegvd_bufferSize')

    global __cusolverDnZhegvd_bufferSize
    __cusolverDnZhegvd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZhegvd_bufferSize')
    if __cusolverDnZhegvd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZhegvd_bufferSize = dlsym(handle, 'cusolverDnZhegvd_bufferSize')

    global __cusolverDnSsygvd
    __cusolverDnSsygvd = dlsym(RTLD_DEFAULT, 'cusolverDnSsygvd')
    if __cusolverDnSsygvd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsygvd = dlsym(handle, 'cusolverDnSsygvd')

    global __cusolverDnDsygvd
    __cusolverDnDsygvd = dlsym(RTLD_DEFAULT, 'cusolverDnDsygvd')
    if __cusolverDnDsygvd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsygvd = dlsym(handle, 'cusolverDnDsygvd')

    global __cusolverDnChegvd
    __cusolverDnChegvd = dlsym(RTLD_DEFAULT, 'cusolverDnChegvd')
    if __cusolverDnChegvd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnChegvd = dlsym(handle, 'cusolverDnChegvd')

    global __cusolverDnZhegvd
    __cusolverDnZhegvd = dlsym(RTLD_DEFAULT, 'cusolverDnZhegvd')
    if __cusolverDnZhegvd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZhegvd = dlsym(handle, 'cusolverDnZhegvd')

    global __cusolverDnCreateSyevjInfo
    __cusolverDnCreateSyevjInfo = dlsym(RTLD_DEFAULT, 'cusolverDnCreateSyevjInfo')
    if __cusolverDnCreateSyevjInfo == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCreateSyevjInfo = dlsym(handle, 'cusolverDnCreateSyevjInfo')

    global __cusolverDnDestroySyevjInfo
    __cusolverDnDestroySyevjInfo = dlsym(RTLD_DEFAULT, 'cusolverDnDestroySyevjInfo')
    if __cusolverDnDestroySyevjInfo == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDestroySyevjInfo = dlsym(handle, 'cusolverDnDestroySyevjInfo')

    global __cusolverDnXsyevjSetTolerance
    __cusolverDnXsyevjSetTolerance = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevjSetTolerance')
    if __cusolverDnXsyevjSetTolerance == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevjSetTolerance = dlsym(handle, 'cusolverDnXsyevjSetTolerance')

    global __cusolverDnXsyevjSetMaxSweeps
    __cusolverDnXsyevjSetMaxSweeps = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevjSetMaxSweeps')
    if __cusolverDnXsyevjSetMaxSweeps == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevjSetMaxSweeps = dlsym(handle, 'cusolverDnXsyevjSetMaxSweeps')

    global __cusolverDnXsyevjSetSortEig
    __cusolverDnXsyevjSetSortEig = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevjSetSortEig')
    if __cusolverDnXsyevjSetSortEig == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevjSetSortEig = dlsym(handle, 'cusolverDnXsyevjSetSortEig')

    global __cusolverDnXsyevjGetResidual
    __cusolverDnXsyevjGetResidual = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevjGetResidual')
    if __cusolverDnXsyevjGetResidual == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevjGetResidual = dlsym(handle, 'cusolverDnXsyevjGetResidual')

    global __cusolverDnXsyevjGetSweeps
    __cusolverDnXsyevjGetSweeps = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevjGetSweeps')
    if __cusolverDnXsyevjGetSweeps == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevjGetSweeps = dlsym(handle, 'cusolverDnXsyevjGetSweeps')

    global __cusolverDnSsyevjBatched_bufferSize
    __cusolverDnSsyevjBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSsyevjBatched_bufferSize')
    if __cusolverDnSsyevjBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsyevjBatched_bufferSize = dlsym(handle, 'cusolverDnSsyevjBatched_bufferSize')

    global __cusolverDnDsyevjBatched_bufferSize
    __cusolverDnDsyevjBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDsyevjBatched_bufferSize')
    if __cusolverDnDsyevjBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsyevjBatched_bufferSize = dlsym(handle, 'cusolverDnDsyevjBatched_bufferSize')

    global __cusolverDnCheevjBatched_bufferSize
    __cusolverDnCheevjBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCheevjBatched_bufferSize')
    if __cusolverDnCheevjBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCheevjBatched_bufferSize = dlsym(handle, 'cusolverDnCheevjBatched_bufferSize')

    global __cusolverDnZheevjBatched_bufferSize
    __cusolverDnZheevjBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZheevjBatched_bufferSize')
    if __cusolverDnZheevjBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZheevjBatched_bufferSize = dlsym(handle, 'cusolverDnZheevjBatched_bufferSize')

    global __cusolverDnSsyevjBatched
    __cusolverDnSsyevjBatched = dlsym(RTLD_DEFAULT, 'cusolverDnSsyevjBatched')
    if __cusolverDnSsyevjBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsyevjBatched = dlsym(handle, 'cusolverDnSsyevjBatched')

    global __cusolverDnDsyevjBatched
    __cusolverDnDsyevjBatched = dlsym(RTLD_DEFAULT, 'cusolverDnDsyevjBatched')
    if __cusolverDnDsyevjBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsyevjBatched = dlsym(handle, 'cusolverDnDsyevjBatched')

    global __cusolverDnCheevjBatched
    __cusolverDnCheevjBatched = dlsym(RTLD_DEFAULT, 'cusolverDnCheevjBatched')
    if __cusolverDnCheevjBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCheevjBatched = dlsym(handle, 'cusolverDnCheevjBatched')

    global __cusolverDnZheevjBatched
    __cusolverDnZheevjBatched = dlsym(RTLD_DEFAULT, 'cusolverDnZheevjBatched')
    if __cusolverDnZheevjBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZheevjBatched = dlsym(handle, 'cusolverDnZheevjBatched')

    global __cusolverDnSsyevj_bufferSize
    __cusolverDnSsyevj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSsyevj_bufferSize')
    if __cusolverDnSsyevj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsyevj_bufferSize = dlsym(handle, 'cusolverDnSsyevj_bufferSize')

    global __cusolverDnDsyevj_bufferSize
    __cusolverDnDsyevj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDsyevj_bufferSize')
    if __cusolverDnDsyevj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsyevj_bufferSize = dlsym(handle, 'cusolverDnDsyevj_bufferSize')

    global __cusolverDnCheevj_bufferSize
    __cusolverDnCheevj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCheevj_bufferSize')
    if __cusolverDnCheevj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCheevj_bufferSize = dlsym(handle, 'cusolverDnCheevj_bufferSize')

    global __cusolverDnZheevj_bufferSize
    __cusolverDnZheevj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZheevj_bufferSize')
    if __cusolverDnZheevj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZheevj_bufferSize = dlsym(handle, 'cusolverDnZheevj_bufferSize')

    global __cusolverDnSsyevj
    __cusolverDnSsyevj = dlsym(RTLD_DEFAULT, 'cusolverDnSsyevj')
    if __cusolverDnSsyevj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsyevj = dlsym(handle, 'cusolverDnSsyevj')

    global __cusolverDnDsyevj
    __cusolverDnDsyevj = dlsym(RTLD_DEFAULT, 'cusolverDnDsyevj')
    if __cusolverDnDsyevj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsyevj = dlsym(handle, 'cusolverDnDsyevj')

    global __cusolverDnCheevj
    __cusolverDnCheevj = dlsym(RTLD_DEFAULT, 'cusolverDnCheevj')
    if __cusolverDnCheevj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCheevj = dlsym(handle, 'cusolverDnCheevj')

    global __cusolverDnZheevj
    __cusolverDnZheevj = dlsym(RTLD_DEFAULT, 'cusolverDnZheevj')
    if __cusolverDnZheevj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZheevj = dlsym(handle, 'cusolverDnZheevj')

    global __cusolverDnSsygvj_bufferSize
    __cusolverDnSsygvj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSsygvj_bufferSize')
    if __cusolverDnSsygvj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsygvj_bufferSize = dlsym(handle, 'cusolverDnSsygvj_bufferSize')

    global __cusolverDnDsygvj_bufferSize
    __cusolverDnDsygvj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDsygvj_bufferSize')
    if __cusolverDnDsygvj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsygvj_bufferSize = dlsym(handle, 'cusolverDnDsygvj_bufferSize')

    global __cusolverDnChegvj_bufferSize
    __cusolverDnChegvj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnChegvj_bufferSize')
    if __cusolverDnChegvj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnChegvj_bufferSize = dlsym(handle, 'cusolverDnChegvj_bufferSize')

    global __cusolverDnZhegvj_bufferSize
    __cusolverDnZhegvj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZhegvj_bufferSize')
    if __cusolverDnZhegvj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZhegvj_bufferSize = dlsym(handle, 'cusolverDnZhegvj_bufferSize')

    global __cusolverDnSsygvj
    __cusolverDnSsygvj = dlsym(RTLD_DEFAULT, 'cusolverDnSsygvj')
    if __cusolverDnSsygvj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSsygvj = dlsym(handle, 'cusolverDnSsygvj')

    global __cusolverDnDsygvj
    __cusolverDnDsygvj = dlsym(RTLD_DEFAULT, 'cusolverDnDsygvj')
    if __cusolverDnDsygvj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDsygvj = dlsym(handle, 'cusolverDnDsygvj')

    global __cusolverDnChegvj
    __cusolverDnChegvj = dlsym(RTLD_DEFAULT, 'cusolverDnChegvj')
    if __cusolverDnChegvj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnChegvj = dlsym(handle, 'cusolverDnChegvj')

    global __cusolverDnZhegvj
    __cusolverDnZhegvj = dlsym(RTLD_DEFAULT, 'cusolverDnZhegvj')
    if __cusolverDnZhegvj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZhegvj = dlsym(handle, 'cusolverDnZhegvj')

    global __cusolverDnCreateGesvdjInfo
    __cusolverDnCreateGesvdjInfo = dlsym(RTLD_DEFAULT, 'cusolverDnCreateGesvdjInfo')
    if __cusolverDnCreateGesvdjInfo == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCreateGesvdjInfo = dlsym(handle, 'cusolverDnCreateGesvdjInfo')

    global __cusolverDnDestroyGesvdjInfo
    __cusolverDnDestroyGesvdjInfo = dlsym(RTLD_DEFAULT, 'cusolverDnDestroyGesvdjInfo')
    if __cusolverDnDestroyGesvdjInfo == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDestroyGesvdjInfo = dlsym(handle, 'cusolverDnDestroyGesvdjInfo')

    global __cusolverDnXgesvdjSetTolerance
    __cusolverDnXgesvdjSetTolerance = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvdjSetTolerance')
    if __cusolverDnXgesvdjSetTolerance == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvdjSetTolerance = dlsym(handle, 'cusolverDnXgesvdjSetTolerance')

    global __cusolverDnXgesvdjSetMaxSweeps
    __cusolverDnXgesvdjSetMaxSweeps = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvdjSetMaxSweeps')
    if __cusolverDnXgesvdjSetMaxSweeps == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvdjSetMaxSweeps = dlsym(handle, 'cusolverDnXgesvdjSetMaxSweeps')

    global __cusolverDnXgesvdjSetSortEig
    __cusolverDnXgesvdjSetSortEig = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvdjSetSortEig')
    if __cusolverDnXgesvdjSetSortEig == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvdjSetSortEig = dlsym(handle, 'cusolverDnXgesvdjSetSortEig')

    global __cusolverDnXgesvdjGetResidual
    __cusolverDnXgesvdjGetResidual = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvdjGetResidual')
    if __cusolverDnXgesvdjGetResidual == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvdjGetResidual = dlsym(handle, 'cusolverDnXgesvdjGetResidual')

    global __cusolverDnXgesvdjGetSweeps
    __cusolverDnXgesvdjGetSweeps = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvdjGetSweeps')
    if __cusolverDnXgesvdjGetSweeps == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvdjGetSweeps = dlsym(handle, 'cusolverDnXgesvdjGetSweeps')

    global __cusolverDnSgesvdjBatched_bufferSize
    __cusolverDnSgesvdjBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSgesvdjBatched_bufferSize')
    if __cusolverDnSgesvdjBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgesvdjBatched_bufferSize = dlsym(handle, 'cusolverDnSgesvdjBatched_bufferSize')

    global __cusolverDnDgesvdjBatched_bufferSize
    __cusolverDnDgesvdjBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDgesvdjBatched_bufferSize')
    if __cusolverDnDgesvdjBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgesvdjBatched_bufferSize = dlsym(handle, 'cusolverDnDgesvdjBatched_bufferSize')

    global __cusolverDnCgesvdjBatched_bufferSize
    __cusolverDnCgesvdjBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCgesvdjBatched_bufferSize')
    if __cusolverDnCgesvdjBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgesvdjBatched_bufferSize = dlsym(handle, 'cusolverDnCgesvdjBatched_bufferSize')

    global __cusolverDnZgesvdjBatched_bufferSize
    __cusolverDnZgesvdjBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZgesvdjBatched_bufferSize')
    if __cusolverDnZgesvdjBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgesvdjBatched_bufferSize = dlsym(handle, 'cusolverDnZgesvdjBatched_bufferSize')

    global __cusolverDnSgesvdjBatched
    __cusolverDnSgesvdjBatched = dlsym(RTLD_DEFAULT, 'cusolverDnSgesvdjBatched')
    if __cusolverDnSgesvdjBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgesvdjBatched = dlsym(handle, 'cusolverDnSgesvdjBatched')

    global __cusolverDnDgesvdjBatched
    __cusolverDnDgesvdjBatched = dlsym(RTLD_DEFAULT, 'cusolverDnDgesvdjBatched')
    if __cusolverDnDgesvdjBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgesvdjBatched = dlsym(handle, 'cusolverDnDgesvdjBatched')

    global __cusolverDnCgesvdjBatched
    __cusolverDnCgesvdjBatched = dlsym(RTLD_DEFAULT, 'cusolverDnCgesvdjBatched')
    if __cusolverDnCgesvdjBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgesvdjBatched = dlsym(handle, 'cusolverDnCgesvdjBatched')

    global __cusolverDnZgesvdjBatched
    __cusolverDnZgesvdjBatched = dlsym(RTLD_DEFAULT, 'cusolverDnZgesvdjBatched')
    if __cusolverDnZgesvdjBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgesvdjBatched = dlsym(handle, 'cusolverDnZgesvdjBatched')

    global __cusolverDnSgesvdj_bufferSize
    __cusolverDnSgesvdj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSgesvdj_bufferSize')
    if __cusolverDnSgesvdj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgesvdj_bufferSize = dlsym(handle, 'cusolverDnSgesvdj_bufferSize')

    global __cusolverDnDgesvdj_bufferSize
    __cusolverDnDgesvdj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDgesvdj_bufferSize')
    if __cusolverDnDgesvdj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgesvdj_bufferSize = dlsym(handle, 'cusolverDnDgesvdj_bufferSize')

    global __cusolverDnCgesvdj_bufferSize
    __cusolverDnCgesvdj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCgesvdj_bufferSize')
    if __cusolverDnCgesvdj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgesvdj_bufferSize = dlsym(handle, 'cusolverDnCgesvdj_bufferSize')

    global __cusolverDnZgesvdj_bufferSize
    __cusolverDnZgesvdj_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZgesvdj_bufferSize')
    if __cusolverDnZgesvdj_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgesvdj_bufferSize = dlsym(handle, 'cusolverDnZgesvdj_bufferSize')

    global __cusolverDnSgesvdj
    __cusolverDnSgesvdj = dlsym(RTLD_DEFAULT, 'cusolverDnSgesvdj')
    if __cusolverDnSgesvdj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgesvdj = dlsym(handle, 'cusolverDnSgesvdj')

    global __cusolverDnDgesvdj
    __cusolverDnDgesvdj = dlsym(RTLD_DEFAULT, 'cusolverDnDgesvdj')
    if __cusolverDnDgesvdj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgesvdj = dlsym(handle, 'cusolverDnDgesvdj')

    global __cusolverDnCgesvdj
    __cusolverDnCgesvdj = dlsym(RTLD_DEFAULT, 'cusolverDnCgesvdj')
    if __cusolverDnCgesvdj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgesvdj = dlsym(handle, 'cusolverDnCgesvdj')

    global __cusolverDnZgesvdj
    __cusolverDnZgesvdj = dlsym(RTLD_DEFAULT, 'cusolverDnZgesvdj')
    if __cusolverDnZgesvdj == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgesvdj = dlsym(handle, 'cusolverDnZgesvdj')

    global __cusolverDnSgesvdaStridedBatched_bufferSize
    __cusolverDnSgesvdaStridedBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnSgesvdaStridedBatched_bufferSize')
    if __cusolverDnSgesvdaStridedBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgesvdaStridedBatched_bufferSize = dlsym(handle, 'cusolverDnSgesvdaStridedBatched_bufferSize')

    global __cusolverDnDgesvdaStridedBatched_bufferSize
    __cusolverDnDgesvdaStridedBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnDgesvdaStridedBatched_bufferSize')
    if __cusolverDnDgesvdaStridedBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgesvdaStridedBatched_bufferSize = dlsym(handle, 'cusolverDnDgesvdaStridedBatched_bufferSize')

    global __cusolverDnCgesvdaStridedBatched_bufferSize
    __cusolverDnCgesvdaStridedBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnCgesvdaStridedBatched_bufferSize')
    if __cusolverDnCgesvdaStridedBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgesvdaStridedBatched_bufferSize = dlsym(handle, 'cusolverDnCgesvdaStridedBatched_bufferSize')

    global __cusolverDnZgesvdaStridedBatched_bufferSize
    __cusolverDnZgesvdaStridedBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnZgesvdaStridedBatched_bufferSize')
    if __cusolverDnZgesvdaStridedBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgesvdaStridedBatched_bufferSize = dlsym(handle, 'cusolverDnZgesvdaStridedBatched_bufferSize')

    global __cusolverDnSgesvdaStridedBatched
    __cusolverDnSgesvdaStridedBatched = dlsym(RTLD_DEFAULT, 'cusolverDnSgesvdaStridedBatched')
    if __cusolverDnSgesvdaStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSgesvdaStridedBatched = dlsym(handle, 'cusolverDnSgesvdaStridedBatched')

    global __cusolverDnDgesvdaStridedBatched
    __cusolverDnDgesvdaStridedBatched = dlsym(RTLD_DEFAULT, 'cusolverDnDgesvdaStridedBatched')
    if __cusolverDnDgesvdaStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDgesvdaStridedBatched = dlsym(handle, 'cusolverDnDgesvdaStridedBatched')

    global __cusolverDnCgesvdaStridedBatched
    __cusolverDnCgesvdaStridedBatched = dlsym(RTLD_DEFAULT, 'cusolverDnCgesvdaStridedBatched')
    if __cusolverDnCgesvdaStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCgesvdaStridedBatched = dlsym(handle, 'cusolverDnCgesvdaStridedBatched')

    global __cusolverDnZgesvdaStridedBatched
    __cusolverDnZgesvdaStridedBatched = dlsym(RTLD_DEFAULT, 'cusolverDnZgesvdaStridedBatched')
    if __cusolverDnZgesvdaStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnZgesvdaStridedBatched = dlsym(handle, 'cusolverDnZgesvdaStridedBatched')

    global __cusolverDnCreateParams
    __cusolverDnCreateParams = dlsym(RTLD_DEFAULT, 'cusolverDnCreateParams')
    if __cusolverDnCreateParams == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnCreateParams = dlsym(handle, 'cusolverDnCreateParams')

    global __cusolverDnDestroyParams
    __cusolverDnDestroyParams = dlsym(RTLD_DEFAULT, 'cusolverDnDestroyParams')
    if __cusolverDnDestroyParams == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnDestroyParams = dlsym(handle, 'cusolverDnDestroyParams')

    global __cusolverDnSetAdvOptions
    __cusolverDnSetAdvOptions = dlsym(RTLD_DEFAULT, 'cusolverDnSetAdvOptions')
    if __cusolverDnSetAdvOptions == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSetAdvOptions = dlsym(handle, 'cusolverDnSetAdvOptions')

    global __cusolverDnXpotrf_bufferSize
    __cusolverDnXpotrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXpotrf_bufferSize')
    if __cusolverDnXpotrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXpotrf_bufferSize = dlsym(handle, 'cusolverDnXpotrf_bufferSize')

    global __cusolverDnXpotrf
    __cusolverDnXpotrf = dlsym(RTLD_DEFAULT, 'cusolverDnXpotrf')
    if __cusolverDnXpotrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXpotrf = dlsym(handle, 'cusolverDnXpotrf')

    global __cusolverDnXpotrs
    __cusolverDnXpotrs = dlsym(RTLD_DEFAULT, 'cusolverDnXpotrs')
    if __cusolverDnXpotrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXpotrs = dlsym(handle, 'cusolverDnXpotrs')

    global __cusolverDnXgeqrf_bufferSize
    __cusolverDnXgeqrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXgeqrf_bufferSize')
    if __cusolverDnXgeqrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgeqrf_bufferSize = dlsym(handle, 'cusolverDnXgeqrf_bufferSize')

    global __cusolverDnXgeqrf
    __cusolverDnXgeqrf = dlsym(RTLD_DEFAULT, 'cusolverDnXgeqrf')
    if __cusolverDnXgeqrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgeqrf = dlsym(handle, 'cusolverDnXgeqrf')

    global __cusolverDnXgetrf_bufferSize
    __cusolverDnXgetrf_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXgetrf_bufferSize')
    if __cusolverDnXgetrf_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgetrf_bufferSize = dlsym(handle, 'cusolverDnXgetrf_bufferSize')

    global __cusolverDnXgetrf
    __cusolverDnXgetrf = dlsym(RTLD_DEFAULT, 'cusolverDnXgetrf')
    if __cusolverDnXgetrf == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgetrf = dlsym(handle, 'cusolverDnXgetrf')

    global __cusolverDnXgetrs
    __cusolverDnXgetrs = dlsym(RTLD_DEFAULT, 'cusolverDnXgetrs')
    if __cusolverDnXgetrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgetrs = dlsym(handle, 'cusolverDnXgetrs')

    global __cusolverDnXsyevd_bufferSize
    __cusolverDnXsyevd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevd_bufferSize')
    if __cusolverDnXsyevd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevd_bufferSize = dlsym(handle, 'cusolverDnXsyevd_bufferSize')

    global __cusolverDnXsyevd
    __cusolverDnXsyevd = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevd')
    if __cusolverDnXsyevd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevd = dlsym(handle, 'cusolverDnXsyevd')

    global __cusolverDnXsyevdx_bufferSize
    __cusolverDnXsyevdx_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevdx_bufferSize')
    if __cusolverDnXsyevdx_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevdx_bufferSize = dlsym(handle, 'cusolverDnXsyevdx_bufferSize')

    global __cusolverDnXsyevdx
    __cusolverDnXsyevdx = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevdx')
    if __cusolverDnXsyevdx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevdx = dlsym(handle, 'cusolverDnXsyevdx')

    global __cusolverDnXgesvd_bufferSize
    __cusolverDnXgesvd_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvd_bufferSize')
    if __cusolverDnXgesvd_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvd_bufferSize = dlsym(handle, 'cusolverDnXgesvd_bufferSize')

    global __cusolverDnXgesvd
    __cusolverDnXgesvd = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvd')
    if __cusolverDnXgesvd == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvd = dlsym(handle, 'cusolverDnXgesvd')

    global __cusolverDnXgesvdp_bufferSize
    __cusolverDnXgesvdp_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvdp_bufferSize')
    if __cusolverDnXgesvdp_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvdp_bufferSize = dlsym(handle, 'cusolverDnXgesvdp_bufferSize')

    global __cusolverDnXgesvdp
    __cusolverDnXgesvdp = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvdp')
    if __cusolverDnXgesvdp == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvdp = dlsym(handle, 'cusolverDnXgesvdp')

    global __cusolverDnXgesvdr_bufferSize
    __cusolverDnXgesvdr_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvdr_bufferSize')
    if __cusolverDnXgesvdr_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvdr_bufferSize = dlsym(handle, 'cusolverDnXgesvdr_bufferSize')

    global __cusolverDnXgesvdr
    __cusolverDnXgesvdr = dlsym(RTLD_DEFAULT, 'cusolverDnXgesvdr')
    if __cusolverDnXgesvdr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgesvdr = dlsym(handle, 'cusolverDnXgesvdr')

    global __cusolverDnXsytrs_bufferSize
    __cusolverDnXsytrs_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXsytrs_bufferSize')
    if __cusolverDnXsytrs_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsytrs_bufferSize = dlsym(handle, 'cusolverDnXsytrs_bufferSize')

    global __cusolverDnXsytrs
    __cusolverDnXsytrs = dlsym(RTLD_DEFAULT, 'cusolverDnXsytrs')
    if __cusolverDnXsytrs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsytrs = dlsym(handle, 'cusolverDnXsytrs')

    global __cusolverDnXtrtri_bufferSize
    __cusolverDnXtrtri_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXtrtri_bufferSize')
    if __cusolverDnXtrtri_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXtrtri_bufferSize = dlsym(handle, 'cusolverDnXtrtri_bufferSize')

    global __cusolverDnXtrtri
    __cusolverDnXtrtri = dlsym(RTLD_DEFAULT, 'cusolverDnXtrtri')
    if __cusolverDnXtrtri == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXtrtri = dlsym(handle, 'cusolverDnXtrtri')

    global __cusolverDnLoggerSetCallback
    __cusolverDnLoggerSetCallback = dlsym(RTLD_DEFAULT, 'cusolverDnLoggerSetCallback')
    if __cusolverDnLoggerSetCallback == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnLoggerSetCallback = dlsym(handle, 'cusolverDnLoggerSetCallback')

    global __cusolverDnLoggerSetFile
    __cusolverDnLoggerSetFile = dlsym(RTLD_DEFAULT, 'cusolverDnLoggerSetFile')
    if __cusolverDnLoggerSetFile == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnLoggerSetFile = dlsym(handle, 'cusolverDnLoggerSetFile')

    global __cusolverDnLoggerOpenFile
    __cusolverDnLoggerOpenFile = dlsym(RTLD_DEFAULT, 'cusolverDnLoggerOpenFile')
    if __cusolverDnLoggerOpenFile == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnLoggerOpenFile = dlsym(handle, 'cusolverDnLoggerOpenFile')

    global __cusolverDnLoggerSetLevel
    __cusolverDnLoggerSetLevel = dlsym(RTLD_DEFAULT, 'cusolverDnLoggerSetLevel')
    if __cusolverDnLoggerSetLevel == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnLoggerSetLevel = dlsym(handle, 'cusolverDnLoggerSetLevel')

    global __cusolverDnLoggerSetMask
    __cusolverDnLoggerSetMask = dlsym(RTLD_DEFAULT, 'cusolverDnLoggerSetMask')
    if __cusolverDnLoggerSetMask == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnLoggerSetMask = dlsym(handle, 'cusolverDnLoggerSetMask')

    global __cusolverDnLoggerForceDisable
    __cusolverDnLoggerForceDisable = dlsym(RTLD_DEFAULT, 'cusolverDnLoggerForceDisable')
    if __cusolverDnLoggerForceDisable == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnLoggerForceDisable = dlsym(handle, 'cusolverDnLoggerForceDisable')

    global __cusolverDnSetDeterministicMode
    __cusolverDnSetDeterministicMode = dlsym(RTLD_DEFAULT, 'cusolverDnSetDeterministicMode')
    if __cusolverDnSetDeterministicMode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnSetDeterministicMode = dlsym(handle, 'cusolverDnSetDeterministicMode')

    global __cusolverDnGetDeterministicMode
    __cusolverDnGetDeterministicMode = dlsym(RTLD_DEFAULT, 'cusolverDnGetDeterministicMode')
    if __cusolverDnGetDeterministicMode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnGetDeterministicMode = dlsym(handle, 'cusolverDnGetDeterministicMode')

    global __cusolverDnXlarft_bufferSize
    __cusolverDnXlarft_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXlarft_bufferSize')
    if __cusolverDnXlarft_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXlarft_bufferSize = dlsym(handle, 'cusolverDnXlarft_bufferSize')

    global __cusolverDnXlarft
    __cusolverDnXlarft = dlsym(RTLD_DEFAULT, 'cusolverDnXlarft')
    if __cusolverDnXlarft == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXlarft = dlsym(handle, 'cusolverDnXlarft')

    global __cusolverDnXsyevBatched_bufferSize
    __cusolverDnXsyevBatched_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevBatched_bufferSize')
    if __cusolverDnXsyevBatched_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevBatched_bufferSize = dlsym(handle, 'cusolverDnXsyevBatched_bufferSize')

    global __cusolverDnXsyevBatched
    __cusolverDnXsyevBatched = dlsym(RTLD_DEFAULT, 'cusolverDnXsyevBatched')
    if __cusolverDnXsyevBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXsyevBatched = dlsym(handle, 'cusolverDnXsyevBatched')

    global __cusolverDnXgeev_bufferSize
    __cusolverDnXgeev_bufferSize = dlsym(RTLD_DEFAULT, 'cusolverDnXgeev_bufferSize')
    if __cusolverDnXgeev_bufferSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgeev_bufferSize = dlsym(handle, 'cusolverDnXgeev_bufferSize')

    global __cusolverDnXgeev
    __cusolverDnXgeev = dlsym(RTLD_DEFAULT, 'cusolverDnXgeev')
    if __cusolverDnXgeev == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverDnXgeev = dlsym(handle, 'cusolverDnXgeev')

    __py_cusolverDn_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cusolverDn()
    cdef dict data = {}

    global __cusolverDnCreate
    data["__cusolverDnCreate"] = <intptr_t>__cusolverDnCreate

    global __cusolverDnDestroy
    data["__cusolverDnDestroy"] = <intptr_t>__cusolverDnDestroy

    global __cusolverDnSetStream
    data["__cusolverDnSetStream"] = <intptr_t>__cusolverDnSetStream

    global __cusolverDnGetStream
    data["__cusolverDnGetStream"] = <intptr_t>__cusolverDnGetStream

    global __cusolverDnIRSParamsCreate
    data["__cusolverDnIRSParamsCreate"] = <intptr_t>__cusolverDnIRSParamsCreate

    global __cusolverDnIRSParamsDestroy
    data["__cusolverDnIRSParamsDestroy"] = <intptr_t>__cusolverDnIRSParamsDestroy

    global __cusolverDnIRSParamsSetRefinementSolver
    data["__cusolverDnIRSParamsSetRefinementSolver"] = <intptr_t>__cusolverDnIRSParamsSetRefinementSolver

    global __cusolverDnIRSParamsSetSolverMainPrecision
    data["__cusolverDnIRSParamsSetSolverMainPrecision"] = <intptr_t>__cusolverDnIRSParamsSetSolverMainPrecision

    global __cusolverDnIRSParamsSetSolverLowestPrecision
    data["__cusolverDnIRSParamsSetSolverLowestPrecision"] = <intptr_t>__cusolverDnIRSParamsSetSolverLowestPrecision

    global __cusolverDnIRSParamsSetSolverPrecisions
    data["__cusolverDnIRSParamsSetSolverPrecisions"] = <intptr_t>__cusolverDnIRSParamsSetSolverPrecisions

    global __cusolverDnIRSParamsSetTol
    data["__cusolverDnIRSParamsSetTol"] = <intptr_t>__cusolverDnIRSParamsSetTol

    global __cusolverDnIRSParamsSetTolInner
    data["__cusolverDnIRSParamsSetTolInner"] = <intptr_t>__cusolverDnIRSParamsSetTolInner

    global __cusolverDnIRSParamsSetMaxIters
    data["__cusolverDnIRSParamsSetMaxIters"] = <intptr_t>__cusolverDnIRSParamsSetMaxIters

    global __cusolverDnIRSParamsSetMaxItersInner
    data["__cusolverDnIRSParamsSetMaxItersInner"] = <intptr_t>__cusolverDnIRSParamsSetMaxItersInner

    global __cusolverDnIRSParamsGetMaxIters
    data["__cusolverDnIRSParamsGetMaxIters"] = <intptr_t>__cusolverDnIRSParamsGetMaxIters

    global __cusolverDnIRSParamsEnableFallback
    data["__cusolverDnIRSParamsEnableFallback"] = <intptr_t>__cusolverDnIRSParamsEnableFallback

    global __cusolverDnIRSParamsDisableFallback
    data["__cusolverDnIRSParamsDisableFallback"] = <intptr_t>__cusolverDnIRSParamsDisableFallback

    global __cusolverDnIRSInfosDestroy
    data["__cusolverDnIRSInfosDestroy"] = <intptr_t>__cusolverDnIRSInfosDestroy

    global __cusolverDnIRSInfosCreate
    data["__cusolverDnIRSInfosCreate"] = <intptr_t>__cusolverDnIRSInfosCreate

    global __cusolverDnIRSInfosGetNiters
    data["__cusolverDnIRSInfosGetNiters"] = <intptr_t>__cusolverDnIRSInfosGetNiters

    global __cusolverDnIRSInfosGetOuterNiters
    data["__cusolverDnIRSInfosGetOuterNiters"] = <intptr_t>__cusolverDnIRSInfosGetOuterNiters

    global __cusolverDnIRSInfosRequestResidual
    data["__cusolverDnIRSInfosRequestResidual"] = <intptr_t>__cusolverDnIRSInfosRequestResidual

    global __cusolverDnIRSInfosGetResidualHistory
    data["__cusolverDnIRSInfosGetResidualHistory"] = <intptr_t>__cusolverDnIRSInfosGetResidualHistory

    global __cusolverDnIRSInfosGetMaxIters
    data["__cusolverDnIRSInfosGetMaxIters"] = <intptr_t>__cusolverDnIRSInfosGetMaxIters

    global __cusolverDnZZgesv
    data["__cusolverDnZZgesv"] = <intptr_t>__cusolverDnZZgesv

    global __cusolverDnZCgesv
    data["__cusolverDnZCgesv"] = <intptr_t>__cusolverDnZCgesv

    global __cusolverDnZKgesv
    data["__cusolverDnZKgesv"] = <intptr_t>__cusolverDnZKgesv

    global __cusolverDnZEgesv
    data["__cusolverDnZEgesv"] = <intptr_t>__cusolverDnZEgesv

    global __cusolverDnZYgesv
    data["__cusolverDnZYgesv"] = <intptr_t>__cusolverDnZYgesv

    global __cusolverDnCCgesv
    data["__cusolverDnCCgesv"] = <intptr_t>__cusolverDnCCgesv

    global __cusolverDnCEgesv
    data["__cusolverDnCEgesv"] = <intptr_t>__cusolverDnCEgesv

    global __cusolverDnCKgesv
    data["__cusolverDnCKgesv"] = <intptr_t>__cusolverDnCKgesv

    global __cusolverDnCYgesv
    data["__cusolverDnCYgesv"] = <intptr_t>__cusolverDnCYgesv

    global __cusolverDnDDgesv
    data["__cusolverDnDDgesv"] = <intptr_t>__cusolverDnDDgesv

    global __cusolverDnDSgesv
    data["__cusolverDnDSgesv"] = <intptr_t>__cusolverDnDSgesv

    global __cusolverDnDHgesv
    data["__cusolverDnDHgesv"] = <intptr_t>__cusolverDnDHgesv

    global __cusolverDnDBgesv
    data["__cusolverDnDBgesv"] = <intptr_t>__cusolverDnDBgesv

    global __cusolverDnDXgesv
    data["__cusolverDnDXgesv"] = <intptr_t>__cusolverDnDXgesv

    global __cusolverDnSSgesv
    data["__cusolverDnSSgesv"] = <intptr_t>__cusolverDnSSgesv

    global __cusolverDnSHgesv
    data["__cusolverDnSHgesv"] = <intptr_t>__cusolverDnSHgesv

    global __cusolverDnSBgesv
    data["__cusolverDnSBgesv"] = <intptr_t>__cusolverDnSBgesv

    global __cusolverDnSXgesv
    data["__cusolverDnSXgesv"] = <intptr_t>__cusolverDnSXgesv

    global __cusolverDnZZgesv_bufferSize
    data["__cusolverDnZZgesv_bufferSize"] = <intptr_t>__cusolverDnZZgesv_bufferSize

    global __cusolverDnZCgesv_bufferSize
    data["__cusolverDnZCgesv_bufferSize"] = <intptr_t>__cusolverDnZCgesv_bufferSize

    global __cusolverDnZKgesv_bufferSize
    data["__cusolverDnZKgesv_bufferSize"] = <intptr_t>__cusolverDnZKgesv_bufferSize

    global __cusolverDnZEgesv_bufferSize
    data["__cusolverDnZEgesv_bufferSize"] = <intptr_t>__cusolverDnZEgesv_bufferSize

    global __cusolverDnZYgesv_bufferSize
    data["__cusolverDnZYgesv_bufferSize"] = <intptr_t>__cusolverDnZYgesv_bufferSize

    global __cusolverDnCCgesv_bufferSize
    data["__cusolverDnCCgesv_bufferSize"] = <intptr_t>__cusolverDnCCgesv_bufferSize

    global __cusolverDnCKgesv_bufferSize
    data["__cusolverDnCKgesv_bufferSize"] = <intptr_t>__cusolverDnCKgesv_bufferSize

    global __cusolverDnCEgesv_bufferSize
    data["__cusolverDnCEgesv_bufferSize"] = <intptr_t>__cusolverDnCEgesv_bufferSize

    global __cusolverDnCYgesv_bufferSize
    data["__cusolverDnCYgesv_bufferSize"] = <intptr_t>__cusolverDnCYgesv_bufferSize

    global __cusolverDnDDgesv_bufferSize
    data["__cusolverDnDDgesv_bufferSize"] = <intptr_t>__cusolverDnDDgesv_bufferSize

    global __cusolverDnDSgesv_bufferSize
    data["__cusolverDnDSgesv_bufferSize"] = <intptr_t>__cusolverDnDSgesv_bufferSize

    global __cusolverDnDHgesv_bufferSize
    data["__cusolverDnDHgesv_bufferSize"] = <intptr_t>__cusolverDnDHgesv_bufferSize

    global __cusolverDnDBgesv_bufferSize
    data["__cusolverDnDBgesv_bufferSize"] = <intptr_t>__cusolverDnDBgesv_bufferSize

    global __cusolverDnDXgesv_bufferSize
    data["__cusolverDnDXgesv_bufferSize"] = <intptr_t>__cusolverDnDXgesv_bufferSize

    global __cusolverDnSSgesv_bufferSize
    data["__cusolverDnSSgesv_bufferSize"] = <intptr_t>__cusolverDnSSgesv_bufferSize

    global __cusolverDnSHgesv_bufferSize
    data["__cusolverDnSHgesv_bufferSize"] = <intptr_t>__cusolverDnSHgesv_bufferSize

    global __cusolverDnSBgesv_bufferSize
    data["__cusolverDnSBgesv_bufferSize"] = <intptr_t>__cusolverDnSBgesv_bufferSize

    global __cusolverDnSXgesv_bufferSize
    data["__cusolverDnSXgesv_bufferSize"] = <intptr_t>__cusolverDnSXgesv_bufferSize

    global __cusolverDnZZgels
    data["__cusolverDnZZgels"] = <intptr_t>__cusolverDnZZgels

    global __cusolverDnZCgels
    data["__cusolverDnZCgels"] = <intptr_t>__cusolverDnZCgels

    global __cusolverDnZKgels
    data["__cusolverDnZKgels"] = <intptr_t>__cusolverDnZKgels

    global __cusolverDnZEgels
    data["__cusolverDnZEgels"] = <intptr_t>__cusolverDnZEgels

    global __cusolverDnZYgels
    data["__cusolverDnZYgels"] = <intptr_t>__cusolverDnZYgels

    global __cusolverDnCCgels
    data["__cusolverDnCCgels"] = <intptr_t>__cusolverDnCCgels

    global __cusolverDnCKgels
    data["__cusolverDnCKgels"] = <intptr_t>__cusolverDnCKgels

    global __cusolverDnCEgels
    data["__cusolverDnCEgels"] = <intptr_t>__cusolverDnCEgels

    global __cusolverDnCYgels
    data["__cusolverDnCYgels"] = <intptr_t>__cusolverDnCYgels

    global __cusolverDnDDgels
    data["__cusolverDnDDgels"] = <intptr_t>__cusolverDnDDgels

    global __cusolverDnDSgels
    data["__cusolverDnDSgels"] = <intptr_t>__cusolverDnDSgels

    global __cusolverDnDHgels
    data["__cusolverDnDHgels"] = <intptr_t>__cusolverDnDHgels

    global __cusolverDnDBgels
    data["__cusolverDnDBgels"] = <intptr_t>__cusolverDnDBgels

    global __cusolverDnDXgels
    data["__cusolverDnDXgels"] = <intptr_t>__cusolverDnDXgels

    global __cusolverDnSSgels
    data["__cusolverDnSSgels"] = <intptr_t>__cusolverDnSSgels

    global __cusolverDnSHgels
    data["__cusolverDnSHgels"] = <intptr_t>__cusolverDnSHgels

    global __cusolverDnSBgels
    data["__cusolverDnSBgels"] = <intptr_t>__cusolverDnSBgels

    global __cusolverDnSXgels
    data["__cusolverDnSXgels"] = <intptr_t>__cusolverDnSXgels

    global __cusolverDnZZgels_bufferSize
    data["__cusolverDnZZgels_bufferSize"] = <intptr_t>__cusolverDnZZgels_bufferSize

    global __cusolverDnZCgels_bufferSize
    data["__cusolverDnZCgels_bufferSize"] = <intptr_t>__cusolverDnZCgels_bufferSize

    global __cusolverDnZKgels_bufferSize
    data["__cusolverDnZKgels_bufferSize"] = <intptr_t>__cusolverDnZKgels_bufferSize

    global __cusolverDnZEgels_bufferSize
    data["__cusolverDnZEgels_bufferSize"] = <intptr_t>__cusolverDnZEgels_bufferSize

    global __cusolverDnZYgels_bufferSize
    data["__cusolverDnZYgels_bufferSize"] = <intptr_t>__cusolverDnZYgels_bufferSize

    global __cusolverDnCCgels_bufferSize
    data["__cusolverDnCCgels_bufferSize"] = <intptr_t>__cusolverDnCCgels_bufferSize

    global __cusolverDnCKgels_bufferSize
    data["__cusolverDnCKgels_bufferSize"] = <intptr_t>__cusolverDnCKgels_bufferSize

    global __cusolverDnCEgels_bufferSize
    data["__cusolverDnCEgels_bufferSize"] = <intptr_t>__cusolverDnCEgels_bufferSize

    global __cusolverDnCYgels_bufferSize
    data["__cusolverDnCYgels_bufferSize"] = <intptr_t>__cusolverDnCYgels_bufferSize

    global __cusolverDnDDgels_bufferSize
    data["__cusolverDnDDgels_bufferSize"] = <intptr_t>__cusolverDnDDgels_bufferSize

    global __cusolverDnDSgels_bufferSize
    data["__cusolverDnDSgels_bufferSize"] = <intptr_t>__cusolverDnDSgels_bufferSize

    global __cusolverDnDHgels_bufferSize
    data["__cusolverDnDHgels_bufferSize"] = <intptr_t>__cusolverDnDHgels_bufferSize

    global __cusolverDnDBgels_bufferSize
    data["__cusolverDnDBgels_bufferSize"] = <intptr_t>__cusolverDnDBgels_bufferSize

    global __cusolverDnDXgels_bufferSize
    data["__cusolverDnDXgels_bufferSize"] = <intptr_t>__cusolverDnDXgels_bufferSize

    global __cusolverDnSSgels_bufferSize
    data["__cusolverDnSSgels_bufferSize"] = <intptr_t>__cusolverDnSSgels_bufferSize

    global __cusolverDnSHgels_bufferSize
    data["__cusolverDnSHgels_bufferSize"] = <intptr_t>__cusolverDnSHgels_bufferSize

    global __cusolverDnSBgels_bufferSize
    data["__cusolverDnSBgels_bufferSize"] = <intptr_t>__cusolverDnSBgels_bufferSize

    global __cusolverDnSXgels_bufferSize
    data["__cusolverDnSXgels_bufferSize"] = <intptr_t>__cusolverDnSXgels_bufferSize

    global __cusolverDnIRSXgesv
    data["__cusolverDnIRSXgesv"] = <intptr_t>__cusolverDnIRSXgesv

    global __cusolverDnIRSXgesv_bufferSize
    data["__cusolverDnIRSXgesv_bufferSize"] = <intptr_t>__cusolverDnIRSXgesv_bufferSize

    global __cusolverDnIRSXgels
    data["__cusolverDnIRSXgels"] = <intptr_t>__cusolverDnIRSXgels

    global __cusolverDnIRSXgels_bufferSize
    data["__cusolverDnIRSXgels_bufferSize"] = <intptr_t>__cusolverDnIRSXgels_bufferSize

    global __cusolverDnSpotrf_bufferSize
    data["__cusolverDnSpotrf_bufferSize"] = <intptr_t>__cusolverDnSpotrf_bufferSize

    global __cusolverDnDpotrf_bufferSize
    data["__cusolverDnDpotrf_bufferSize"] = <intptr_t>__cusolverDnDpotrf_bufferSize

    global __cusolverDnCpotrf_bufferSize
    data["__cusolverDnCpotrf_bufferSize"] = <intptr_t>__cusolverDnCpotrf_bufferSize

    global __cusolverDnZpotrf_bufferSize
    data["__cusolverDnZpotrf_bufferSize"] = <intptr_t>__cusolverDnZpotrf_bufferSize

    global __cusolverDnSpotrf
    data["__cusolverDnSpotrf"] = <intptr_t>__cusolverDnSpotrf

    global __cusolverDnDpotrf
    data["__cusolverDnDpotrf"] = <intptr_t>__cusolverDnDpotrf

    global __cusolverDnCpotrf
    data["__cusolverDnCpotrf"] = <intptr_t>__cusolverDnCpotrf

    global __cusolverDnZpotrf
    data["__cusolverDnZpotrf"] = <intptr_t>__cusolverDnZpotrf

    global __cusolverDnSpotrs
    data["__cusolverDnSpotrs"] = <intptr_t>__cusolverDnSpotrs

    global __cusolverDnDpotrs
    data["__cusolverDnDpotrs"] = <intptr_t>__cusolverDnDpotrs

    global __cusolverDnCpotrs
    data["__cusolverDnCpotrs"] = <intptr_t>__cusolverDnCpotrs

    global __cusolverDnZpotrs
    data["__cusolverDnZpotrs"] = <intptr_t>__cusolverDnZpotrs

    global __cusolverDnSpotrfBatched
    data["__cusolverDnSpotrfBatched"] = <intptr_t>__cusolverDnSpotrfBatched

    global __cusolverDnDpotrfBatched
    data["__cusolverDnDpotrfBatched"] = <intptr_t>__cusolverDnDpotrfBatched

    global __cusolverDnCpotrfBatched
    data["__cusolverDnCpotrfBatched"] = <intptr_t>__cusolverDnCpotrfBatched

    global __cusolverDnZpotrfBatched
    data["__cusolverDnZpotrfBatched"] = <intptr_t>__cusolverDnZpotrfBatched

    global __cusolverDnSpotrsBatched
    data["__cusolverDnSpotrsBatched"] = <intptr_t>__cusolverDnSpotrsBatched

    global __cusolverDnDpotrsBatched
    data["__cusolverDnDpotrsBatched"] = <intptr_t>__cusolverDnDpotrsBatched

    global __cusolverDnCpotrsBatched
    data["__cusolverDnCpotrsBatched"] = <intptr_t>__cusolverDnCpotrsBatched

    global __cusolverDnZpotrsBatched
    data["__cusolverDnZpotrsBatched"] = <intptr_t>__cusolverDnZpotrsBatched

    global __cusolverDnSpotri_bufferSize
    data["__cusolverDnSpotri_bufferSize"] = <intptr_t>__cusolverDnSpotri_bufferSize

    global __cusolverDnDpotri_bufferSize
    data["__cusolverDnDpotri_bufferSize"] = <intptr_t>__cusolverDnDpotri_bufferSize

    global __cusolverDnCpotri_bufferSize
    data["__cusolverDnCpotri_bufferSize"] = <intptr_t>__cusolverDnCpotri_bufferSize

    global __cusolverDnZpotri_bufferSize
    data["__cusolverDnZpotri_bufferSize"] = <intptr_t>__cusolverDnZpotri_bufferSize

    global __cusolverDnSpotri
    data["__cusolverDnSpotri"] = <intptr_t>__cusolverDnSpotri

    global __cusolverDnDpotri
    data["__cusolverDnDpotri"] = <intptr_t>__cusolverDnDpotri

    global __cusolverDnCpotri
    data["__cusolverDnCpotri"] = <intptr_t>__cusolverDnCpotri

    global __cusolverDnZpotri
    data["__cusolverDnZpotri"] = <intptr_t>__cusolverDnZpotri

    global __cusolverDnSlauum_bufferSize
    data["__cusolverDnSlauum_bufferSize"] = <intptr_t>__cusolverDnSlauum_bufferSize

    global __cusolverDnDlauum_bufferSize
    data["__cusolverDnDlauum_bufferSize"] = <intptr_t>__cusolverDnDlauum_bufferSize

    global __cusolverDnClauum_bufferSize
    data["__cusolverDnClauum_bufferSize"] = <intptr_t>__cusolverDnClauum_bufferSize

    global __cusolverDnZlauum_bufferSize
    data["__cusolverDnZlauum_bufferSize"] = <intptr_t>__cusolverDnZlauum_bufferSize

    global __cusolverDnSlauum
    data["__cusolverDnSlauum"] = <intptr_t>__cusolverDnSlauum

    global __cusolverDnDlauum
    data["__cusolverDnDlauum"] = <intptr_t>__cusolverDnDlauum

    global __cusolverDnClauum
    data["__cusolverDnClauum"] = <intptr_t>__cusolverDnClauum

    global __cusolverDnZlauum
    data["__cusolverDnZlauum"] = <intptr_t>__cusolverDnZlauum

    global __cusolverDnSgetrf_bufferSize
    data["__cusolverDnSgetrf_bufferSize"] = <intptr_t>__cusolverDnSgetrf_bufferSize

    global __cusolverDnDgetrf_bufferSize
    data["__cusolverDnDgetrf_bufferSize"] = <intptr_t>__cusolverDnDgetrf_bufferSize

    global __cusolverDnCgetrf_bufferSize
    data["__cusolverDnCgetrf_bufferSize"] = <intptr_t>__cusolverDnCgetrf_bufferSize

    global __cusolverDnZgetrf_bufferSize
    data["__cusolverDnZgetrf_bufferSize"] = <intptr_t>__cusolverDnZgetrf_bufferSize

    global __cusolverDnSgetrf
    data["__cusolverDnSgetrf"] = <intptr_t>__cusolverDnSgetrf

    global __cusolverDnDgetrf
    data["__cusolverDnDgetrf"] = <intptr_t>__cusolverDnDgetrf

    global __cusolverDnCgetrf
    data["__cusolverDnCgetrf"] = <intptr_t>__cusolverDnCgetrf

    global __cusolverDnZgetrf
    data["__cusolverDnZgetrf"] = <intptr_t>__cusolverDnZgetrf

    global __cusolverDnSlaswp
    data["__cusolverDnSlaswp"] = <intptr_t>__cusolverDnSlaswp

    global __cusolverDnDlaswp
    data["__cusolverDnDlaswp"] = <intptr_t>__cusolverDnDlaswp

    global __cusolverDnClaswp
    data["__cusolverDnClaswp"] = <intptr_t>__cusolverDnClaswp

    global __cusolverDnZlaswp
    data["__cusolverDnZlaswp"] = <intptr_t>__cusolverDnZlaswp

    global __cusolverDnSgetrs
    data["__cusolverDnSgetrs"] = <intptr_t>__cusolverDnSgetrs

    global __cusolverDnDgetrs
    data["__cusolverDnDgetrs"] = <intptr_t>__cusolverDnDgetrs

    global __cusolverDnCgetrs
    data["__cusolverDnCgetrs"] = <intptr_t>__cusolverDnCgetrs

    global __cusolverDnZgetrs
    data["__cusolverDnZgetrs"] = <intptr_t>__cusolverDnZgetrs

    global __cusolverDnSgeqrf_bufferSize
    data["__cusolverDnSgeqrf_bufferSize"] = <intptr_t>__cusolverDnSgeqrf_bufferSize

    global __cusolverDnDgeqrf_bufferSize
    data["__cusolverDnDgeqrf_bufferSize"] = <intptr_t>__cusolverDnDgeqrf_bufferSize

    global __cusolverDnCgeqrf_bufferSize
    data["__cusolverDnCgeqrf_bufferSize"] = <intptr_t>__cusolverDnCgeqrf_bufferSize

    global __cusolverDnZgeqrf_bufferSize
    data["__cusolverDnZgeqrf_bufferSize"] = <intptr_t>__cusolverDnZgeqrf_bufferSize

    global __cusolverDnSgeqrf
    data["__cusolverDnSgeqrf"] = <intptr_t>__cusolverDnSgeqrf

    global __cusolverDnDgeqrf
    data["__cusolverDnDgeqrf"] = <intptr_t>__cusolverDnDgeqrf

    global __cusolverDnCgeqrf
    data["__cusolverDnCgeqrf"] = <intptr_t>__cusolverDnCgeqrf

    global __cusolverDnZgeqrf
    data["__cusolverDnZgeqrf"] = <intptr_t>__cusolverDnZgeqrf

    global __cusolverDnSorgqr_bufferSize
    data["__cusolverDnSorgqr_bufferSize"] = <intptr_t>__cusolverDnSorgqr_bufferSize

    global __cusolverDnDorgqr_bufferSize
    data["__cusolverDnDorgqr_bufferSize"] = <intptr_t>__cusolverDnDorgqr_bufferSize

    global __cusolverDnCungqr_bufferSize
    data["__cusolverDnCungqr_bufferSize"] = <intptr_t>__cusolverDnCungqr_bufferSize

    global __cusolverDnZungqr_bufferSize
    data["__cusolverDnZungqr_bufferSize"] = <intptr_t>__cusolverDnZungqr_bufferSize

    global __cusolverDnSorgqr
    data["__cusolverDnSorgqr"] = <intptr_t>__cusolverDnSorgqr

    global __cusolverDnDorgqr
    data["__cusolverDnDorgqr"] = <intptr_t>__cusolverDnDorgqr

    global __cusolverDnCungqr
    data["__cusolverDnCungqr"] = <intptr_t>__cusolverDnCungqr

    global __cusolverDnZungqr
    data["__cusolverDnZungqr"] = <intptr_t>__cusolverDnZungqr

    global __cusolverDnSormqr_bufferSize
    data["__cusolverDnSormqr_bufferSize"] = <intptr_t>__cusolverDnSormqr_bufferSize

    global __cusolverDnDormqr_bufferSize
    data["__cusolverDnDormqr_bufferSize"] = <intptr_t>__cusolverDnDormqr_bufferSize

    global __cusolverDnCunmqr_bufferSize
    data["__cusolverDnCunmqr_bufferSize"] = <intptr_t>__cusolverDnCunmqr_bufferSize

    global __cusolverDnZunmqr_bufferSize
    data["__cusolverDnZunmqr_bufferSize"] = <intptr_t>__cusolverDnZunmqr_bufferSize

    global __cusolverDnSormqr
    data["__cusolverDnSormqr"] = <intptr_t>__cusolverDnSormqr

    global __cusolverDnDormqr
    data["__cusolverDnDormqr"] = <intptr_t>__cusolverDnDormqr

    global __cusolverDnCunmqr
    data["__cusolverDnCunmqr"] = <intptr_t>__cusolverDnCunmqr

    global __cusolverDnZunmqr
    data["__cusolverDnZunmqr"] = <intptr_t>__cusolverDnZunmqr

    global __cusolverDnSsytrf_bufferSize
    data["__cusolverDnSsytrf_bufferSize"] = <intptr_t>__cusolverDnSsytrf_bufferSize

    global __cusolverDnDsytrf_bufferSize
    data["__cusolverDnDsytrf_bufferSize"] = <intptr_t>__cusolverDnDsytrf_bufferSize

    global __cusolverDnCsytrf_bufferSize
    data["__cusolverDnCsytrf_bufferSize"] = <intptr_t>__cusolverDnCsytrf_bufferSize

    global __cusolverDnZsytrf_bufferSize
    data["__cusolverDnZsytrf_bufferSize"] = <intptr_t>__cusolverDnZsytrf_bufferSize

    global __cusolverDnSsytrf
    data["__cusolverDnSsytrf"] = <intptr_t>__cusolverDnSsytrf

    global __cusolverDnDsytrf
    data["__cusolverDnDsytrf"] = <intptr_t>__cusolverDnDsytrf

    global __cusolverDnCsytrf
    data["__cusolverDnCsytrf"] = <intptr_t>__cusolverDnCsytrf

    global __cusolverDnZsytrf
    data["__cusolverDnZsytrf"] = <intptr_t>__cusolverDnZsytrf

    global __cusolverDnSsytri_bufferSize
    data["__cusolverDnSsytri_bufferSize"] = <intptr_t>__cusolverDnSsytri_bufferSize

    global __cusolverDnDsytri_bufferSize
    data["__cusolverDnDsytri_bufferSize"] = <intptr_t>__cusolverDnDsytri_bufferSize

    global __cusolverDnCsytri_bufferSize
    data["__cusolverDnCsytri_bufferSize"] = <intptr_t>__cusolverDnCsytri_bufferSize

    global __cusolverDnZsytri_bufferSize
    data["__cusolverDnZsytri_bufferSize"] = <intptr_t>__cusolverDnZsytri_bufferSize

    global __cusolverDnSsytri
    data["__cusolverDnSsytri"] = <intptr_t>__cusolverDnSsytri

    global __cusolverDnDsytri
    data["__cusolverDnDsytri"] = <intptr_t>__cusolverDnDsytri

    global __cusolverDnCsytri
    data["__cusolverDnCsytri"] = <intptr_t>__cusolverDnCsytri

    global __cusolverDnZsytri
    data["__cusolverDnZsytri"] = <intptr_t>__cusolverDnZsytri

    global __cusolverDnSgebrd_bufferSize
    data["__cusolverDnSgebrd_bufferSize"] = <intptr_t>__cusolverDnSgebrd_bufferSize

    global __cusolverDnDgebrd_bufferSize
    data["__cusolverDnDgebrd_bufferSize"] = <intptr_t>__cusolverDnDgebrd_bufferSize

    global __cusolverDnCgebrd_bufferSize
    data["__cusolverDnCgebrd_bufferSize"] = <intptr_t>__cusolverDnCgebrd_bufferSize

    global __cusolverDnZgebrd_bufferSize
    data["__cusolverDnZgebrd_bufferSize"] = <intptr_t>__cusolverDnZgebrd_bufferSize

    global __cusolverDnSgebrd
    data["__cusolverDnSgebrd"] = <intptr_t>__cusolverDnSgebrd

    global __cusolverDnDgebrd
    data["__cusolverDnDgebrd"] = <intptr_t>__cusolverDnDgebrd

    global __cusolverDnCgebrd
    data["__cusolverDnCgebrd"] = <intptr_t>__cusolverDnCgebrd

    global __cusolverDnZgebrd
    data["__cusolverDnZgebrd"] = <intptr_t>__cusolverDnZgebrd

    global __cusolverDnSorgbr_bufferSize
    data["__cusolverDnSorgbr_bufferSize"] = <intptr_t>__cusolverDnSorgbr_bufferSize

    global __cusolverDnDorgbr_bufferSize
    data["__cusolverDnDorgbr_bufferSize"] = <intptr_t>__cusolverDnDorgbr_bufferSize

    global __cusolverDnCungbr_bufferSize
    data["__cusolverDnCungbr_bufferSize"] = <intptr_t>__cusolverDnCungbr_bufferSize

    global __cusolverDnZungbr_bufferSize
    data["__cusolverDnZungbr_bufferSize"] = <intptr_t>__cusolverDnZungbr_bufferSize

    global __cusolverDnSorgbr
    data["__cusolverDnSorgbr"] = <intptr_t>__cusolverDnSorgbr

    global __cusolverDnDorgbr
    data["__cusolverDnDorgbr"] = <intptr_t>__cusolverDnDorgbr

    global __cusolverDnCungbr
    data["__cusolverDnCungbr"] = <intptr_t>__cusolverDnCungbr

    global __cusolverDnZungbr
    data["__cusolverDnZungbr"] = <intptr_t>__cusolverDnZungbr

    global __cusolverDnSsytrd_bufferSize
    data["__cusolverDnSsytrd_bufferSize"] = <intptr_t>__cusolverDnSsytrd_bufferSize

    global __cusolverDnDsytrd_bufferSize
    data["__cusolverDnDsytrd_bufferSize"] = <intptr_t>__cusolverDnDsytrd_bufferSize

    global __cusolverDnChetrd_bufferSize
    data["__cusolverDnChetrd_bufferSize"] = <intptr_t>__cusolverDnChetrd_bufferSize

    global __cusolverDnZhetrd_bufferSize
    data["__cusolverDnZhetrd_bufferSize"] = <intptr_t>__cusolverDnZhetrd_bufferSize

    global __cusolverDnSsytrd
    data["__cusolverDnSsytrd"] = <intptr_t>__cusolverDnSsytrd

    global __cusolverDnDsytrd
    data["__cusolverDnDsytrd"] = <intptr_t>__cusolverDnDsytrd

    global __cusolverDnChetrd
    data["__cusolverDnChetrd"] = <intptr_t>__cusolverDnChetrd

    global __cusolverDnZhetrd
    data["__cusolverDnZhetrd"] = <intptr_t>__cusolverDnZhetrd

    global __cusolverDnSorgtr_bufferSize
    data["__cusolverDnSorgtr_bufferSize"] = <intptr_t>__cusolverDnSorgtr_bufferSize

    global __cusolverDnDorgtr_bufferSize
    data["__cusolverDnDorgtr_bufferSize"] = <intptr_t>__cusolverDnDorgtr_bufferSize

    global __cusolverDnCungtr_bufferSize
    data["__cusolverDnCungtr_bufferSize"] = <intptr_t>__cusolverDnCungtr_bufferSize

    global __cusolverDnZungtr_bufferSize
    data["__cusolverDnZungtr_bufferSize"] = <intptr_t>__cusolverDnZungtr_bufferSize

    global __cusolverDnSorgtr
    data["__cusolverDnSorgtr"] = <intptr_t>__cusolverDnSorgtr

    global __cusolverDnDorgtr
    data["__cusolverDnDorgtr"] = <intptr_t>__cusolverDnDorgtr

    global __cusolverDnCungtr
    data["__cusolverDnCungtr"] = <intptr_t>__cusolverDnCungtr

    global __cusolverDnZungtr
    data["__cusolverDnZungtr"] = <intptr_t>__cusolverDnZungtr

    global __cusolverDnSormtr_bufferSize
    data["__cusolverDnSormtr_bufferSize"] = <intptr_t>__cusolverDnSormtr_bufferSize

    global __cusolverDnDormtr_bufferSize
    data["__cusolverDnDormtr_bufferSize"] = <intptr_t>__cusolverDnDormtr_bufferSize

    global __cusolverDnCunmtr_bufferSize
    data["__cusolverDnCunmtr_bufferSize"] = <intptr_t>__cusolverDnCunmtr_bufferSize

    global __cusolverDnZunmtr_bufferSize
    data["__cusolverDnZunmtr_bufferSize"] = <intptr_t>__cusolverDnZunmtr_bufferSize

    global __cusolverDnSormtr
    data["__cusolverDnSormtr"] = <intptr_t>__cusolverDnSormtr

    global __cusolverDnDormtr
    data["__cusolverDnDormtr"] = <intptr_t>__cusolverDnDormtr

    global __cusolverDnCunmtr
    data["__cusolverDnCunmtr"] = <intptr_t>__cusolverDnCunmtr

    global __cusolverDnZunmtr
    data["__cusolverDnZunmtr"] = <intptr_t>__cusolverDnZunmtr

    global __cusolverDnSgesvd_bufferSize
    data["__cusolverDnSgesvd_bufferSize"] = <intptr_t>__cusolverDnSgesvd_bufferSize

    global __cusolverDnDgesvd_bufferSize
    data["__cusolverDnDgesvd_bufferSize"] = <intptr_t>__cusolverDnDgesvd_bufferSize

    global __cusolverDnCgesvd_bufferSize
    data["__cusolverDnCgesvd_bufferSize"] = <intptr_t>__cusolverDnCgesvd_bufferSize

    global __cusolverDnZgesvd_bufferSize
    data["__cusolverDnZgesvd_bufferSize"] = <intptr_t>__cusolverDnZgesvd_bufferSize

    global __cusolverDnSgesvd
    data["__cusolverDnSgesvd"] = <intptr_t>__cusolverDnSgesvd

    global __cusolverDnDgesvd
    data["__cusolverDnDgesvd"] = <intptr_t>__cusolverDnDgesvd

    global __cusolverDnCgesvd
    data["__cusolverDnCgesvd"] = <intptr_t>__cusolverDnCgesvd

    global __cusolverDnZgesvd
    data["__cusolverDnZgesvd"] = <intptr_t>__cusolverDnZgesvd

    global __cusolverDnSsyevd_bufferSize
    data["__cusolverDnSsyevd_bufferSize"] = <intptr_t>__cusolverDnSsyevd_bufferSize

    global __cusolverDnDsyevd_bufferSize
    data["__cusolverDnDsyevd_bufferSize"] = <intptr_t>__cusolverDnDsyevd_bufferSize

    global __cusolverDnCheevd_bufferSize
    data["__cusolverDnCheevd_bufferSize"] = <intptr_t>__cusolverDnCheevd_bufferSize

    global __cusolverDnZheevd_bufferSize
    data["__cusolverDnZheevd_bufferSize"] = <intptr_t>__cusolverDnZheevd_bufferSize

    global __cusolverDnSsyevd
    data["__cusolverDnSsyevd"] = <intptr_t>__cusolverDnSsyevd

    global __cusolverDnDsyevd
    data["__cusolverDnDsyevd"] = <intptr_t>__cusolverDnDsyevd

    global __cusolverDnCheevd
    data["__cusolverDnCheevd"] = <intptr_t>__cusolverDnCheevd

    global __cusolverDnZheevd
    data["__cusolverDnZheevd"] = <intptr_t>__cusolverDnZheevd

    global __cusolverDnSsyevdx_bufferSize
    data["__cusolverDnSsyevdx_bufferSize"] = <intptr_t>__cusolverDnSsyevdx_bufferSize

    global __cusolverDnDsyevdx_bufferSize
    data["__cusolverDnDsyevdx_bufferSize"] = <intptr_t>__cusolverDnDsyevdx_bufferSize

    global __cusolverDnCheevdx_bufferSize
    data["__cusolverDnCheevdx_bufferSize"] = <intptr_t>__cusolverDnCheevdx_bufferSize

    global __cusolverDnZheevdx_bufferSize
    data["__cusolverDnZheevdx_bufferSize"] = <intptr_t>__cusolverDnZheevdx_bufferSize

    global __cusolverDnSsyevdx
    data["__cusolverDnSsyevdx"] = <intptr_t>__cusolverDnSsyevdx

    global __cusolverDnDsyevdx
    data["__cusolverDnDsyevdx"] = <intptr_t>__cusolverDnDsyevdx

    global __cusolverDnCheevdx
    data["__cusolverDnCheevdx"] = <intptr_t>__cusolverDnCheevdx

    global __cusolverDnZheevdx
    data["__cusolverDnZheevdx"] = <intptr_t>__cusolverDnZheevdx

    global __cusolverDnSsygvdx_bufferSize
    data["__cusolverDnSsygvdx_bufferSize"] = <intptr_t>__cusolverDnSsygvdx_bufferSize

    global __cusolverDnDsygvdx_bufferSize
    data["__cusolverDnDsygvdx_bufferSize"] = <intptr_t>__cusolverDnDsygvdx_bufferSize

    global __cusolverDnChegvdx_bufferSize
    data["__cusolverDnChegvdx_bufferSize"] = <intptr_t>__cusolverDnChegvdx_bufferSize

    global __cusolverDnZhegvdx_bufferSize
    data["__cusolverDnZhegvdx_bufferSize"] = <intptr_t>__cusolverDnZhegvdx_bufferSize

    global __cusolverDnSsygvdx
    data["__cusolverDnSsygvdx"] = <intptr_t>__cusolverDnSsygvdx

    global __cusolverDnDsygvdx
    data["__cusolverDnDsygvdx"] = <intptr_t>__cusolverDnDsygvdx

    global __cusolverDnChegvdx
    data["__cusolverDnChegvdx"] = <intptr_t>__cusolverDnChegvdx

    global __cusolverDnZhegvdx
    data["__cusolverDnZhegvdx"] = <intptr_t>__cusolverDnZhegvdx

    global __cusolverDnSsygvd_bufferSize
    data["__cusolverDnSsygvd_bufferSize"] = <intptr_t>__cusolverDnSsygvd_bufferSize

    global __cusolverDnDsygvd_bufferSize
    data["__cusolverDnDsygvd_bufferSize"] = <intptr_t>__cusolverDnDsygvd_bufferSize

    global __cusolverDnChegvd_bufferSize
    data["__cusolverDnChegvd_bufferSize"] = <intptr_t>__cusolverDnChegvd_bufferSize

    global __cusolverDnZhegvd_bufferSize
    data["__cusolverDnZhegvd_bufferSize"] = <intptr_t>__cusolverDnZhegvd_bufferSize

    global __cusolverDnSsygvd
    data["__cusolverDnSsygvd"] = <intptr_t>__cusolverDnSsygvd

    global __cusolverDnDsygvd
    data["__cusolverDnDsygvd"] = <intptr_t>__cusolverDnDsygvd

    global __cusolverDnChegvd
    data["__cusolverDnChegvd"] = <intptr_t>__cusolverDnChegvd

    global __cusolverDnZhegvd
    data["__cusolverDnZhegvd"] = <intptr_t>__cusolverDnZhegvd

    global __cusolverDnCreateSyevjInfo
    data["__cusolverDnCreateSyevjInfo"] = <intptr_t>__cusolverDnCreateSyevjInfo

    global __cusolverDnDestroySyevjInfo
    data["__cusolverDnDestroySyevjInfo"] = <intptr_t>__cusolverDnDestroySyevjInfo

    global __cusolverDnXsyevjSetTolerance
    data["__cusolverDnXsyevjSetTolerance"] = <intptr_t>__cusolverDnXsyevjSetTolerance

    global __cusolverDnXsyevjSetMaxSweeps
    data["__cusolverDnXsyevjSetMaxSweeps"] = <intptr_t>__cusolverDnXsyevjSetMaxSweeps

    global __cusolverDnXsyevjSetSortEig
    data["__cusolverDnXsyevjSetSortEig"] = <intptr_t>__cusolverDnXsyevjSetSortEig

    global __cusolverDnXsyevjGetResidual
    data["__cusolverDnXsyevjGetResidual"] = <intptr_t>__cusolverDnXsyevjGetResidual

    global __cusolverDnXsyevjGetSweeps
    data["__cusolverDnXsyevjGetSweeps"] = <intptr_t>__cusolverDnXsyevjGetSweeps

    global __cusolverDnSsyevjBatched_bufferSize
    data["__cusolverDnSsyevjBatched_bufferSize"] = <intptr_t>__cusolverDnSsyevjBatched_bufferSize

    global __cusolverDnDsyevjBatched_bufferSize
    data["__cusolverDnDsyevjBatched_bufferSize"] = <intptr_t>__cusolverDnDsyevjBatched_bufferSize

    global __cusolverDnCheevjBatched_bufferSize
    data["__cusolverDnCheevjBatched_bufferSize"] = <intptr_t>__cusolverDnCheevjBatched_bufferSize

    global __cusolverDnZheevjBatched_bufferSize
    data["__cusolverDnZheevjBatched_bufferSize"] = <intptr_t>__cusolverDnZheevjBatched_bufferSize

    global __cusolverDnSsyevjBatched
    data["__cusolverDnSsyevjBatched"] = <intptr_t>__cusolverDnSsyevjBatched

    global __cusolverDnDsyevjBatched
    data["__cusolverDnDsyevjBatched"] = <intptr_t>__cusolverDnDsyevjBatched

    global __cusolverDnCheevjBatched
    data["__cusolverDnCheevjBatched"] = <intptr_t>__cusolverDnCheevjBatched

    global __cusolverDnZheevjBatched
    data["__cusolverDnZheevjBatched"] = <intptr_t>__cusolverDnZheevjBatched

    global __cusolverDnSsyevj_bufferSize
    data["__cusolverDnSsyevj_bufferSize"] = <intptr_t>__cusolverDnSsyevj_bufferSize

    global __cusolverDnDsyevj_bufferSize
    data["__cusolverDnDsyevj_bufferSize"] = <intptr_t>__cusolverDnDsyevj_bufferSize

    global __cusolverDnCheevj_bufferSize
    data["__cusolverDnCheevj_bufferSize"] = <intptr_t>__cusolverDnCheevj_bufferSize

    global __cusolverDnZheevj_bufferSize
    data["__cusolverDnZheevj_bufferSize"] = <intptr_t>__cusolverDnZheevj_bufferSize

    global __cusolverDnSsyevj
    data["__cusolverDnSsyevj"] = <intptr_t>__cusolverDnSsyevj

    global __cusolverDnDsyevj
    data["__cusolverDnDsyevj"] = <intptr_t>__cusolverDnDsyevj

    global __cusolverDnCheevj
    data["__cusolverDnCheevj"] = <intptr_t>__cusolverDnCheevj

    global __cusolverDnZheevj
    data["__cusolverDnZheevj"] = <intptr_t>__cusolverDnZheevj

    global __cusolverDnSsygvj_bufferSize
    data["__cusolverDnSsygvj_bufferSize"] = <intptr_t>__cusolverDnSsygvj_bufferSize

    global __cusolverDnDsygvj_bufferSize
    data["__cusolverDnDsygvj_bufferSize"] = <intptr_t>__cusolverDnDsygvj_bufferSize

    global __cusolverDnChegvj_bufferSize
    data["__cusolverDnChegvj_bufferSize"] = <intptr_t>__cusolverDnChegvj_bufferSize

    global __cusolverDnZhegvj_bufferSize
    data["__cusolverDnZhegvj_bufferSize"] = <intptr_t>__cusolverDnZhegvj_bufferSize

    global __cusolverDnSsygvj
    data["__cusolverDnSsygvj"] = <intptr_t>__cusolverDnSsygvj

    global __cusolverDnDsygvj
    data["__cusolverDnDsygvj"] = <intptr_t>__cusolverDnDsygvj

    global __cusolverDnChegvj
    data["__cusolverDnChegvj"] = <intptr_t>__cusolverDnChegvj

    global __cusolverDnZhegvj
    data["__cusolverDnZhegvj"] = <intptr_t>__cusolverDnZhegvj

    global __cusolverDnCreateGesvdjInfo
    data["__cusolverDnCreateGesvdjInfo"] = <intptr_t>__cusolverDnCreateGesvdjInfo

    global __cusolverDnDestroyGesvdjInfo
    data["__cusolverDnDestroyGesvdjInfo"] = <intptr_t>__cusolverDnDestroyGesvdjInfo

    global __cusolverDnXgesvdjSetTolerance
    data["__cusolverDnXgesvdjSetTolerance"] = <intptr_t>__cusolverDnXgesvdjSetTolerance

    global __cusolverDnXgesvdjSetMaxSweeps
    data["__cusolverDnXgesvdjSetMaxSweeps"] = <intptr_t>__cusolverDnXgesvdjSetMaxSweeps

    global __cusolverDnXgesvdjSetSortEig
    data["__cusolverDnXgesvdjSetSortEig"] = <intptr_t>__cusolverDnXgesvdjSetSortEig

    global __cusolverDnXgesvdjGetResidual
    data["__cusolverDnXgesvdjGetResidual"] = <intptr_t>__cusolverDnXgesvdjGetResidual

    global __cusolverDnXgesvdjGetSweeps
    data["__cusolverDnXgesvdjGetSweeps"] = <intptr_t>__cusolverDnXgesvdjGetSweeps

    global __cusolverDnSgesvdjBatched_bufferSize
    data["__cusolverDnSgesvdjBatched_bufferSize"] = <intptr_t>__cusolverDnSgesvdjBatched_bufferSize

    global __cusolverDnDgesvdjBatched_bufferSize
    data["__cusolverDnDgesvdjBatched_bufferSize"] = <intptr_t>__cusolverDnDgesvdjBatched_bufferSize

    global __cusolverDnCgesvdjBatched_bufferSize
    data["__cusolverDnCgesvdjBatched_bufferSize"] = <intptr_t>__cusolverDnCgesvdjBatched_bufferSize

    global __cusolverDnZgesvdjBatched_bufferSize
    data["__cusolverDnZgesvdjBatched_bufferSize"] = <intptr_t>__cusolverDnZgesvdjBatched_bufferSize

    global __cusolverDnSgesvdjBatched
    data["__cusolverDnSgesvdjBatched"] = <intptr_t>__cusolverDnSgesvdjBatched

    global __cusolverDnDgesvdjBatched
    data["__cusolverDnDgesvdjBatched"] = <intptr_t>__cusolverDnDgesvdjBatched

    global __cusolverDnCgesvdjBatched
    data["__cusolverDnCgesvdjBatched"] = <intptr_t>__cusolverDnCgesvdjBatched

    global __cusolverDnZgesvdjBatched
    data["__cusolverDnZgesvdjBatched"] = <intptr_t>__cusolverDnZgesvdjBatched

    global __cusolverDnSgesvdj_bufferSize
    data["__cusolverDnSgesvdj_bufferSize"] = <intptr_t>__cusolverDnSgesvdj_bufferSize

    global __cusolverDnDgesvdj_bufferSize
    data["__cusolverDnDgesvdj_bufferSize"] = <intptr_t>__cusolverDnDgesvdj_bufferSize

    global __cusolverDnCgesvdj_bufferSize
    data["__cusolverDnCgesvdj_bufferSize"] = <intptr_t>__cusolverDnCgesvdj_bufferSize

    global __cusolverDnZgesvdj_bufferSize
    data["__cusolverDnZgesvdj_bufferSize"] = <intptr_t>__cusolverDnZgesvdj_bufferSize

    global __cusolverDnSgesvdj
    data["__cusolverDnSgesvdj"] = <intptr_t>__cusolverDnSgesvdj

    global __cusolverDnDgesvdj
    data["__cusolverDnDgesvdj"] = <intptr_t>__cusolverDnDgesvdj

    global __cusolverDnCgesvdj
    data["__cusolverDnCgesvdj"] = <intptr_t>__cusolverDnCgesvdj

    global __cusolverDnZgesvdj
    data["__cusolverDnZgesvdj"] = <intptr_t>__cusolverDnZgesvdj

    global __cusolverDnSgesvdaStridedBatched_bufferSize
    data["__cusolverDnSgesvdaStridedBatched_bufferSize"] = <intptr_t>__cusolverDnSgesvdaStridedBatched_bufferSize

    global __cusolverDnDgesvdaStridedBatched_bufferSize
    data["__cusolverDnDgesvdaStridedBatched_bufferSize"] = <intptr_t>__cusolverDnDgesvdaStridedBatched_bufferSize

    global __cusolverDnCgesvdaStridedBatched_bufferSize
    data["__cusolverDnCgesvdaStridedBatched_bufferSize"] = <intptr_t>__cusolverDnCgesvdaStridedBatched_bufferSize

    global __cusolverDnZgesvdaStridedBatched_bufferSize
    data["__cusolverDnZgesvdaStridedBatched_bufferSize"] = <intptr_t>__cusolverDnZgesvdaStridedBatched_bufferSize

    global __cusolverDnSgesvdaStridedBatched
    data["__cusolverDnSgesvdaStridedBatched"] = <intptr_t>__cusolverDnSgesvdaStridedBatched

    global __cusolverDnDgesvdaStridedBatched
    data["__cusolverDnDgesvdaStridedBatched"] = <intptr_t>__cusolverDnDgesvdaStridedBatched

    global __cusolverDnCgesvdaStridedBatched
    data["__cusolverDnCgesvdaStridedBatched"] = <intptr_t>__cusolverDnCgesvdaStridedBatched

    global __cusolverDnZgesvdaStridedBatched
    data["__cusolverDnZgesvdaStridedBatched"] = <intptr_t>__cusolverDnZgesvdaStridedBatched

    global __cusolverDnCreateParams
    data["__cusolverDnCreateParams"] = <intptr_t>__cusolverDnCreateParams

    global __cusolverDnDestroyParams
    data["__cusolverDnDestroyParams"] = <intptr_t>__cusolverDnDestroyParams

    global __cusolverDnSetAdvOptions
    data["__cusolverDnSetAdvOptions"] = <intptr_t>__cusolverDnSetAdvOptions

    global __cusolverDnXpotrf_bufferSize
    data["__cusolverDnXpotrf_bufferSize"] = <intptr_t>__cusolverDnXpotrf_bufferSize

    global __cusolverDnXpotrf
    data["__cusolverDnXpotrf"] = <intptr_t>__cusolverDnXpotrf

    global __cusolverDnXpotrs
    data["__cusolverDnXpotrs"] = <intptr_t>__cusolverDnXpotrs

    global __cusolverDnXgeqrf_bufferSize
    data["__cusolverDnXgeqrf_bufferSize"] = <intptr_t>__cusolverDnXgeqrf_bufferSize

    global __cusolverDnXgeqrf
    data["__cusolverDnXgeqrf"] = <intptr_t>__cusolverDnXgeqrf

    global __cusolverDnXgetrf_bufferSize
    data["__cusolverDnXgetrf_bufferSize"] = <intptr_t>__cusolverDnXgetrf_bufferSize

    global __cusolverDnXgetrf
    data["__cusolverDnXgetrf"] = <intptr_t>__cusolverDnXgetrf

    global __cusolverDnXgetrs
    data["__cusolverDnXgetrs"] = <intptr_t>__cusolverDnXgetrs

    global __cusolverDnXsyevd_bufferSize
    data["__cusolverDnXsyevd_bufferSize"] = <intptr_t>__cusolverDnXsyevd_bufferSize

    global __cusolverDnXsyevd
    data["__cusolverDnXsyevd"] = <intptr_t>__cusolverDnXsyevd

    global __cusolverDnXsyevdx_bufferSize
    data["__cusolverDnXsyevdx_bufferSize"] = <intptr_t>__cusolverDnXsyevdx_bufferSize

    global __cusolverDnXsyevdx
    data["__cusolverDnXsyevdx"] = <intptr_t>__cusolverDnXsyevdx

    global __cusolverDnXgesvd_bufferSize
    data["__cusolverDnXgesvd_bufferSize"] = <intptr_t>__cusolverDnXgesvd_bufferSize

    global __cusolverDnXgesvd
    data["__cusolverDnXgesvd"] = <intptr_t>__cusolverDnXgesvd

    global __cusolverDnXgesvdp_bufferSize
    data["__cusolverDnXgesvdp_bufferSize"] = <intptr_t>__cusolverDnXgesvdp_bufferSize

    global __cusolverDnXgesvdp
    data["__cusolverDnXgesvdp"] = <intptr_t>__cusolverDnXgesvdp

    global __cusolverDnXgesvdr_bufferSize
    data["__cusolverDnXgesvdr_bufferSize"] = <intptr_t>__cusolverDnXgesvdr_bufferSize

    global __cusolverDnXgesvdr
    data["__cusolverDnXgesvdr"] = <intptr_t>__cusolverDnXgesvdr

    global __cusolverDnXsytrs_bufferSize
    data["__cusolverDnXsytrs_bufferSize"] = <intptr_t>__cusolverDnXsytrs_bufferSize

    global __cusolverDnXsytrs
    data["__cusolverDnXsytrs"] = <intptr_t>__cusolverDnXsytrs

    global __cusolverDnXtrtri_bufferSize
    data["__cusolverDnXtrtri_bufferSize"] = <intptr_t>__cusolverDnXtrtri_bufferSize

    global __cusolverDnXtrtri
    data["__cusolverDnXtrtri"] = <intptr_t>__cusolverDnXtrtri

    global __cusolverDnLoggerSetCallback
    data["__cusolverDnLoggerSetCallback"] = <intptr_t>__cusolverDnLoggerSetCallback

    global __cusolverDnLoggerSetFile
    data["__cusolverDnLoggerSetFile"] = <intptr_t>__cusolverDnLoggerSetFile

    global __cusolverDnLoggerOpenFile
    data["__cusolverDnLoggerOpenFile"] = <intptr_t>__cusolverDnLoggerOpenFile

    global __cusolverDnLoggerSetLevel
    data["__cusolverDnLoggerSetLevel"] = <intptr_t>__cusolverDnLoggerSetLevel

    global __cusolverDnLoggerSetMask
    data["__cusolverDnLoggerSetMask"] = <intptr_t>__cusolverDnLoggerSetMask

    global __cusolverDnLoggerForceDisable
    data["__cusolverDnLoggerForceDisable"] = <intptr_t>__cusolverDnLoggerForceDisable

    global __cusolverDnSetDeterministicMode
    data["__cusolverDnSetDeterministicMode"] = <intptr_t>__cusolverDnSetDeterministicMode

    global __cusolverDnGetDeterministicMode
    data["__cusolverDnGetDeterministicMode"] = <intptr_t>__cusolverDnGetDeterministicMode

    global __cusolverDnXlarft_bufferSize
    data["__cusolverDnXlarft_bufferSize"] = <intptr_t>__cusolverDnXlarft_bufferSize

    global __cusolverDnXlarft
    data["__cusolverDnXlarft"] = <intptr_t>__cusolverDnXlarft

    global __cusolverDnXsyevBatched_bufferSize
    data["__cusolverDnXsyevBatched_bufferSize"] = <intptr_t>__cusolverDnXsyevBatched_bufferSize

    global __cusolverDnXsyevBatched
    data["__cusolverDnXsyevBatched"] = <intptr_t>__cusolverDnXsyevBatched

    global __cusolverDnXgeev_bufferSize
    data["__cusolverDnXgeev_bufferSize"] = <intptr_t>__cusolverDnXgeev_bufferSize

    global __cusolverDnXgeev
    data["__cusolverDnXgeev"] = <intptr_t>__cusolverDnXgeev

    func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]


###############################################################################
# Wrapper functions
###############################################################################

cdef cusolverStatus_t _cusolverDnCreate(cusolverDnHandle_t* handle) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCreate
    _check_or_init_cusolverDn()
    if __cusolverDnCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCreate is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t*) noexcept nogil>__cusolverDnCreate)(
        handle)


cdef cusolverStatus_t _cusolverDnDestroy(cusolverDnHandle_t handle) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDestroy
    _check_or_init_cusolverDn()
    if __cusolverDnDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDestroy is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t) noexcept nogil>__cusolverDnDestroy)(
        handle)


cdef cusolverStatus_t _cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSetStream
    _check_or_init_cusolverDn()
    if __cusolverDnSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSetStream is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cudaStream_t) noexcept nogil>__cusolverDnSetStream)(
        handle, streamId)


cdef cusolverStatus_t _cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t* streamId) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnGetStream
    _check_or_init_cusolverDn()
    if __cusolverDnGetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnGetStream is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cudaStream_t*) noexcept nogil>__cusolverDnGetStream)(
        handle, streamId)


cdef cusolverStatus_t _cusolverDnIRSParamsCreate(cusolverDnIRSParams_t* params_ptr) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsCreate
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsCreate is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t*) noexcept nogil>__cusolverDnIRSParamsCreate)(
        params_ptr)


cdef cusolverStatus_t _cusolverDnIRSParamsDestroy(cusolverDnIRSParams_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsDestroy
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsDestroy is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t) noexcept nogil>__cusolverDnIRSParamsDestroy)(
        params)


cdef cusolverStatus_t _cusolverDnIRSParamsSetRefinementSolver(cusolverDnIRSParams_t params, cusolverIRSRefinement_t refinement_solver) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsSetRefinementSolver
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetRefinementSolver == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetRefinementSolver is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverIRSRefinement_t) noexcept nogil>__cusolverDnIRSParamsSetRefinementSolver)(
        params, refinement_solver)


cdef cusolverStatus_t _cusolverDnIRSParamsSetSolverMainPrecision(cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsSetSolverMainPrecision
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetSolverMainPrecision == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetSolverMainPrecision is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverPrecType_t) noexcept nogil>__cusolverDnIRSParamsSetSolverMainPrecision)(
        params, solver_main_precision)


cdef cusolverStatus_t _cusolverDnIRSParamsSetSolverLowestPrecision(cusolverDnIRSParams_t params, cusolverPrecType_t solver_lowest_precision) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsSetSolverLowestPrecision
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetSolverLowestPrecision == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetSolverLowestPrecision is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverPrecType_t) noexcept nogil>__cusolverDnIRSParamsSetSolverLowestPrecision)(
        params, solver_lowest_precision)


cdef cusolverStatus_t _cusolverDnIRSParamsSetSolverPrecisions(cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision, cusolverPrecType_t solver_lowest_precision) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsSetSolverPrecisions
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetSolverPrecisions == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetSolverPrecisions is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverPrecType_t, cusolverPrecType_t) noexcept nogil>__cusolverDnIRSParamsSetSolverPrecisions)(
        params, solver_main_precision, solver_lowest_precision)


cdef cusolverStatus_t _cusolverDnIRSParamsSetTol(cusolverDnIRSParams_t params, double val) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsSetTol
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetTol == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetTol is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, double) noexcept nogil>__cusolverDnIRSParamsSetTol)(
        params, val)


cdef cusolverStatus_t _cusolverDnIRSParamsSetTolInner(cusolverDnIRSParams_t params, double val) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsSetTolInner
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetTolInner == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetTolInner is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, double) noexcept nogil>__cusolverDnIRSParamsSetTolInner)(
        params, val)


cdef cusolverStatus_t _cusolverDnIRSParamsSetMaxIters(cusolverDnIRSParams_t params, cusolver_int_t maxiters) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsSetMaxIters
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetMaxIters == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetMaxIters is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolver_int_t) noexcept nogil>__cusolverDnIRSParamsSetMaxIters)(
        params, maxiters)


cdef cusolverStatus_t _cusolverDnIRSParamsSetMaxItersInner(cusolverDnIRSParams_t params, cusolver_int_t maxiters_inner) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsSetMaxItersInner
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetMaxItersInner == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetMaxItersInner is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolver_int_t) noexcept nogil>__cusolverDnIRSParamsSetMaxItersInner)(
        params, maxiters_inner)


cdef cusolverStatus_t _cusolverDnIRSParamsGetMaxIters(cusolverDnIRSParams_t params, cusolver_int_t* maxiters) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsGetMaxIters
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsGetMaxIters == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsGetMaxIters is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolver_int_t*) noexcept nogil>__cusolverDnIRSParamsGetMaxIters)(
        params, maxiters)


cdef cusolverStatus_t _cusolverDnIRSParamsEnableFallback(cusolverDnIRSParams_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsEnableFallback
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsEnableFallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsEnableFallback is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t) noexcept nogil>__cusolverDnIRSParamsEnableFallback)(
        params)


cdef cusolverStatus_t _cusolverDnIRSParamsDisableFallback(cusolverDnIRSParams_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSParamsDisableFallback
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsDisableFallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsDisableFallback is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t) noexcept nogil>__cusolverDnIRSParamsDisableFallback)(
        params)


cdef cusolverStatus_t _cusolverDnIRSInfosDestroy(cusolverDnIRSInfos_t infos) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSInfosDestroy
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosDestroy is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t) noexcept nogil>__cusolverDnIRSInfosDestroy)(
        infos)


cdef cusolverStatus_t _cusolverDnIRSInfosCreate(cusolverDnIRSInfos_t* infos_ptr) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSInfosCreate
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosCreate is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t*) noexcept nogil>__cusolverDnIRSInfosCreate)(
        infos_ptr)


cdef cusolverStatus_t _cusolverDnIRSInfosGetNiters(cusolverDnIRSInfos_t infos, cusolver_int_t* niters) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSInfosGetNiters
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosGetNiters == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosGetNiters is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t, cusolver_int_t*) noexcept nogil>__cusolverDnIRSInfosGetNiters)(
        infos, niters)


cdef cusolverStatus_t _cusolverDnIRSInfosGetOuterNiters(cusolverDnIRSInfos_t infos, cusolver_int_t* outer_niters) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSInfosGetOuterNiters
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosGetOuterNiters == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosGetOuterNiters is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t, cusolver_int_t*) noexcept nogil>__cusolverDnIRSInfosGetOuterNiters)(
        infos, outer_niters)


cdef cusolverStatus_t _cusolverDnIRSInfosRequestResidual(cusolverDnIRSInfos_t infos) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSInfosRequestResidual
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosRequestResidual == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosRequestResidual is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t) noexcept nogil>__cusolverDnIRSInfosRequestResidual)(
        infos)


cdef cusolverStatus_t _cusolverDnIRSInfosGetResidualHistory(cusolverDnIRSInfos_t infos, void** residual_history) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSInfosGetResidualHistory
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosGetResidualHistory == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosGetResidualHistory is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t, void**) noexcept nogil>__cusolverDnIRSInfosGetResidualHistory)(
        infos, residual_history)


cdef cusolverStatus_t _cusolverDnIRSInfosGetMaxIters(cusolverDnIRSInfos_t infos, cusolver_int_t* maxiters) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSInfosGetMaxIters
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosGetMaxIters == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosGetMaxIters is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t, cusolver_int_t*) noexcept nogil>__cusolverDnIRSInfosGetMaxIters)(
        infos, maxiters)


cdef cusolverStatus_t _cusolverDnZZgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZZgesv
    _check_or_init_cusolverDn()
    if __cusolverDnZZgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZZgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnZZgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZCgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZCgesv
    _check_or_init_cusolverDn()
    if __cusolverDnZCgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZCgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnZCgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZKgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZKgesv
    _check_or_init_cusolverDn()
    if __cusolverDnZKgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZKgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnZKgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZEgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZEgesv
    _check_or_init_cusolverDn()
    if __cusolverDnZEgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZEgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnZEgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZYgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZYgesv
    _check_or_init_cusolverDn()
    if __cusolverDnZYgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZYgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnZYgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCCgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCCgesv
    _check_or_init_cusolverDn()
    if __cusolverDnCCgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCCgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnCCgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCEgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCEgesv
    _check_or_init_cusolverDn()
    if __cusolverDnCEgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCEgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnCEgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCKgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCKgesv
    _check_or_init_cusolverDn()
    if __cusolverDnCKgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCKgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnCKgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCYgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCYgesv
    _check_or_init_cusolverDn()
    if __cusolverDnCYgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCYgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnCYgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDDgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDDgesv
    _check_or_init_cusolverDn()
    if __cusolverDnDDgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDDgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnDDgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDSgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDSgesv
    _check_or_init_cusolverDn()
    if __cusolverDnDSgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDSgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnDSgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDHgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDHgesv
    _check_or_init_cusolverDn()
    if __cusolverDnDHgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDHgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnDHgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDBgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDBgesv
    _check_or_init_cusolverDn()
    if __cusolverDnDBgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDBgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnDBgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDXgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDXgesv
    _check_or_init_cusolverDn()
    if __cusolverDnDXgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDXgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnDXgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSSgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSSgesv
    _check_or_init_cusolverDn()
    if __cusolverDnSSgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSSgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnSSgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSHgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSHgesv
    _check_or_init_cusolverDn()
    if __cusolverDnSHgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSHgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnSHgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSBgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSBgesv
    _check_or_init_cusolverDn()
    if __cusolverDnSBgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSBgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnSBgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSXgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSXgesv
    _check_or_init_cusolverDn()
    if __cusolverDnSXgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSXgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnSXgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZZgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZZgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZZgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZZgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnZZgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZCgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZCgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZCgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZCgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnZCgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZKgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZKgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZKgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZKgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnZKgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZEgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZEgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZEgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZEgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnZEgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZYgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZYgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZYgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZYgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnZYgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCCgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCCgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCCgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCCgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnCCgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCKgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCKgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCKgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCKgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnCKgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCEgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCEgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCEgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCEgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnCEgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCYgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCYgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCYgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCYgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnCYgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDDgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDDgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDDgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDDgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnDDgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDSgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDSgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDSgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDSgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnDSgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDHgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDHgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDHgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDHgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnDHgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDBgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDBgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDBgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDBgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnDBgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDXgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDXgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDXgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDXgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnDXgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSSgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSSgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSSgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSSgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnSSgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSHgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSHgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSHgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSHgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnSHgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSBgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSBgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSBgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSBgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnSBgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSXgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSXgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSXgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSXgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnSXgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZZgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZZgels
    _check_or_init_cusolverDn()
    if __cusolverDnZZgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZZgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnZZgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZCgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZCgels
    _check_or_init_cusolverDn()
    if __cusolverDnZCgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZCgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnZCgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZKgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZKgels
    _check_or_init_cusolverDn()
    if __cusolverDnZKgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZKgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnZKgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZEgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZEgels
    _check_or_init_cusolverDn()
    if __cusolverDnZEgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZEgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnZEgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZYgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZYgels
    _check_or_init_cusolverDn()
    if __cusolverDnZYgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZYgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnZYgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCCgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCCgels
    _check_or_init_cusolverDn()
    if __cusolverDnCCgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCCgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnCCgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCKgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCKgels
    _check_or_init_cusolverDn()
    if __cusolverDnCKgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCKgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnCKgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCEgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCEgels
    _check_or_init_cusolverDn()
    if __cusolverDnCEgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCEgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnCEgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCYgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCYgels
    _check_or_init_cusolverDn()
    if __cusolverDnCYgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCYgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnCYgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDDgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDDgels
    _check_or_init_cusolverDn()
    if __cusolverDnDDgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDDgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnDDgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDSgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDSgels
    _check_or_init_cusolverDn()
    if __cusolverDnDSgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDSgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnDSgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDHgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDHgels
    _check_or_init_cusolverDn()
    if __cusolverDnDHgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDHgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnDHgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDBgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDBgels
    _check_or_init_cusolverDn()
    if __cusolverDnDBgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDBgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnDBgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDXgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDXgels
    _check_or_init_cusolverDn()
    if __cusolverDnDXgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDXgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnDXgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSSgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSSgels
    _check_or_init_cusolverDn()
    if __cusolverDnSSgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSSgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnSSgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSHgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSHgels
    _check_or_init_cusolverDn()
    if __cusolverDnSHgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSHgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnSHgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSBgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSBgels
    _check_or_init_cusolverDn()
    if __cusolverDnSBgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSBgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnSBgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSXgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSXgels
    _check_or_init_cusolverDn()
    if __cusolverDnSXgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSXgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnSXgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZZgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZZgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZZgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZZgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnZZgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZCgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZCgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZCgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZCgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnZCgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZKgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZKgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZKgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZKgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnZKgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZEgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZEgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZEgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZEgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnZEgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZYgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZYgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZYgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZYgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnZYgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCCgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCCgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCCgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCCgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnCCgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCKgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCKgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCKgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCKgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnCKgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCEgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCEgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCEgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCEgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnCEgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCYgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCYgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCYgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCYgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnCYgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDDgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDDgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDDgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDDgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnDDgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDSgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDSgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDSgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDSgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnDSgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDHgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDHgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDHgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDHgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnDHgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDBgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDBgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDBgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDBgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnDBgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDXgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDXgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDXgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDXgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnDXgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSSgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSSgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSSgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSSgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnSSgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSHgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSHgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSHgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSHgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnSHgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSBgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSBgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSBgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSBgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnSBgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSXgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSXgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSXgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSXgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) noexcept nogil>__cusolverDnSXgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnIRSXgesv(cusolverDnHandle_t handle, cusolverDnIRSParams_t gesv_irs_params, cusolverDnIRSInfos_t gesv_irs_infos, cusolver_int_t n, cusolver_int_t nrhs, void* dA, cusolver_int_t ldda, void* dB, cusolver_int_t lddb, void* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* niters, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSXgesv
    _check_or_init_cusolverDn()
    if __cusolverDnIRSXgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSXgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t, cusolver_int_t, cusolver_int_t, void*, cusolver_int_t, void*, cusolver_int_t, void*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnIRSXgesv)(
        handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info)


cdef cusolverStatus_t _cusolverDnIRSXgesv_bufferSize(cusolverDnHandle_t handle, cusolverDnIRSParams_t params, cusolver_int_t n, cusolver_int_t nrhs, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSXgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnIRSXgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSXgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t, cusolver_int_t, size_t*) noexcept nogil>__cusolverDnIRSXgesv_bufferSize)(
        handle, params, n, nrhs, lwork_bytes)


cdef cusolverStatus_t _cusolverDnIRSXgels(cusolverDnHandle_t handle, cusolverDnIRSParams_t gels_irs_params, cusolverDnIRSInfos_t gels_irs_infos, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, void* dA, cusolver_int_t ldda, void* dB, cusolver_int_t lddb, void* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* niters, cusolver_int_t* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSXgels
    _check_or_init_cusolverDn()
    if __cusolverDnIRSXgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSXgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, void*, cusolver_int_t, void*, cusolver_int_t, void*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) noexcept nogil>__cusolverDnIRSXgels)(
        handle, gels_irs_params, gels_irs_infos, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info)


cdef cusolverStatus_t _cusolverDnIRSXgels_bufferSize(cusolverDnHandle_t handle, cusolverDnIRSParams_t params, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, size_t* lwork_bytes) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnIRSXgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnIRSXgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSXgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, size_t*) noexcept nogil>__cusolverDnIRSXgels_bufferSize)(
        handle, params, m, n, nrhs, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSpotrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSpotrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, int*) noexcept nogil>__cusolverDnSpotrf_bufferSize)(
        handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDpotrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDpotrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, int*) noexcept nogil>__cusolverDnDpotrf_bufferSize)(
        handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCpotrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCpotrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnCpotrf_bufferSize)(
        handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZpotrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZpotrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZpotrf_bufferSize)(
        handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* Workspace, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSpotrf
    _check_or_init_cusolverDn()
    if __cusolverDnSpotrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, float*, int, int*) noexcept nogil>__cusolverDnSpotrf)(
        handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* Workspace, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDpotrf
    _check_or_init_cusolverDn()
    if __cusolverDnDpotrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, double*, int, int*) noexcept nogil>__cusolverDnDpotrf)(
        handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnCpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* Workspace, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCpotrf
    _check_or_init_cusolverDn()
    if __cusolverDnCpotrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnCpotrf)(
        handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* Workspace, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZpotrf
    _check_or_init_cusolverDn()
    if __cusolverDnZpotrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZpotrf)(
        handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float* A, int lda, float* B, int ldb, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSpotrs
    _check_or_init_cusolverDn()
    if __cusolverDnSpotrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const float*, int, float*, int, int*) noexcept nogil>__cusolverDnSpotrs)(
        handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDpotrs
    _check_or_init_cusolverDn()
    if __cusolverDnDpotrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const double*, int, double*, int, int*) noexcept nogil>__cusolverDnDpotrs)(
        handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnCpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuComplex* A, int lda, cuComplex* B, int ldb, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCpotrs
    _check_or_init_cusolverDn()
    if __cusolverDnCpotrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const cuComplex*, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnCpotrs)(
        handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnZpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZpotrs
    _check_or_init_cusolverDn()
    if __cusolverDnZpotrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const cuDoubleComplex*, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZpotrs)(
        handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnSpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* Aarray[], int lda, int* infoArray, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSpotrfBatched
    _check_or_init_cusolverDn()
    if __cusolverDnSpotrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotrfBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float**, int, int*, int) noexcept nogil>__cusolverDnSpotrfBatched)(
        handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t _cusolverDnDpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* Aarray[], int lda, int* infoArray, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDpotrfBatched
    _check_or_init_cusolverDn()
    if __cusolverDnDpotrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotrfBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double**, int, int*, int) noexcept nogil>__cusolverDnDpotrfBatched)(
        handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t _cusolverDnCpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* Aarray[], int lda, int* infoArray, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCpotrfBatched
    _check_or_init_cusolverDn()
    if __cusolverDnCpotrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotrfBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex**, int, int*, int) noexcept nogil>__cusolverDnCpotrfBatched)(
        handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t _cusolverDnZpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* Aarray[], int lda, int* infoArray, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZpotrfBatched
    _check_or_init_cusolverDn()
    if __cusolverDnZpotrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotrfBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex**, int, int*, int) noexcept nogil>__cusolverDnZpotrfBatched)(
        handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t _cusolverDnSpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, float* A[], int lda, float* B[], int ldb, int* d_info, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSpotrsBatched
    _check_or_init_cusolverDn()
    if __cusolverDnSpotrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotrsBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, float**, int, float**, int, int*, int) noexcept nogil>__cusolverDnSpotrsBatched)(
        handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t _cusolverDnDpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, double* A[], int lda, double* B[], int ldb, int* d_info, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDpotrsBatched
    _check_or_init_cusolverDn()
    if __cusolverDnDpotrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotrsBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, double**, int, double**, int, int*, int) noexcept nogil>__cusolverDnDpotrsBatched)(
        handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t _cusolverDnCpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, cuComplex* A[], int lda, cuComplex* B[], int ldb, int* d_info, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCpotrsBatched
    _check_or_init_cusolverDn()
    if __cusolverDnCpotrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotrsBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, cuComplex**, int, cuComplex**, int, int*, int) noexcept nogil>__cusolverDnCpotrsBatched)(
        handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t _cusolverDnZpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, cuDoubleComplex* A[], int lda, cuDoubleComplex* B[], int ldb, int* d_info, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZpotrsBatched
    _check_or_init_cusolverDn()
    if __cusolverDnZpotrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotrsBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, cuDoubleComplex**, int, cuDoubleComplex**, int, int*, int) noexcept nogil>__cusolverDnZpotrsBatched)(
        handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t _cusolverDnSpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSpotri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSpotri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, int*) noexcept nogil>__cusolverDnSpotri_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnDpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDpotri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDpotri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, int*) noexcept nogil>__cusolverDnDpotri_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnCpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCpotri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCpotri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnCpotri_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnZpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZpotri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZpotri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZpotri_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnSpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSpotri
    _check_or_init_cusolverDn()
    if __cusolverDnSpotri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, float*, int, int*) noexcept nogil>__cusolverDnSpotri)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDpotri
    _check_or_init_cusolverDn()
    if __cusolverDnDpotri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, double*, int, int*) noexcept nogil>__cusolverDnDpotri)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnCpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCpotri
    _check_or_init_cusolverDn()
    if __cusolverDnCpotri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnCpotri)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZpotri
    _check_or_init_cusolverDn()
    if __cusolverDnZpotri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZpotri)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSlauum_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSlauum_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSlauum_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, int*) noexcept nogil>__cusolverDnSlauum_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnDlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDlauum_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDlauum_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDlauum_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, int*) noexcept nogil>__cusolverDnDlauum_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnClauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnClauum_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnClauum_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnClauum_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnClauum_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnZlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZlauum_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZlauum_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZlauum_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZlauum_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnSlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSlauum
    _check_or_init_cusolverDn()
    if __cusolverDnSlauum == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSlauum is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, float*, int, int*) noexcept nogil>__cusolverDnSlauum)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDlauum
    _check_or_init_cusolverDn()
    if __cusolverDnDlauum == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDlauum is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, double*, int, int*) noexcept nogil>__cusolverDnDlauum)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnClauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnClauum
    _check_or_init_cusolverDn()
    if __cusolverDnClauum == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnClauum is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnClauum)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZlauum
    _check_or_init_cusolverDn()
    if __cusolverDnZlauum == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZlauum is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZlauum)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgetrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgetrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgetrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float*, int, int*) noexcept nogil>__cusolverDnSgetrf_bufferSize)(
        handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgetrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgetrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgetrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, int*) noexcept nogil>__cusolverDnDgetrf_bufferSize)(
        handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgetrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgetrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgetrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnCgetrf_bufferSize)(
        handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgetrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgetrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgetrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZgetrf_bufferSize)(
        handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* Workspace, int* devIpiv, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgetrf
    _check_or_init_cusolverDn()
    if __cusolverDnSgetrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgetrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float*, int, float*, int*, int*) noexcept nogil>__cusolverDnSgetrf)(
        handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t _cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* Workspace, int* devIpiv, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgetrf
    _check_or_init_cusolverDn()
    if __cusolverDnDgetrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgetrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, double*, int*, int*) noexcept nogil>__cusolverDnDgetrf)(
        handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t _cusolverDnCgetrf(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, cuComplex* Workspace, int* devIpiv, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgetrf
    _check_or_init_cusolverDn()
    if __cusolverDnCgetrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgetrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex*, int, cuComplex*, int*, int*) noexcept nogil>__cusolverDnCgetrf)(
        handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t _cusolverDnZgetrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* Workspace, int* devIpiv, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgetrf
    _check_or_init_cusolverDn()
    if __cusolverDnZgetrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgetrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex*, int, cuDoubleComplex*, int*, int*) noexcept nogil>__cusolverDnZgetrf)(
        handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t _cusolverDnSlaswp(cusolverDnHandle_t handle, int n, float* A, int lda, int k1, int k2, const int* devIpiv, int incx) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSlaswp
    _check_or_init_cusolverDn()
    if __cusolverDnSlaswp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSlaswp is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, float*, int, int, int, const int*, int) noexcept nogil>__cusolverDnSlaswp)(
        handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t _cusolverDnDlaswp(cusolverDnHandle_t handle, int n, double* A, int lda, int k1, int k2, const int* devIpiv, int incx) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDlaswp
    _check_or_init_cusolverDn()
    if __cusolverDnDlaswp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDlaswp is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, double*, int, int, int, const int*, int) noexcept nogil>__cusolverDnDlaswp)(
        handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t _cusolverDnClaswp(cusolverDnHandle_t handle, int n, cuComplex* A, int lda, int k1, int k2, const int* devIpiv, int incx) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnClaswp
    _check_or_init_cusolverDn()
    if __cusolverDnClaswp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnClaswp is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, cuComplex*, int, int, int, const int*, int) noexcept nogil>__cusolverDnClaswp)(
        handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t _cusolverDnZlaswp(cusolverDnHandle_t handle, int n, cuDoubleComplex* A, int lda, int k1, int k2, const int* devIpiv, int incx) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZlaswp
    _check_or_init_cusolverDn()
    if __cusolverDnZlaswp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZlaswp is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, cuDoubleComplex*, int, int, int, const int*, int) noexcept nogil>__cusolverDnZlaswp)(
        handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t _cusolverDnSgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgetrs
    _check_or_init_cusolverDn()
    if __cusolverDnSgetrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgetrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const float*, int, const int*, float*, int, int*) noexcept nogil>__cusolverDnSgetrs)(
        handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnDgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgetrs
    _check_or_init_cusolverDn()
    if __cusolverDnDgetrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgetrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const double*, int, const int*, double*, int, int*) noexcept nogil>__cusolverDnDgetrs)(
        handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnCgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* A, int lda, const int* devIpiv, cuComplex* B, int ldb, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgetrs
    _check_or_init_cusolverDn()
    if __cusolverDnCgetrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgetrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const cuComplex*, int, const int*, cuComplex*, int, int*) noexcept nogil>__cusolverDnCgetrs)(
        handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnZgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* A, int lda, const int* devIpiv, cuDoubleComplex* B, int ldb, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgetrs
    _check_or_init_cusolverDn()
    if __cusolverDnZgetrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgetrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const cuDoubleComplex*, int, const int*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZgetrs)(
        handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgeqrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgeqrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgeqrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float*, int, int*) noexcept nogil>__cusolverDnSgeqrf_bufferSize)(
        handle, m, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgeqrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgeqrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgeqrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, int*) noexcept nogil>__cusolverDnDgeqrf_bufferSize)(
        handle, m, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnCgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgeqrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgeqrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgeqrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnCgeqrf_bufferSize)(
        handle, m, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnZgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgeqrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgeqrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgeqrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZgeqrf_bufferSize)(
        handle, m, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* TAU, float* Workspace, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgeqrf
    _check_or_init_cusolverDn()
    if __cusolverDnSgeqrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgeqrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float*, int, float*, float*, int, int*) noexcept nogil>__cusolverDnSgeqrf)(
        handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* TAU, double* Workspace, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgeqrf
    _check_or_init_cusolverDn()
    if __cusolverDnDgeqrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgeqrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, double*, double*, int, int*) noexcept nogil>__cusolverDnDgeqrf)(
        handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnCgeqrf(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, cuComplex* TAU, cuComplex* Workspace, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgeqrf
    _check_or_init_cusolverDn()
    if __cusolverDnCgeqrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgeqrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex*, int, cuComplex*, cuComplex*, int, int*) noexcept nogil>__cusolverDnCgeqrf)(
        handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZgeqrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* TAU, cuDoubleComplex* Workspace, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgeqrf
    _check_or_init_cusolverDn()
    if __cusolverDnZgeqrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgeqrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex*, int, cuDoubleComplex*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZgeqrf)(
        handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const float* A, int lda, const float* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSorgqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSorgqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const float*, int, const float*, int*) noexcept nogil>__cusolverDnSorgqr_bufferSize)(
        handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnDorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const double* A, int lda, const double* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDorgqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDorgqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const double*, int, const double*, int*) noexcept nogil>__cusolverDnDorgqr_bufferSize)(
        handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnCungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCungqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCungqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const cuComplex*, int, const cuComplex*, int*) noexcept nogil>__cusolverDnCungqr_bufferSize)(
        handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnZungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZungqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZungqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int*) noexcept nogil>__cusolverDnZungqr_bufferSize)(
        handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnSorgqr(cusolverDnHandle_t handle, int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSorgqr
    _check_or_init_cusolverDn()
    if __cusolverDnSorgqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, float*, int, const float*, float*, int, int*) noexcept nogil>__cusolverDnSorgqr)(
        handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDorgqr(cusolverDnHandle_t handle, int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDorgqr
    _check_or_init_cusolverDn()
    if __cusolverDnDorgqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, double*, int, const double*, double*, int, int*) noexcept nogil>__cusolverDnDorgqr)(
        handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCungqr(cusolverDnHandle_t handle, int m, int n, int k, cuComplex* A, int lda, const cuComplex* tau, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCungqr
    _check_or_init_cusolverDn()
    if __cusolverDnCungqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, cuComplex*, int, const cuComplex*, cuComplex*, int, int*) noexcept nogil>__cusolverDnCungqr)(
        handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZungqr(cusolverDnHandle_t handle, int m, int n, int k, cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZungqr
    _check_or_init_cusolverDn()
    if __cusolverDnZungqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZungqr)(
        handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float* A, int lda, const float* tau, const float* C, int ldc, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSormqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSormqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSormqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const float*, int, const float*, const float*, int, int*) noexcept nogil>__cusolverDnSormqr_bufferSize)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double* A, int lda, const double* tau, const double* C, int ldc, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDormqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDormqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDormqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const double*, int, const double*, const double*, int, int*) noexcept nogil>__cusolverDnDormqr_bufferSize)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnCunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, const cuComplex* C, int ldc, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCunmqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCunmqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCunmqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const cuComplex*, int, const cuComplex*, const cuComplex*, int, int*) noexcept nogil>__cusolverDnCunmqr_bufferSize)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnZunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, const cuDoubleComplex* C, int ldc, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZunmqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZunmqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZunmqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, const cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZunmqr_bufferSize)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnSormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, float* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSormqr
    _check_or_init_cusolverDn()
    if __cusolverDnSormqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSormqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const float*, int, const float*, float*, int, float*, int, int*) noexcept nogil>__cusolverDnSormqr)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, double* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDormqr
    _check_or_init_cusolverDn()
    if __cusolverDnDormqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDormqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const double*, int, const double*, double*, int, double*, int, int*) noexcept nogil>__cusolverDnDormqr)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnCunmqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, cuComplex* C, int ldc, cuComplex* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCunmqr
    _check_or_init_cusolverDn()
    if __cusolverDnCunmqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCunmqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const cuComplex*, int, const cuComplex*, cuComplex*, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnCunmqr)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZunmqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* C, int ldc, cuDoubleComplex* work, int lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZunmqr
    _check_or_init_cusolverDn()
    if __cusolverDnZunmqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZunmqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZunmqr)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSsytrf_bufferSize(cusolverDnHandle_t handle, int n, float* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsytrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsytrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, float*, int, int*) noexcept nogil>__cusolverDnSsytrf_bufferSize)(
        handle, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnDsytrf_bufferSize(cusolverDnHandle_t handle, int n, double* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsytrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsytrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, double*, int, int*) noexcept nogil>__cusolverDnDsytrf_bufferSize)(
        handle, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnCsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuComplex* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCsytrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCsytrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCsytrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnCsytrf_bufferSize)(
        handle, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnZsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuDoubleComplex* A, int lda, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZsytrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZsytrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZsytrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZsytrf_bufferSize)(
        handle, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnSsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* ipiv, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsytrf
    _check_or_init_cusolverDn()
    if __cusolverDnSsytrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, int*, float*, int, int*) noexcept nogil>__cusolverDnSsytrf)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* ipiv, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsytrf
    _check_or_init_cusolverDn()
    if __cusolverDnDsytrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, int*, double*, int, int*) noexcept nogil>__cusolverDnDsytrf)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* ipiv, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCsytrf
    _check_or_init_cusolverDn()
    if __cusolverDnCsytrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCsytrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, int*, cuComplex*, int, int*) noexcept nogil>__cusolverDnCsytrf)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* ipiv, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZsytrf
    _check_or_init_cusolverDn()
    if __cusolverDnZsytrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZsytrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, int*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZsytrf)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, const int* ipiv, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsytri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsytri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, const int*, int*) noexcept nogil>__cusolverDnSsytri_bufferSize)(
        handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t _cusolverDnDsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, const int* ipiv, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsytri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsytri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, const int*, int*) noexcept nogil>__cusolverDnDsytri_bufferSize)(
        handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t _cusolverDnCsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, const int* ipiv, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCsytri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCsytri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCsytri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, const int*, int*) noexcept nogil>__cusolverDnCsytri_bufferSize)(
        handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t _cusolverDnZsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, const int* ipiv, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZsytri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZsytri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZsytri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, const int*, int*) noexcept nogil>__cusolverDnZsytri_bufferSize)(
        handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t _cusolverDnSsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, const int* ipiv, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsytri
    _check_or_init_cusolverDn()
    if __cusolverDnSsytri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, const int*, float*, int, int*) noexcept nogil>__cusolverDnSsytri)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, const int* ipiv, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsytri
    _check_or_init_cusolverDn()
    if __cusolverDnDsytri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, const int*, double*, int, int*) noexcept nogil>__cusolverDnDsytri)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, const int* ipiv, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCsytri
    _check_or_init_cusolverDn()
    if __cusolverDnCsytri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCsytri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, const int*, cuComplex*, int, int*) noexcept nogil>__cusolverDnCsytri)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, const int* ipiv, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZsytri
    _check_or_init_cusolverDn()
    if __cusolverDnZsytri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZsytri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, const int*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZsytri)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgebrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgebrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgebrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) noexcept nogil>__cusolverDnSgebrd_bufferSize)(
        handle, m, n, Lwork)


cdef cusolverStatus_t _cusolverDnDgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgebrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgebrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgebrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) noexcept nogil>__cusolverDnDgebrd_bufferSize)(
        handle, m, n, Lwork)


cdef cusolverStatus_t _cusolverDnCgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgebrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgebrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgebrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) noexcept nogil>__cusolverDnCgebrd_bufferSize)(
        handle, m, n, Lwork)


cdef cusolverStatus_t _cusolverDnZgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgebrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgebrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgebrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) noexcept nogil>__cusolverDnZgebrd_bufferSize)(
        handle, m, n, Lwork)


cdef cusolverStatus_t _cusolverDnSgebrd(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* D, float* E, float* TAUQ, float* TAUP, float* Work, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgebrd
    _check_or_init_cusolverDn()
    if __cusolverDnSgebrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgebrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float*, int, float*, float*, float*, float*, float*, int, int*) noexcept nogil>__cusolverDnSgebrd)(
        handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDgebrd(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* D, double* E, double* TAUQ, double* TAUP, double* Work, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgebrd
    _check_or_init_cusolverDn()
    if __cusolverDnDgebrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgebrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, double*, double*, double*, double*, double*, int, int*) noexcept nogil>__cusolverDnDgebrd)(
        handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnCgebrd(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, float* D, float* E, cuComplex* TAUQ, cuComplex* TAUP, cuComplex* Work, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgebrd
    _check_or_init_cusolverDn()
    if __cusolverDnCgebrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgebrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex*, int, float*, float*, cuComplex*, cuComplex*, cuComplex*, int, int*) noexcept nogil>__cusolverDnCgebrd)(
        handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZgebrd(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, double* D, double* E, cuDoubleComplex* TAUQ, cuDoubleComplex* TAUP, cuDoubleComplex* Work, int Lwork, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgebrd
    _check_or_init_cusolverDn()
    if __cusolverDnZgebrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgebrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex*, int, double*, double*, cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZgebrd)(
        handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSorgbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const float* A, int lda, const float* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSorgbr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSorgbr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgbr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const float*, int, const float*, int*) noexcept nogil>__cusolverDnSorgbr_bufferSize)(
        handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnDorgbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const double* A, int lda, const double* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDorgbr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDorgbr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgbr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const double*, int, const double*, int*) noexcept nogil>__cusolverDnDorgbr_bufferSize)(
        handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnCungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCungbr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCungbr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungbr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const cuComplex*, int, const cuComplex*, int*) noexcept nogil>__cusolverDnCungbr_bufferSize)(
        handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnZungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZungbr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZungbr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungbr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int*) noexcept nogil>__cusolverDnZungbr_bufferSize)(
        handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnSorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSorgbr
    _check_or_init_cusolverDn()
    if __cusolverDnSorgbr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgbr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, float*, int, const float*, float*, int, int*) noexcept nogil>__cusolverDnSorgbr)(
        handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDorgbr
    _check_or_init_cusolverDn()
    if __cusolverDnDorgbr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgbr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, double*, int, const double*, double*, int, int*) noexcept nogil>__cusolverDnDorgbr)(
        handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, cuComplex* A, int lda, const cuComplex* tau, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCungbr
    _check_or_init_cusolverDn()
    if __cusolverDnCungbr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungbr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, cuComplex*, int, const cuComplex*, cuComplex*, int, int*) noexcept nogil>__cusolverDnCungbr)(
        handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZungbr
    _check_or_init_cusolverDn()
    if __cusolverDnZungbr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungbr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZungbr)(
        handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, const float* d, const float* e, const float* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsytrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsytrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const float*, int, const float*, const float*, const float*, int*) noexcept nogil>__cusolverDnSsytrd_bufferSize)(
        handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t _cusolverDnDsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, const double* d, const double* e, const double* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsytrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsytrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const double*, int, const double*, const double*, const double*, int*) noexcept nogil>__cusolverDnDsytrd_bufferSize)(
        handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t _cusolverDnChetrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* d, const float* e, const cuComplex* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnChetrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnChetrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChetrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuComplex*, int, const float*, const float*, const cuComplex*, int*) noexcept nogil>__cusolverDnChetrd_bufferSize)(
        handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t _cusolverDnZhetrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* d, const double* e, const cuDoubleComplex* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZhetrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZhetrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhetrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const double*, const double*, const cuDoubleComplex*, int*) noexcept nogil>__cusolverDnZhetrd_bufferSize)(
        handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t _cusolverDnSsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* d, float* e, float* tau, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsytrd
    _check_or_init_cusolverDn()
    if __cusolverDnSsytrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, float*, float*, float*, float*, int, int*) noexcept nogil>__cusolverDnSsytrd)(
        handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* d, double* e, double* tau, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsytrd
    _check_or_init_cusolverDn()
    if __cusolverDnDsytrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, double*, double*, double*, double*, int, int*) noexcept nogil>__cusolverDnDsytrd)(
        handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnChetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* d, float* e, cuComplex* tau, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnChetrd
    _check_or_init_cusolverDn()
    if __cusolverDnChetrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChetrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, float*, float*, cuComplex*, cuComplex*, int, int*) noexcept nogil>__cusolverDnChetrd)(
        handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZhetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* d, double* e, cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZhetrd
    _check_or_init_cusolverDn()
    if __cusolverDnZhetrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhetrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, double*, double*, cuDoubleComplex*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZhetrd)(
        handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSorgtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, const float* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSorgtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSorgtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const float*, int, const float*, int*) noexcept nogil>__cusolverDnSorgtr_bufferSize)(
        handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnDorgtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, const double* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDorgtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDorgtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const double*, int, const double*, int*) noexcept nogil>__cusolverDnDorgtr_bufferSize)(
        handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnCungtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCungtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCungtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuComplex*, int, const cuComplex*, int*) noexcept nogil>__cusolverDnCungtr_bufferSize)(
        handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnZungtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZungtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZungtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int*) noexcept nogil>__cusolverDnZungtr_bufferSize)(
        handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnSorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, const float* tau, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSorgtr
    _check_or_init_cusolverDn()
    if __cusolverDnSorgtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, const float*, float*, int, int*) noexcept nogil>__cusolverDnSorgtr)(
        handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, const double* tau, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDorgtr
    _check_or_init_cusolverDn()
    if __cusolverDnDorgtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, const double*, double*, int, int*) noexcept nogil>__cusolverDnDorgtr)(
        handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, const cuComplex* tau, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCungtr
    _check_or_init_cusolverDn()
    if __cusolverDnCungtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, const cuComplex*, cuComplex*, int, int*) noexcept nogil>__cusolverDnCungtr)(
        handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZungtr
    _check_or_init_cusolverDn()
    if __cusolverDnZungtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZungtr)(
        handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const float* A, int lda, const float* tau, const float* C, int ldc, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSormtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSormtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSormtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, int, const float*, const float*, int, int*) noexcept nogil>__cusolverDnSormtr_bufferSize)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnDormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const double* A, int lda, const double* tau, const double* C, int ldc, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDormtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDormtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDormtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, int, const double*, const double*, int, int*) noexcept nogil>__cusolverDnDormtr_bufferSize)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnCunmtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const cuComplex* A, int lda, const cuComplex* tau, const cuComplex* C, int ldc, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCunmtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCunmtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCunmtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, int, const cuComplex*, const cuComplex*, int, int*) noexcept nogil>__cusolverDnCunmtr_bufferSize)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnZunmtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, const cuDoubleComplex* C, int ldc, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZunmtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZunmtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZunmtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, const cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZunmtr_bufferSize)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnSormtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, float* A, int lda, float* tau, float* C, int ldc, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSormtr
    _check_or_init_cusolverDn()
    if __cusolverDnSormtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSormtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, float*, int, float*, float*, int, float*, int, int*) noexcept nogil>__cusolverDnSormtr)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDormtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, double* A, int lda, double* tau, double* C, int ldc, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDormtr
    _check_or_init_cusolverDn()
    if __cusolverDnDormtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDormtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, double*, int, double*, double*, int, double*, int, int*) noexcept nogil>__cusolverDnDormtr)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCunmtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, cuComplex* A, int lda, cuComplex* tau, cuComplex* C, int ldc, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCunmtr
    _check_or_init_cusolverDn()
    if __cusolverDnCunmtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCunmtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, cuComplex*, int, cuComplex*, cuComplex*, int, cuComplex*, int, int*) noexcept nogil>__cusolverDnCunmtr)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZunmtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* tau, cuDoubleComplex* C, int ldc, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZunmtr
    _check_or_init_cusolverDn()
    if __cusolverDnZunmtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZunmtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, cuDoubleComplex*, int, cuDoubleComplex*, cuDoubleComplex*, int, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZunmtr)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgesvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) noexcept nogil>__cusolverDnSgesvd_bufferSize)(
        handle, m, n, lwork)


cdef cusolverStatus_t _cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgesvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) noexcept nogil>__cusolverDnDgesvd_bufferSize)(
        handle, m, n, lwork)


cdef cusolverStatus_t _cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgesvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) noexcept nogil>__cusolverDnCgesvd_bufferSize)(
        handle, m, n, lwork)


cdef cusolverStatus_t _cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgesvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) noexcept nogil>__cusolverDnZgesvd_bufferSize)(
        handle, m, n, lwork)


cdef cusolverStatus_t _cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* VT, int ldvt, float* work, int lwork, float* rwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgesvd
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, float*, int, float*, float*, int, float*, int, float*, int, float*, int*) noexcept nogil>__cusolverDnSgesvd)(
        handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t _cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt, double* work, int lwork, double* rwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgesvd
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, double*, int, double*, double*, int, double*, int, double*, int, double*, int*) noexcept nogil>__cusolverDnDgesvd)(
        handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t _cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* VT, int ldvt, cuComplex* work, int lwork, float* rwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgesvd
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, cuComplex*, int, float*, cuComplex*, int, cuComplex*, int, cuComplex*, int, float*, int*) noexcept nogil>__cusolverDnCgesvd)(
        handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t _cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* VT, int ldvt, cuDoubleComplex* work, int lwork, double* rwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgesvd
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, cuDoubleComplex*, int, cuDoubleComplex*, int, double*, int*) noexcept nogil>__cusolverDnZgesvd)(
        handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t _cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsyevd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const float*, int, const float*, int*) noexcept nogil>__cusolverDnSsyevd_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t _cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsyevd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const double*, int, const double*, int*) noexcept nogil>__cusolverDnDsyevd_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t _cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCheevd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCheevd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuComplex*, int, const float*, int*) noexcept nogil>__cusolverDnCheevd_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t _cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZheevd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZheevd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const double*, int*) noexcept nogil>__cusolverDnZheevd_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t _cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsyevd
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float*, int, float*, float*, int, int*) noexcept nogil>__cusolverDnSsyevd)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsyevd
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double*, int, double*, double*, int, int*) noexcept nogil>__cusolverDnDsyevd)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCheevd
    _check_or_init_cusolverDn()
    if __cusolverDnCheevd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex*, int, float*, cuComplex*, int, int*) noexcept nogil>__cusolverDnCheevd)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZheevd
    _check_or_init_cusolverDn()
    if __cusolverDnZheevd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZheevd)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float* A, int lda, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsyevdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const float*, int, float, float, int, int, int*, const float*, int*) noexcept nogil>__cusolverDnSsyevdx_bufferSize)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnDsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double* A, int lda, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsyevdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const double*, int, double, double, int, int, int*, const double*, int*) noexcept nogil>__cusolverDnDsyevdx_bufferSize)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnCheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCheevdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCheevdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const cuComplex*, int, float, float, int, int, int*, const float*, int*) noexcept nogil>__cusolverDnCheevdx_bufferSize)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnZheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZheevdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZheevdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const cuDoubleComplex*, int, double, double, int, int, int*, const double*, int*) noexcept nogil>__cusolverDnZheevdx_bufferSize)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnSsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, float* A, int lda, float vl, float vu, int il, int iu, int* meig, float* W, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsyevdx
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, float*, int, float, float, int, int, int*, float*, float*, int, int*) noexcept nogil>__cusolverDnSsyevdx)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, double* A, int lda, double vl, double vu, int il, int iu, int* meig, double* W, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsyevdx
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, double*, int, double, double, int, int, int*, double*, double*, int, int*) noexcept nogil>__cusolverDnDsyevdx)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float vl, float vu, int il, int iu, int* meig, float* W, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCheevdx
    _check_or_init_cusolverDn()
    if __cusolverDnCheevdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, cuComplex*, int, float, float, int, int, int*, float*, cuComplex*, int, int*) noexcept nogil>__cusolverDnCheevdx)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double vl, double vu, int il, int iu, int* meig, double* W, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZheevdx
    _check_or_init_cusolverDn()
    if __cusolverDnZheevdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, cuDoubleComplex*, int, double, double, int, int, int*, double*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZheevdx)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSsygvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsygvdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const float*, int, const float*, int, float, float, int, int, int*, const float*, int*) noexcept nogil>__cusolverDnSsygvdx_bufferSize)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnDsygvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsygvdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const double*, int, const double*, int, double, double, int, int, int*, const double*, int*) noexcept nogil>__cusolverDnDsygvdx_bufferSize)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnChegvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* B, int ldb, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnChegvdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnChegvdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const cuComplex*, int, const cuComplex*, int, float, float, int, int, int*, const float*, int*) noexcept nogil>__cusolverDnChegvdx_bufferSize)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnZhegvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZhegvdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, double, double, int, int, int*, const double*, int*) noexcept nogil>__cusolverDnZhegvdx_bufferSize)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnSsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float vl, float vu, int il, int iu, int* meig, float* W, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsygvdx
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, float*, int, float*, int, float, float, int, int, int*, float*, float*, int, int*) noexcept nogil>__cusolverDnSsygvdx)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double vl, double vu, int il, int iu, int* meig, double* W, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsygvdx
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, double*, int, double*, int, double, double, int, int, int*, double*, double*, int, int*) noexcept nogil>__cusolverDnDsygvdx)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnChegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb, float vl, float vu, int il, int iu, int* meig, float* W, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnChegvdx
    _check_or_init_cusolverDn()
    if __cusolverDnChegvdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, float, float, int, int, int*, float*, cuComplex*, int, int*) noexcept nogil>__cusolverDnChegvdx)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZhegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, double vl, double vu, int il, int iu, int* meig, double* W, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZhegvdx
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, double, double, int, int, int*, double*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZhegvdx)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, const float* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsygvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const float*, int, const float*, int, const float*, int*) noexcept nogil>__cusolverDnSsygvd_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t _cusolverDnDsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, const double* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsygvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const double*, int, const double*, int, const double*, int*) noexcept nogil>__cusolverDnDsygvd_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t _cusolverDnChegvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnChegvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnChegvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const cuComplex*, int, const cuComplex*, int, const float*, int*) noexcept nogil>__cusolverDnChegvd_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t _cusolverDnZhegvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* W, int* lwork) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZhegvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const double*, int*) noexcept nogil>__cusolverDnZhegvd_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t _cusolverDnSsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float* W, float* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsygvd
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, float*, int, float*, int, float*, float*, int, int*) noexcept nogil>__cusolverDnSsygvd)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double* W, double* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsygvd
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, double*, int, double*, int, double*, double*, int, int*) noexcept nogil>__cusolverDnDsygvd)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnChegvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb, float* W, cuComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnChegvd
    _check_or_init_cusolverDn()
    if __cusolverDnChegvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, float*, cuComplex*, int, int*) noexcept nogil>__cusolverDnChegvd)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZhegvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, double* W, cuDoubleComplex* work, int lwork, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZhegvd
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, int*) noexcept nogil>__cusolverDnZhegvd)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCreateSyevjInfo(syevjInfo_t* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCreateSyevjInfo
    _check_or_init_cusolverDn()
    if __cusolverDnCreateSyevjInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCreateSyevjInfo is not found")
    return (<cusolverStatus_t (*)(syevjInfo_t*) noexcept nogil>__cusolverDnCreateSyevjInfo)(
        info)


cdef cusolverStatus_t _cusolverDnDestroySyevjInfo(syevjInfo_t info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDestroySyevjInfo
    _check_or_init_cusolverDn()
    if __cusolverDnDestroySyevjInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDestroySyevjInfo is not found")
    return (<cusolverStatus_t (*)(syevjInfo_t) noexcept nogil>__cusolverDnDestroySyevjInfo)(
        info)


cdef cusolverStatus_t _cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevjSetTolerance
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevjSetTolerance == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevjSetTolerance is not found")
    return (<cusolverStatus_t (*)(syevjInfo_t, double) noexcept nogil>__cusolverDnXsyevjSetTolerance)(
        info, tolerance)


cdef cusolverStatus_t _cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevjSetMaxSweeps
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevjSetMaxSweeps == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevjSetMaxSweeps is not found")
    return (<cusolverStatus_t (*)(syevjInfo_t, int) noexcept nogil>__cusolverDnXsyevjSetMaxSweeps)(
        info, max_sweeps)


cdef cusolverStatus_t _cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevjSetSortEig
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevjSetSortEig == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevjSetSortEig is not found")
    return (<cusolverStatus_t (*)(syevjInfo_t, int) noexcept nogil>__cusolverDnXsyevjSetSortEig)(
        info, sort_eig)


cdef cusolverStatus_t _cusolverDnXsyevjGetResidual(cusolverDnHandle_t handle, syevjInfo_t info, double* residual) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevjGetResidual
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevjGetResidual == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevjGetResidual is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, syevjInfo_t, double*) noexcept nogil>__cusolverDnXsyevjGetResidual)(
        handle, info, residual)


cdef cusolverStatus_t _cusolverDnXsyevjGetSweeps(cusolverDnHandle_t handle, syevjInfo_t info, int* executed_sweeps) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevjGetSweeps
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevjGetSweeps == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevjGetSweeps is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, syevjInfo_t, int*) noexcept nogil>__cusolverDnXsyevjGetSweeps)(
        handle, info, executed_sweeps)


cdef cusolverStatus_t _cusolverDnSsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork, syevjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsyevjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const float*, int, const float*, int*, syevjInfo_t, int) noexcept nogil>__cusolverDnSsyevjBatched_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork, syevjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsyevjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const double*, int, const double*, int*, syevjInfo_t, int) noexcept nogil>__cusolverDnDsyevjBatched_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnCheevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork, syevjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCheevjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCheevjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuComplex*, int, const float*, int*, syevjInfo_t, int) noexcept nogil>__cusolverDnCheevjBatched_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnZheevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork, syevjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZheevjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZheevjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const double*, int*, syevjInfo_t, int) noexcept nogil>__cusolverDnZheevjBatched_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnSsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info, syevjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsyevjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float*, int, float*, float*, int, int*, syevjInfo_t, int) noexcept nogil>__cusolverDnSsyevjBatched)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnDsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info, syevjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsyevjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double*, int, double*, double*, int, int*, syevjInfo_t, int) noexcept nogil>__cusolverDnDsyevjBatched)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnCheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCheevjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnCheevjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex*, int, float*, cuComplex*, int, int*, syevjInfo_t, int) noexcept nogil>__cusolverDnCheevjBatched)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnZheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZheevjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnZheevjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, int*, syevjInfo_t, int) noexcept nogil>__cusolverDnZheevjBatched)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnSsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsyevj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const float*, int, const float*, int*, syevjInfo_t) noexcept nogil>__cusolverDnSsyevj_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t _cusolverDnDsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsyevj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const double*, int, const double*, int*, syevjInfo_t) noexcept nogil>__cusolverDnDsyevj_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t _cusolverDnCheevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCheevj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCheevj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuComplex*, int, const float*, int*, syevjInfo_t) noexcept nogil>__cusolverDnCheevj_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t _cusolverDnZheevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZheevj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZheevj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const double*, int*, syevjInfo_t) noexcept nogil>__cusolverDnZheevj_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t _cusolverDnSsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsyevj
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float*, int, float*, float*, int, int*, syevjInfo_t) noexcept nogil>__cusolverDnSsyevj)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnDsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsyevj
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double*, int, double*, double*, int, int*, syevjInfo_t) noexcept nogil>__cusolverDnDsyevj)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnCheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCheevj
    _check_or_init_cusolverDn()
    if __cusolverDnCheevj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex*, int, float*, cuComplex*, int, int*, syevjInfo_t) noexcept nogil>__cusolverDnCheevj)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnZheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZheevj
    _check_or_init_cusolverDn()
    if __cusolverDnZheevj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, int*, syevjInfo_t) noexcept nogil>__cusolverDnZheevj)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnSsygvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, const float* W, int* lwork, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsygvj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const float*, int, const float*, int, const float*, int*, syevjInfo_t) noexcept nogil>__cusolverDnSsygvj_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t _cusolverDnDsygvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, const double* W, int* lwork, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsygvj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const double*, int, const double*, int, const double*, int*, syevjInfo_t) noexcept nogil>__cusolverDnDsygvj_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t _cusolverDnChegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* W, int* lwork, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnChegvj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnChegvj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const cuComplex*, int, const cuComplex*, int, const float*, int*, syevjInfo_t) noexcept nogil>__cusolverDnChegvj_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t _cusolverDnZhegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* W, int* lwork, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZhegvj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const double*, int*, syevjInfo_t) noexcept nogil>__cusolverDnZhegvj_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t _cusolverDnSsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float* W, float* work, int lwork, int* info, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSsygvj
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, float*, int, float*, int, float*, float*, int, int*, syevjInfo_t) noexcept nogil>__cusolverDnSsygvj)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnDsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double* W, double* work, int lwork, int* info, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDsygvj
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, double*, int, double*, int, double*, double*, int, int*, syevjInfo_t) noexcept nogil>__cusolverDnDsygvj)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnChegvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnChegvj
    _check_or_init_cusolverDn()
    if __cusolverDnChegvj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, float*, cuComplex*, int, int*, syevjInfo_t) noexcept nogil>__cusolverDnChegvj)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnZhegvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZhegvj
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, int*, syevjInfo_t) noexcept nogil>__cusolverDnZhegvj)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCreateGesvdjInfo
    _check_or_init_cusolverDn()
    if __cusolverDnCreateGesvdjInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCreateGesvdjInfo is not found")
    return (<cusolverStatus_t (*)(gesvdjInfo_t*) noexcept nogil>__cusolverDnCreateGesvdjInfo)(
        info)


cdef cusolverStatus_t _cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDestroyGesvdjInfo
    _check_or_init_cusolverDn()
    if __cusolverDnDestroyGesvdjInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDestroyGesvdjInfo is not found")
    return (<cusolverStatus_t (*)(gesvdjInfo_t) noexcept nogil>__cusolverDnDestroyGesvdjInfo)(
        info)


cdef cusolverStatus_t _cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvdjSetTolerance
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdjSetTolerance == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdjSetTolerance is not found")
    return (<cusolverStatus_t (*)(gesvdjInfo_t, double) noexcept nogil>__cusolverDnXgesvdjSetTolerance)(
        info, tolerance)


cdef cusolverStatus_t _cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int max_sweeps) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvdjSetMaxSweeps
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdjSetMaxSweeps == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdjSetMaxSweeps is not found")
    return (<cusolverStatus_t (*)(gesvdjInfo_t, int) noexcept nogil>__cusolverDnXgesvdjSetMaxSweeps)(
        info, max_sweeps)


cdef cusolverStatus_t _cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_svd) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvdjSetSortEig
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdjSetSortEig == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdjSetSortEig is not found")
    return (<cusolverStatus_t (*)(gesvdjInfo_t, int) noexcept nogil>__cusolverDnXgesvdjSetSortEig)(
        info, sort_svd)


cdef cusolverStatus_t _cusolverDnXgesvdjGetResidual(cusolverDnHandle_t handle, gesvdjInfo_t info, double* residual) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvdjGetResidual
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdjGetResidual == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdjGetResidual is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, gesvdjInfo_t, double*) noexcept nogil>__cusolverDnXgesvdjGetResidual)(
        handle, info, residual)


cdef cusolverStatus_t _cusolverDnXgesvdjGetSweeps(cusolverDnHandle_t handle, gesvdjInfo_t info, int* executed_sweeps) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvdjGetSweeps
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdjGetSweeps == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdjGetSweeps is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, gesvdjInfo_t, int*) noexcept nogil>__cusolverDnXgesvdjGetSweeps)(
        handle, info, executed_sweeps)


cdef cusolverStatus_t _cusolverDnSgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const float* A, int lda, const float* S, const float* U, int ldu, const float* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgesvdjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const float*, int, const float*, const float*, int, const float*, int, int*, gesvdjInfo_t, int) noexcept nogil>__cusolverDnSgesvdjBatched_bufferSize)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnDgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const double* A, int lda, const double* S, const double* U, int ldu, const double* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgesvdjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const double*, int, const double*, const double*, int, const double*, int, int*, gesvdjInfo_t, int) noexcept nogil>__cusolverDnDgesvdjBatched_bufferSize)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnCgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuComplex* A, int lda, const float* S, const cuComplex* U, int ldu, const cuComplex* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgesvdjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const cuComplex*, int, const float*, const cuComplex*, int, const cuComplex*, int, int*, gesvdjInfo_t, int) noexcept nogil>__cusolverDnCgesvdjBatched_bufferSize)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnZgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuDoubleComplex* A, int lda, const double* S, const cuDoubleComplex* U, int ldu, const cuDoubleComplex* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgesvdjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const cuDoubleComplex*, int, const double*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, int*, gesvdjInfo_t, int) noexcept nogil>__cusolverDnZgesvdjBatched_bufferSize)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnSgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgesvdjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, float*, int, float*, float*, int, float*, int, float*, int, int*, gesvdjInfo_t, int) noexcept nogil>__cusolverDnSgesvdjBatched)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnDgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgesvdjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, double*, int, double*, double*, int, double*, int, double*, int, int*, gesvdjInfo_t, int) noexcept nogil>__cusolverDnDgesvdjBatched)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnCgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* V, int ldv, cuComplex* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgesvdjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, cuComplex*, int, float*, cuComplex*, int, cuComplex*, int, cuComplex*, int, int*, gesvdjInfo_t, int) noexcept nogil>__cusolverDnCgesvdjBatched)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnZgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* V, int ldv, cuDoubleComplex* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgesvdjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, cuDoubleComplex*, int, cuDoubleComplex*, int, int*, gesvdjInfo_t, int) noexcept nogil>__cusolverDnZgesvdjBatched)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnSgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const float* A, int lda, const float* S, const float* U, int ldu, const float* V, int ldv, int* lwork, gesvdjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgesvdj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float*, int, const float*, const float*, int, const float*, int, int*, gesvdjInfo_t) noexcept nogil>__cusolverDnSgesvdj_bufferSize)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t _cusolverDnDgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const double* A, int lda, const double* S, const double* U, int ldu, const double* V, int ldv, int* lwork, gesvdjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgesvdj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double*, int, const double*, const double*, int, const double*, int, int*, gesvdjInfo_t) noexcept nogil>__cusolverDnDgesvdj_bufferSize)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t _cusolverDnCgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuComplex* A, int lda, const float* S, const cuComplex* U, int ldu, const cuComplex* V, int ldv, int* lwork, gesvdjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgesvdj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex*, int, const float*, const cuComplex*, int, const cuComplex*, int, int*, gesvdjInfo_t) noexcept nogil>__cusolverDnCgesvdj_bufferSize)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t _cusolverDnZgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuDoubleComplex* A, int lda, const double* S, const cuDoubleComplex* U, int ldu, const cuDoubleComplex* V, int ldv, int* lwork, gesvdjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgesvdj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuDoubleComplex*, int, const double*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, int*, gesvdjInfo_t) noexcept nogil>__cusolverDnZgesvdj_bufferSize)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t _cusolverDnSgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, int* info, gesvdjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgesvdj
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, float*, int, float*, float*, int, float*, int, float*, int, int*, gesvdjInfo_t) noexcept nogil>__cusolverDnSgesvdj)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnDgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, int* info, gesvdjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgesvdj
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, double*, int, double*, double*, int, double*, int, double*, int, int*, gesvdjInfo_t) noexcept nogil>__cusolverDnDgesvdj)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnCgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* V, int ldv, cuComplex* work, int lwork, int* info, gesvdjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgesvdj
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, cuComplex*, int, float*, cuComplex*, int, cuComplex*, int, cuComplex*, int, int*, gesvdjInfo_t) noexcept nogil>__cusolverDnCgesvdj)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnZgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* V, int ldv, cuDoubleComplex* work, int lwork, int* info, gesvdjInfo_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgesvdj
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, cuDoubleComplex*, int, cuDoubleComplex*, int, int*, gesvdjInfo_t) noexcept nogil>__cusolverDnZgesvdj)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnSgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float* d_A, int lda, long long int strideA, const float* d_S, long long int strideS, const float* d_U, int ldu, long long int strideU, const float* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgesvdaStridedBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdaStridedBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdaStridedBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float*, int, long long int, const float*, long long int, const float*, int, long long int, const float*, int, long long int, int*, int) noexcept nogil>__cusolverDnSgesvdaStridedBatched_bufferSize)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t _cusolverDnDgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double* d_A, int lda, long long int strideA, const double* d_S, long long int strideS, const double* d_U, int ldu, long long int strideU, const double* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgesvdaStridedBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdaStridedBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdaStridedBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double*, int, long long int, const double*, long long int, const double*, int, long long int, const double*, int, long long int, int*, int) noexcept nogil>__cusolverDnDgesvdaStridedBatched_bufferSize)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t _cusolverDnCgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex* d_A, int lda, long long int strideA, const float* d_S, long long int strideS, const cuComplex* d_U, int ldu, long long int strideU, const cuComplex* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgesvdaStridedBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdaStridedBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdaStridedBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex*, int, long long int, const float*, long long int, const cuComplex*, int, long long int, const cuComplex*, int, long long int, int*, int) noexcept nogil>__cusolverDnCgesvdaStridedBatched_bufferSize)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t _cusolverDnZgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex* d_A, int lda, long long int strideA, const double* d_S, long long int strideS, const cuDoubleComplex* d_U, int ldu, long long int strideU, const cuDoubleComplex* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgesvdaStridedBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdaStridedBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdaStridedBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuDoubleComplex*, int, long long int, const double*, long long int, const cuDoubleComplex*, int, long long int, const cuDoubleComplex*, int, long long int, int*, int) noexcept nogil>__cusolverDnZgesvdaStridedBatched_bufferSize)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t _cusolverDnSgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float* d_A, int lda, long long int strideA, float* d_S, long long int strideS, float* d_U, int ldu, long long int strideU, float* d_V, int ldv, long long int strideV, float* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSgesvdaStridedBatched
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdaStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdaStridedBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float*, int, long long int, float*, long long int, float*, int, long long int, float*, int, long long int, float*, int, int*, double*, int) noexcept nogil>__cusolverDnSgesvdaStridedBatched)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t _cusolverDnDgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double* d_A, int lda, long long int strideA, double* d_S, long long int strideS, double* d_U, int ldu, long long int strideU, double* d_V, int ldv, long long int strideV, double* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDgesvdaStridedBatched
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdaStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdaStridedBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double*, int, long long int, double*, long long int, double*, int, long long int, double*, int, long long int, double*, int, int*, double*, int) noexcept nogil>__cusolverDnDgesvdaStridedBatched)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t _cusolverDnCgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex* d_A, int lda, long long int strideA, float* d_S, long long int strideS, cuComplex* d_U, int ldu, long long int strideU, cuComplex* d_V, int ldv, long long int strideV, cuComplex* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCgesvdaStridedBatched
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdaStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdaStridedBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex*, int, long long int, float*, long long int, cuComplex*, int, long long int, cuComplex*, int, long long int, cuComplex*, int, int*, double*, int) noexcept nogil>__cusolverDnCgesvdaStridedBatched)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t _cusolverDnZgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex* d_A, int lda, long long int strideA, double* d_S, long long int strideS, cuDoubleComplex* d_U, int ldu, long long int strideU, cuDoubleComplex* d_V, int ldv, long long int strideV, cuDoubleComplex* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnZgesvdaStridedBatched
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdaStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdaStridedBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuDoubleComplex*, int, long long int, double*, long long int, cuDoubleComplex*, int, long long int, cuDoubleComplex*, int, long long int, cuDoubleComplex*, int, int*, double*, int) noexcept nogil>__cusolverDnZgesvdaStridedBatched)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t _cusolverDnCreateParams(cusolverDnParams_t* params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnCreateParams
    _check_or_init_cusolverDn()
    if __cusolverDnCreateParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCreateParams is not found")
    return (<cusolverStatus_t (*)(cusolverDnParams_t*) noexcept nogil>__cusolverDnCreateParams)(
        params)


cdef cusolverStatus_t _cusolverDnDestroyParams(cusolverDnParams_t params) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnDestroyParams
    _check_or_init_cusolverDn()
    if __cusolverDnDestroyParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDestroyParams is not found")
    return (<cusolverStatus_t (*)(cusolverDnParams_t) noexcept nogil>__cusolverDnDestroyParams)(
        params)


cdef cusolverStatus_t _cusolverDnSetAdvOptions(cusolverDnParams_t params, cusolverDnFunction_t function, cusolverAlgMode_t algo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSetAdvOptions
    _check_or_init_cusolverDn()
    if __cusolverDnSetAdvOptions == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSetAdvOptions is not found")
    return (<cusolverStatus_t (*)(cusolverDnParams_t, cusolverDnFunction_t, cusolverAlgMode_t) noexcept nogil>__cusolverDnSetAdvOptions)(
        params, function, algo)


cdef cusolverStatus_t _cusolverDnXpotrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXpotrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXpotrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXpotrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) noexcept nogil>__cusolverDnXpotrf_bufferSize)(
        handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXpotrf(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXpotrf
    _check_or_init_cusolverDn()
    if __cusolverDnXpotrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXpotrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t, int*) noexcept nogil>__cusolverDnXpotrf)(
        handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXpotrs(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeB, void* B, int64_t ldb, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXpotrs
    _check_or_init_cusolverDn()
    if __cusolverDnXpotrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXpotrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, void*, int64_t, int*) noexcept nogil>__cusolverDnXpotrs)(
        handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info)


cdef cusolverStatus_t _cusolverDnXgeqrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeTau, const void* tau, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgeqrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgeqrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgeqrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, size_t*, size_t*) noexcept nogil>__cusolverDnXgeqrf_bufferSize)(
        handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgeqrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeTau, void* tau, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgeqrf
    _check_or_init_cusolverDn()
    if __cusolverDnXgeqrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgeqrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, size_t, void*, size_t, int*) noexcept nogil>__cusolverDnXgeqrf)(
        handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXgetrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgetrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgetrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgetrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) noexcept nogil>__cusolverDnXgetrf_bufferSize)(
        handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgetrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, int64_t* ipiv, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgetrf
    _check_or_init_cusolverDn()
    if __cusolverDnXgetrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgetrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, void*, int64_t, int64_t*, cudaDataType, void*, size_t, void*, size_t, int*) noexcept nogil>__cusolverDnXgetrf)(
        handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXgetrs(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasOperation_t trans, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, const int64_t* ipiv, cudaDataType dataTypeB, void* B, int64_t ldb, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgetrs
    _check_or_init_cusolverDn()
    if __cusolverDnXgetrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgetrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasOperation_t, int64_t, int64_t, cudaDataType, const void*, int64_t, const int64_t*, cudaDataType, void*, int64_t, int*) noexcept nogil>__cusolverDnXgetrs)(
        handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info)


cdef cusolverStatus_t _cusolverDnXsyevd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeW, const void* W, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, size_t*, size_t*) noexcept nogil>__cusolverDnXsyevd_bufferSize)(
        handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXsyevd(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevd
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, size_t, void*, size_t, int*) noexcept nogil>__cusolverDnXsyevd)(
        handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, void* vl, void* vu, int64_t il, int64_t iu, int64_t* h_meig, cudaDataType dataTypeW, const void* W, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int64_t, cudaDataType, const void*, int64_t, void*, void*, int64_t, int64_t, int64_t*, cudaDataType, const void*, cudaDataType, size_t*, size_t*) noexcept nogil>__cusolverDnXsyevdx_bufferSize)(
        handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXsyevdx(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, void* vl, void* vu, int64_t il, int64_t iu, int64_t* meig64, cudaDataType dataTypeW, void* W, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevdx
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int64_t, cudaDataType, void*, int64_t, void*, void*, int64_t, int64_t, int64_t*, cudaDataType, void*, cudaDataType, void*, size_t, void*, size_t, int*) noexcept nogil>__cusolverDnXsyevdx)(
        handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXgesvd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeS, const void* S, cudaDataType dataTypeU, const void* U, int64_t ldu, cudaDataType dataTypeVT, const void* VT, int64_t ldvt, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, const void*, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) noexcept nogil>__cusolverDnXgesvd_bufferSize)(
        handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgesvd(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeS, void* S, cudaDataType dataTypeU, void* U, int64_t ldu, cudaDataType dataTypeVT, void* VT, int64_t ldvt, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvd
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t, int*) noexcept nogil>__cusolverDnXgesvd)(
        handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXgesvdp_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, int econ, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeS, const void* S, cudaDataType dataTypeU, const void* U, int64_t ldu, cudaDataType dataTypeV, const void* V, int64_t ldv, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvdp_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdp_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdp_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, int, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, const void*, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) noexcept nogil>__cusolverDnXgesvdp_bufferSize)(
        handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgesvdp(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, int econ, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeS, void* S, cudaDataType dataTypeU, void* U, int64_t ldu, cudaDataType dataTypeV, void* V, int64_t ldv, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* d_info, double* h_err_sigma) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvdp
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdp is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, int, int64_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t, int*, double*) noexcept nogil>__cusolverDnXgesvdp)(
        handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info, h_err_sigma)


cdef cusolverStatus_t _cusolverDnXgesvdr_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeSrand, const void* Srand, cudaDataType dataTypeUrand, const void* Urand, int64_t ldUrand, cudaDataType dataTypeVrand, const void* Vrand, int64_t ldVrand, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvdr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t, int64_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, const void*, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) noexcept nogil>__cusolverDnXgesvdr_bufferSize)(
        handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgesvdr(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeSrand, void* Srand, cudaDataType dataTypeUrand, void* Urand, int64_t ldUrand, cudaDataType dataTypeVrand, void* Vrand, int64_t ldVrand, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* d_info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgesvdr
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t, int64_t, int64_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t, int*) noexcept nogil>__cusolverDnXgesvdr)(
        handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)


cdef cusolverStatus_t _cusolverDnXsytrs_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, const int64_t* ipiv, cudaDataType dataTypeB, void* B, int64_t ldb, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsytrs_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXsytrs_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsytrs_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int64_t, int64_t, cudaDataType, const void*, int64_t, const int64_t*, cudaDataType, void*, int64_t, size_t*, size_t*) noexcept nogil>__cusolverDnXsytrs_bufferSize)(
        handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXsytrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, const int64_t* ipiv, cudaDataType dataTypeB, void* B, int64_t ldb, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsytrs
    _check_or_init_cusolverDn()
    if __cusolverDnXsytrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsytrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int64_t, int64_t, cudaDataType, const void*, int64_t, const int64_t*, cudaDataType, void*, int64_t, void*, size_t, void*, size_t, int*) noexcept nogil>__cusolverDnXsytrs)(
        handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXtrtri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXtrtri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXtrtri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXtrtri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int64_t, cudaDataType, void*, int64_t, size_t*, size_t*) noexcept nogil>__cusolverDnXtrtri_bufferSize)(
        handle, uplo, diag, n, dataTypeA, A, lda, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXtrtri(cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* devInfo) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXtrtri
    _check_or_init_cusolverDn()
    if __cusolverDnXtrtri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXtrtri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int64_t, cudaDataType, void*, int64_t, void*, size_t, void*, size_t, int*) noexcept nogil>__cusolverDnXtrtri)(
        handle, uplo, diag, n, dataTypeA, A, lda, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, devInfo)


cdef cusolverStatus_t _cusolverDnLoggerSetCallback(cusolverDnLoggerCallback_t callback) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnLoggerSetCallback
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerSetCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerSetCallback is not found")
    return (<cusolverStatus_t (*)(cusolverDnLoggerCallback_t) noexcept nogil>__cusolverDnLoggerSetCallback)(
        callback)


cdef cusolverStatus_t _cusolverDnLoggerSetFile(FILE* file) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnLoggerSetFile
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerSetFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerSetFile is not found")
    return (<cusolverStatus_t (*)(FILE*) noexcept nogil>__cusolverDnLoggerSetFile)(
        file)


cdef cusolverStatus_t _cusolverDnLoggerOpenFile(const char* logFile) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnLoggerOpenFile
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerOpenFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerOpenFile is not found")
    return (<cusolverStatus_t (*)(const char*) noexcept nogil>__cusolverDnLoggerOpenFile)(
        logFile)


cdef cusolverStatus_t _cusolverDnLoggerSetLevel(int level) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnLoggerSetLevel
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerSetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerSetLevel is not found")
    return (<cusolverStatus_t (*)(int) noexcept nogil>__cusolverDnLoggerSetLevel)(
        level)


cdef cusolverStatus_t _cusolverDnLoggerSetMask(int mask) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnLoggerSetMask
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerSetMask == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerSetMask is not found")
    return (<cusolverStatus_t (*)(int) noexcept nogil>__cusolverDnLoggerSetMask)(
        mask)


cdef cusolverStatus_t _cusolverDnLoggerForceDisable() except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnLoggerForceDisable
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerForceDisable == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerForceDisable is not found")
    return (<cusolverStatus_t (*)() noexcept nogil>__cusolverDnLoggerForceDisable)(
        )


cdef cusolverStatus_t _cusolverDnSetDeterministicMode(cusolverDnHandle_t handle, cusolverDeterministicMode_t mode) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnSetDeterministicMode
    _check_or_init_cusolverDn()
    if __cusolverDnSetDeterministicMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSetDeterministicMode is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDeterministicMode_t) noexcept nogil>__cusolverDnSetDeterministicMode)(
        handle, mode)


cdef cusolverStatus_t _cusolverDnGetDeterministicMode(cusolverDnHandle_t handle, cusolverDeterministicMode_t* mode) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnGetDeterministicMode
    _check_or_init_cusolverDn()
    if __cusolverDnGetDeterministicMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnGetDeterministicMode is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDeterministicMode_t*) noexcept nogil>__cusolverDnGetDeterministicMode)(
        handle, mode)


cdef cusolverStatus_t _cusolverDnXlarft_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n, int64_t k, cudaDataType dataTypeV, const void* V, int64_t ldv, cudaDataType dataTypeTau, const void* tau, cudaDataType dataTypeT, void* T, int64_t ldt, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXlarft_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXlarft_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXlarft_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverDirectMode_t, cusolverStorevMode_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, void*, int64_t, cudaDataType, size_t*, size_t*) noexcept nogil>__cusolverDnXlarft_bufferSize)(
        handle, params, direct, storev, n, k, dataTypeV, V, ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXlarft(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n, int64_t k, cudaDataType dataTypeV, const void* V, int64_t ldv, cudaDataType dataTypeTau, const void* tau, cudaDataType dataTypeT, void* T, int64_t ldt, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXlarft
    _check_or_init_cusolverDn()
    if __cusolverDnXlarft == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXlarft is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverDirectMode_t, cusolverStorevMode_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t) noexcept nogil>__cusolverDnXlarft)(
        handle, params, direct, storev, n, k, dataTypeV, V, ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXsyevBatched_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeW, const void* W, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost, int64_t batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, size_t*, size_t*, int64_t) noexcept nogil>__cusolverDnXsyevBatched_bufferSize)(
        handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost, batchSize)


cdef cusolverStatus_t _cusolverDnXsyevBatched(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info, int64_t batchSize) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXsyevBatched
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, size_t, void*, size_t, int*, int64_t) noexcept nogil>__cusolverDnXsyevBatched)(
        handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info, batchSize)


cdef cusolverStatus_t _cusolverDnXgeev_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobvl, cusolverEigMode_t jobvr, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeW, const void* W, cudaDataType dataTypeVL, const void* VL, int64_t ldvl, cudaDataType dataTypeVR, const void* VR, int64_t ldvr, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgeev_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgeev_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgeev_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigMode_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, const void*, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) noexcept nogil>__cusolverDnXgeev_bufferSize)(
        handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgeev(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobvl, cusolverEigMode_t jobvr, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType dataTypeVL, void* VL, int64_t ldvl, cudaDataType dataTypeVR, void* VR, int64_t ldvr, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverDnXgeev
    _check_or_init_cusolverDn()
    if __cusolverDnXgeev == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgeev is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigMode_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t, int*) noexcept nogil>__cusolverDnXgeev)(
        handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
