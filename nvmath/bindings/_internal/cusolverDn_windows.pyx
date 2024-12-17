# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cublas cimport load_library as load_cublas
from .cusparse cimport load_library as load_cusparse
from .utils cimport get_cusolver_dso_version_suffix

import os
import site

import win32api

from .utils import FunctionNotFoundError, NotSupportedError


###############################################################################
# Wrapper init
###############################################################################

LOAD_LIBRARY_SEARCH_SYSTEM32     = 0x00000800
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
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


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef load_library(const int driver_ver):
    handle = 0

    for suffix in get_cusolver_dso_version_suffix(driver_ver):
        if len(suffix) == 0:
            continue
        dll_name = f"cusolver64_{suffix}.dll"

        # First check if the DLL has been loaded by 3rd parties
        try:
            handle = win32api.GetModuleHandle(dll_name)
        except:
            pass
        else:
            break

        # Next, check if DLLs are installed via pip
        for sp in get_site_packages():
            mod_path = os.path.join(sp, "nvidia", "cusolver", "bin")
            if not os.path.isdir(mod_path):
                continue
            os.add_dll_directory(mod_path)

        # cuSOLVER also requires additional dependencies...
        load_cublas(driver_ver)
        load_cusparse(driver_ver)

        try:
            handle = win32api.LoadLibraryEx(
                # Note: LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR needs an abs path...
                os.path.join(mod_path, dll_name),
                0, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
        except:
            pass
        else:
            break

        # Finally, try default search
        try:
            handle = win32api.LoadLibrary(dll_name)
        except:
            pass
        else:
            break
    else:
        raise RuntimeError('Failed to load cusolverDn')

    assert handle != 0
    return handle


cdef int _check_or_init_cusolverDn() except -1 nogil:
    global __py_cusolverDn_init
    if __py_cusolverDn_init:
        return 0

    cdef int err, driver_ver
    with gil:
        # Load driver to check version
        try:
            handle = win32api.LoadLibraryEx("nvcuda.dll", 0, LOAD_LIBRARY_SEARCH_SYSTEM32)
        except Exception as e:
            raise NotSupportedError(f'CUDA driver is not found ({e})')
        global __cuDriverGetVersion
        if __cuDriverGetVersion == NULL:
            __cuDriverGetVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'cuDriverGetVersion')
            if __cuDriverGetVersion == NULL:
                raise RuntimeError('something went wrong')
        err = (<int (*)(int*) nogil>__cuDriverGetVersion)(&driver_ver)
        if err != 0:
            raise RuntimeError('something went wrong')

        # Load library
        handle = load_library(driver_ver)

        # Load function
        global __cusolverDnCreate
        try:
            __cusolverDnCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCreate')
        except:
            pass

        global __cusolverDnDestroy
        try:
            __cusolverDnDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDestroy')
        except:
            pass

        global __cusolverDnSetStream
        try:
            __cusolverDnSetStream = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSetStream')
        except:
            pass

        global __cusolverDnGetStream
        try:
            __cusolverDnGetStream = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnGetStream')
        except:
            pass

        global __cusolverDnIRSParamsCreate
        try:
            __cusolverDnIRSParamsCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsCreate')
        except:
            pass

        global __cusolverDnIRSParamsDestroy
        try:
            __cusolverDnIRSParamsDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsDestroy')
        except:
            pass

        global __cusolverDnIRSParamsSetRefinementSolver
        try:
            __cusolverDnIRSParamsSetRefinementSolver = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsSetRefinementSolver')
        except:
            pass

        global __cusolverDnIRSParamsSetSolverMainPrecision
        try:
            __cusolverDnIRSParamsSetSolverMainPrecision = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsSetSolverMainPrecision')
        except:
            pass

        global __cusolverDnIRSParamsSetSolverLowestPrecision
        try:
            __cusolverDnIRSParamsSetSolverLowestPrecision = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsSetSolverLowestPrecision')
        except:
            pass

        global __cusolverDnIRSParamsSetSolverPrecisions
        try:
            __cusolverDnIRSParamsSetSolverPrecisions = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsSetSolverPrecisions')
        except:
            pass

        global __cusolverDnIRSParamsSetTol
        try:
            __cusolverDnIRSParamsSetTol = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsSetTol')
        except:
            pass

        global __cusolverDnIRSParamsSetTolInner
        try:
            __cusolverDnIRSParamsSetTolInner = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsSetTolInner')
        except:
            pass

        global __cusolverDnIRSParamsSetMaxIters
        try:
            __cusolverDnIRSParamsSetMaxIters = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsSetMaxIters')
        except:
            pass

        global __cusolverDnIRSParamsSetMaxItersInner
        try:
            __cusolverDnIRSParamsSetMaxItersInner = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsSetMaxItersInner')
        except:
            pass

        global __cusolverDnIRSParamsGetMaxIters
        try:
            __cusolverDnIRSParamsGetMaxIters = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsGetMaxIters')
        except:
            pass

        global __cusolverDnIRSParamsEnableFallback
        try:
            __cusolverDnIRSParamsEnableFallback = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsEnableFallback')
        except:
            pass

        global __cusolverDnIRSParamsDisableFallback
        try:
            __cusolverDnIRSParamsDisableFallback = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSParamsDisableFallback')
        except:
            pass

        global __cusolverDnIRSInfosDestroy
        try:
            __cusolverDnIRSInfosDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSInfosDestroy')
        except:
            pass

        global __cusolverDnIRSInfosCreate
        try:
            __cusolverDnIRSInfosCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSInfosCreate')
        except:
            pass

        global __cusolverDnIRSInfosGetNiters
        try:
            __cusolverDnIRSInfosGetNiters = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSInfosGetNiters')
        except:
            pass

        global __cusolverDnIRSInfosGetOuterNiters
        try:
            __cusolverDnIRSInfosGetOuterNiters = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSInfosGetOuterNiters')
        except:
            pass

        global __cusolverDnIRSInfosRequestResidual
        try:
            __cusolverDnIRSInfosRequestResidual = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSInfosRequestResidual')
        except:
            pass

        global __cusolverDnIRSInfosGetResidualHistory
        try:
            __cusolverDnIRSInfosGetResidualHistory = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSInfosGetResidualHistory')
        except:
            pass

        global __cusolverDnIRSInfosGetMaxIters
        try:
            __cusolverDnIRSInfosGetMaxIters = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSInfosGetMaxIters')
        except:
            pass

        global __cusolverDnZZgesv
        try:
            __cusolverDnZZgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZZgesv')
        except:
            pass

        global __cusolverDnZCgesv
        try:
            __cusolverDnZCgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZCgesv')
        except:
            pass

        global __cusolverDnZKgesv
        try:
            __cusolverDnZKgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZKgesv')
        except:
            pass

        global __cusolverDnZEgesv
        try:
            __cusolverDnZEgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZEgesv')
        except:
            pass

        global __cusolverDnZYgesv
        try:
            __cusolverDnZYgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZYgesv')
        except:
            pass

        global __cusolverDnCCgesv
        try:
            __cusolverDnCCgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCCgesv')
        except:
            pass

        global __cusolverDnCEgesv
        try:
            __cusolverDnCEgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCEgesv')
        except:
            pass

        global __cusolverDnCKgesv
        try:
            __cusolverDnCKgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCKgesv')
        except:
            pass

        global __cusolverDnCYgesv
        try:
            __cusolverDnCYgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCYgesv')
        except:
            pass

        global __cusolverDnDDgesv
        try:
            __cusolverDnDDgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDDgesv')
        except:
            pass

        global __cusolverDnDSgesv
        try:
            __cusolverDnDSgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDSgesv')
        except:
            pass

        global __cusolverDnDHgesv
        try:
            __cusolverDnDHgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDHgesv')
        except:
            pass

        global __cusolverDnDBgesv
        try:
            __cusolverDnDBgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDBgesv')
        except:
            pass

        global __cusolverDnDXgesv
        try:
            __cusolverDnDXgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDXgesv')
        except:
            pass

        global __cusolverDnSSgesv
        try:
            __cusolverDnSSgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSSgesv')
        except:
            pass

        global __cusolverDnSHgesv
        try:
            __cusolverDnSHgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSHgesv')
        except:
            pass

        global __cusolverDnSBgesv
        try:
            __cusolverDnSBgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSBgesv')
        except:
            pass

        global __cusolverDnSXgesv
        try:
            __cusolverDnSXgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSXgesv')
        except:
            pass

        global __cusolverDnZZgesv_bufferSize
        try:
            __cusolverDnZZgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZZgesv_bufferSize')
        except:
            pass

        global __cusolverDnZCgesv_bufferSize
        try:
            __cusolverDnZCgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZCgesv_bufferSize')
        except:
            pass

        global __cusolverDnZKgesv_bufferSize
        try:
            __cusolverDnZKgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZKgesv_bufferSize')
        except:
            pass

        global __cusolverDnZEgesv_bufferSize
        try:
            __cusolverDnZEgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZEgesv_bufferSize')
        except:
            pass

        global __cusolverDnZYgesv_bufferSize
        try:
            __cusolverDnZYgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZYgesv_bufferSize')
        except:
            pass

        global __cusolverDnCCgesv_bufferSize
        try:
            __cusolverDnCCgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCCgesv_bufferSize')
        except:
            pass

        global __cusolverDnCKgesv_bufferSize
        try:
            __cusolverDnCKgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCKgesv_bufferSize')
        except:
            pass

        global __cusolverDnCEgesv_bufferSize
        try:
            __cusolverDnCEgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCEgesv_bufferSize')
        except:
            pass

        global __cusolverDnCYgesv_bufferSize
        try:
            __cusolverDnCYgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCYgesv_bufferSize')
        except:
            pass

        global __cusolverDnDDgesv_bufferSize
        try:
            __cusolverDnDDgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDDgesv_bufferSize')
        except:
            pass

        global __cusolverDnDSgesv_bufferSize
        try:
            __cusolverDnDSgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDSgesv_bufferSize')
        except:
            pass

        global __cusolverDnDHgesv_bufferSize
        try:
            __cusolverDnDHgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDHgesv_bufferSize')
        except:
            pass

        global __cusolverDnDBgesv_bufferSize
        try:
            __cusolverDnDBgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDBgesv_bufferSize')
        except:
            pass

        global __cusolverDnDXgesv_bufferSize
        try:
            __cusolverDnDXgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDXgesv_bufferSize')
        except:
            pass

        global __cusolverDnSSgesv_bufferSize
        try:
            __cusolverDnSSgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSSgesv_bufferSize')
        except:
            pass

        global __cusolverDnSHgesv_bufferSize
        try:
            __cusolverDnSHgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSHgesv_bufferSize')
        except:
            pass

        global __cusolverDnSBgesv_bufferSize
        try:
            __cusolverDnSBgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSBgesv_bufferSize')
        except:
            pass

        global __cusolverDnSXgesv_bufferSize
        try:
            __cusolverDnSXgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSXgesv_bufferSize')
        except:
            pass

        global __cusolverDnZZgels
        try:
            __cusolverDnZZgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZZgels')
        except:
            pass

        global __cusolverDnZCgels
        try:
            __cusolverDnZCgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZCgels')
        except:
            pass

        global __cusolverDnZKgels
        try:
            __cusolverDnZKgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZKgels')
        except:
            pass

        global __cusolverDnZEgels
        try:
            __cusolverDnZEgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZEgels')
        except:
            pass

        global __cusolverDnZYgels
        try:
            __cusolverDnZYgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZYgels')
        except:
            pass

        global __cusolverDnCCgels
        try:
            __cusolverDnCCgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCCgels')
        except:
            pass

        global __cusolverDnCKgels
        try:
            __cusolverDnCKgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCKgels')
        except:
            pass

        global __cusolverDnCEgels
        try:
            __cusolverDnCEgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCEgels')
        except:
            pass

        global __cusolverDnCYgels
        try:
            __cusolverDnCYgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCYgels')
        except:
            pass

        global __cusolverDnDDgels
        try:
            __cusolverDnDDgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDDgels')
        except:
            pass

        global __cusolverDnDSgels
        try:
            __cusolverDnDSgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDSgels')
        except:
            pass

        global __cusolverDnDHgels
        try:
            __cusolverDnDHgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDHgels')
        except:
            pass

        global __cusolverDnDBgels
        try:
            __cusolverDnDBgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDBgels')
        except:
            pass

        global __cusolverDnDXgels
        try:
            __cusolverDnDXgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDXgels')
        except:
            pass

        global __cusolverDnSSgels
        try:
            __cusolverDnSSgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSSgels')
        except:
            pass

        global __cusolverDnSHgels
        try:
            __cusolverDnSHgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSHgels')
        except:
            pass

        global __cusolverDnSBgels
        try:
            __cusolverDnSBgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSBgels')
        except:
            pass

        global __cusolverDnSXgels
        try:
            __cusolverDnSXgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSXgels')
        except:
            pass

        global __cusolverDnZZgels_bufferSize
        try:
            __cusolverDnZZgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZZgels_bufferSize')
        except:
            pass

        global __cusolverDnZCgels_bufferSize
        try:
            __cusolverDnZCgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZCgels_bufferSize')
        except:
            pass

        global __cusolverDnZKgels_bufferSize
        try:
            __cusolverDnZKgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZKgels_bufferSize')
        except:
            pass

        global __cusolverDnZEgels_bufferSize
        try:
            __cusolverDnZEgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZEgels_bufferSize')
        except:
            pass

        global __cusolverDnZYgels_bufferSize
        try:
            __cusolverDnZYgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZYgels_bufferSize')
        except:
            pass

        global __cusolverDnCCgels_bufferSize
        try:
            __cusolverDnCCgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCCgels_bufferSize')
        except:
            pass

        global __cusolverDnCKgels_bufferSize
        try:
            __cusolverDnCKgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCKgels_bufferSize')
        except:
            pass

        global __cusolverDnCEgels_bufferSize
        try:
            __cusolverDnCEgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCEgels_bufferSize')
        except:
            pass

        global __cusolverDnCYgels_bufferSize
        try:
            __cusolverDnCYgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCYgels_bufferSize')
        except:
            pass

        global __cusolverDnDDgels_bufferSize
        try:
            __cusolverDnDDgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDDgels_bufferSize')
        except:
            pass

        global __cusolverDnDSgels_bufferSize
        try:
            __cusolverDnDSgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDSgels_bufferSize')
        except:
            pass

        global __cusolverDnDHgels_bufferSize
        try:
            __cusolverDnDHgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDHgels_bufferSize')
        except:
            pass

        global __cusolverDnDBgels_bufferSize
        try:
            __cusolverDnDBgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDBgels_bufferSize')
        except:
            pass

        global __cusolverDnDXgels_bufferSize
        try:
            __cusolverDnDXgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDXgels_bufferSize')
        except:
            pass

        global __cusolverDnSSgels_bufferSize
        try:
            __cusolverDnSSgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSSgels_bufferSize')
        except:
            pass

        global __cusolverDnSHgels_bufferSize
        try:
            __cusolverDnSHgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSHgels_bufferSize')
        except:
            pass

        global __cusolverDnSBgels_bufferSize
        try:
            __cusolverDnSBgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSBgels_bufferSize')
        except:
            pass

        global __cusolverDnSXgels_bufferSize
        try:
            __cusolverDnSXgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSXgels_bufferSize')
        except:
            pass

        global __cusolverDnIRSXgesv
        try:
            __cusolverDnIRSXgesv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSXgesv')
        except:
            pass

        global __cusolverDnIRSXgesv_bufferSize
        try:
            __cusolverDnIRSXgesv_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSXgesv_bufferSize')
        except:
            pass

        global __cusolverDnIRSXgels
        try:
            __cusolverDnIRSXgels = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSXgels')
        except:
            pass

        global __cusolverDnIRSXgels_bufferSize
        try:
            __cusolverDnIRSXgels_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnIRSXgels_bufferSize')
        except:
            pass

        global __cusolverDnSpotrf_bufferSize
        try:
            __cusolverDnSpotrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSpotrf_bufferSize')
        except:
            pass

        global __cusolverDnDpotrf_bufferSize
        try:
            __cusolverDnDpotrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDpotrf_bufferSize')
        except:
            pass

        global __cusolverDnCpotrf_bufferSize
        try:
            __cusolverDnCpotrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCpotrf_bufferSize')
        except:
            pass

        global __cusolverDnZpotrf_bufferSize
        try:
            __cusolverDnZpotrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZpotrf_bufferSize')
        except:
            pass

        global __cusolverDnSpotrf
        try:
            __cusolverDnSpotrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSpotrf')
        except:
            pass

        global __cusolverDnDpotrf
        try:
            __cusolverDnDpotrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDpotrf')
        except:
            pass

        global __cusolverDnCpotrf
        try:
            __cusolverDnCpotrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCpotrf')
        except:
            pass

        global __cusolverDnZpotrf
        try:
            __cusolverDnZpotrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZpotrf')
        except:
            pass

        global __cusolverDnSpotrs
        try:
            __cusolverDnSpotrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSpotrs')
        except:
            pass

        global __cusolverDnDpotrs
        try:
            __cusolverDnDpotrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDpotrs')
        except:
            pass

        global __cusolverDnCpotrs
        try:
            __cusolverDnCpotrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCpotrs')
        except:
            pass

        global __cusolverDnZpotrs
        try:
            __cusolverDnZpotrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZpotrs')
        except:
            pass

        global __cusolverDnSpotrfBatched
        try:
            __cusolverDnSpotrfBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSpotrfBatched')
        except:
            pass

        global __cusolverDnDpotrfBatched
        try:
            __cusolverDnDpotrfBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDpotrfBatched')
        except:
            pass

        global __cusolverDnCpotrfBatched
        try:
            __cusolverDnCpotrfBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCpotrfBatched')
        except:
            pass

        global __cusolverDnZpotrfBatched
        try:
            __cusolverDnZpotrfBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZpotrfBatched')
        except:
            pass

        global __cusolverDnSpotrsBatched
        try:
            __cusolverDnSpotrsBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSpotrsBatched')
        except:
            pass

        global __cusolverDnDpotrsBatched
        try:
            __cusolverDnDpotrsBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDpotrsBatched')
        except:
            pass

        global __cusolverDnCpotrsBatched
        try:
            __cusolverDnCpotrsBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCpotrsBatched')
        except:
            pass

        global __cusolverDnZpotrsBatched
        try:
            __cusolverDnZpotrsBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZpotrsBatched')
        except:
            pass

        global __cusolverDnSpotri_bufferSize
        try:
            __cusolverDnSpotri_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSpotri_bufferSize')
        except:
            pass

        global __cusolverDnDpotri_bufferSize
        try:
            __cusolverDnDpotri_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDpotri_bufferSize')
        except:
            pass

        global __cusolverDnCpotri_bufferSize
        try:
            __cusolverDnCpotri_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCpotri_bufferSize')
        except:
            pass

        global __cusolverDnZpotri_bufferSize
        try:
            __cusolverDnZpotri_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZpotri_bufferSize')
        except:
            pass

        global __cusolverDnSpotri
        try:
            __cusolverDnSpotri = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSpotri')
        except:
            pass

        global __cusolverDnDpotri
        try:
            __cusolverDnDpotri = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDpotri')
        except:
            pass

        global __cusolverDnCpotri
        try:
            __cusolverDnCpotri = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCpotri')
        except:
            pass

        global __cusolverDnZpotri
        try:
            __cusolverDnZpotri = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZpotri')
        except:
            pass

        global __cusolverDnSlauum_bufferSize
        try:
            __cusolverDnSlauum_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSlauum_bufferSize')
        except:
            pass

        global __cusolverDnDlauum_bufferSize
        try:
            __cusolverDnDlauum_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDlauum_bufferSize')
        except:
            pass

        global __cusolverDnClauum_bufferSize
        try:
            __cusolverDnClauum_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnClauum_bufferSize')
        except:
            pass

        global __cusolverDnZlauum_bufferSize
        try:
            __cusolverDnZlauum_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZlauum_bufferSize')
        except:
            pass

        global __cusolverDnSlauum
        try:
            __cusolverDnSlauum = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSlauum')
        except:
            pass

        global __cusolverDnDlauum
        try:
            __cusolverDnDlauum = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDlauum')
        except:
            pass

        global __cusolverDnClauum
        try:
            __cusolverDnClauum = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnClauum')
        except:
            pass

        global __cusolverDnZlauum
        try:
            __cusolverDnZlauum = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZlauum')
        except:
            pass

        global __cusolverDnSgetrf_bufferSize
        try:
            __cusolverDnSgetrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgetrf_bufferSize')
        except:
            pass

        global __cusolverDnDgetrf_bufferSize
        try:
            __cusolverDnDgetrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgetrf_bufferSize')
        except:
            pass

        global __cusolverDnCgetrf_bufferSize
        try:
            __cusolverDnCgetrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgetrf_bufferSize')
        except:
            pass

        global __cusolverDnZgetrf_bufferSize
        try:
            __cusolverDnZgetrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgetrf_bufferSize')
        except:
            pass

        global __cusolverDnSgetrf
        try:
            __cusolverDnSgetrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgetrf')
        except:
            pass

        global __cusolverDnDgetrf
        try:
            __cusolverDnDgetrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgetrf')
        except:
            pass

        global __cusolverDnCgetrf
        try:
            __cusolverDnCgetrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgetrf')
        except:
            pass

        global __cusolverDnZgetrf
        try:
            __cusolverDnZgetrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgetrf')
        except:
            pass

        global __cusolverDnSlaswp
        try:
            __cusolverDnSlaswp = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSlaswp')
        except:
            pass

        global __cusolverDnDlaswp
        try:
            __cusolverDnDlaswp = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDlaswp')
        except:
            pass

        global __cusolverDnClaswp
        try:
            __cusolverDnClaswp = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnClaswp')
        except:
            pass

        global __cusolverDnZlaswp
        try:
            __cusolverDnZlaswp = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZlaswp')
        except:
            pass

        global __cusolverDnSgetrs
        try:
            __cusolverDnSgetrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgetrs')
        except:
            pass

        global __cusolverDnDgetrs
        try:
            __cusolverDnDgetrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgetrs')
        except:
            pass

        global __cusolverDnCgetrs
        try:
            __cusolverDnCgetrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgetrs')
        except:
            pass

        global __cusolverDnZgetrs
        try:
            __cusolverDnZgetrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgetrs')
        except:
            pass

        global __cusolverDnSgeqrf_bufferSize
        try:
            __cusolverDnSgeqrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgeqrf_bufferSize')
        except:
            pass

        global __cusolverDnDgeqrf_bufferSize
        try:
            __cusolverDnDgeqrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgeqrf_bufferSize')
        except:
            pass

        global __cusolverDnCgeqrf_bufferSize
        try:
            __cusolverDnCgeqrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgeqrf_bufferSize')
        except:
            pass

        global __cusolverDnZgeqrf_bufferSize
        try:
            __cusolverDnZgeqrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgeqrf_bufferSize')
        except:
            pass

        global __cusolverDnSgeqrf
        try:
            __cusolverDnSgeqrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgeqrf')
        except:
            pass

        global __cusolverDnDgeqrf
        try:
            __cusolverDnDgeqrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgeqrf')
        except:
            pass

        global __cusolverDnCgeqrf
        try:
            __cusolverDnCgeqrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgeqrf')
        except:
            pass

        global __cusolverDnZgeqrf
        try:
            __cusolverDnZgeqrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgeqrf')
        except:
            pass

        global __cusolverDnSorgqr_bufferSize
        try:
            __cusolverDnSorgqr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSorgqr_bufferSize')
        except:
            pass

        global __cusolverDnDorgqr_bufferSize
        try:
            __cusolverDnDorgqr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDorgqr_bufferSize')
        except:
            pass

        global __cusolverDnCungqr_bufferSize
        try:
            __cusolverDnCungqr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCungqr_bufferSize')
        except:
            pass

        global __cusolverDnZungqr_bufferSize
        try:
            __cusolverDnZungqr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZungqr_bufferSize')
        except:
            pass

        global __cusolverDnSorgqr
        try:
            __cusolverDnSorgqr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSorgqr')
        except:
            pass

        global __cusolverDnDorgqr
        try:
            __cusolverDnDorgqr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDorgqr')
        except:
            pass

        global __cusolverDnCungqr
        try:
            __cusolverDnCungqr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCungqr')
        except:
            pass

        global __cusolverDnZungqr
        try:
            __cusolverDnZungqr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZungqr')
        except:
            pass

        global __cusolverDnSormqr_bufferSize
        try:
            __cusolverDnSormqr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSormqr_bufferSize')
        except:
            pass

        global __cusolverDnDormqr_bufferSize
        try:
            __cusolverDnDormqr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDormqr_bufferSize')
        except:
            pass

        global __cusolverDnCunmqr_bufferSize
        try:
            __cusolverDnCunmqr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCunmqr_bufferSize')
        except:
            pass

        global __cusolverDnZunmqr_bufferSize
        try:
            __cusolverDnZunmqr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZunmqr_bufferSize')
        except:
            pass

        global __cusolverDnSormqr
        try:
            __cusolverDnSormqr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSormqr')
        except:
            pass

        global __cusolverDnDormqr
        try:
            __cusolverDnDormqr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDormqr')
        except:
            pass

        global __cusolverDnCunmqr
        try:
            __cusolverDnCunmqr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCunmqr')
        except:
            pass

        global __cusolverDnZunmqr
        try:
            __cusolverDnZunmqr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZunmqr')
        except:
            pass

        global __cusolverDnSsytrf_bufferSize
        try:
            __cusolverDnSsytrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsytrf_bufferSize')
        except:
            pass

        global __cusolverDnDsytrf_bufferSize
        try:
            __cusolverDnDsytrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsytrf_bufferSize')
        except:
            pass

        global __cusolverDnCsytrf_bufferSize
        try:
            __cusolverDnCsytrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCsytrf_bufferSize')
        except:
            pass

        global __cusolverDnZsytrf_bufferSize
        try:
            __cusolverDnZsytrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZsytrf_bufferSize')
        except:
            pass

        global __cusolverDnSsytrf
        try:
            __cusolverDnSsytrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsytrf')
        except:
            pass

        global __cusolverDnDsytrf
        try:
            __cusolverDnDsytrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsytrf')
        except:
            pass

        global __cusolverDnCsytrf
        try:
            __cusolverDnCsytrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCsytrf')
        except:
            pass

        global __cusolverDnZsytrf
        try:
            __cusolverDnZsytrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZsytrf')
        except:
            pass

        global __cusolverDnSsytri_bufferSize
        try:
            __cusolverDnSsytri_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsytri_bufferSize')
        except:
            pass

        global __cusolverDnDsytri_bufferSize
        try:
            __cusolverDnDsytri_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsytri_bufferSize')
        except:
            pass

        global __cusolverDnCsytri_bufferSize
        try:
            __cusolverDnCsytri_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCsytri_bufferSize')
        except:
            pass

        global __cusolverDnZsytri_bufferSize
        try:
            __cusolverDnZsytri_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZsytri_bufferSize')
        except:
            pass

        global __cusolverDnSsytri
        try:
            __cusolverDnSsytri = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsytri')
        except:
            pass

        global __cusolverDnDsytri
        try:
            __cusolverDnDsytri = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsytri')
        except:
            pass

        global __cusolverDnCsytri
        try:
            __cusolverDnCsytri = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCsytri')
        except:
            pass

        global __cusolverDnZsytri
        try:
            __cusolverDnZsytri = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZsytri')
        except:
            pass

        global __cusolverDnSgebrd_bufferSize
        try:
            __cusolverDnSgebrd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgebrd_bufferSize')
        except:
            pass

        global __cusolverDnDgebrd_bufferSize
        try:
            __cusolverDnDgebrd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgebrd_bufferSize')
        except:
            pass

        global __cusolverDnCgebrd_bufferSize
        try:
            __cusolverDnCgebrd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgebrd_bufferSize')
        except:
            pass

        global __cusolverDnZgebrd_bufferSize
        try:
            __cusolverDnZgebrd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgebrd_bufferSize')
        except:
            pass

        global __cusolverDnSgebrd
        try:
            __cusolverDnSgebrd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgebrd')
        except:
            pass

        global __cusolverDnDgebrd
        try:
            __cusolverDnDgebrd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgebrd')
        except:
            pass

        global __cusolverDnCgebrd
        try:
            __cusolverDnCgebrd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgebrd')
        except:
            pass

        global __cusolverDnZgebrd
        try:
            __cusolverDnZgebrd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgebrd')
        except:
            pass

        global __cusolverDnSorgbr_bufferSize
        try:
            __cusolverDnSorgbr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSorgbr_bufferSize')
        except:
            pass

        global __cusolverDnDorgbr_bufferSize
        try:
            __cusolverDnDorgbr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDorgbr_bufferSize')
        except:
            pass

        global __cusolverDnCungbr_bufferSize
        try:
            __cusolverDnCungbr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCungbr_bufferSize')
        except:
            pass

        global __cusolverDnZungbr_bufferSize
        try:
            __cusolverDnZungbr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZungbr_bufferSize')
        except:
            pass

        global __cusolverDnSorgbr
        try:
            __cusolverDnSorgbr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSorgbr')
        except:
            pass

        global __cusolverDnDorgbr
        try:
            __cusolverDnDorgbr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDorgbr')
        except:
            pass

        global __cusolverDnCungbr
        try:
            __cusolverDnCungbr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCungbr')
        except:
            pass

        global __cusolverDnZungbr
        try:
            __cusolverDnZungbr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZungbr')
        except:
            pass

        global __cusolverDnSsytrd_bufferSize
        try:
            __cusolverDnSsytrd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsytrd_bufferSize')
        except:
            pass

        global __cusolverDnDsytrd_bufferSize
        try:
            __cusolverDnDsytrd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsytrd_bufferSize')
        except:
            pass

        global __cusolverDnChetrd_bufferSize
        try:
            __cusolverDnChetrd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnChetrd_bufferSize')
        except:
            pass

        global __cusolverDnZhetrd_bufferSize
        try:
            __cusolverDnZhetrd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZhetrd_bufferSize')
        except:
            pass

        global __cusolverDnSsytrd
        try:
            __cusolverDnSsytrd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsytrd')
        except:
            pass

        global __cusolverDnDsytrd
        try:
            __cusolverDnDsytrd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsytrd')
        except:
            pass

        global __cusolverDnChetrd
        try:
            __cusolverDnChetrd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnChetrd')
        except:
            pass

        global __cusolverDnZhetrd
        try:
            __cusolverDnZhetrd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZhetrd')
        except:
            pass

        global __cusolverDnSorgtr_bufferSize
        try:
            __cusolverDnSorgtr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSorgtr_bufferSize')
        except:
            pass

        global __cusolverDnDorgtr_bufferSize
        try:
            __cusolverDnDorgtr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDorgtr_bufferSize')
        except:
            pass

        global __cusolverDnCungtr_bufferSize
        try:
            __cusolverDnCungtr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCungtr_bufferSize')
        except:
            pass

        global __cusolverDnZungtr_bufferSize
        try:
            __cusolverDnZungtr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZungtr_bufferSize')
        except:
            pass

        global __cusolverDnSorgtr
        try:
            __cusolverDnSorgtr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSorgtr')
        except:
            pass

        global __cusolverDnDorgtr
        try:
            __cusolverDnDorgtr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDorgtr')
        except:
            pass

        global __cusolverDnCungtr
        try:
            __cusolverDnCungtr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCungtr')
        except:
            pass

        global __cusolverDnZungtr
        try:
            __cusolverDnZungtr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZungtr')
        except:
            pass

        global __cusolverDnSormtr_bufferSize
        try:
            __cusolverDnSormtr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSormtr_bufferSize')
        except:
            pass

        global __cusolverDnDormtr_bufferSize
        try:
            __cusolverDnDormtr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDormtr_bufferSize')
        except:
            pass

        global __cusolverDnCunmtr_bufferSize
        try:
            __cusolverDnCunmtr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCunmtr_bufferSize')
        except:
            pass

        global __cusolverDnZunmtr_bufferSize
        try:
            __cusolverDnZunmtr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZunmtr_bufferSize')
        except:
            pass

        global __cusolverDnSormtr
        try:
            __cusolverDnSormtr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSormtr')
        except:
            pass

        global __cusolverDnDormtr
        try:
            __cusolverDnDormtr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDormtr')
        except:
            pass

        global __cusolverDnCunmtr
        try:
            __cusolverDnCunmtr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCunmtr')
        except:
            pass

        global __cusolverDnZunmtr
        try:
            __cusolverDnZunmtr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZunmtr')
        except:
            pass

        global __cusolverDnSgesvd_bufferSize
        try:
            __cusolverDnSgesvd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgesvd_bufferSize')
        except:
            pass

        global __cusolverDnDgesvd_bufferSize
        try:
            __cusolverDnDgesvd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgesvd_bufferSize')
        except:
            pass

        global __cusolverDnCgesvd_bufferSize
        try:
            __cusolverDnCgesvd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgesvd_bufferSize')
        except:
            pass

        global __cusolverDnZgesvd_bufferSize
        try:
            __cusolverDnZgesvd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgesvd_bufferSize')
        except:
            pass

        global __cusolverDnSgesvd
        try:
            __cusolverDnSgesvd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgesvd')
        except:
            pass

        global __cusolverDnDgesvd
        try:
            __cusolverDnDgesvd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgesvd')
        except:
            pass

        global __cusolverDnCgesvd
        try:
            __cusolverDnCgesvd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgesvd')
        except:
            pass

        global __cusolverDnZgesvd
        try:
            __cusolverDnZgesvd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgesvd')
        except:
            pass

        global __cusolverDnSsyevd_bufferSize
        try:
            __cusolverDnSsyevd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsyevd_bufferSize')
        except:
            pass

        global __cusolverDnDsyevd_bufferSize
        try:
            __cusolverDnDsyevd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsyevd_bufferSize')
        except:
            pass

        global __cusolverDnCheevd_bufferSize
        try:
            __cusolverDnCheevd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCheevd_bufferSize')
        except:
            pass

        global __cusolverDnZheevd_bufferSize
        try:
            __cusolverDnZheevd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZheevd_bufferSize')
        except:
            pass

        global __cusolverDnSsyevd
        try:
            __cusolverDnSsyevd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsyevd')
        except:
            pass

        global __cusolverDnDsyevd
        try:
            __cusolverDnDsyevd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsyevd')
        except:
            pass

        global __cusolverDnCheevd
        try:
            __cusolverDnCheevd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCheevd')
        except:
            pass

        global __cusolverDnZheevd
        try:
            __cusolverDnZheevd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZheevd')
        except:
            pass

        global __cusolverDnSsyevdx_bufferSize
        try:
            __cusolverDnSsyevdx_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsyevdx_bufferSize')
        except:
            pass

        global __cusolverDnDsyevdx_bufferSize
        try:
            __cusolverDnDsyevdx_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsyevdx_bufferSize')
        except:
            pass

        global __cusolverDnCheevdx_bufferSize
        try:
            __cusolverDnCheevdx_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCheevdx_bufferSize')
        except:
            pass

        global __cusolverDnZheevdx_bufferSize
        try:
            __cusolverDnZheevdx_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZheevdx_bufferSize')
        except:
            pass

        global __cusolverDnSsyevdx
        try:
            __cusolverDnSsyevdx = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsyevdx')
        except:
            pass

        global __cusolverDnDsyevdx
        try:
            __cusolverDnDsyevdx = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsyevdx')
        except:
            pass

        global __cusolverDnCheevdx
        try:
            __cusolverDnCheevdx = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCheevdx')
        except:
            pass

        global __cusolverDnZheevdx
        try:
            __cusolverDnZheevdx = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZheevdx')
        except:
            pass

        global __cusolverDnSsygvdx_bufferSize
        try:
            __cusolverDnSsygvdx_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsygvdx_bufferSize')
        except:
            pass

        global __cusolverDnDsygvdx_bufferSize
        try:
            __cusolverDnDsygvdx_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsygvdx_bufferSize')
        except:
            pass

        global __cusolverDnChegvdx_bufferSize
        try:
            __cusolverDnChegvdx_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnChegvdx_bufferSize')
        except:
            pass

        global __cusolverDnZhegvdx_bufferSize
        try:
            __cusolverDnZhegvdx_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZhegvdx_bufferSize')
        except:
            pass

        global __cusolverDnSsygvdx
        try:
            __cusolverDnSsygvdx = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsygvdx')
        except:
            pass

        global __cusolverDnDsygvdx
        try:
            __cusolverDnDsygvdx = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsygvdx')
        except:
            pass

        global __cusolverDnChegvdx
        try:
            __cusolverDnChegvdx = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnChegvdx')
        except:
            pass

        global __cusolverDnZhegvdx
        try:
            __cusolverDnZhegvdx = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZhegvdx')
        except:
            pass

        global __cusolverDnSsygvd_bufferSize
        try:
            __cusolverDnSsygvd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsygvd_bufferSize')
        except:
            pass

        global __cusolverDnDsygvd_bufferSize
        try:
            __cusolverDnDsygvd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsygvd_bufferSize')
        except:
            pass

        global __cusolverDnChegvd_bufferSize
        try:
            __cusolverDnChegvd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnChegvd_bufferSize')
        except:
            pass

        global __cusolverDnZhegvd_bufferSize
        try:
            __cusolverDnZhegvd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZhegvd_bufferSize')
        except:
            pass

        global __cusolverDnSsygvd
        try:
            __cusolverDnSsygvd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsygvd')
        except:
            pass

        global __cusolverDnDsygvd
        try:
            __cusolverDnDsygvd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsygvd')
        except:
            pass

        global __cusolverDnChegvd
        try:
            __cusolverDnChegvd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnChegvd')
        except:
            pass

        global __cusolverDnZhegvd
        try:
            __cusolverDnZhegvd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZhegvd')
        except:
            pass

        global __cusolverDnCreateSyevjInfo
        try:
            __cusolverDnCreateSyevjInfo = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCreateSyevjInfo')
        except:
            pass

        global __cusolverDnDestroySyevjInfo
        try:
            __cusolverDnDestroySyevjInfo = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDestroySyevjInfo')
        except:
            pass

        global __cusolverDnXsyevjSetTolerance
        try:
            __cusolverDnXsyevjSetTolerance = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevjSetTolerance')
        except:
            pass

        global __cusolverDnXsyevjSetMaxSweeps
        try:
            __cusolverDnXsyevjSetMaxSweeps = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevjSetMaxSweeps')
        except:
            pass

        global __cusolverDnXsyevjSetSortEig
        try:
            __cusolverDnXsyevjSetSortEig = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevjSetSortEig')
        except:
            pass

        global __cusolverDnXsyevjGetResidual
        try:
            __cusolverDnXsyevjGetResidual = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevjGetResidual')
        except:
            pass

        global __cusolverDnXsyevjGetSweeps
        try:
            __cusolverDnXsyevjGetSweeps = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevjGetSweeps')
        except:
            pass

        global __cusolverDnSsyevjBatched_bufferSize
        try:
            __cusolverDnSsyevjBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsyevjBatched_bufferSize')
        except:
            pass

        global __cusolverDnDsyevjBatched_bufferSize
        try:
            __cusolverDnDsyevjBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsyevjBatched_bufferSize')
        except:
            pass

        global __cusolverDnCheevjBatched_bufferSize
        try:
            __cusolverDnCheevjBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCheevjBatched_bufferSize')
        except:
            pass

        global __cusolverDnZheevjBatched_bufferSize
        try:
            __cusolverDnZheevjBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZheevjBatched_bufferSize')
        except:
            pass

        global __cusolverDnSsyevjBatched
        try:
            __cusolverDnSsyevjBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsyevjBatched')
        except:
            pass

        global __cusolverDnDsyevjBatched
        try:
            __cusolverDnDsyevjBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsyevjBatched')
        except:
            pass

        global __cusolverDnCheevjBatched
        try:
            __cusolverDnCheevjBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCheevjBatched')
        except:
            pass

        global __cusolverDnZheevjBatched
        try:
            __cusolverDnZheevjBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZheevjBatched')
        except:
            pass

        global __cusolverDnSsyevj_bufferSize
        try:
            __cusolverDnSsyevj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsyevj_bufferSize')
        except:
            pass

        global __cusolverDnDsyevj_bufferSize
        try:
            __cusolverDnDsyevj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsyevj_bufferSize')
        except:
            pass

        global __cusolverDnCheevj_bufferSize
        try:
            __cusolverDnCheevj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCheevj_bufferSize')
        except:
            pass

        global __cusolverDnZheevj_bufferSize
        try:
            __cusolverDnZheevj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZheevj_bufferSize')
        except:
            pass

        global __cusolverDnSsyevj
        try:
            __cusolverDnSsyevj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsyevj')
        except:
            pass

        global __cusolverDnDsyevj
        try:
            __cusolverDnDsyevj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsyevj')
        except:
            pass

        global __cusolverDnCheevj
        try:
            __cusolverDnCheevj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCheevj')
        except:
            pass

        global __cusolverDnZheevj
        try:
            __cusolverDnZheevj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZheevj')
        except:
            pass

        global __cusolverDnSsygvj_bufferSize
        try:
            __cusolverDnSsygvj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsygvj_bufferSize')
        except:
            pass

        global __cusolverDnDsygvj_bufferSize
        try:
            __cusolverDnDsygvj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsygvj_bufferSize')
        except:
            pass

        global __cusolverDnChegvj_bufferSize
        try:
            __cusolverDnChegvj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnChegvj_bufferSize')
        except:
            pass

        global __cusolverDnZhegvj_bufferSize
        try:
            __cusolverDnZhegvj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZhegvj_bufferSize')
        except:
            pass

        global __cusolverDnSsygvj
        try:
            __cusolverDnSsygvj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSsygvj')
        except:
            pass

        global __cusolverDnDsygvj
        try:
            __cusolverDnDsygvj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDsygvj')
        except:
            pass

        global __cusolverDnChegvj
        try:
            __cusolverDnChegvj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnChegvj')
        except:
            pass

        global __cusolverDnZhegvj
        try:
            __cusolverDnZhegvj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZhegvj')
        except:
            pass

        global __cusolverDnCreateGesvdjInfo
        try:
            __cusolverDnCreateGesvdjInfo = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCreateGesvdjInfo')
        except:
            pass

        global __cusolverDnDestroyGesvdjInfo
        try:
            __cusolverDnDestroyGesvdjInfo = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDestroyGesvdjInfo')
        except:
            pass

        global __cusolverDnXgesvdjSetTolerance
        try:
            __cusolverDnXgesvdjSetTolerance = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvdjSetTolerance')
        except:
            pass

        global __cusolverDnXgesvdjSetMaxSweeps
        try:
            __cusolverDnXgesvdjSetMaxSweeps = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvdjSetMaxSweeps')
        except:
            pass

        global __cusolverDnXgesvdjSetSortEig
        try:
            __cusolverDnXgesvdjSetSortEig = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvdjSetSortEig')
        except:
            pass

        global __cusolverDnXgesvdjGetResidual
        try:
            __cusolverDnXgesvdjGetResidual = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvdjGetResidual')
        except:
            pass

        global __cusolverDnXgesvdjGetSweeps
        try:
            __cusolverDnXgesvdjGetSweeps = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvdjGetSweeps')
        except:
            pass

        global __cusolverDnSgesvdjBatched_bufferSize
        try:
            __cusolverDnSgesvdjBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgesvdjBatched_bufferSize')
        except:
            pass

        global __cusolverDnDgesvdjBatched_bufferSize
        try:
            __cusolverDnDgesvdjBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgesvdjBatched_bufferSize')
        except:
            pass

        global __cusolverDnCgesvdjBatched_bufferSize
        try:
            __cusolverDnCgesvdjBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgesvdjBatched_bufferSize')
        except:
            pass

        global __cusolverDnZgesvdjBatched_bufferSize
        try:
            __cusolverDnZgesvdjBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgesvdjBatched_bufferSize')
        except:
            pass

        global __cusolverDnSgesvdjBatched
        try:
            __cusolverDnSgesvdjBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgesvdjBatched')
        except:
            pass

        global __cusolverDnDgesvdjBatched
        try:
            __cusolverDnDgesvdjBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgesvdjBatched')
        except:
            pass

        global __cusolverDnCgesvdjBatched
        try:
            __cusolverDnCgesvdjBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgesvdjBatched')
        except:
            pass

        global __cusolverDnZgesvdjBatched
        try:
            __cusolverDnZgesvdjBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgesvdjBatched')
        except:
            pass

        global __cusolverDnSgesvdj_bufferSize
        try:
            __cusolverDnSgesvdj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgesvdj_bufferSize')
        except:
            pass

        global __cusolverDnDgesvdj_bufferSize
        try:
            __cusolverDnDgesvdj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgesvdj_bufferSize')
        except:
            pass

        global __cusolverDnCgesvdj_bufferSize
        try:
            __cusolverDnCgesvdj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgesvdj_bufferSize')
        except:
            pass

        global __cusolverDnZgesvdj_bufferSize
        try:
            __cusolverDnZgesvdj_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgesvdj_bufferSize')
        except:
            pass

        global __cusolverDnSgesvdj
        try:
            __cusolverDnSgesvdj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgesvdj')
        except:
            pass

        global __cusolverDnDgesvdj
        try:
            __cusolverDnDgesvdj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgesvdj')
        except:
            pass

        global __cusolverDnCgesvdj
        try:
            __cusolverDnCgesvdj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgesvdj')
        except:
            pass

        global __cusolverDnZgesvdj
        try:
            __cusolverDnZgesvdj = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgesvdj')
        except:
            pass

        global __cusolverDnSgesvdaStridedBatched_bufferSize
        try:
            __cusolverDnSgesvdaStridedBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgesvdaStridedBatched_bufferSize')
        except:
            pass

        global __cusolverDnDgesvdaStridedBatched_bufferSize
        try:
            __cusolverDnDgesvdaStridedBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgesvdaStridedBatched_bufferSize')
        except:
            pass

        global __cusolverDnCgesvdaStridedBatched_bufferSize
        try:
            __cusolverDnCgesvdaStridedBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgesvdaStridedBatched_bufferSize')
        except:
            pass

        global __cusolverDnZgesvdaStridedBatched_bufferSize
        try:
            __cusolverDnZgesvdaStridedBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgesvdaStridedBatched_bufferSize')
        except:
            pass

        global __cusolverDnSgesvdaStridedBatched
        try:
            __cusolverDnSgesvdaStridedBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSgesvdaStridedBatched')
        except:
            pass

        global __cusolverDnDgesvdaStridedBatched
        try:
            __cusolverDnDgesvdaStridedBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDgesvdaStridedBatched')
        except:
            pass

        global __cusolverDnCgesvdaStridedBatched
        try:
            __cusolverDnCgesvdaStridedBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCgesvdaStridedBatched')
        except:
            pass

        global __cusolverDnZgesvdaStridedBatched
        try:
            __cusolverDnZgesvdaStridedBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnZgesvdaStridedBatched')
        except:
            pass

        global __cusolverDnCreateParams
        try:
            __cusolverDnCreateParams = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnCreateParams')
        except:
            pass

        global __cusolverDnDestroyParams
        try:
            __cusolverDnDestroyParams = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnDestroyParams')
        except:
            pass

        global __cusolverDnSetAdvOptions
        try:
            __cusolverDnSetAdvOptions = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSetAdvOptions')
        except:
            pass

        global __cusolverDnXpotrf_bufferSize
        try:
            __cusolverDnXpotrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXpotrf_bufferSize')
        except:
            pass

        global __cusolverDnXpotrf
        try:
            __cusolverDnXpotrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXpotrf')
        except:
            pass

        global __cusolverDnXpotrs
        try:
            __cusolverDnXpotrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXpotrs')
        except:
            pass

        global __cusolverDnXgeqrf_bufferSize
        try:
            __cusolverDnXgeqrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgeqrf_bufferSize')
        except:
            pass

        global __cusolverDnXgeqrf
        try:
            __cusolverDnXgeqrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgeqrf')
        except:
            pass

        global __cusolverDnXgetrf_bufferSize
        try:
            __cusolverDnXgetrf_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgetrf_bufferSize')
        except:
            pass

        global __cusolverDnXgetrf
        try:
            __cusolverDnXgetrf = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgetrf')
        except:
            pass

        global __cusolverDnXgetrs
        try:
            __cusolverDnXgetrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgetrs')
        except:
            pass

        global __cusolverDnXsyevd_bufferSize
        try:
            __cusolverDnXsyevd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevd_bufferSize')
        except:
            pass

        global __cusolverDnXsyevd
        try:
            __cusolverDnXsyevd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevd')
        except:
            pass

        global __cusolverDnXsyevdx_bufferSize
        try:
            __cusolverDnXsyevdx_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevdx_bufferSize')
        except:
            pass

        global __cusolverDnXsyevdx
        try:
            __cusolverDnXsyevdx = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevdx')
        except:
            pass

        global __cusolverDnXgesvd_bufferSize
        try:
            __cusolverDnXgesvd_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvd_bufferSize')
        except:
            pass

        global __cusolverDnXgesvd
        try:
            __cusolverDnXgesvd = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvd')
        except:
            pass

        global __cusolverDnXgesvdp_bufferSize
        try:
            __cusolverDnXgesvdp_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvdp_bufferSize')
        except:
            pass

        global __cusolverDnXgesvdp
        try:
            __cusolverDnXgesvdp = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvdp')
        except:
            pass

        global __cusolverDnXgesvdr_bufferSize
        try:
            __cusolverDnXgesvdr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvdr_bufferSize')
        except:
            pass

        global __cusolverDnXgesvdr
        try:
            __cusolverDnXgesvdr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgesvdr')
        except:
            pass

        global __cusolverDnXsytrs_bufferSize
        try:
            __cusolverDnXsytrs_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsytrs_bufferSize')
        except:
            pass

        global __cusolverDnXsytrs
        try:
            __cusolverDnXsytrs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsytrs')
        except:
            pass

        global __cusolverDnXtrtri_bufferSize
        try:
            __cusolverDnXtrtri_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXtrtri_bufferSize')
        except:
            pass

        global __cusolverDnXtrtri
        try:
            __cusolverDnXtrtri = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXtrtri')
        except:
            pass

        global __cusolverDnLoggerSetCallback
        try:
            __cusolverDnLoggerSetCallback = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnLoggerSetCallback')
        except:
            pass

        global __cusolverDnLoggerSetFile
        try:
            __cusolverDnLoggerSetFile = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnLoggerSetFile')
        except:
            pass

        global __cusolverDnLoggerOpenFile
        try:
            __cusolverDnLoggerOpenFile = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnLoggerOpenFile')
        except:
            pass

        global __cusolverDnLoggerSetLevel
        try:
            __cusolverDnLoggerSetLevel = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnLoggerSetLevel')
        except:
            pass

        global __cusolverDnLoggerSetMask
        try:
            __cusolverDnLoggerSetMask = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnLoggerSetMask')
        except:
            pass

        global __cusolverDnLoggerForceDisable
        try:
            __cusolverDnLoggerForceDisable = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnLoggerForceDisable')
        except:
            pass

        global __cusolverDnSetDeterministicMode
        try:
            __cusolverDnSetDeterministicMode = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnSetDeterministicMode')
        except:
            pass

        global __cusolverDnGetDeterministicMode
        try:
            __cusolverDnGetDeterministicMode = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnGetDeterministicMode')
        except:
            pass

        global __cusolverDnXlarft_bufferSize
        try:
            __cusolverDnXlarft_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXlarft_bufferSize')
        except:
            pass

        global __cusolverDnXlarft
        try:
            __cusolverDnXlarft = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXlarft')
        except:
            pass

        global __cusolverDnXsyevBatched_bufferSize
        try:
            __cusolverDnXsyevBatched_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevBatched_bufferSize')
        except:
            pass

        global __cusolverDnXsyevBatched
        try:
            __cusolverDnXsyevBatched = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXsyevBatched')
        except:
            pass

        global __cusolverDnXgeev_bufferSize
        try:
            __cusolverDnXgeev_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgeev_bufferSize')
        except:
            pass

        global __cusolverDnXgeev
        try:
            __cusolverDnXgeev = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverDnXgeev')
        except:
            pass

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

cdef cusolverStatus_t _cusolverDnCreate(cusolverDnHandle_t* handle) except* nogil:
    global __cusolverDnCreate
    _check_or_init_cusolverDn()
    if __cusolverDnCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCreate is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t*) nogil>__cusolverDnCreate)(
        handle)


cdef cusolverStatus_t _cusolverDnDestroy(cusolverDnHandle_t handle) except* nogil:
    global __cusolverDnDestroy
    _check_or_init_cusolverDn()
    if __cusolverDnDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDestroy is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t) nogil>__cusolverDnDestroy)(
        handle)


cdef cusolverStatus_t _cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId) except* nogil:
    global __cusolverDnSetStream
    _check_or_init_cusolverDn()
    if __cusolverDnSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSetStream is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cudaStream_t) nogil>__cusolverDnSetStream)(
        handle, streamId)


cdef cusolverStatus_t _cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t* streamId) except* nogil:
    global __cusolverDnGetStream
    _check_or_init_cusolverDn()
    if __cusolverDnGetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnGetStream is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cudaStream_t*) nogil>__cusolverDnGetStream)(
        handle, streamId)


cdef cusolverStatus_t _cusolverDnIRSParamsCreate(cusolverDnIRSParams_t* params_ptr) except* nogil:
    global __cusolverDnIRSParamsCreate
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsCreate is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t*) nogil>__cusolverDnIRSParamsCreate)(
        params_ptr)


cdef cusolverStatus_t _cusolverDnIRSParamsDestroy(cusolverDnIRSParams_t params) except* nogil:
    global __cusolverDnIRSParamsDestroy
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsDestroy is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t) nogil>__cusolverDnIRSParamsDestroy)(
        params)


cdef cusolverStatus_t _cusolverDnIRSParamsSetRefinementSolver(cusolverDnIRSParams_t params, cusolverIRSRefinement_t refinement_solver) except* nogil:
    global __cusolverDnIRSParamsSetRefinementSolver
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetRefinementSolver == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetRefinementSolver is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverIRSRefinement_t) nogil>__cusolverDnIRSParamsSetRefinementSolver)(
        params, refinement_solver)


cdef cusolverStatus_t _cusolverDnIRSParamsSetSolverMainPrecision(cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision) except* nogil:
    global __cusolverDnIRSParamsSetSolverMainPrecision
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetSolverMainPrecision == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetSolverMainPrecision is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverPrecType_t) nogil>__cusolverDnIRSParamsSetSolverMainPrecision)(
        params, solver_main_precision)


cdef cusolverStatus_t _cusolverDnIRSParamsSetSolverLowestPrecision(cusolverDnIRSParams_t params, cusolverPrecType_t solver_lowest_precision) except* nogil:
    global __cusolverDnIRSParamsSetSolverLowestPrecision
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetSolverLowestPrecision == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetSolverLowestPrecision is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverPrecType_t) nogil>__cusolverDnIRSParamsSetSolverLowestPrecision)(
        params, solver_lowest_precision)


cdef cusolverStatus_t _cusolverDnIRSParamsSetSolverPrecisions(cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision, cusolverPrecType_t solver_lowest_precision) except* nogil:
    global __cusolverDnIRSParamsSetSolverPrecisions
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetSolverPrecisions == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetSolverPrecisions is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverPrecType_t, cusolverPrecType_t) nogil>__cusolverDnIRSParamsSetSolverPrecisions)(
        params, solver_main_precision, solver_lowest_precision)


cdef cusolverStatus_t _cusolverDnIRSParamsSetTol(cusolverDnIRSParams_t params, double val) except* nogil:
    global __cusolverDnIRSParamsSetTol
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetTol == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetTol is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, double) nogil>__cusolverDnIRSParamsSetTol)(
        params, val)


cdef cusolverStatus_t _cusolverDnIRSParamsSetTolInner(cusolverDnIRSParams_t params, double val) except* nogil:
    global __cusolverDnIRSParamsSetTolInner
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetTolInner == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetTolInner is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, double) nogil>__cusolverDnIRSParamsSetTolInner)(
        params, val)


cdef cusolverStatus_t _cusolverDnIRSParamsSetMaxIters(cusolverDnIRSParams_t params, cusolver_int_t maxiters) except* nogil:
    global __cusolverDnIRSParamsSetMaxIters
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetMaxIters == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetMaxIters is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolver_int_t) nogil>__cusolverDnIRSParamsSetMaxIters)(
        params, maxiters)


cdef cusolverStatus_t _cusolverDnIRSParamsSetMaxItersInner(cusolverDnIRSParams_t params, cusolver_int_t maxiters_inner) except* nogil:
    global __cusolverDnIRSParamsSetMaxItersInner
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsSetMaxItersInner == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsSetMaxItersInner is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolver_int_t) nogil>__cusolverDnIRSParamsSetMaxItersInner)(
        params, maxiters_inner)


cdef cusolverStatus_t _cusolverDnIRSParamsGetMaxIters(cusolverDnIRSParams_t params, cusolver_int_t* maxiters) except* nogil:
    global __cusolverDnIRSParamsGetMaxIters
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsGetMaxIters == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsGetMaxIters is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolver_int_t*) nogil>__cusolverDnIRSParamsGetMaxIters)(
        params, maxiters)


cdef cusolverStatus_t _cusolverDnIRSParamsEnableFallback(cusolverDnIRSParams_t params) except* nogil:
    global __cusolverDnIRSParamsEnableFallback
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsEnableFallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsEnableFallback is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t) nogil>__cusolverDnIRSParamsEnableFallback)(
        params)


cdef cusolverStatus_t _cusolverDnIRSParamsDisableFallback(cusolverDnIRSParams_t params) except* nogil:
    global __cusolverDnIRSParamsDisableFallback
    _check_or_init_cusolverDn()
    if __cusolverDnIRSParamsDisableFallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSParamsDisableFallback is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSParams_t) nogil>__cusolverDnIRSParamsDisableFallback)(
        params)


cdef cusolverStatus_t _cusolverDnIRSInfosDestroy(cusolverDnIRSInfos_t infos) except* nogil:
    global __cusolverDnIRSInfosDestroy
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosDestroy is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t) nogil>__cusolverDnIRSInfosDestroy)(
        infos)


cdef cusolverStatus_t _cusolverDnIRSInfosCreate(cusolverDnIRSInfos_t* infos_ptr) except* nogil:
    global __cusolverDnIRSInfosCreate
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosCreate is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t*) nogil>__cusolverDnIRSInfosCreate)(
        infos_ptr)


cdef cusolverStatus_t _cusolverDnIRSInfosGetNiters(cusolverDnIRSInfos_t infos, cusolver_int_t* niters) except* nogil:
    global __cusolverDnIRSInfosGetNiters
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosGetNiters == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosGetNiters is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t, cusolver_int_t*) nogil>__cusolverDnIRSInfosGetNiters)(
        infos, niters)


cdef cusolverStatus_t _cusolverDnIRSInfosGetOuterNiters(cusolverDnIRSInfos_t infos, cusolver_int_t* outer_niters) except* nogil:
    global __cusolverDnIRSInfosGetOuterNiters
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosGetOuterNiters == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosGetOuterNiters is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t, cusolver_int_t*) nogil>__cusolverDnIRSInfosGetOuterNiters)(
        infos, outer_niters)


cdef cusolverStatus_t _cusolverDnIRSInfosRequestResidual(cusolverDnIRSInfos_t infos) except* nogil:
    global __cusolverDnIRSInfosRequestResidual
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosRequestResidual == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosRequestResidual is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t) nogil>__cusolverDnIRSInfosRequestResidual)(
        infos)


cdef cusolverStatus_t _cusolverDnIRSInfosGetResidualHistory(cusolverDnIRSInfos_t infos, void** residual_history) except* nogil:
    global __cusolverDnIRSInfosGetResidualHistory
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosGetResidualHistory == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosGetResidualHistory is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t, void**) nogil>__cusolverDnIRSInfosGetResidualHistory)(
        infos, residual_history)


cdef cusolverStatus_t _cusolverDnIRSInfosGetMaxIters(cusolverDnIRSInfos_t infos, cusolver_int_t* maxiters) except* nogil:
    global __cusolverDnIRSInfosGetMaxIters
    _check_or_init_cusolverDn()
    if __cusolverDnIRSInfosGetMaxIters == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSInfosGetMaxIters is not found")
    return (<cusolverStatus_t (*)(cusolverDnIRSInfos_t, cusolver_int_t*) nogil>__cusolverDnIRSInfosGetMaxIters)(
        infos, maxiters)


cdef cusolverStatus_t _cusolverDnZZgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnZZgesv
    _check_or_init_cusolverDn()
    if __cusolverDnZZgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZZgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnZZgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZCgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnZCgesv
    _check_or_init_cusolverDn()
    if __cusolverDnZCgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZCgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnZCgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZKgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnZKgesv
    _check_or_init_cusolverDn()
    if __cusolverDnZKgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZKgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnZKgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZEgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnZEgesv
    _check_or_init_cusolverDn()
    if __cusolverDnZEgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZEgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnZEgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZYgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnZYgesv
    _check_or_init_cusolverDn()
    if __cusolverDnZYgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZYgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnZYgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCCgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnCCgesv
    _check_or_init_cusolverDn()
    if __cusolverDnCCgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCCgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnCCgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCEgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnCEgesv
    _check_or_init_cusolverDn()
    if __cusolverDnCEgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCEgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnCEgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCKgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnCKgesv
    _check_or_init_cusolverDn()
    if __cusolverDnCKgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCKgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnCKgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCYgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnCYgesv
    _check_or_init_cusolverDn()
    if __cusolverDnCYgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCYgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnCYgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDDgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnDDgesv
    _check_or_init_cusolverDn()
    if __cusolverDnDDgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDDgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnDDgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDSgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnDSgesv
    _check_or_init_cusolverDn()
    if __cusolverDnDSgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDSgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnDSgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDHgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnDHgesv
    _check_or_init_cusolverDn()
    if __cusolverDnDHgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDHgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnDHgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDBgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnDBgesv
    _check_or_init_cusolverDn()
    if __cusolverDnDBgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDBgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnDBgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDXgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnDXgesv
    _check_or_init_cusolverDn()
    if __cusolverDnDXgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDXgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnDXgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSSgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnSSgesv
    _check_or_init_cusolverDn()
    if __cusolverDnSSgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSSgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnSSgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSHgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnSHgesv
    _check_or_init_cusolverDn()
    if __cusolverDnSHgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSHgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnSHgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSBgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnSBgesv
    _check_or_init_cusolverDn()
    if __cusolverDnSBgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSBgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnSBgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSXgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnSXgesv
    _check_or_init_cusolverDn()
    if __cusolverDnSXgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSXgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnSXgesv)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZZgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnZZgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZZgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZZgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnZZgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZCgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnZCgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZCgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZCgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnZCgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZKgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnZKgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZKgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZKgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnZKgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZEgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnZEgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZEgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZEgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnZEgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZYgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnZYgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZYgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZYgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cusolver_int_t*, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnZYgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCCgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnCCgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCCgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCCgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnCCgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCKgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnCKgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCKgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCKgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnCKgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCEgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnCEgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCEgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCEgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnCEgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCYgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnCYgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCYgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCYgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cusolver_int_t*, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnCYgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDDgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnDDgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDDgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDDgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnDDgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDSgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnDSgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDSgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDSgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnDSgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDHgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnDHgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDHgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDHgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnDHgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDBgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnDBgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDBgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDBgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnDBgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDXgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnDXgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDXgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDXgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, cusolver_int_t*, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnDXgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSSgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnSSgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSSgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSSgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnSSgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSHgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnSHgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSHgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSHgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnSHgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSBgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnSBgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSBgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSBgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnSBgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSXgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, cusolver_int_t* dipiv, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnSXgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSXgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSXgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, cusolver_int_t*, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnSXgesv_bufferSize)(
        handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZZgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnZZgels
    _check_or_init_cusolverDn()
    if __cusolverDnZZgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZZgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnZZgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZCgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnZCgels
    _check_or_init_cusolverDn()
    if __cusolverDnZCgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZCgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnZCgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZKgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnZKgels
    _check_or_init_cusolverDn()
    if __cusolverDnZKgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZKgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnZKgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZEgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnZEgels
    _check_or_init_cusolverDn()
    if __cusolverDnZEgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZEgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnZEgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZYgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnZYgels
    _check_or_init_cusolverDn()
    if __cusolverDnZYgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZYgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnZYgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCCgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnCCgels
    _check_or_init_cusolverDn()
    if __cusolverDnCCgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCCgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnCCgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCKgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnCKgels
    _check_or_init_cusolverDn()
    if __cusolverDnCKgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCKgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnCKgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCEgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnCEgels
    _check_or_init_cusolverDn()
    if __cusolverDnCEgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCEgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnCEgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnCYgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnCYgels
    _check_or_init_cusolverDn()
    if __cusolverDnCYgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCYgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnCYgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDDgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnDDgels
    _check_or_init_cusolverDn()
    if __cusolverDnDDgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDDgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnDDgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDSgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnDSgels
    _check_or_init_cusolverDn()
    if __cusolverDnDSgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDSgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnDSgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDHgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnDHgels
    _check_or_init_cusolverDn()
    if __cusolverDnDHgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDHgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnDHgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDBgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnDBgels
    _check_or_init_cusolverDn()
    if __cusolverDnDBgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDBgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnDBgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnDXgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnDXgels
    _check_or_init_cusolverDn()
    if __cusolverDnDXgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDXgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnDXgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSSgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnSSgels
    _check_or_init_cusolverDn()
    if __cusolverDnSSgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSSgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnSSgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSHgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnSHgels
    _check_or_init_cusolverDn()
    if __cusolverDnSHgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSHgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnSHgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSBgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnSBgels
    _check_or_init_cusolverDn()
    if __cusolverDnSBgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSBgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnSBgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnSXgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* iter, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnSXgels
    _check_or_init_cusolverDn()
    if __cusolverDnSXgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSXgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnSXgels)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)


cdef cusolverStatus_t _cusolverDnZZgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnZZgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZZgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZZgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnZZgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZCgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnZCgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZCgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZCgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnZCgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZKgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnZKgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZKgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZKgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnZKgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZEgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnZEgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZEgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZEgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnZEgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnZYgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex* dA, cusolver_int_t ldda, cuDoubleComplex* dB, cusolver_int_t lddb, cuDoubleComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnZYgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZYgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZYgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, cuDoubleComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnZYgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCCgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnCCgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCCgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCCgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnCCgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCKgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnCKgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCKgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCKgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnCKgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCEgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnCEgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCEgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCEgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnCEgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnCYgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex* dA, cusolver_int_t ldda, cuComplex* dB, cusolver_int_t lddb, cuComplex* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnCYgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCYgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCYgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, cuComplex*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnCYgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDDgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnDDgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDDgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDDgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnDDgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDSgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnDSgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDSgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDSgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnDSgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDHgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnDHgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDHgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDHgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnDHgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDBgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnDBgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDBgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDBgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnDBgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnDXgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double* dA, cusolver_int_t ldda, double* dB, cusolver_int_t lddb, double* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnDXgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDXgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDXgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, double*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnDXgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSSgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnSSgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSSgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSSgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnSSgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSHgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnSHgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSHgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSHgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnSHgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSBgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnSBgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSBgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSBgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnSBgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSXgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float* dA, cusolver_int_t ldda, float* dB, cusolver_int_t lddb, float* dX, cusolver_int_t lddx, void* dWorkspace, size_t* lwork_bytes) except* nogil:
    global __cusolverDnSXgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSXgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSXgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, float*, cusolver_int_t, void*, size_t*) nogil>__cusolverDnSXgels_bufferSize)(
        handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)


cdef cusolverStatus_t _cusolverDnIRSXgesv(cusolverDnHandle_t handle, cusolverDnIRSParams_t gesv_irs_params, cusolverDnIRSInfos_t gesv_irs_infos, cusolver_int_t n, cusolver_int_t nrhs, void* dA, cusolver_int_t ldda, void* dB, cusolver_int_t lddb, void* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* niters, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnIRSXgesv
    _check_or_init_cusolverDn()
    if __cusolverDnIRSXgesv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSXgesv is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t, cusolver_int_t, cusolver_int_t, void*, cusolver_int_t, void*, cusolver_int_t, void*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnIRSXgesv)(
        handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info)


cdef cusolverStatus_t _cusolverDnIRSXgesv_bufferSize(cusolverDnHandle_t handle, cusolverDnIRSParams_t params, cusolver_int_t n, cusolver_int_t nrhs, size_t* lwork_bytes) except* nogil:
    global __cusolverDnIRSXgesv_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnIRSXgesv_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSXgesv_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t, cusolver_int_t, size_t*) nogil>__cusolverDnIRSXgesv_bufferSize)(
        handle, params, n, nrhs, lwork_bytes)


cdef cusolverStatus_t _cusolverDnIRSXgels(cusolverDnHandle_t handle, cusolverDnIRSParams_t gels_irs_params, cusolverDnIRSInfos_t gels_irs_infos, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, void* dA, cusolver_int_t ldda, void* dB, cusolver_int_t lddb, void* dX, cusolver_int_t lddx, void* dWorkspace, size_t lwork_bytes, cusolver_int_t* niters, cusolver_int_t* d_info) except* nogil:
    global __cusolverDnIRSXgels
    _check_or_init_cusolverDn()
    if __cusolverDnIRSXgels == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSXgels is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, void*, cusolver_int_t, void*, cusolver_int_t, void*, cusolver_int_t, void*, size_t, cusolver_int_t*, cusolver_int_t*) nogil>__cusolverDnIRSXgels)(
        handle, gels_irs_params, gels_irs_infos, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info)


cdef cusolverStatus_t _cusolverDnIRSXgels_bufferSize(cusolverDnHandle_t handle, cusolverDnIRSParams_t params, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, size_t* lwork_bytes) except* nogil:
    global __cusolverDnIRSXgels_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnIRSXgels_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnIRSXgels_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, size_t*) nogil>__cusolverDnIRSXgels_bufferSize)(
        handle, params, m, n, nrhs, lwork_bytes)


cdef cusolverStatus_t _cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* Lwork) except* nogil:
    global __cusolverDnSpotrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSpotrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, int*) nogil>__cusolverDnSpotrf_bufferSize)(
        handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* Lwork) except* nogil:
    global __cusolverDnDpotrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDpotrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, int*) nogil>__cusolverDnDpotrf_bufferSize)(
        handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* Lwork) except* nogil:
    global __cusolverDnCpotrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCpotrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, int*) nogil>__cusolverDnCpotrf_bufferSize)(
        handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* Lwork) except* nogil:
    global __cusolverDnZpotrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZpotrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZpotrf_bufferSize)(
        handle, uplo, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* Workspace, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnSpotrf
    _check_or_init_cusolverDn()
    if __cusolverDnSpotrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, float*, int, int*) nogil>__cusolverDnSpotrf)(
        handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* Workspace, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnDpotrf
    _check_or_init_cusolverDn()
    if __cusolverDnDpotrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, double*, int, int*) nogil>__cusolverDnDpotrf)(
        handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnCpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* Workspace, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnCpotrf
    _check_or_init_cusolverDn()
    if __cusolverDnCpotrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, int*) nogil>__cusolverDnCpotrf)(
        handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* Workspace, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnZpotrf
    _check_or_init_cusolverDn()
    if __cusolverDnZpotrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZpotrf)(
        handle, uplo, n, A, lda, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float* A, int lda, float* B, int ldb, int* devInfo) except* nogil:
    global __cusolverDnSpotrs
    _check_or_init_cusolverDn()
    if __cusolverDnSpotrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const float*, int, float*, int, int*) nogil>__cusolverDnSpotrs)(
        handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* devInfo) except* nogil:
    global __cusolverDnDpotrs
    _check_or_init_cusolverDn()
    if __cusolverDnDpotrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const double*, int, double*, int, int*) nogil>__cusolverDnDpotrs)(
        handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnCpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuComplex* A, int lda, cuComplex* B, int ldb, int* devInfo) except* nogil:
    global __cusolverDnCpotrs
    _check_or_init_cusolverDn()
    if __cusolverDnCpotrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const cuComplex*, int, cuComplex*, int, int*) nogil>__cusolverDnCpotrs)(
        handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnZpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, int* devInfo) except* nogil:
    global __cusolverDnZpotrs
    _check_or_init_cusolverDn()
    if __cusolverDnZpotrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const cuDoubleComplex*, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZpotrs)(
        handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnSpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* Aarray[], int lda, int* infoArray, int batchSize) except* nogil:
    global __cusolverDnSpotrfBatched
    _check_or_init_cusolverDn()
    if __cusolverDnSpotrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotrfBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float**, int, int*, int) nogil>__cusolverDnSpotrfBatched)(
        handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t _cusolverDnDpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* Aarray[], int lda, int* infoArray, int batchSize) except* nogil:
    global __cusolverDnDpotrfBatched
    _check_or_init_cusolverDn()
    if __cusolverDnDpotrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotrfBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double**, int, int*, int) nogil>__cusolverDnDpotrfBatched)(
        handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t _cusolverDnCpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* Aarray[], int lda, int* infoArray, int batchSize) except* nogil:
    global __cusolverDnCpotrfBatched
    _check_or_init_cusolverDn()
    if __cusolverDnCpotrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotrfBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex**, int, int*, int) nogil>__cusolverDnCpotrfBatched)(
        handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t _cusolverDnZpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* Aarray[], int lda, int* infoArray, int batchSize) except* nogil:
    global __cusolverDnZpotrfBatched
    _check_or_init_cusolverDn()
    if __cusolverDnZpotrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotrfBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex**, int, int*, int) nogil>__cusolverDnZpotrfBatched)(
        handle, uplo, n, Aarray, lda, infoArray, batchSize)


cdef cusolverStatus_t _cusolverDnSpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, float* A[], int lda, float* B[], int ldb, int* d_info, int batchSize) except* nogil:
    global __cusolverDnSpotrsBatched
    _check_or_init_cusolverDn()
    if __cusolverDnSpotrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotrsBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, float**, int, float**, int, int*, int) nogil>__cusolverDnSpotrsBatched)(
        handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t _cusolverDnDpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, double* A[], int lda, double* B[], int ldb, int* d_info, int batchSize) except* nogil:
    global __cusolverDnDpotrsBatched
    _check_or_init_cusolverDn()
    if __cusolverDnDpotrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotrsBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, double**, int, double**, int, int*, int) nogil>__cusolverDnDpotrsBatched)(
        handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t _cusolverDnCpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, cuComplex* A[], int lda, cuComplex* B[], int ldb, int* d_info, int batchSize) except* nogil:
    global __cusolverDnCpotrsBatched
    _check_or_init_cusolverDn()
    if __cusolverDnCpotrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotrsBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, cuComplex**, int, cuComplex**, int, int*, int) nogil>__cusolverDnCpotrsBatched)(
        handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t _cusolverDnZpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, cuDoubleComplex* A[], int lda, cuDoubleComplex* B[], int ldb, int* d_info, int batchSize) except* nogil:
    global __cusolverDnZpotrsBatched
    _check_or_init_cusolverDn()
    if __cusolverDnZpotrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotrsBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, cuDoubleComplex**, int, cuDoubleComplex**, int, int*, int) nogil>__cusolverDnZpotrsBatched)(
        handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)


cdef cusolverStatus_t _cusolverDnSpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork) except* nogil:
    global __cusolverDnSpotri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSpotri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, int*) nogil>__cusolverDnSpotri_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnDpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork) except* nogil:
    global __cusolverDnDpotri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDpotri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, int*) nogil>__cusolverDnDpotri_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnCpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* lwork) except* nogil:
    global __cusolverDnCpotri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCpotri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, int*) nogil>__cusolverDnCpotri_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnZpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* lwork) except* nogil:
    global __cusolverDnZpotri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZpotri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZpotri_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnSpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnSpotri
    _check_or_init_cusolverDn()
    if __cusolverDnSpotri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSpotri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, float*, int, int*) nogil>__cusolverDnSpotri)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnDpotri
    _check_or_init_cusolverDn()
    if __cusolverDnDpotri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDpotri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, double*, int, int*) nogil>__cusolverDnDpotri)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnCpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnCpotri
    _check_or_init_cusolverDn()
    if __cusolverDnCpotri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCpotri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, int*) nogil>__cusolverDnCpotri)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnZpotri
    _check_or_init_cusolverDn()
    if __cusolverDnZpotri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZpotri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZpotri)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork) except* nogil:
    global __cusolverDnSlauum_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSlauum_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSlauum_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, int*) nogil>__cusolverDnSlauum_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnDlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork) except* nogil:
    global __cusolverDnDlauum_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDlauum_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDlauum_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, int*) nogil>__cusolverDnDlauum_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnClauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* lwork) except* nogil:
    global __cusolverDnClauum_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnClauum_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnClauum_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, int*) nogil>__cusolverDnClauum_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnZlauum_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* lwork) except* nogil:
    global __cusolverDnZlauum_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZlauum_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZlauum_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZlauum_bufferSize)(
        handle, uplo, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnSlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnSlauum
    _check_or_init_cusolverDn()
    if __cusolverDnSlauum == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSlauum is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, float*, int, int*) nogil>__cusolverDnSlauum)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnDlauum
    _check_or_init_cusolverDn()
    if __cusolverDnDlauum == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDlauum is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, double*, int, int*) nogil>__cusolverDnDlauum)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnClauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnClauum
    _check_or_init_cusolverDn()
    if __cusolverDnClauum == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnClauum is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, int*) nogil>__cusolverDnClauum)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnZlauum
    _check_or_init_cusolverDn()
    if __cusolverDnZlauum == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZlauum is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZlauum)(
        handle, uplo, n, A, lda, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* Lwork) except* nogil:
    global __cusolverDnSgetrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgetrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgetrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float*, int, int*) nogil>__cusolverDnSgetrf_bufferSize)(
        handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* Lwork) except* nogil:
    global __cusolverDnDgetrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgetrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgetrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, int*) nogil>__cusolverDnDgetrf_bufferSize)(
        handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, int* Lwork) except* nogil:
    global __cusolverDnCgetrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgetrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgetrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex*, int, int*) nogil>__cusolverDnCgetrf_bufferSize)(
        handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, int* Lwork) except* nogil:
    global __cusolverDnZgetrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgetrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgetrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZgetrf_bufferSize)(
        handle, m, n, A, lda, Lwork)


cdef cusolverStatus_t _cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* Workspace, int* devIpiv, int* devInfo) except* nogil:
    global __cusolverDnSgetrf
    _check_or_init_cusolverDn()
    if __cusolverDnSgetrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgetrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float*, int, float*, int*, int*) nogil>__cusolverDnSgetrf)(
        handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t _cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* Workspace, int* devIpiv, int* devInfo) except* nogil:
    global __cusolverDnDgetrf
    _check_or_init_cusolverDn()
    if __cusolverDnDgetrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgetrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, double*, int*, int*) nogil>__cusolverDnDgetrf)(
        handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t _cusolverDnCgetrf(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, cuComplex* Workspace, int* devIpiv, int* devInfo) except* nogil:
    global __cusolverDnCgetrf
    _check_or_init_cusolverDn()
    if __cusolverDnCgetrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgetrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex*, int, cuComplex*, int*, int*) nogil>__cusolverDnCgetrf)(
        handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t _cusolverDnZgetrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* Workspace, int* devIpiv, int* devInfo) except* nogil:
    global __cusolverDnZgetrf
    _check_or_init_cusolverDn()
    if __cusolverDnZgetrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgetrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex*, int, cuDoubleComplex*, int*, int*) nogil>__cusolverDnZgetrf)(
        handle, m, n, A, lda, Workspace, devIpiv, devInfo)


cdef cusolverStatus_t _cusolverDnSlaswp(cusolverDnHandle_t handle, int n, float* A, int lda, int k1, int k2, const int* devIpiv, int incx) except* nogil:
    global __cusolverDnSlaswp
    _check_or_init_cusolverDn()
    if __cusolverDnSlaswp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSlaswp is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, float*, int, int, int, const int*, int) nogil>__cusolverDnSlaswp)(
        handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t _cusolverDnDlaswp(cusolverDnHandle_t handle, int n, double* A, int lda, int k1, int k2, const int* devIpiv, int incx) except* nogil:
    global __cusolverDnDlaswp
    _check_or_init_cusolverDn()
    if __cusolverDnDlaswp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDlaswp is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, double*, int, int, int, const int*, int) nogil>__cusolverDnDlaswp)(
        handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t _cusolverDnClaswp(cusolverDnHandle_t handle, int n, cuComplex* A, int lda, int k1, int k2, const int* devIpiv, int incx) except* nogil:
    global __cusolverDnClaswp
    _check_or_init_cusolverDn()
    if __cusolverDnClaswp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnClaswp is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, cuComplex*, int, int, int, const int*, int) nogil>__cusolverDnClaswp)(
        handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t _cusolverDnZlaswp(cusolverDnHandle_t handle, int n, cuDoubleComplex* A, int lda, int k1, int k2, const int* devIpiv, int incx) except* nogil:
    global __cusolverDnZlaswp
    _check_or_init_cusolverDn()
    if __cusolverDnZlaswp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZlaswp is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, cuDoubleComplex*, int, int, int, const int*, int) nogil>__cusolverDnZlaswp)(
        handle, n, A, lda, k1, k2, devIpiv, incx)


cdef cusolverStatus_t _cusolverDnSgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo) except* nogil:
    global __cusolverDnSgetrs
    _check_or_init_cusolverDn()
    if __cusolverDnSgetrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgetrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const float*, int, const int*, float*, int, int*) nogil>__cusolverDnSgetrs)(
        handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnDgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo) except* nogil:
    global __cusolverDnDgetrs
    _check_or_init_cusolverDn()
    if __cusolverDnDgetrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgetrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const double*, int, const int*, double*, int, int*) nogil>__cusolverDnDgetrs)(
        handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnCgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* A, int lda, const int* devIpiv, cuComplex* B, int ldb, int* devInfo) except* nogil:
    global __cusolverDnCgetrs
    _check_or_init_cusolverDn()
    if __cusolverDnCgetrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgetrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const cuComplex*, int, const int*, cuComplex*, int, int*) nogil>__cusolverDnCgetrs)(
        handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnZgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* A, int lda, const int* devIpiv, cuDoubleComplex* B, int ldb, int* devInfo) except* nogil:
    global __cusolverDnZgetrs
    _check_or_init_cusolverDn()
    if __cusolverDnZgetrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgetrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const cuDoubleComplex*, int, const int*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZgetrs)(
        handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)


cdef cusolverStatus_t _cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork) except* nogil:
    global __cusolverDnSgeqrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgeqrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgeqrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float*, int, int*) nogil>__cusolverDnSgeqrf_bufferSize)(
        handle, m, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork) except* nogil:
    global __cusolverDnDgeqrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgeqrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgeqrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, int*) nogil>__cusolverDnDgeqrf_bufferSize)(
        handle, m, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnCgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, int* lwork) except* nogil:
    global __cusolverDnCgeqrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgeqrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgeqrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex*, int, int*) nogil>__cusolverDnCgeqrf_bufferSize)(
        handle, m, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnZgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, int* lwork) except* nogil:
    global __cusolverDnZgeqrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgeqrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgeqrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZgeqrf_bufferSize)(
        handle, m, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* TAU, float* Workspace, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnSgeqrf
    _check_or_init_cusolverDn()
    if __cusolverDnSgeqrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgeqrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float*, int, float*, float*, int, int*) nogil>__cusolverDnSgeqrf)(
        handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* TAU, double* Workspace, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnDgeqrf
    _check_or_init_cusolverDn()
    if __cusolverDnDgeqrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgeqrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, double*, double*, int, int*) nogil>__cusolverDnDgeqrf)(
        handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnCgeqrf(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, cuComplex* TAU, cuComplex* Workspace, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnCgeqrf
    _check_or_init_cusolverDn()
    if __cusolverDnCgeqrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgeqrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex*, int, cuComplex*, cuComplex*, int, int*) nogil>__cusolverDnCgeqrf)(
        handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZgeqrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* TAU, cuDoubleComplex* Workspace, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnZgeqrf
    _check_or_init_cusolverDn()
    if __cusolverDnZgeqrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgeqrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex*, int, cuDoubleComplex*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZgeqrf)(
        handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const float* A, int lda, const float* tau, int* lwork) except* nogil:
    global __cusolverDnSorgqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSorgqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const float*, int, const float*, int*) nogil>__cusolverDnSorgqr_bufferSize)(
        handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnDorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const double* A, int lda, const double* tau, int* lwork) except* nogil:
    global __cusolverDnDorgqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDorgqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const double*, int, const double*, int*) nogil>__cusolverDnDorgqr_bufferSize)(
        handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnCungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, int* lwork) except* nogil:
    global __cusolverDnCungqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCungqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const cuComplex*, int, const cuComplex*, int*) nogil>__cusolverDnCungqr_bufferSize)(
        handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnZungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, int* lwork) except* nogil:
    global __cusolverDnZungqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZungqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int*) nogil>__cusolverDnZungqr_bufferSize)(
        handle, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnSorgqr(cusolverDnHandle_t handle, int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSorgqr
    _check_or_init_cusolverDn()
    if __cusolverDnSorgqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, float*, int, const float*, float*, int, int*) nogil>__cusolverDnSorgqr)(
        handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDorgqr(cusolverDnHandle_t handle, int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDorgqr
    _check_or_init_cusolverDn()
    if __cusolverDnDorgqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, double*, int, const double*, double*, int, int*) nogil>__cusolverDnDorgqr)(
        handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCungqr(cusolverDnHandle_t handle, int m, int n, int k, cuComplex* A, int lda, const cuComplex* tau, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnCungqr
    _check_or_init_cusolverDn()
    if __cusolverDnCungqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, cuComplex*, int, const cuComplex*, cuComplex*, int, int*) nogil>__cusolverDnCungqr)(
        handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZungqr(cusolverDnHandle_t handle, int m, int n, int k, cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZungqr
    _check_or_init_cusolverDn()
    if __cusolverDnZungqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZungqr)(
        handle, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float* A, int lda, const float* tau, const float* C, int ldc, int* lwork) except* nogil:
    global __cusolverDnSormqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSormqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSormqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const float*, int, const float*, const float*, int, int*) nogil>__cusolverDnSormqr_bufferSize)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double* A, int lda, const double* tau, const double* C, int ldc, int* lwork) except* nogil:
    global __cusolverDnDormqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDormqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDormqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const double*, int, const double*, const double*, int, int*) nogil>__cusolverDnDormqr_bufferSize)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnCunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, const cuComplex* C, int ldc, int* lwork) except* nogil:
    global __cusolverDnCunmqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCunmqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCunmqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const cuComplex*, int, const cuComplex*, const cuComplex*, int, int*) nogil>__cusolverDnCunmqr_bufferSize)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnZunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, const cuDoubleComplex* C, int ldc, int* lwork) except* nogil:
    global __cusolverDnZunmqr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZunmqr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZunmqr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, const cuDoubleComplex*, int, int*) nogil>__cusolverDnZunmqr_bufferSize)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnSormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, float* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnSormqr
    _check_or_init_cusolverDn()
    if __cusolverDnSormqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSormqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const float*, int, const float*, float*, int, float*, int, int*) nogil>__cusolverDnSormqr)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, double* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnDormqr
    _check_or_init_cusolverDn()
    if __cusolverDnDormqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDormqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const double*, int, const double*, double*, int, double*, int, int*) nogil>__cusolverDnDormqr)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnCunmqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, cuComplex* C, int ldc, cuComplex* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnCunmqr
    _check_or_init_cusolverDn()
    if __cusolverDnCunmqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCunmqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const cuComplex*, int, const cuComplex*, cuComplex*, int, cuComplex*, int, int*) nogil>__cusolverDnCunmqr)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZunmqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* C, int ldc, cuDoubleComplex* work, int lwork, int* devInfo) except* nogil:
    global __cusolverDnZunmqr
    _check_or_init_cusolverDn()
    if __cusolverDnZunmqr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZunmqr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZunmqr)(
        handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSsytrf_bufferSize(cusolverDnHandle_t handle, int n, float* A, int lda, int* lwork) except* nogil:
    global __cusolverDnSsytrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsytrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, float*, int, int*) nogil>__cusolverDnSsytrf_bufferSize)(
        handle, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnDsytrf_bufferSize(cusolverDnHandle_t handle, int n, double* A, int lda, int* lwork) except* nogil:
    global __cusolverDnDsytrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsytrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, double*, int, int*) nogil>__cusolverDnDsytrf_bufferSize)(
        handle, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnCsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuComplex* A, int lda, int* lwork) except* nogil:
    global __cusolverDnCsytrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCsytrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCsytrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, cuComplex*, int, int*) nogil>__cusolverDnCsytrf_bufferSize)(
        handle, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnZsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuDoubleComplex* A, int lda, int* lwork) except* nogil:
    global __cusolverDnZsytrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZsytrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZsytrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZsytrf_bufferSize)(
        handle, n, A, lda, lwork)


cdef cusolverStatus_t _cusolverDnSsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* ipiv, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSsytrf
    _check_or_init_cusolverDn()
    if __cusolverDnSsytrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, int*, float*, int, int*) nogil>__cusolverDnSsytrf)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* ipiv, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDsytrf
    _check_or_init_cusolverDn()
    if __cusolverDnDsytrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, int*, double*, int, int*) nogil>__cusolverDnDsytrf)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, int* ipiv, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnCsytrf
    _check_or_init_cusolverDn()
    if __cusolverDnCsytrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCsytrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, int*, cuComplex*, int, int*) nogil>__cusolverDnCsytrf)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, int* ipiv, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZsytrf
    _check_or_init_cusolverDn()
    if __cusolverDnZsytrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZsytrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, int*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZsytrf)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, const int* ipiv, int* lwork) except* nogil:
    global __cusolverDnSsytri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsytri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, const int*, int*) nogil>__cusolverDnSsytri_bufferSize)(
        handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t _cusolverDnDsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, const int* ipiv, int* lwork) except* nogil:
    global __cusolverDnDsytri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsytri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, const int*, int*) nogil>__cusolverDnDsytri_bufferSize)(
        handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t _cusolverDnCsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, const int* ipiv, int* lwork) except* nogil:
    global __cusolverDnCsytri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCsytri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCsytri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, const int*, int*) nogil>__cusolverDnCsytri_bufferSize)(
        handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t _cusolverDnZsytri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, const int* ipiv, int* lwork) except* nogil:
    global __cusolverDnZsytri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZsytri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZsytri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, const int*, int*) nogil>__cusolverDnZsytri_bufferSize)(
        handle, uplo, n, A, lda, ipiv, lwork)


cdef cusolverStatus_t _cusolverDnSsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, const int* ipiv, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSsytri
    _check_or_init_cusolverDn()
    if __cusolverDnSsytri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, const int*, float*, int, int*) nogil>__cusolverDnSsytri)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, const int* ipiv, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDsytri
    _check_or_init_cusolverDn()
    if __cusolverDnDsytri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, const int*, double*, int, int*) nogil>__cusolverDnDsytri)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, const int* ipiv, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnCsytri
    _check_or_init_cusolverDn()
    if __cusolverDnCsytri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCsytri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, const int*, cuComplex*, int, int*) nogil>__cusolverDnCsytri)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, const int* ipiv, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZsytri
    _check_or_init_cusolverDn()
    if __cusolverDnZsytri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZsytri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, const int*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZsytri)(
        handle, uplo, n, A, lda, ipiv, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except* nogil:
    global __cusolverDnSgebrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgebrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgebrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) nogil>__cusolverDnSgebrd_bufferSize)(
        handle, m, n, Lwork)


cdef cusolverStatus_t _cusolverDnDgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except* nogil:
    global __cusolverDnDgebrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgebrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgebrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) nogil>__cusolverDnDgebrd_bufferSize)(
        handle, m, n, Lwork)


cdef cusolverStatus_t _cusolverDnCgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except* nogil:
    global __cusolverDnCgebrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgebrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgebrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) nogil>__cusolverDnCgebrd_bufferSize)(
        handle, m, n, Lwork)


cdef cusolverStatus_t _cusolverDnZgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* Lwork) except* nogil:
    global __cusolverDnZgebrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgebrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgebrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) nogil>__cusolverDnZgebrd_bufferSize)(
        handle, m, n, Lwork)


cdef cusolverStatus_t _cusolverDnSgebrd(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* D, float* E, float* TAUQ, float* TAUP, float* Work, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnSgebrd
    _check_or_init_cusolverDn()
    if __cusolverDnSgebrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgebrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float*, int, float*, float*, float*, float*, float*, int, int*) nogil>__cusolverDnSgebrd)(
        handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnDgebrd(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* D, double* E, double* TAUQ, double* TAUP, double* Work, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnDgebrd
    _check_or_init_cusolverDn()
    if __cusolverDnDgebrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgebrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, double*, double*, double*, double*, double*, int, int*) nogil>__cusolverDnDgebrd)(
        handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnCgebrd(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, float* D, float* E, cuComplex* TAUQ, cuComplex* TAUP, cuComplex* Work, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnCgebrd
    _check_or_init_cusolverDn()
    if __cusolverDnCgebrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgebrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex*, int, float*, float*, cuComplex*, cuComplex*, cuComplex*, int, int*) nogil>__cusolverDnCgebrd)(
        handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnZgebrd(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, double* D, double* E, cuDoubleComplex* TAUQ, cuDoubleComplex* TAUP, cuDoubleComplex* Work, int Lwork, int* devInfo) except* nogil:
    global __cusolverDnZgebrd
    _check_or_init_cusolverDn()
    if __cusolverDnZgebrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgebrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex*, int, double*, double*, cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZgebrd)(
        handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)


cdef cusolverStatus_t _cusolverDnSorgbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const float* A, int lda, const float* tau, int* lwork) except* nogil:
    global __cusolverDnSorgbr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSorgbr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgbr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const float*, int, const float*, int*) nogil>__cusolverDnSorgbr_bufferSize)(
        handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnDorgbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const double* A, int lda, const double* tau, int* lwork) except* nogil:
    global __cusolverDnDorgbr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDorgbr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgbr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const double*, int, const double*, int*) nogil>__cusolverDnDorgbr_bufferSize)(
        handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnCungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const cuComplex* A, int lda, const cuComplex* tau, int* lwork) except* nogil:
    global __cusolverDnCungbr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCungbr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungbr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const cuComplex*, int, const cuComplex*, int*) nogil>__cusolverDnCungbr_bufferSize)(
        handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnZungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, int* lwork) except* nogil:
    global __cusolverDnZungbr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZungbr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungbr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int*) nogil>__cusolverDnZungbr_bufferSize)(
        handle, side, m, n, k, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnSorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSorgbr
    _check_or_init_cusolverDn()
    if __cusolverDnSorgbr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgbr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, float*, int, const float*, float*, int, int*) nogil>__cusolverDnSorgbr)(
        handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDorgbr
    _check_or_init_cusolverDn()
    if __cusolverDnDorgbr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgbr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, double*, int, const double*, double*, int, int*) nogil>__cusolverDnDorgbr)(
        handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, cuComplex* A, int lda, const cuComplex* tau, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnCungbr
    _check_or_init_cusolverDn()
    if __cusolverDnCungbr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungbr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, cuComplex*, int, const cuComplex*, cuComplex*, int, int*) nogil>__cusolverDnCungbr)(
        handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZungbr
    _check_or_init_cusolverDn()
    if __cusolverDnZungbr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungbr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZungbr)(
        handle, side, m, n, k, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, const float* d, const float* e, const float* tau, int* lwork) except* nogil:
    global __cusolverDnSsytrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsytrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const float*, int, const float*, const float*, const float*, int*) nogil>__cusolverDnSsytrd_bufferSize)(
        handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t _cusolverDnDsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, const double* d, const double* e, const double* tau, int* lwork) except* nogil:
    global __cusolverDnDsytrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsytrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const double*, int, const double*, const double*, const double*, int*) nogil>__cusolverDnDsytrd_bufferSize)(
        handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t _cusolverDnChetrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* d, const float* e, const cuComplex* tau, int* lwork) except* nogil:
    global __cusolverDnChetrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnChetrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChetrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuComplex*, int, const float*, const float*, const cuComplex*, int*) nogil>__cusolverDnChetrd_bufferSize)(
        handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t _cusolverDnZhetrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* d, const double* e, const cuDoubleComplex* tau, int* lwork) except* nogil:
    global __cusolverDnZhetrd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZhetrd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhetrd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const double*, const double*, const cuDoubleComplex*, int*) nogil>__cusolverDnZhetrd_bufferSize)(
        handle, uplo, n, A, lda, d, e, tau, lwork)


cdef cusolverStatus_t _cusolverDnSsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* d, float* e, float* tau, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSsytrd
    _check_or_init_cusolverDn()
    if __cusolverDnSsytrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsytrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, float*, float*, float*, float*, int, int*) nogil>__cusolverDnSsytrd)(
        handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* d, double* e, double* tau, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDsytrd
    _check_or_init_cusolverDn()
    if __cusolverDnDsytrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsytrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, double*, double*, double*, double*, int, int*) nogil>__cusolverDnDsytrd)(
        handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnChetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* d, float* e, cuComplex* tau, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnChetrd
    _check_or_init_cusolverDn()
    if __cusolverDnChetrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChetrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, float*, float*, cuComplex*, cuComplex*, int, int*) nogil>__cusolverDnChetrd)(
        handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZhetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* d, double* e, cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZhetrd
    _check_or_init_cusolverDn()
    if __cusolverDnZhetrd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhetrd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, double*, double*, cuDoubleComplex*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZhetrd)(
        handle, uplo, n, A, lda, d, e, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSorgtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, const float* tau, int* lwork) except* nogil:
    global __cusolverDnSorgtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSorgtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const float*, int, const float*, int*) nogil>__cusolverDnSorgtr_bufferSize)(
        handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnDorgtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, const double* tau, int* lwork) except* nogil:
    global __cusolverDnDorgtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDorgtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const double*, int, const double*, int*) nogil>__cusolverDnDorgtr_bufferSize)(
        handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnCungtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* tau, int* lwork) except* nogil:
    global __cusolverDnCungtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCungtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuComplex*, int, const cuComplex*, int*) nogil>__cusolverDnCungtr_bufferSize)(
        handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnZungtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, int* lwork) except* nogil:
    global __cusolverDnZungtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZungtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int*) nogil>__cusolverDnZungtr_bufferSize)(
        handle, uplo, n, A, lda, tau, lwork)


cdef cusolverStatus_t _cusolverDnSorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, const float* tau, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSorgtr
    _check_or_init_cusolverDn()
    if __cusolverDnSorgtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSorgtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float*, int, const float*, float*, int, int*) nogil>__cusolverDnSorgtr)(
        handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, const double* tau, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDorgtr
    _check_or_init_cusolverDn()
    if __cusolverDnDorgtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDorgtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double*, int, const double*, double*, int, int*) nogil>__cusolverDnDorgtr)(
        handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex* A, int lda, const cuComplex* tau, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnCungtr
    _check_or_init_cusolverDn()
    if __cusolverDnCungtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCungtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex*, int, const cuComplex*, cuComplex*, int, int*) nogil>__cusolverDnCungtr)(
        handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZungtr
    _check_or_init_cusolverDn()
    if __cusolverDnZungtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZungtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZungtr)(
        handle, uplo, n, A, lda, tau, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const float* A, int lda, const float* tau, const float* C, int ldc, int* lwork) except* nogil:
    global __cusolverDnSormtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSormtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSormtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, int, const float*, const float*, int, int*) nogil>__cusolverDnSormtr_bufferSize)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnDormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const double* A, int lda, const double* tau, const double* C, int ldc, int* lwork) except* nogil:
    global __cusolverDnDormtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDormtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDormtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, int, const double*, const double*, int, int*) nogil>__cusolverDnDormtr_bufferSize)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnCunmtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const cuComplex* A, int lda, const cuComplex* tau, const cuComplex* C, int ldc, int* lwork) except* nogil:
    global __cusolverDnCunmtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCunmtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCunmtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, int, const cuComplex*, const cuComplex*, int, int*) nogil>__cusolverDnCunmtr_bufferSize)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnZunmtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* tau, const cuDoubleComplex* C, int ldc, int* lwork) except* nogil:
    global __cusolverDnZunmtr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZunmtr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZunmtr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, const cuDoubleComplex*, int, int*) nogil>__cusolverDnZunmtr_bufferSize)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)


cdef cusolverStatus_t _cusolverDnSormtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, float* A, int lda, float* tau, float* C, int ldc, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSormtr
    _check_or_init_cusolverDn()
    if __cusolverDnSormtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSormtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, float*, int, float*, float*, int, float*, int, int*) nogil>__cusolverDnSormtr)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDormtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, double* A, int lda, double* tau, double* C, int ldc, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDormtr
    _check_or_init_cusolverDn()
    if __cusolverDnDormtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDormtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, double*, int, double*, double*, int, double*, int, int*) nogil>__cusolverDnDormtr)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCunmtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, cuComplex* A, int lda, cuComplex* tau, cuComplex* C, int ldc, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnCunmtr
    _check_or_init_cusolverDn()
    if __cusolverDnCunmtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCunmtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, cuComplex*, int, cuComplex*, cuComplex*, int, cuComplex*, int, int*) nogil>__cusolverDnCunmtr)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZunmtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* tau, cuDoubleComplex* C, int ldc, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZunmtr
    _check_or_init_cusolverDn()
    if __cusolverDnZunmtr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZunmtr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int, cuDoubleComplex*, int, cuDoubleComplex*, cuDoubleComplex*, int, cuDoubleComplex*, int, int*) nogil>__cusolverDnZunmtr)(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except* nogil:
    global __cusolverDnSgesvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) nogil>__cusolverDnSgesvd_bufferSize)(
        handle, m, n, lwork)


cdef cusolverStatus_t _cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except* nogil:
    global __cusolverDnDgesvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) nogil>__cusolverDnDgesvd_bufferSize)(
        handle, m, n, lwork)


cdef cusolverStatus_t _cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except* nogil:
    global __cusolverDnCgesvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) nogil>__cusolverDnCgesvd_bufferSize)(
        handle, m, n, lwork)


cdef cusolverStatus_t _cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int* lwork) except* nogil:
    global __cusolverDnZgesvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int*) nogil>__cusolverDnZgesvd_bufferSize)(
        handle, m, n, lwork)


cdef cusolverStatus_t _cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* VT, int ldvt, float* work, int lwork, float* rwork, int* info) except* nogil:
    global __cusolverDnSgesvd
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, float*, int, float*, float*, int, float*, int, float*, int, float*, int*) nogil>__cusolverDnSgesvd)(
        handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t _cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt, double* work, int lwork, double* rwork, int* info) except* nogil:
    global __cusolverDnDgesvd
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, double*, int, double*, double*, int, double*, int, double*, int, double*, int*) nogil>__cusolverDnDgesvd)(
        handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t _cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* VT, int ldvt, cuComplex* work, int lwork, float* rwork, int* info) except* nogil:
    global __cusolverDnCgesvd
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, cuComplex*, int, float*, cuComplex*, int, cuComplex*, int, cuComplex*, int, float*, int*) nogil>__cusolverDnCgesvd)(
        handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t _cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* VT, int ldvt, cuDoubleComplex* work, int lwork, double* rwork, int* info) except* nogil:
    global __cusolverDnZgesvd
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, cuDoubleComplex*, int, cuDoubleComplex*, int, double*, int*) nogil>__cusolverDnZgesvd)(
        handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)


cdef cusolverStatus_t _cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork) except* nogil:
    global __cusolverDnSsyevd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const float*, int, const float*, int*) nogil>__cusolverDnSsyevd_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t _cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork) except* nogil:
    global __cusolverDnDsyevd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const double*, int, const double*, int*) nogil>__cusolverDnDsyevd_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t _cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork) except* nogil:
    global __cusolverDnCheevd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCheevd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuComplex*, int, const float*, int*) nogil>__cusolverDnCheevd_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t _cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork) except* nogil:
    global __cusolverDnZheevd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZheevd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const double*, int*) nogil>__cusolverDnZheevd_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork)


cdef cusolverStatus_t _cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSsyevd
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float*, int, float*, float*, int, int*) nogil>__cusolverDnSsyevd)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDsyevd
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double*, int, double*, double*, int, int*) nogil>__cusolverDnDsyevd)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnCheevd
    _check_or_init_cusolverDn()
    if __cusolverDnCheevd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex*, int, float*, cuComplex*, int, int*) nogil>__cusolverDnCheevd)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZheevd
    _check_or_init_cusolverDn()
    if __cusolverDnZheevd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZheevd)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float* A, int lda, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except* nogil:
    global __cusolverDnSsyevdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const float*, int, float, float, int, int, int*, const float*, int*) nogil>__cusolverDnSsyevdx_bufferSize)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnDsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double* A, int lda, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except* nogil:
    global __cusolverDnDsyevdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const double*, int, double, double, int, int, int*, const double*, int*) nogil>__cusolverDnDsyevdx_bufferSize)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnCheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except* nogil:
    global __cusolverDnCheevdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCheevdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const cuComplex*, int, float, float, int, int, int*, const float*, int*) nogil>__cusolverDnCheevdx_bufferSize)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnZheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except* nogil:
    global __cusolverDnZheevdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZheevdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const cuDoubleComplex*, int, double, double, int, int, int*, const double*, int*) nogil>__cusolverDnZheevdx_bufferSize)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnSsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, float* A, int lda, float vl, float vu, int il, int iu, int* meig, float* W, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSsyevdx
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, float*, int, float, float, int, int, int*, float*, float*, int, int*) nogil>__cusolverDnSsyevdx)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, double* A, int lda, double vl, double vu, int il, int iu, int* meig, double* W, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDsyevdx
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, double*, int, double, double, int, int, int*, double*, double*, int, int*) nogil>__cusolverDnDsyevdx)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float vl, float vu, int il, int iu, int* meig, float* W, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnCheevdx
    _check_or_init_cusolverDn()
    if __cusolverDnCheevdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, cuComplex*, int, float, float, int, int, int*, float*, cuComplex*, int, int*) nogil>__cusolverDnCheevdx)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double vl, double vu, int il, int iu, int* meig, double* W, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZheevdx
    _check_or_init_cusolverDn()
    if __cusolverDnZheevdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, cuDoubleComplex*, int, double, double, int, int, int*, double*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZheevdx)(
        handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSsygvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except* nogil:
    global __cusolverDnSsygvdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const float*, int, const float*, int, float, float, int, int, int*, const float*, int*) nogil>__cusolverDnSsygvdx_bufferSize)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnDsygvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except* nogil:
    global __cusolverDnDsygvdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const double*, int, const double*, int, double, double, int, int, int*, const double*, int*) nogil>__cusolverDnDsygvdx_bufferSize)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnChegvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* B, int ldb, float vl, float vu, int il, int iu, int* meig, const float* W, int* lwork) except* nogil:
    global __cusolverDnChegvdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnChegvdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const cuComplex*, int, const cuComplex*, int, float, float, int, int, int*, const float*, int*) nogil>__cusolverDnChegvdx_bufferSize)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnZhegvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, double vl, double vu, int il, int iu, int* meig, const double* W, int* lwork) except* nogil:
    global __cusolverDnZhegvdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, double, double, int, int, int*, const double*, int*) nogil>__cusolverDnZhegvdx_bufferSize)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)


cdef cusolverStatus_t _cusolverDnSsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float vl, float vu, int il, int iu, int* meig, float* W, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSsygvdx
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, float*, int, float*, int, float, float, int, int, int*, float*, float*, int, int*) nogil>__cusolverDnSsygvdx)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double vl, double vu, int il, int iu, int* meig, double* W, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDsygvdx
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, double*, int, double*, int, double, double, int, int, int*, double*, double*, int, int*) nogil>__cusolverDnDsygvdx)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnChegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb, float vl, float vu, int il, int iu, int* meig, float* W, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnChegvdx
    _check_or_init_cusolverDn()
    if __cusolverDnChegvdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, float, float, int, int, int*, float*, cuComplex*, int, int*) nogil>__cusolverDnChegvdx)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZhegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, double vl, double vu, int il, int iu, int* meig, double* W, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZhegvdx
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, double, double, int, int, int*, double*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZhegvdx)(
        handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnSsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, const float* W, int* lwork) except* nogil:
    global __cusolverDnSsygvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const float*, int, const float*, int, const float*, int*) nogil>__cusolverDnSsygvd_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t _cusolverDnDsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, const double* W, int* lwork) except* nogil:
    global __cusolverDnDsygvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const double*, int, const double*, int, const double*, int*) nogil>__cusolverDnDsygvd_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t _cusolverDnChegvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* W, int* lwork) except* nogil:
    global __cusolverDnChegvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnChegvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const cuComplex*, int, const cuComplex*, int, const float*, int*) nogil>__cusolverDnChegvd_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t _cusolverDnZhegvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* W, int* lwork) except* nogil:
    global __cusolverDnZhegvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const double*, int*) nogil>__cusolverDnZhegvd_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)


cdef cusolverStatus_t _cusolverDnSsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float* W, float* work, int lwork, int* info) except* nogil:
    global __cusolverDnSsygvd
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, float*, int, float*, int, float*, float*, int, int*) nogil>__cusolverDnSsygvd)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnDsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double* W, double* work, int lwork, int* info) except* nogil:
    global __cusolverDnDsygvd
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, double*, int, double*, int, double*, double*, int, int*) nogil>__cusolverDnDsygvd)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnChegvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb, float* W, cuComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnChegvd
    _check_or_init_cusolverDn()
    if __cusolverDnChegvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, float*, cuComplex*, int, int*) nogil>__cusolverDnChegvd)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnZhegvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, double* W, cuDoubleComplex* work, int lwork, int* info) except* nogil:
    global __cusolverDnZhegvd
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, int*) nogil>__cusolverDnZhegvd)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)


cdef cusolverStatus_t _cusolverDnCreateSyevjInfo(syevjInfo_t* info) except* nogil:
    global __cusolverDnCreateSyevjInfo
    _check_or_init_cusolverDn()
    if __cusolverDnCreateSyevjInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCreateSyevjInfo is not found")
    return (<cusolverStatus_t (*)(syevjInfo_t*) nogil>__cusolverDnCreateSyevjInfo)(
        info)


cdef cusolverStatus_t _cusolverDnDestroySyevjInfo(syevjInfo_t info) except* nogil:
    global __cusolverDnDestroySyevjInfo
    _check_or_init_cusolverDn()
    if __cusolverDnDestroySyevjInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDestroySyevjInfo is not found")
    return (<cusolverStatus_t (*)(syevjInfo_t) nogil>__cusolverDnDestroySyevjInfo)(
        info)


cdef cusolverStatus_t _cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance) except* nogil:
    global __cusolverDnXsyevjSetTolerance
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevjSetTolerance == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevjSetTolerance is not found")
    return (<cusolverStatus_t (*)(syevjInfo_t, double) nogil>__cusolverDnXsyevjSetTolerance)(
        info, tolerance)


cdef cusolverStatus_t _cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps) except* nogil:
    global __cusolverDnXsyevjSetMaxSweeps
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevjSetMaxSweeps == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevjSetMaxSweeps is not found")
    return (<cusolverStatus_t (*)(syevjInfo_t, int) nogil>__cusolverDnXsyevjSetMaxSweeps)(
        info, max_sweeps)


cdef cusolverStatus_t _cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig) except* nogil:
    global __cusolverDnXsyevjSetSortEig
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevjSetSortEig == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevjSetSortEig is not found")
    return (<cusolverStatus_t (*)(syevjInfo_t, int) nogil>__cusolverDnXsyevjSetSortEig)(
        info, sort_eig)


cdef cusolverStatus_t _cusolverDnXsyevjGetResidual(cusolverDnHandle_t handle, syevjInfo_t info, double* residual) except* nogil:
    global __cusolverDnXsyevjGetResidual
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevjGetResidual == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevjGetResidual is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, syevjInfo_t, double*) nogil>__cusolverDnXsyevjGetResidual)(
        handle, info, residual)


cdef cusolverStatus_t _cusolverDnXsyevjGetSweeps(cusolverDnHandle_t handle, syevjInfo_t info, int* executed_sweeps) except* nogil:
    global __cusolverDnXsyevjGetSweeps
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevjGetSweeps == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevjGetSweeps is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, syevjInfo_t, int*) nogil>__cusolverDnXsyevjGetSweeps)(
        handle, info, executed_sweeps)


cdef cusolverStatus_t _cusolverDnSsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork, syevjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnSsyevjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const float*, int, const float*, int*, syevjInfo_t, int) nogil>__cusolverDnSsyevjBatched_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork, syevjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnDsyevjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const double*, int, const double*, int*, syevjInfo_t, int) nogil>__cusolverDnDsyevjBatched_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnCheevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork, syevjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnCheevjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCheevjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuComplex*, int, const float*, int*, syevjInfo_t, int) nogil>__cusolverDnCheevjBatched_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnZheevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork, syevjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnZheevjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZheevjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const double*, int*, syevjInfo_t, int) nogil>__cusolverDnZheevjBatched_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnSsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info, syevjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnSsyevjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float*, int, float*, float*, int, int*, syevjInfo_t, int) nogil>__cusolverDnSsyevjBatched)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnDsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info, syevjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnDsyevjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double*, int, double*, double*, int, int*, syevjInfo_t, int) nogil>__cusolverDnDsyevjBatched)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnCheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnCheevjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnCheevjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex*, int, float*, cuComplex*, int, int*, syevjInfo_t, int) nogil>__cusolverDnCheevjBatched)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnZheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnZheevjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnZheevjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, int*, syevjInfo_t, int) nogil>__cusolverDnZheevjBatched)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnSsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork, syevjInfo_t params) except* nogil:
    global __cusolverDnSsyevj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const float*, int, const float*, int*, syevjInfo_t) nogil>__cusolverDnSsyevj_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t _cusolverDnDsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork, syevjInfo_t params) except* nogil:
    global __cusolverDnDsyevj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const double*, int, const double*, int*, syevjInfo_t) nogil>__cusolverDnDsyevj_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t _cusolverDnCheevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const float* W, int* lwork, syevjInfo_t params) except* nogil:
    global __cusolverDnCheevj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCheevj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuComplex*, int, const float*, int*, syevjInfo_t) nogil>__cusolverDnCheevj_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t _cusolverDnZheevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const double* W, int* lwork, syevjInfo_t params) except* nogil:
    global __cusolverDnZheevj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZheevj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const double*, int*, syevjInfo_t) nogil>__cusolverDnZheevj_bufferSize)(
        handle, jobz, uplo, n, A, lda, W, lwork, params)


cdef cusolverStatus_t _cusolverDnSsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    global __cusolverDnSsyevj
    _check_or_init_cusolverDn()
    if __cusolverDnSsyevj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsyevj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float*, int, float*, float*, int, int*, syevjInfo_t) nogil>__cusolverDnSsyevj)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnDsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    global __cusolverDnDsyevj
    _check_or_init_cusolverDn()
    if __cusolverDnDsyevj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsyevj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double*, int, double*, double*, int, int*, syevjInfo_t) nogil>__cusolverDnDsyevj)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnCheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    global __cusolverDnCheevj
    _check_or_init_cusolverDn()
    if __cusolverDnCheevj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCheevj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex*, int, float*, cuComplex*, int, int*, syevjInfo_t) nogil>__cusolverDnCheevj)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnZheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    global __cusolverDnZheevj
    _check_or_init_cusolverDn()
    if __cusolverDnZheevj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZheevj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, int*, syevjInfo_t) nogil>__cusolverDnZheevj)(
        handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnSsygvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, const float* W, int* lwork, syevjInfo_t params) except* nogil:
    global __cusolverDnSsygvj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const float*, int, const float*, int, const float*, int*, syevjInfo_t) nogil>__cusolverDnSsygvj_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t _cusolverDnDsygvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, const double* W, int* lwork, syevjInfo_t params) except* nogil:
    global __cusolverDnDsygvj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const double*, int, const double*, int, const double*, int*, syevjInfo_t) nogil>__cusolverDnDsygvj_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t _cusolverDnChegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* W, int* lwork, syevjInfo_t params) except* nogil:
    global __cusolverDnChegvj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnChegvj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const cuComplex*, int, const cuComplex*, int, const float*, int*, syevjInfo_t) nogil>__cusolverDnChegvj_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t _cusolverDnZhegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* W, int* lwork, syevjInfo_t params) except* nogil:
    global __cusolverDnZhegvj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const double*, int*, syevjInfo_t) nogil>__cusolverDnZhegvj_bufferSize)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)


cdef cusolverStatus_t _cusolverDnSsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float* W, float* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    global __cusolverDnSsygvj
    _check_or_init_cusolverDn()
    if __cusolverDnSsygvj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSsygvj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, float*, int, float*, int, float*, float*, int, int*, syevjInfo_t) nogil>__cusolverDnSsygvj)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnDsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double* W, double* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    global __cusolverDnDsygvj
    _check_or_init_cusolverDn()
    if __cusolverDnDsygvj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDsygvj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, double*, int, double*, int, double*, double*, int, int*, syevjInfo_t) nogil>__cusolverDnDsygvj)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnChegvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb, float* W, cuComplex* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    global __cusolverDnChegvj
    _check_or_init_cusolverDn()
    if __cusolverDnChegvj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnChegvj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex*, int, cuComplex*, int, float*, cuComplex*, int, int*, syevjInfo_t) nogil>__cusolverDnChegvj)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnZhegvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb, double* W, cuDoubleComplex* work, int lwork, int* info, syevjInfo_t params) except* nogil:
    global __cusolverDnZhegvj
    _check_or_init_cusolverDn()
    if __cusolverDnZhegvj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZhegvj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, int*, syevjInfo_t) nogil>__cusolverDnZhegvj)(
        handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info) except* nogil:
    global __cusolverDnCreateGesvdjInfo
    _check_or_init_cusolverDn()
    if __cusolverDnCreateGesvdjInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCreateGesvdjInfo is not found")
    return (<cusolverStatus_t (*)(gesvdjInfo_t*) nogil>__cusolverDnCreateGesvdjInfo)(
        info)


cdef cusolverStatus_t _cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info) except* nogil:
    global __cusolverDnDestroyGesvdjInfo
    _check_or_init_cusolverDn()
    if __cusolverDnDestroyGesvdjInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDestroyGesvdjInfo is not found")
    return (<cusolverStatus_t (*)(gesvdjInfo_t) nogil>__cusolverDnDestroyGesvdjInfo)(
        info)


cdef cusolverStatus_t _cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance) except* nogil:
    global __cusolverDnXgesvdjSetTolerance
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdjSetTolerance == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdjSetTolerance is not found")
    return (<cusolverStatus_t (*)(gesvdjInfo_t, double) nogil>__cusolverDnXgesvdjSetTolerance)(
        info, tolerance)


cdef cusolverStatus_t _cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int max_sweeps) except* nogil:
    global __cusolverDnXgesvdjSetMaxSweeps
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdjSetMaxSweeps == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdjSetMaxSweeps is not found")
    return (<cusolverStatus_t (*)(gesvdjInfo_t, int) nogil>__cusolverDnXgesvdjSetMaxSweeps)(
        info, max_sweeps)


cdef cusolverStatus_t _cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_svd) except* nogil:
    global __cusolverDnXgesvdjSetSortEig
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdjSetSortEig == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdjSetSortEig is not found")
    return (<cusolverStatus_t (*)(gesvdjInfo_t, int) nogil>__cusolverDnXgesvdjSetSortEig)(
        info, sort_svd)


cdef cusolverStatus_t _cusolverDnXgesvdjGetResidual(cusolverDnHandle_t handle, gesvdjInfo_t info, double* residual) except* nogil:
    global __cusolverDnXgesvdjGetResidual
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdjGetResidual == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdjGetResidual is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, gesvdjInfo_t, double*) nogil>__cusolverDnXgesvdjGetResidual)(
        handle, info, residual)


cdef cusolverStatus_t _cusolverDnXgesvdjGetSweeps(cusolverDnHandle_t handle, gesvdjInfo_t info, int* executed_sweeps) except* nogil:
    global __cusolverDnXgesvdjGetSweeps
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdjGetSweeps == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdjGetSweeps is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, gesvdjInfo_t, int*) nogil>__cusolverDnXgesvdjGetSweeps)(
        handle, info, executed_sweeps)


cdef cusolverStatus_t _cusolverDnSgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const float* A, int lda, const float* S, const float* U, int ldu, const float* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnSgesvdjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const float*, int, const float*, const float*, int, const float*, int, int*, gesvdjInfo_t, int) nogil>__cusolverDnSgesvdjBatched_bufferSize)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnDgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const double* A, int lda, const double* S, const double* U, int ldu, const double* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnDgesvdjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const double*, int, const double*, const double*, int, const double*, int, int*, gesvdjInfo_t, int) nogil>__cusolverDnDgesvdjBatched_bufferSize)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnCgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuComplex* A, int lda, const float* S, const cuComplex* U, int ldu, const cuComplex* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnCgesvdjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const cuComplex*, int, const float*, const cuComplex*, int, const cuComplex*, int, int*, gesvdjInfo_t, int) nogil>__cusolverDnCgesvdjBatched_bufferSize)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnZgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuDoubleComplex* A, int lda, const double* S, const cuDoubleComplex* U, int ldu, const cuDoubleComplex* V, int ldv, int* lwork, gesvdjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnZgesvdjBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdjBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdjBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const cuDoubleComplex*, int, const double*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, int*, gesvdjInfo_t, int) nogil>__cusolverDnZgesvdjBatched_bufferSize)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)


cdef cusolverStatus_t _cusolverDnSgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnSgesvdjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, float*, int, float*, float*, int, float*, int, float*, int, int*, gesvdjInfo_t, int) nogil>__cusolverDnSgesvdjBatched)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnDgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnDgesvdjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, double*, int, double*, double*, int, double*, int, double*, int, int*, gesvdjInfo_t, int) nogil>__cusolverDnDgesvdjBatched)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnCgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* V, int ldv, cuComplex* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnCgesvdjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, cuComplex*, int, float*, cuComplex*, int, cuComplex*, int, cuComplex*, int, int*, gesvdjInfo_t, int) nogil>__cusolverDnCgesvdjBatched)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnZgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* V, int ldv, cuDoubleComplex* work, int lwork, int* info, gesvdjInfo_t params, int batchSize) except* nogil:
    global __cusolverDnZgesvdjBatched
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdjBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdjBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, cuDoubleComplex*, int, cuDoubleComplex*, int, int*, gesvdjInfo_t, int) nogil>__cusolverDnZgesvdjBatched)(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)


cdef cusolverStatus_t _cusolverDnSgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const float* A, int lda, const float* S, const float* U, int ldu, const float* V, int ldv, int* lwork, gesvdjInfo_t params) except* nogil:
    global __cusolverDnSgesvdj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float*, int, const float*, const float*, int, const float*, int, int*, gesvdjInfo_t) nogil>__cusolverDnSgesvdj_bufferSize)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t _cusolverDnDgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const double* A, int lda, const double* S, const double* U, int ldu, const double* V, int ldv, int* lwork, gesvdjInfo_t params) except* nogil:
    global __cusolverDnDgesvdj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double*, int, const double*, const double*, int, const double*, int, int*, gesvdjInfo_t) nogil>__cusolverDnDgesvdj_bufferSize)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t _cusolverDnCgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuComplex* A, int lda, const float* S, const cuComplex* U, int ldu, const cuComplex* V, int ldv, int* lwork, gesvdjInfo_t params) except* nogil:
    global __cusolverDnCgesvdj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex*, int, const float*, const cuComplex*, int, const cuComplex*, int, int*, gesvdjInfo_t) nogil>__cusolverDnCgesvdj_bufferSize)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t _cusolverDnZgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuDoubleComplex* A, int lda, const double* S, const cuDoubleComplex* U, int ldu, const cuDoubleComplex* V, int ldv, int* lwork, gesvdjInfo_t params) except* nogil:
    global __cusolverDnZgesvdj_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdj_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdj_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuDoubleComplex*, int, const double*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, int*, gesvdjInfo_t) nogil>__cusolverDnZgesvdj_bufferSize)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)


cdef cusolverStatus_t _cusolverDnSgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, int* info, gesvdjInfo_t params) except* nogil:
    global __cusolverDnSgesvdj
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, float*, int, float*, float*, int, float*, int, float*, int, int*, gesvdjInfo_t) nogil>__cusolverDnSgesvdj)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnDgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, int* info, gesvdjInfo_t params) except* nogil:
    global __cusolverDnDgesvdj
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, double*, int, double*, double*, int, double*, int, double*, int, int*, gesvdjInfo_t) nogil>__cusolverDnDgesvdj)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnCgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, cuComplex* A, int lda, float* S, cuComplex* U, int ldu, cuComplex* V, int ldv, cuComplex* work, int lwork, int* info, gesvdjInfo_t params) except* nogil:
    global __cusolverDnCgesvdj
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, cuComplex*, int, float*, cuComplex*, int, cuComplex*, int, cuComplex*, int, int*, gesvdjInfo_t) nogil>__cusolverDnCgesvdj)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnZgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, cuDoubleComplex* A, int lda, double* S, cuDoubleComplex* U, int ldu, cuDoubleComplex* V, int ldv, cuDoubleComplex* work, int lwork, int* info, gesvdjInfo_t params) except* nogil:
    global __cusolverDnZgesvdj
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdj == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdj is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, cuDoubleComplex*, int, double*, cuDoubleComplex*, int, cuDoubleComplex*, int, cuDoubleComplex*, int, int*, gesvdjInfo_t) nogil>__cusolverDnZgesvdj)(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)


cdef cusolverStatus_t _cusolverDnSgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float* d_A, int lda, long long int strideA, const float* d_S, long long int strideS, const float* d_U, int ldu, long long int strideU, const float* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except* nogil:
    global __cusolverDnSgesvdaStridedBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdaStridedBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdaStridedBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float*, int, long long int, const float*, long long int, const float*, int, long long int, const float*, int, long long int, int*, int) nogil>__cusolverDnSgesvdaStridedBatched_bufferSize)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t _cusolverDnDgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double* d_A, int lda, long long int strideA, const double* d_S, long long int strideS, const double* d_U, int ldu, long long int strideU, const double* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except* nogil:
    global __cusolverDnDgesvdaStridedBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdaStridedBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdaStridedBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double*, int, long long int, const double*, long long int, const double*, int, long long int, const double*, int, long long int, int*, int) nogil>__cusolverDnDgesvdaStridedBatched_bufferSize)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t _cusolverDnCgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex* d_A, int lda, long long int strideA, const float* d_S, long long int strideS, const cuComplex* d_U, int ldu, long long int strideU, const cuComplex* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except* nogil:
    global __cusolverDnCgesvdaStridedBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdaStridedBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdaStridedBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex*, int, long long int, const float*, long long int, const cuComplex*, int, long long int, const cuComplex*, int, long long int, int*, int) nogil>__cusolverDnCgesvdaStridedBatched_bufferSize)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t _cusolverDnZgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex* d_A, int lda, long long int strideA, const double* d_S, long long int strideS, const cuDoubleComplex* d_U, int ldu, long long int strideU, const cuDoubleComplex* d_V, int ldv, long long int strideV, int* lwork, int batchSize) except* nogil:
    global __cusolverDnZgesvdaStridedBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdaStridedBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdaStridedBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuDoubleComplex*, int, long long int, const double*, long long int, const cuDoubleComplex*, int, long long int, const cuDoubleComplex*, int, long long int, int*, int) nogil>__cusolverDnZgesvdaStridedBatched_bufferSize)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)


cdef cusolverStatus_t _cusolverDnSgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float* d_A, int lda, long long int strideA, float* d_S, long long int strideS, float* d_U, int ldu, long long int strideU, float* d_V, int ldv, long long int strideV, float* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except* nogil:
    global __cusolverDnSgesvdaStridedBatched
    _check_or_init_cusolverDn()
    if __cusolverDnSgesvdaStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSgesvdaStridedBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float*, int, long long int, float*, long long int, float*, int, long long int, float*, int, long long int, float*, int, int*, double*, int) nogil>__cusolverDnSgesvdaStridedBatched)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t _cusolverDnDgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double* d_A, int lda, long long int strideA, double* d_S, long long int strideS, double* d_U, int ldu, long long int strideU, double* d_V, int ldv, long long int strideV, double* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except* nogil:
    global __cusolverDnDgesvdaStridedBatched
    _check_or_init_cusolverDn()
    if __cusolverDnDgesvdaStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDgesvdaStridedBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double*, int, long long int, double*, long long int, double*, int, long long int, double*, int, long long int, double*, int, int*, double*, int) nogil>__cusolverDnDgesvdaStridedBatched)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t _cusolverDnCgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex* d_A, int lda, long long int strideA, float* d_S, long long int strideS, cuComplex* d_U, int ldu, long long int strideU, cuComplex* d_V, int ldv, long long int strideV, cuComplex* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except* nogil:
    global __cusolverDnCgesvdaStridedBatched
    _check_or_init_cusolverDn()
    if __cusolverDnCgesvdaStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCgesvdaStridedBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex*, int, long long int, float*, long long int, cuComplex*, int, long long int, cuComplex*, int, long long int, cuComplex*, int, int*, double*, int) nogil>__cusolverDnCgesvdaStridedBatched)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t _cusolverDnZgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex* d_A, int lda, long long int strideA, double* d_S, long long int strideS, cuDoubleComplex* d_U, int ldu, long long int strideU, cuDoubleComplex* d_V, int ldv, long long int strideV, cuDoubleComplex* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) except* nogil:
    global __cusolverDnZgesvdaStridedBatched
    _check_or_init_cusolverDn()
    if __cusolverDnZgesvdaStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnZgesvdaStridedBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuDoubleComplex*, int, long long int, double*, long long int, cuDoubleComplex*, int, long long int, cuDoubleComplex*, int, long long int, cuDoubleComplex*, int, int*, double*, int) nogil>__cusolverDnZgesvdaStridedBatched)(
        handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)


cdef cusolverStatus_t _cusolverDnCreateParams(cusolverDnParams_t* params) except* nogil:
    global __cusolverDnCreateParams
    _check_or_init_cusolverDn()
    if __cusolverDnCreateParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnCreateParams is not found")
    return (<cusolverStatus_t (*)(cusolverDnParams_t*) nogil>__cusolverDnCreateParams)(
        params)


cdef cusolverStatus_t _cusolverDnDestroyParams(cusolverDnParams_t params) except* nogil:
    global __cusolverDnDestroyParams
    _check_or_init_cusolverDn()
    if __cusolverDnDestroyParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnDestroyParams is not found")
    return (<cusolverStatus_t (*)(cusolverDnParams_t) nogil>__cusolverDnDestroyParams)(
        params)


cdef cusolverStatus_t _cusolverDnSetAdvOptions(cusolverDnParams_t params, cusolverDnFunction_t function, cusolverAlgMode_t algo) except* nogil:
    global __cusolverDnSetAdvOptions
    _check_or_init_cusolverDn()
    if __cusolverDnSetAdvOptions == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSetAdvOptions is not found")
    return (<cusolverStatus_t (*)(cusolverDnParams_t, cusolverDnFunction_t, cusolverAlgMode_t) nogil>__cusolverDnSetAdvOptions)(
        params, function, algo)


cdef cusolverStatus_t _cusolverDnXpotrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXpotrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXpotrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXpotrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) nogil>__cusolverDnXpotrf_bufferSize)(
        handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXpotrf(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    global __cusolverDnXpotrf
    _check_or_init_cusolverDn()
    if __cusolverDnXpotrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXpotrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t, int*) nogil>__cusolverDnXpotrf)(
        handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXpotrs(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeB, void* B, int64_t ldb, int* info) except* nogil:
    global __cusolverDnXpotrs
    _check_or_init_cusolverDn()
    if __cusolverDnXpotrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXpotrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, void*, int64_t, int*) nogil>__cusolverDnXpotrs)(
        handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info)


cdef cusolverStatus_t _cusolverDnXgeqrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeTau, const void* tau, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXgeqrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgeqrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgeqrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, size_t*, size_t*) nogil>__cusolverDnXgeqrf_bufferSize)(
        handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgeqrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeTau, void* tau, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    global __cusolverDnXgeqrf
    _check_or_init_cusolverDn()
    if __cusolverDnXgeqrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgeqrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, size_t, void*, size_t, int*) nogil>__cusolverDnXgeqrf)(
        handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXgetrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXgetrf_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgetrf_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgetrf_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) nogil>__cusolverDnXgetrf_bufferSize)(
        handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgetrf(cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, int64_t* ipiv, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    global __cusolverDnXgetrf
    _check_or_init_cusolverDn()
    if __cusolverDnXgetrf == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgetrf is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, void*, int64_t, int64_t*, cudaDataType, void*, size_t, void*, size_t, int*) nogil>__cusolverDnXgetrf)(
        handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXgetrs(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasOperation_t trans, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, const int64_t* ipiv, cudaDataType dataTypeB, void* B, int64_t ldb, int* info) except* nogil:
    global __cusolverDnXgetrs
    _check_or_init_cusolverDn()
    if __cusolverDnXgetrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgetrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasOperation_t, int64_t, int64_t, cudaDataType, const void*, int64_t, const int64_t*, cudaDataType, void*, int64_t, int*) nogil>__cusolverDnXgetrs)(
        handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info)


cdef cusolverStatus_t _cusolverDnXsyevd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeW, const void* W, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXsyevd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, size_t*, size_t*) nogil>__cusolverDnXsyevd_bufferSize)(
        handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXsyevd(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    global __cusolverDnXsyevd
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, size_t, void*, size_t, int*) nogil>__cusolverDnXsyevd)(
        handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, void* vl, void* vu, int64_t il, int64_t iu, int64_t* h_meig, cudaDataType dataTypeW, const void* W, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXsyevdx_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevdx_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevdx_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int64_t, cudaDataType, const void*, int64_t, void*, void*, int64_t, int64_t, int64_t*, cudaDataType, const void*, cudaDataType, size_t*, size_t*) nogil>__cusolverDnXsyevdx_bufferSize)(
        handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXsyevdx(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, void* vl, void* vu, int64_t il, int64_t iu, int64_t* meig64, cudaDataType dataTypeW, void* W, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    global __cusolverDnXsyevdx
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevdx == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevdx is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int64_t, cudaDataType, void*, int64_t, void*, void*, int64_t, int64_t, int64_t*, cudaDataType, void*, cudaDataType, void*, size_t, void*, size_t, int*) nogil>__cusolverDnXsyevdx)(
        handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXgesvd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeS, const void* S, cudaDataType dataTypeU, const void* U, int64_t ldu, cudaDataType dataTypeVT, const void* VT, int64_t ldvt, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXgesvd_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvd_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, const void*, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) nogil>__cusolverDnXgesvd_bufferSize)(
        handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgesvd(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeS, void* S, cudaDataType dataTypeU, void* U, int64_t ldu, cudaDataType dataTypeVT, void* VT, int64_t ldvt, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    global __cusolverDnXgesvd
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvd == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvd is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t, int*) nogil>__cusolverDnXgesvd)(
        handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXgesvdp_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, int econ, int64_t m, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeS, const void* S, cudaDataType dataTypeU, const void* U, int64_t ldu, cudaDataType dataTypeV, const void* V, int64_t ldv, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXgesvdp_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdp_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdp_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, int, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, const void*, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) nogil>__cusolverDnXgesvdp_bufferSize)(
        handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgesvdp(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, int econ, int64_t m, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeS, void* S, cudaDataType dataTypeU, void* U, int64_t ldu, cudaDataType dataTypeV, void* V, int64_t ldv, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* d_info, double* h_err_sigma) except* nogil:
    global __cusolverDnXgesvdp
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdp is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, int, int64_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t, int*, double*) nogil>__cusolverDnXgesvdp)(
        handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info, h_err_sigma)


cdef cusolverStatus_t _cusolverDnXgesvdr_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeSrand, const void* Srand, cudaDataType dataTypeUrand, const void* Urand, int64_t ldUrand, cudaDataType dataTypeVrand, const void* Vrand, int64_t ldVrand, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXgesvdr_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdr_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t, int64_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, const void*, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) nogil>__cusolverDnXgesvdr_bufferSize)(
        handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgesvdr(cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeSrand, void* Srand, cudaDataType dataTypeUrand, void* Urand, int64_t ldUrand, cudaDataType dataTypeVrand, void* Vrand, int64_t ldVrand, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* d_info) except* nogil:
    global __cusolverDnXgesvdr
    _check_or_init_cusolverDn()
    if __cusolverDnXgesvdr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgesvdr is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t, int64_t, int64_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t, int*) nogil>__cusolverDnXgesvdr)(
        handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)


cdef cusolverStatus_t _cusolverDnXsytrs_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, const int64_t* ipiv, cudaDataType dataTypeB, void* B, int64_t ldb, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXsytrs_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXsytrs_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsytrs_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int64_t, int64_t, cudaDataType, const void*, int64_t, const int64_t*, cudaDataType, void*, int64_t, size_t*, size_t*) nogil>__cusolverDnXsytrs_bufferSize)(
        handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXsytrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void* A, int64_t lda, const int64_t* ipiv, cudaDataType dataTypeB, void* B, int64_t ldb, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    global __cusolverDnXsytrs
    _check_or_init_cusolverDn()
    if __cusolverDnXsytrs == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsytrs is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int64_t, int64_t, cudaDataType, const void*, int64_t, const int64_t*, cudaDataType, void*, int64_t, void*, size_t, void*, size_t, int*) nogil>__cusolverDnXsytrs)(
        handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)


cdef cusolverStatus_t _cusolverDnXtrtri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXtrtri_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXtrtri_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXtrtri_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int64_t, cudaDataType, void*, int64_t, size_t*, size_t*) nogil>__cusolverDnXtrtri_bufferSize)(
        handle, uplo, diag, n, dataTypeA, A, lda, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXtrtri(cusolverDnHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* devInfo) except* nogil:
    global __cusolverDnXtrtri
    _check_or_init_cusolverDn()
    if __cusolverDnXtrtri == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXtrtri is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int64_t, cudaDataType, void*, int64_t, void*, size_t, void*, size_t, int*) nogil>__cusolverDnXtrtri)(
        handle, uplo, diag, n, dataTypeA, A, lda, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, devInfo)


cdef cusolverStatus_t _cusolverDnLoggerSetCallback(cusolverDnLoggerCallback_t callback) except* nogil:
    global __cusolverDnLoggerSetCallback
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerSetCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerSetCallback is not found")
    return (<cusolverStatus_t (*)(cusolverDnLoggerCallback_t) nogil>__cusolverDnLoggerSetCallback)(
        callback)


cdef cusolverStatus_t _cusolverDnLoggerSetFile(FILE* file) except* nogil:
    global __cusolverDnLoggerSetFile
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerSetFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerSetFile is not found")
    return (<cusolverStatus_t (*)(FILE*) nogil>__cusolverDnLoggerSetFile)(
        file)


cdef cusolverStatus_t _cusolverDnLoggerOpenFile(const char* logFile) except* nogil:
    global __cusolverDnLoggerOpenFile
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerOpenFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerOpenFile is not found")
    return (<cusolverStatus_t (*)(const char*) nogil>__cusolverDnLoggerOpenFile)(
        logFile)


cdef cusolverStatus_t _cusolverDnLoggerSetLevel(int level) except* nogil:
    global __cusolverDnLoggerSetLevel
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerSetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerSetLevel is not found")
    return (<cusolverStatus_t (*)(int) nogil>__cusolverDnLoggerSetLevel)(
        level)


cdef cusolverStatus_t _cusolverDnLoggerSetMask(int mask) except* nogil:
    global __cusolverDnLoggerSetMask
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerSetMask == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerSetMask is not found")
    return (<cusolverStatus_t (*)(int) nogil>__cusolverDnLoggerSetMask)(
        mask)


cdef cusolverStatus_t _cusolverDnLoggerForceDisable() except* nogil:
    global __cusolverDnLoggerForceDisable
    _check_or_init_cusolverDn()
    if __cusolverDnLoggerForceDisable == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnLoggerForceDisable is not found")
    return (<cusolverStatus_t (*)() nogil>__cusolverDnLoggerForceDisable)(
        )


cdef cusolverStatus_t _cusolverDnSetDeterministicMode(cusolverDnHandle_t handle, cusolverDeterministicMode_t mode) except* nogil:
    global __cusolverDnSetDeterministicMode
    _check_or_init_cusolverDn()
    if __cusolverDnSetDeterministicMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnSetDeterministicMode is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDeterministicMode_t) nogil>__cusolverDnSetDeterministicMode)(
        handle, mode)


cdef cusolverStatus_t _cusolverDnGetDeterministicMode(cusolverDnHandle_t handle, cusolverDeterministicMode_t* mode) except* nogil:
    global __cusolverDnGetDeterministicMode
    _check_or_init_cusolverDn()
    if __cusolverDnGetDeterministicMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnGetDeterministicMode is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDeterministicMode_t*) nogil>__cusolverDnGetDeterministicMode)(
        handle, mode)


cdef cusolverStatus_t _cusolverDnXlarft_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n, int64_t k, cudaDataType dataTypeV, const void* V, int64_t ldv, cudaDataType dataTypeTau, const void* tau, cudaDataType dataTypeT, void* T, int64_t ldt, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXlarft_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXlarft_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXlarft_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverDirectMode_t, cusolverStorevMode_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, void*, int64_t, cudaDataType, size_t*, size_t*) nogil>__cusolverDnXlarft_bufferSize)(
        handle, params, direct, storev, n, k, dataTypeV, V, ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXlarft(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverDirectMode_t direct, cusolverStorevMode_t storev, int64_t n, int64_t k, cudaDataType dataTypeV, const void* V, int64_t ldv, cudaDataType dataTypeTau, const void* tau, cudaDataType dataTypeT, void* T, int64_t ldt, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXlarft
    _check_or_init_cusolverDn()
    if __cusolverDnXlarft == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXlarft is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverDirectMode_t, cusolverStorevMode_t, int64_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t) nogil>__cusolverDnXlarft)(
        handle, params, direct, storev, n, k, dataTypeV, V, ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXsyevBatched_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeW, const void* W, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost, int64_t batchSize) except* nogil:
    global __cusolverDnXsyevBatched_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevBatched_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevBatched_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, size_t*, size_t*, int64_t) nogil>__cusolverDnXsyevBatched_bufferSize)(
        handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost, batchSize)


cdef cusolverStatus_t _cusolverDnXsyevBatched(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info, int64_t batchSize) except* nogil:
    global __cusolverDnXsyevBatched
    _check_or_init_cusolverDn()
    if __cusolverDnXsyevBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXsyevBatched is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, size_t, void*, size_t, int*, int64_t) nogil>__cusolverDnXsyevBatched)(
        handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info, batchSize)


cdef cusolverStatus_t _cusolverDnXgeev_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobvl, cusolverEigMode_t jobvr, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda, cudaDataType dataTypeW, const void* W, cudaDataType dataTypeVL, const void* VL, int64_t ldvl, cudaDataType dataTypeVR, const void* VR, int64_t ldvr, cudaDataType computeType, size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost) except* nogil:
    global __cusolverDnXgeev_bufferSize
    _check_or_init_cusolverDn()
    if __cusolverDnXgeev_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgeev_bufferSize is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigMode_t, int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*, cudaDataType, const void*, int64_t, cudaDataType, const void*, int64_t, cudaDataType, size_t*, size_t*) nogil>__cusolverDnXgeev_bufferSize)(
        handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)


cdef cusolverStatus_t _cusolverDnXgeev(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobvl, cusolverEigMode_t jobvr, int64_t n, cudaDataType dataTypeA, void* A, int64_t lda, cudaDataType dataTypeW, void* W, cudaDataType dataTypeVL, void* VL, int64_t ldvl, cudaDataType dataTypeVR, void* VR, int64_t ldvr, cudaDataType computeType, void* bufferOnDevice, size_t workspaceInBytesOnDevice, void* bufferOnHost, size_t workspaceInBytesOnHost, int* info) except* nogil:
    global __cusolverDnXgeev
    _check_or_init_cusolverDn()
    if __cusolverDnXgeev == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverDnXgeev is not found")
    return (<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigMode_t, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*, int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t, int*) nogil>__cusolverDnXgeev)(
        handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
