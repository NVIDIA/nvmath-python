# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 13.2.0, generator version 0.3.1.dev1301+g7215ac36e. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

import os
import site
import threading

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib

from libc.stddef cimport wchar_t
from libc.stdint cimport uintptr_t
from cpython cimport PyUnicode_AsWideCharString, PyMem_Free

# You must 'from .utils import NotSupportedError' before using this template

cdef extern from "windows.h" nogil:
    ctypedef void* HMODULE
    ctypedef void* HANDLE
    ctypedef void* FARPROC
    ctypedef unsigned long DWORD
    ctypedef const wchar_t *LPCWSTR
    ctypedef const char *LPCSTR

    cdef DWORD LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800
    cdef DWORD LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
    cdef DWORD LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100

    HMODULE _LoadLibraryExW "LoadLibraryExW"(
        LPCWSTR lpLibFileName,
        HANDLE hFile,
        DWORD dwFlags
    )

    FARPROC _GetProcAddress "GetProcAddress"(HMODULE hModule, LPCSTR lpProcName)

cdef inline uintptr_t LoadLibraryExW(str path, HANDLE hFile, DWORD dwFlags):
    cdef uintptr_t result
    cdef wchar_t* wpath = PyUnicode_AsWideCharString(path, NULL)
    with nogil:
        result = <uintptr_t>_LoadLibraryExW(
            wpath,
            hFile,
            dwFlags
        )
    PyMem_Free(wpath)
    return result

cdef inline void *GetProcAddress(uintptr_t hModule, const char* lpProcName) nogil:
    return _GetProcAddress(<HMODULE>hModule, lpProcName)

cdef int get_cuda_version():
    cdef int err, driver_ver = 0

    # Load driver to check version
    handle = LoadLibraryExW("nvcuda.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32)
    if handle == 0:
        raise NotSupportedError('CUDA driver is not found')
    cuDriverGetVersion = GetProcAddress(handle, 'cuDriverGetVersion')
    if cuDriverGetVersion == NULL:
        raise RuntimeError('Did not find cuDriverGetVersion symbol in nvcuda.dll')
    err = (<int (*)(int*) noexcept nogil>cuDriverGetVersion)(&driver_ver)
    if err != 0:
        raise RuntimeError(f'cuDriverGetVersion returned error code {err}')

    return driver_ver


###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_cublas_init = False

cdef void* __cublasCreate_v2 = NULL
cdef void* __cublasDestroy_v2 = NULL
cdef void* __cublasGetVersion_v2 = NULL
cdef void* __cublasGetProperty = NULL
cdef void* __cublasGetCudartVersion = NULL
cdef void* __cublasSetWorkspace_v2 = NULL
cdef void* __cublasSetStream_v2 = NULL
cdef void* __cublasGetStream_v2 = NULL
cdef void* __cublasGetPointerMode_v2 = NULL
cdef void* __cublasSetPointerMode_v2 = NULL
cdef void* __cublasGetAtomicsMode = NULL
cdef void* __cublasSetAtomicsMode = NULL
cdef void* __cublasGetMathMode = NULL
cdef void* __cublasSetMathMode = NULL
cdef void* __cublasLoggerConfigure = NULL
cdef void* __cublasSetLoggerCallback = NULL
cdef void* __cublasGetLoggerCallback = NULL
cdef void* __cublasSetVector = NULL
cdef void* __cublasGetVector = NULL
cdef void* __cublasSetMatrix = NULL
cdef void* __cublasGetMatrix = NULL
cdef void* __cublasSetVectorAsync = NULL
cdef void* __cublasGetVectorAsync = NULL
cdef void* __cublasSetMatrixAsync = NULL
cdef void* __cublasGetMatrixAsync = NULL
cdef void* __cublasNrm2Ex = NULL
cdef void* __cublasSnrm2_v2 = NULL
cdef void* __cublasDnrm2_v2 = NULL
cdef void* __cublasScnrm2_v2 = NULL
cdef void* __cublasDznrm2_v2 = NULL
cdef void* __cublasDotEx = NULL
cdef void* __cublasDotcEx = NULL
cdef void* __cublasSdot_v2 = NULL
cdef void* __cublasDdot_v2 = NULL
cdef void* __cublasCdotu_v2 = NULL
cdef void* __cublasCdotc_v2 = NULL
cdef void* __cublasZdotu_v2 = NULL
cdef void* __cublasZdotc_v2 = NULL
cdef void* __cublasScalEx = NULL
cdef void* __cublasSscal_v2 = NULL
cdef void* __cublasDscal_v2 = NULL
cdef void* __cublasCscal_v2 = NULL
cdef void* __cublasCsscal_v2 = NULL
cdef void* __cublasZscal_v2 = NULL
cdef void* __cublasZdscal_v2 = NULL
cdef void* __cublasAxpyEx = NULL
cdef void* __cublasSaxpy_v2 = NULL
cdef void* __cublasDaxpy_v2 = NULL
cdef void* __cublasCaxpy_v2 = NULL
cdef void* __cublasZaxpy_v2 = NULL
cdef void* __cublasCopyEx = NULL
cdef void* __cublasScopy_v2 = NULL
cdef void* __cublasDcopy_v2 = NULL
cdef void* __cublasCcopy_v2 = NULL
cdef void* __cublasZcopy_v2 = NULL
cdef void* __cublasSswap_v2 = NULL
cdef void* __cublasDswap_v2 = NULL
cdef void* __cublasCswap_v2 = NULL
cdef void* __cublasZswap_v2 = NULL
cdef void* __cublasSwapEx = NULL
cdef void* __cublasIsamax_v2 = NULL
cdef void* __cublasIdamax_v2 = NULL
cdef void* __cublasIcamax_v2 = NULL
cdef void* __cublasIzamax_v2 = NULL
cdef void* __cublasIamaxEx = NULL
cdef void* __cublasIsamin_v2 = NULL
cdef void* __cublasIdamin_v2 = NULL
cdef void* __cublasIcamin_v2 = NULL
cdef void* __cublasIzamin_v2 = NULL
cdef void* __cublasIaminEx = NULL
cdef void* __cublasAsumEx = NULL
cdef void* __cublasSasum_v2 = NULL
cdef void* __cublasDasum_v2 = NULL
cdef void* __cublasScasum_v2 = NULL
cdef void* __cublasDzasum_v2 = NULL
cdef void* __cublasSrot_v2 = NULL
cdef void* __cublasDrot_v2 = NULL
cdef void* __cublasCrot_v2 = NULL
cdef void* __cublasCsrot_v2 = NULL
cdef void* __cublasZrot_v2 = NULL
cdef void* __cublasZdrot_v2 = NULL
cdef void* __cublasRotEx = NULL
cdef void* __cublasSrotg_v2 = NULL
cdef void* __cublasDrotg_v2 = NULL
cdef void* __cublasCrotg_v2 = NULL
cdef void* __cublasZrotg_v2 = NULL
cdef void* __cublasRotgEx = NULL
cdef void* __cublasSrotm_v2 = NULL
cdef void* __cublasDrotm_v2 = NULL
cdef void* __cublasRotmEx = NULL
cdef void* __cublasSrotmg_v2 = NULL
cdef void* __cublasDrotmg_v2 = NULL
cdef void* __cublasRotmgEx = NULL
cdef void* __cublasSgemv_v2 = NULL
cdef void* __cublasDgemv_v2 = NULL
cdef void* __cublasCgemv_v2 = NULL
cdef void* __cublasZgemv_v2 = NULL
cdef void* __cublasSgbmv_v2 = NULL
cdef void* __cublasDgbmv_v2 = NULL
cdef void* __cublasCgbmv_v2 = NULL
cdef void* __cublasZgbmv_v2 = NULL
cdef void* __cublasStrmv_v2 = NULL
cdef void* __cublasDtrmv_v2 = NULL
cdef void* __cublasCtrmv_v2 = NULL
cdef void* __cublasZtrmv_v2 = NULL
cdef void* __cublasStbmv_v2 = NULL
cdef void* __cublasDtbmv_v2 = NULL
cdef void* __cublasCtbmv_v2 = NULL
cdef void* __cublasZtbmv_v2 = NULL
cdef void* __cublasStpmv_v2 = NULL
cdef void* __cublasDtpmv_v2 = NULL
cdef void* __cublasCtpmv_v2 = NULL
cdef void* __cublasZtpmv_v2 = NULL
cdef void* __cublasStrsv_v2 = NULL
cdef void* __cublasDtrsv_v2 = NULL
cdef void* __cublasCtrsv_v2 = NULL
cdef void* __cublasZtrsv_v2 = NULL
cdef void* __cublasStpsv_v2 = NULL
cdef void* __cublasDtpsv_v2 = NULL
cdef void* __cublasCtpsv_v2 = NULL
cdef void* __cublasZtpsv_v2 = NULL
cdef void* __cublasStbsv_v2 = NULL
cdef void* __cublasDtbsv_v2 = NULL
cdef void* __cublasCtbsv_v2 = NULL
cdef void* __cublasZtbsv_v2 = NULL
cdef void* __cublasSsymv_v2 = NULL
cdef void* __cublasDsymv_v2 = NULL
cdef void* __cublasCsymv_v2 = NULL
cdef void* __cublasZsymv_v2 = NULL
cdef void* __cublasChemv_v2 = NULL
cdef void* __cublasZhemv_v2 = NULL
cdef void* __cublasSsbmv_v2 = NULL
cdef void* __cublasDsbmv_v2 = NULL
cdef void* __cublasChbmv_v2 = NULL
cdef void* __cublasZhbmv_v2 = NULL
cdef void* __cublasSspmv_v2 = NULL
cdef void* __cublasDspmv_v2 = NULL
cdef void* __cublasChpmv_v2 = NULL
cdef void* __cublasZhpmv_v2 = NULL
cdef void* __cublasSger_v2 = NULL
cdef void* __cublasDger_v2 = NULL
cdef void* __cublasCgeru_v2 = NULL
cdef void* __cublasCgerc_v2 = NULL
cdef void* __cublasZgeru_v2 = NULL
cdef void* __cublasZgerc_v2 = NULL
cdef void* __cublasSsyr_v2 = NULL
cdef void* __cublasDsyr_v2 = NULL
cdef void* __cublasCsyr_v2 = NULL
cdef void* __cublasZsyr_v2 = NULL
cdef void* __cublasCher_v2 = NULL
cdef void* __cublasZher_v2 = NULL
cdef void* __cublasSspr_v2 = NULL
cdef void* __cublasDspr_v2 = NULL
cdef void* __cublasChpr_v2 = NULL
cdef void* __cublasZhpr_v2 = NULL
cdef void* __cublasSsyr2_v2 = NULL
cdef void* __cublasDsyr2_v2 = NULL
cdef void* __cublasCsyr2_v2 = NULL
cdef void* __cublasZsyr2_v2 = NULL
cdef void* __cublasCher2_v2 = NULL
cdef void* __cublasZher2_v2 = NULL
cdef void* __cublasSspr2_v2 = NULL
cdef void* __cublasDspr2_v2 = NULL
cdef void* __cublasChpr2_v2 = NULL
cdef void* __cublasZhpr2_v2 = NULL
cdef void* __cublasSgemm_v2 = NULL
cdef void* __cublasDgemm_v2 = NULL
cdef void* __cublasCgemm_v2 = NULL
cdef void* __cublasCgemm3m = NULL
cdef void* __cublasCgemm3mEx = NULL
cdef void* __cublasZgemm_v2 = NULL
cdef void* __cublasZgemm3m = NULL
cdef void* __cublasSgemmEx = NULL
cdef void* __cublasGemmEx = NULL
cdef void* __cublasCgemmEx = NULL
cdef void* __cublasUint8gemmBias = NULL
cdef void* __cublasSsyrk_v2 = NULL
cdef void* __cublasDsyrk_v2 = NULL
cdef void* __cublasCsyrk_v2 = NULL
cdef void* __cublasZsyrk_v2 = NULL
cdef void* __cublasCsyrkEx = NULL
cdef void* __cublasCsyrk3mEx = NULL
cdef void* __cublasCherk_v2 = NULL
cdef void* __cublasZherk_v2 = NULL
cdef void* __cublasCherkEx = NULL
cdef void* __cublasCherk3mEx = NULL
cdef void* __cublasSsyr2k_v2 = NULL
cdef void* __cublasDsyr2k_v2 = NULL
cdef void* __cublasCsyr2k_v2 = NULL
cdef void* __cublasZsyr2k_v2 = NULL
cdef void* __cublasCher2k_v2 = NULL
cdef void* __cublasZher2k_v2 = NULL
cdef void* __cublasSsyrkx = NULL
cdef void* __cublasDsyrkx = NULL
cdef void* __cublasCsyrkx = NULL
cdef void* __cublasZsyrkx = NULL
cdef void* __cublasCherkx = NULL
cdef void* __cublasZherkx = NULL
cdef void* __cublasSsymm_v2 = NULL
cdef void* __cublasDsymm_v2 = NULL
cdef void* __cublasCsymm_v2 = NULL
cdef void* __cublasZsymm_v2 = NULL
cdef void* __cublasChemm_v2 = NULL
cdef void* __cublasZhemm_v2 = NULL
cdef void* __cublasStrsm_v2 = NULL
cdef void* __cublasDtrsm_v2 = NULL
cdef void* __cublasCtrsm_v2 = NULL
cdef void* __cublasZtrsm_v2 = NULL
cdef void* __cublasStrmm_v2 = NULL
cdef void* __cublasDtrmm_v2 = NULL
cdef void* __cublasCtrmm_v2 = NULL
cdef void* __cublasZtrmm_v2 = NULL
cdef void* __cublasSgemmBatched = NULL
cdef void* __cublasDgemmBatched = NULL
cdef void* __cublasCgemmBatched = NULL
cdef void* __cublasCgemm3mBatched = NULL
cdef void* __cublasZgemmBatched = NULL
cdef void* __cublasGemmBatchedEx = NULL
cdef void* __cublasGemmStridedBatchedEx = NULL
cdef void* __cublasSgemmStridedBatched = NULL
cdef void* __cublasDgemmStridedBatched = NULL
cdef void* __cublasCgemmStridedBatched = NULL
cdef void* __cublasCgemm3mStridedBatched = NULL
cdef void* __cublasZgemmStridedBatched = NULL
cdef void* __cublasSgeam = NULL
cdef void* __cublasDgeam = NULL
cdef void* __cublasCgeam = NULL
cdef void* __cublasZgeam = NULL
cdef void* __cublasSgetrfBatched = NULL
cdef void* __cublasDgetrfBatched = NULL
cdef void* __cublasCgetrfBatched = NULL
cdef void* __cublasZgetrfBatched = NULL
cdef void* __cublasSgetriBatched = NULL
cdef void* __cublasDgetriBatched = NULL
cdef void* __cublasCgetriBatched = NULL
cdef void* __cublasZgetriBatched = NULL
cdef void* __cublasSgetrsBatched = NULL
cdef void* __cublasDgetrsBatched = NULL
cdef void* __cublasCgetrsBatched = NULL
cdef void* __cublasZgetrsBatched = NULL
cdef void* __cublasStrsmBatched = NULL
cdef void* __cublasDtrsmBatched = NULL
cdef void* __cublasCtrsmBatched = NULL
cdef void* __cublasZtrsmBatched = NULL
cdef void* __cublasSmatinvBatched = NULL
cdef void* __cublasDmatinvBatched = NULL
cdef void* __cublasCmatinvBatched = NULL
cdef void* __cublasZmatinvBatched = NULL
cdef void* __cublasSgeqrfBatched = NULL
cdef void* __cublasDgeqrfBatched = NULL
cdef void* __cublasCgeqrfBatched = NULL
cdef void* __cublasZgeqrfBatched = NULL
cdef void* __cublasSgelsBatched = NULL
cdef void* __cublasDgelsBatched = NULL
cdef void* __cublasCgelsBatched = NULL
cdef void* __cublasZgelsBatched = NULL
cdef void* __cublasSdgmm = NULL
cdef void* __cublasDdgmm = NULL
cdef void* __cublasCdgmm = NULL
cdef void* __cublasZdgmm = NULL
cdef void* __cublasStpttr = NULL
cdef void* __cublasDtpttr = NULL
cdef void* __cublasCtpttr = NULL
cdef void* __cublasZtpttr = NULL
cdef void* __cublasStrttp = NULL
cdef void* __cublasDtrttp = NULL
cdef void* __cublasCtrttp = NULL
cdef void* __cublasZtrttp = NULL
cdef void* __cublasGetSmCountTarget = NULL
cdef void* __cublasSetSmCountTarget = NULL
cdef void* __cublasGetStatusName = NULL
cdef void* __cublasGetStatusString = NULL
cdef void* __cublasSgemvBatched = NULL
cdef void* __cublasDgemvBatched = NULL
cdef void* __cublasCgemvBatched = NULL
cdef void* __cublasZgemvBatched = NULL
cdef void* __cublasSgemvStridedBatched = NULL
cdef void* __cublasDgemvStridedBatched = NULL
cdef void* __cublasCgemvStridedBatched = NULL
cdef void* __cublasZgemvStridedBatched = NULL
cdef void* __cublasSetVector_64 = NULL
cdef void* __cublasGetVector_64 = NULL
cdef void* __cublasSetMatrix_64 = NULL
cdef void* __cublasGetMatrix_64 = NULL
cdef void* __cublasSetVectorAsync_64 = NULL
cdef void* __cublasGetVectorAsync_64 = NULL
cdef void* __cublasSetMatrixAsync_64 = NULL
cdef void* __cublasGetMatrixAsync_64 = NULL
cdef void* __cublasNrm2Ex_64 = NULL
cdef void* __cublasSnrm2_v2_64 = NULL
cdef void* __cublasDnrm2_v2_64 = NULL
cdef void* __cublasScnrm2_v2_64 = NULL
cdef void* __cublasDznrm2_v2_64 = NULL
cdef void* __cublasDotEx_64 = NULL
cdef void* __cublasDotcEx_64 = NULL
cdef void* __cublasSdot_v2_64 = NULL
cdef void* __cublasDdot_v2_64 = NULL
cdef void* __cublasCdotu_v2_64 = NULL
cdef void* __cublasCdotc_v2_64 = NULL
cdef void* __cublasZdotu_v2_64 = NULL
cdef void* __cublasZdotc_v2_64 = NULL
cdef void* __cublasScalEx_64 = NULL
cdef void* __cublasSscal_v2_64 = NULL
cdef void* __cublasDscal_v2_64 = NULL
cdef void* __cublasCscal_v2_64 = NULL
cdef void* __cublasCsscal_v2_64 = NULL
cdef void* __cublasZscal_v2_64 = NULL
cdef void* __cublasZdscal_v2_64 = NULL
cdef void* __cublasAxpyEx_64 = NULL
cdef void* __cublasSaxpy_v2_64 = NULL
cdef void* __cublasDaxpy_v2_64 = NULL
cdef void* __cublasCaxpy_v2_64 = NULL
cdef void* __cublasZaxpy_v2_64 = NULL
cdef void* __cublasCopyEx_64 = NULL
cdef void* __cublasScopy_v2_64 = NULL
cdef void* __cublasDcopy_v2_64 = NULL
cdef void* __cublasCcopy_v2_64 = NULL
cdef void* __cublasZcopy_v2_64 = NULL
cdef void* __cublasSswap_v2_64 = NULL
cdef void* __cublasDswap_v2_64 = NULL
cdef void* __cublasCswap_v2_64 = NULL
cdef void* __cublasZswap_v2_64 = NULL
cdef void* __cublasSwapEx_64 = NULL
cdef void* __cublasIsamax_v2_64 = NULL
cdef void* __cublasIdamax_v2_64 = NULL
cdef void* __cublasIcamax_v2_64 = NULL
cdef void* __cublasIzamax_v2_64 = NULL
cdef void* __cublasIamaxEx_64 = NULL
cdef void* __cublasIsamin_v2_64 = NULL
cdef void* __cublasIdamin_v2_64 = NULL
cdef void* __cublasIcamin_v2_64 = NULL
cdef void* __cublasIzamin_v2_64 = NULL
cdef void* __cublasIaminEx_64 = NULL
cdef void* __cublasAsumEx_64 = NULL
cdef void* __cublasSasum_v2_64 = NULL
cdef void* __cublasDasum_v2_64 = NULL
cdef void* __cublasScasum_v2_64 = NULL
cdef void* __cublasDzasum_v2_64 = NULL
cdef void* __cublasSrot_v2_64 = NULL
cdef void* __cublasDrot_v2_64 = NULL
cdef void* __cublasCrot_v2_64 = NULL
cdef void* __cublasCsrot_v2_64 = NULL
cdef void* __cublasZrot_v2_64 = NULL
cdef void* __cublasZdrot_v2_64 = NULL
cdef void* __cublasRotEx_64 = NULL
cdef void* __cublasSrotm_v2_64 = NULL
cdef void* __cublasDrotm_v2_64 = NULL
cdef void* __cublasRotmEx_64 = NULL
cdef void* __cublasSgemv_v2_64 = NULL
cdef void* __cublasDgemv_v2_64 = NULL
cdef void* __cublasCgemv_v2_64 = NULL
cdef void* __cublasZgemv_v2_64 = NULL
cdef void* __cublasSgbmv_v2_64 = NULL
cdef void* __cublasDgbmv_v2_64 = NULL
cdef void* __cublasCgbmv_v2_64 = NULL
cdef void* __cublasZgbmv_v2_64 = NULL
cdef void* __cublasStrmv_v2_64 = NULL
cdef void* __cublasDtrmv_v2_64 = NULL
cdef void* __cublasCtrmv_v2_64 = NULL
cdef void* __cublasZtrmv_v2_64 = NULL
cdef void* __cublasStbmv_v2_64 = NULL
cdef void* __cublasDtbmv_v2_64 = NULL
cdef void* __cublasCtbmv_v2_64 = NULL
cdef void* __cublasZtbmv_v2_64 = NULL
cdef void* __cublasStpmv_v2_64 = NULL
cdef void* __cublasDtpmv_v2_64 = NULL
cdef void* __cublasCtpmv_v2_64 = NULL
cdef void* __cublasZtpmv_v2_64 = NULL
cdef void* __cublasStrsv_v2_64 = NULL
cdef void* __cublasDtrsv_v2_64 = NULL
cdef void* __cublasCtrsv_v2_64 = NULL
cdef void* __cublasZtrsv_v2_64 = NULL
cdef void* __cublasStpsv_v2_64 = NULL
cdef void* __cublasDtpsv_v2_64 = NULL
cdef void* __cublasCtpsv_v2_64 = NULL
cdef void* __cublasZtpsv_v2_64 = NULL
cdef void* __cublasStbsv_v2_64 = NULL
cdef void* __cublasDtbsv_v2_64 = NULL
cdef void* __cublasCtbsv_v2_64 = NULL
cdef void* __cublasZtbsv_v2_64 = NULL
cdef void* __cublasSsymv_v2_64 = NULL
cdef void* __cublasDsymv_v2_64 = NULL
cdef void* __cublasCsymv_v2_64 = NULL
cdef void* __cublasZsymv_v2_64 = NULL
cdef void* __cublasChemv_v2_64 = NULL
cdef void* __cublasZhemv_v2_64 = NULL
cdef void* __cublasSsbmv_v2_64 = NULL
cdef void* __cublasDsbmv_v2_64 = NULL
cdef void* __cublasChbmv_v2_64 = NULL
cdef void* __cublasZhbmv_v2_64 = NULL
cdef void* __cublasSspmv_v2_64 = NULL
cdef void* __cublasDspmv_v2_64 = NULL
cdef void* __cublasChpmv_v2_64 = NULL
cdef void* __cublasZhpmv_v2_64 = NULL
cdef void* __cublasSger_v2_64 = NULL
cdef void* __cublasDger_v2_64 = NULL
cdef void* __cublasCgeru_v2_64 = NULL
cdef void* __cublasCgerc_v2_64 = NULL
cdef void* __cublasZgeru_v2_64 = NULL
cdef void* __cublasZgerc_v2_64 = NULL
cdef void* __cublasSsyr_v2_64 = NULL
cdef void* __cublasDsyr_v2_64 = NULL
cdef void* __cublasCsyr_v2_64 = NULL
cdef void* __cublasZsyr_v2_64 = NULL
cdef void* __cublasCher_v2_64 = NULL
cdef void* __cublasZher_v2_64 = NULL
cdef void* __cublasSspr_v2_64 = NULL
cdef void* __cublasDspr_v2_64 = NULL
cdef void* __cublasChpr_v2_64 = NULL
cdef void* __cublasZhpr_v2_64 = NULL
cdef void* __cublasSsyr2_v2_64 = NULL
cdef void* __cublasDsyr2_v2_64 = NULL
cdef void* __cublasCsyr2_v2_64 = NULL
cdef void* __cublasZsyr2_v2_64 = NULL
cdef void* __cublasCher2_v2_64 = NULL
cdef void* __cublasZher2_v2_64 = NULL
cdef void* __cublasSspr2_v2_64 = NULL
cdef void* __cublasDspr2_v2_64 = NULL
cdef void* __cublasChpr2_v2_64 = NULL
cdef void* __cublasZhpr2_v2_64 = NULL
cdef void* __cublasSgemvBatched_64 = NULL
cdef void* __cublasDgemvBatched_64 = NULL
cdef void* __cublasCgemvBatched_64 = NULL
cdef void* __cublasZgemvBatched_64 = NULL
cdef void* __cublasSgemvStridedBatched_64 = NULL
cdef void* __cublasDgemvStridedBatched_64 = NULL
cdef void* __cublasCgemvStridedBatched_64 = NULL
cdef void* __cublasZgemvStridedBatched_64 = NULL
cdef void* __cublasSgemm_v2_64 = NULL
cdef void* __cublasDgemm_v2_64 = NULL
cdef void* __cublasCgemm_v2_64 = NULL
cdef void* __cublasCgemm3m_64 = NULL
cdef void* __cublasCgemm3mEx_64 = NULL
cdef void* __cublasZgemm_v2_64 = NULL
cdef void* __cublasZgemm3m_64 = NULL
cdef void* __cublasSgemmEx_64 = NULL
cdef void* __cublasGemmEx_64 = NULL
cdef void* __cublasCgemmEx_64 = NULL
cdef void* __cublasSsyrk_v2_64 = NULL
cdef void* __cublasDsyrk_v2_64 = NULL
cdef void* __cublasCsyrk_v2_64 = NULL
cdef void* __cublasZsyrk_v2_64 = NULL
cdef void* __cublasCsyrkEx_64 = NULL
cdef void* __cublasCsyrk3mEx_64 = NULL
cdef void* __cublasCherk_v2_64 = NULL
cdef void* __cublasZherk_v2_64 = NULL
cdef void* __cublasCherkEx_64 = NULL
cdef void* __cublasCherk3mEx_64 = NULL
cdef void* __cublasSsyr2k_v2_64 = NULL
cdef void* __cublasDsyr2k_v2_64 = NULL
cdef void* __cublasCsyr2k_v2_64 = NULL
cdef void* __cublasZsyr2k_v2_64 = NULL
cdef void* __cublasCher2k_v2_64 = NULL
cdef void* __cublasZher2k_v2_64 = NULL
cdef void* __cublasSsyrkx_64 = NULL
cdef void* __cublasDsyrkx_64 = NULL
cdef void* __cublasCsyrkx_64 = NULL
cdef void* __cublasZsyrkx_64 = NULL
cdef void* __cublasCherkx_64 = NULL
cdef void* __cublasZherkx_64 = NULL
cdef void* __cublasSsymm_v2_64 = NULL
cdef void* __cublasDsymm_v2_64 = NULL
cdef void* __cublasCsymm_v2_64 = NULL
cdef void* __cublasZsymm_v2_64 = NULL
cdef void* __cublasChemm_v2_64 = NULL
cdef void* __cublasZhemm_v2_64 = NULL
cdef void* __cublasStrsm_v2_64 = NULL
cdef void* __cublasDtrsm_v2_64 = NULL
cdef void* __cublasCtrsm_v2_64 = NULL
cdef void* __cublasZtrsm_v2_64 = NULL
cdef void* __cublasStrmm_v2_64 = NULL
cdef void* __cublasDtrmm_v2_64 = NULL
cdef void* __cublasCtrmm_v2_64 = NULL
cdef void* __cublasZtrmm_v2_64 = NULL
cdef void* __cublasSgemmBatched_64 = NULL
cdef void* __cublasDgemmBatched_64 = NULL
cdef void* __cublasCgemmBatched_64 = NULL
cdef void* __cublasCgemm3mBatched_64 = NULL
cdef void* __cublasZgemmBatched_64 = NULL
cdef void* __cublasSgemmStridedBatched_64 = NULL
cdef void* __cublasDgemmStridedBatched_64 = NULL
cdef void* __cublasCgemmStridedBatched_64 = NULL
cdef void* __cublasCgemm3mStridedBatched_64 = NULL
cdef void* __cublasZgemmStridedBatched_64 = NULL
cdef void* __cublasGemmBatchedEx_64 = NULL
cdef void* __cublasGemmStridedBatchedEx_64 = NULL
cdef void* __cublasSgeam_64 = NULL
cdef void* __cublasDgeam_64 = NULL
cdef void* __cublasCgeam_64 = NULL
cdef void* __cublasZgeam_64 = NULL
cdef void* __cublasStrsmBatched_64 = NULL
cdef void* __cublasDtrsmBatched_64 = NULL
cdef void* __cublasCtrsmBatched_64 = NULL
cdef void* __cublasZtrsmBatched_64 = NULL
cdef void* __cublasSdgmm_64 = NULL
cdef void* __cublasDdgmm_64 = NULL
cdef void* __cublasCdgmm_64 = NULL
cdef void* __cublasZdgmm_64 = NULL
cdef void* __cublasSgemmGroupedBatched = NULL
cdef void* __cublasSgemmGroupedBatched_64 = NULL
cdef void* __cublasDgemmGroupedBatched = NULL
cdef void* __cublasDgemmGroupedBatched_64 = NULL
cdef void* __cublasGemmGroupedBatchedEx = NULL
cdef void* __cublasGemmGroupedBatchedEx_64 = NULL
cdef void* __cublasGetEmulationStrategy = NULL
cdef void* __cublasSetEmulationStrategy = NULL
cdef void* __cublasGetEmulationSpecialValuesSupport = NULL
cdef void* __cublasSetEmulationSpecialValuesSupport = NULL
cdef void* __cublasGetFixedPointEmulationMantissaControl = NULL
cdef void* __cublasSetFixedPointEmulationMantissaControl = NULL
cdef void* __cublasGetFixedPointEmulationMaxMantissaBitCount = NULL
cdef void* __cublasSetFixedPointEmulationMaxMantissaBitCount = NULL
cdef void* __cublasGetFixedPointEmulationMantissaBitOffset = NULL
cdef void* __cublasSetFixedPointEmulationMantissaBitOffset = NULL
cdef void* __cublasGetFixedPointEmulationMantissaBitCountPointer = NULL
cdef void* __cublasSetFixedPointEmulationMantissaBitCountPointer = NULL


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef void* load_library(const int driver_ver) except* with gil:
    handle = load_nvidia_dynamic_lib("cublas")._handle_uint
    return <void*><intptr_t>handle

cdef int _check_or_init_cublas() except -1 nogil:
    global __py_cublas_init
    if __py_cublas_init:
        return 0

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_cublas_init:
            return 0

        driver_ver = get_cuda_version()

        # Load library
        handle = <intptr_t>load_library(driver_ver)

        # Load function
        global __cublasCreate_v2
        __cublasCreate_v2 = GetProcAddress(handle, 'cublasCreate_v2')

        global __cublasDestroy_v2
        __cublasDestroy_v2 = GetProcAddress(handle, 'cublasDestroy_v2')

        global __cublasGetVersion_v2
        __cublasGetVersion_v2 = GetProcAddress(handle, 'cublasGetVersion_v2')

        global __cublasGetProperty
        __cublasGetProperty = GetProcAddress(handle, 'cublasGetProperty')

        global __cublasGetCudartVersion
        __cublasGetCudartVersion = GetProcAddress(handle, 'cublasGetCudartVersion')

        global __cublasSetWorkspace_v2
        __cublasSetWorkspace_v2 = GetProcAddress(handle, 'cublasSetWorkspace_v2')

        global __cublasSetStream_v2
        __cublasSetStream_v2 = GetProcAddress(handle, 'cublasSetStream_v2')

        global __cublasGetStream_v2
        __cublasGetStream_v2 = GetProcAddress(handle, 'cublasGetStream_v2')

        global __cublasGetPointerMode_v2
        __cublasGetPointerMode_v2 = GetProcAddress(handle, 'cublasGetPointerMode_v2')

        global __cublasSetPointerMode_v2
        __cublasSetPointerMode_v2 = GetProcAddress(handle, 'cublasSetPointerMode_v2')

        global __cublasGetAtomicsMode
        __cublasGetAtomicsMode = GetProcAddress(handle, 'cublasGetAtomicsMode')

        global __cublasSetAtomicsMode
        __cublasSetAtomicsMode = GetProcAddress(handle, 'cublasSetAtomicsMode')

        global __cublasGetMathMode
        __cublasGetMathMode = GetProcAddress(handle, 'cublasGetMathMode')

        global __cublasSetMathMode
        __cublasSetMathMode = GetProcAddress(handle, 'cublasSetMathMode')

        global __cublasLoggerConfigure
        __cublasLoggerConfigure = GetProcAddress(handle, 'cublasLoggerConfigure')

        global __cublasSetLoggerCallback
        __cublasSetLoggerCallback = GetProcAddress(handle, 'cublasSetLoggerCallback')

        global __cublasGetLoggerCallback
        __cublasGetLoggerCallback = GetProcAddress(handle, 'cublasGetLoggerCallback')

        global __cublasSetVector
        __cublasSetVector = GetProcAddress(handle, 'cublasSetVector')

        global __cublasGetVector
        __cublasGetVector = GetProcAddress(handle, 'cublasGetVector')

        global __cublasSetMatrix
        __cublasSetMatrix = GetProcAddress(handle, 'cublasSetMatrix')

        global __cublasGetMatrix
        __cublasGetMatrix = GetProcAddress(handle, 'cublasGetMatrix')

        global __cublasSetVectorAsync
        __cublasSetVectorAsync = GetProcAddress(handle, 'cublasSetVectorAsync')

        global __cublasGetVectorAsync
        __cublasGetVectorAsync = GetProcAddress(handle, 'cublasGetVectorAsync')

        global __cublasSetMatrixAsync
        __cublasSetMatrixAsync = GetProcAddress(handle, 'cublasSetMatrixAsync')

        global __cublasGetMatrixAsync
        __cublasGetMatrixAsync = GetProcAddress(handle, 'cublasGetMatrixAsync')

        global __cublasNrm2Ex
        __cublasNrm2Ex = GetProcAddress(handle, 'cublasNrm2Ex')

        global __cublasSnrm2_v2
        __cublasSnrm2_v2 = GetProcAddress(handle, 'cublasSnrm2_v2')

        global __cublasDnrm2_v2
        __cublasDnrm2_v2 = GetProcAddress(handle, 'cublasDnrm2_v2')

        global __cublasScnrm2_v2
        __cublasScnrm2_v2 = GetProcAddress(handle, 'cublasScnrm2_v2')

        global __cublasDznrm2_v2
        __cublasDznrm2_v2 = GetProcAddress(handle, 'cublasDznrm2_v2')

        global __cublasDotEx
        __cublasDotEx = GetProcAddress(handle, 'cublasDotEx')

        global __cublasDotcEx
        __cublasDotcEx = GetProcAddress(handle, 'cublasDotcEx')

        global __cublasSdot_v2
        __cublasSdot_v2 = GetProcAddress(handle, 'cublasSdot_v2')

        global __cublasDdot_v2
        __cublasDdot_v2 = GetProcAddress(handle, 'cublasDdot_v2')

        global __cublasCdotu_v2
        __cublasCdotu_v2 = GetProcAddress(handle, 'cublasCdotu_v2')

        global __cublasCdotc_v2
        __cublasCdotc_v2 = GetProcAddress(handle, 'cublasCdotc_v2')

        global __cublasZdotu_v2
        __cublasZdotu_v2 = GetProcAddress(handle, 'cublasZdotu_v2')

        global __cublasZdotc_v2
        __cublasZdotc_v2 = GetProcAddress(handle, 'cublasZdotc_v2')

        global __cublasScalEx
        __cublasScalEx = GetProcAddress(handle, 'cublasScalEx')

        global __cublasSscal_v2
        __cublasSscal_v2 = GetProcAddress(handle, 'cublasSscal_v2')

        global __cublasDscal_v2
        __cublasDscal_v2 = GetProcAddress(handle, 'cublasDscal_v2')

        global __cublasCscal_v2
        __cublasCscal_v2 = GetProcAddress(handle, 'cublasCscal_v2')

        global __cublasCsscal_v2
        __cublasCsscal_v2 = GetProcAddress(handle, 'cublasCsscal_v2')

        global __cublasZscal_v2
        __cublasZscal_v2 = GetProcAddress(handle, 'cublasZscal_v2')

        global __cublasZdscal_v2
        __cublasZdscal_v2 = GetProcAddress(handle, 'cublasZdscal_v2')

        global __cublasAxpyEx
        __cublasAxpyEx = GetProcAddress(handle, 'cublasAxpyEx')

        global __cublasSaxpy_v2
        __cublasSaxpy_v2 = GetProcAddress(handle, 'cublasSaxpy_v2')

        global __cublasDaxpy_v2
        __cublasDaxpy_v2 = GetProcAddress(handle, 'cublasDaxpy_v2')

        global __cublasCaxpy_v2
        __cublasCaxpy_v2 = GetProcAddress(handle, 'cublasCaxpy_v2')

        global __cublasZaxpy_v2
        __cublasZaxpy_v2 = GetProcAddress(handle, 'cublasZaxpy_v2')

        global __cublasCopyEx
        __cublasCopyEx = GetProcAddress(handle, 'cublasCopyEx')

        global __cublasScopy_v2
        __cublasScopy_v2 = GetProcAddress(handle, 'cublasScopy_v2')

        global __cublasDcopy_v2
        __cublasDcopy_v2 = GetProcAddress(handle, 'cublasDcopy_v2')

        global __cublasCcopy_v2
        __cublasCcopy_v2 = GetProcAddress(handle, 'cublasCcopy_v2')

        global __cublasZcopy_v2
        __cublasZcopy_v2 = GetProcAddress(handle, 'cublasZcopy_v2')

        global __cublasSswap_v2
        __cublasSswap_v2 = GetProcAddress(handle, 'cublasSswap_v2')

        global __cublasDswap_v2
        __cublasDswap_v2 = GetProcAddress(handle, 'cublasDswap_v2')

        global __cublasCswap_v2
        __cublasCswap_v2 = GetProcAddress(handle, 'cublasCswap_v2')

        global __cublasZswap_v2
        __cublasZswap_v2 = GetProcAddress(handle, 'cublasZswap_v2')

        global __cublasSwapEx
        __cublasSwapEx = GetProcAddress(handle, 'cublasSwapEx')

        global __cublasIsamax_v2
        __cublasIsamax_v2 = GetProcAddress(handle, 'cublasIsamax_v2')

        global __cublasIdamax_v2
        __cublasIdamax_v2 = GetProcAddress(handle, 'cublasIdamax_v2')

        global __cublasIcamax_v2
        __cublasIcamax_v2 = GetProcAddress(handle, 'cublasIcamax_v2')

        global __cublasIzamax_v2
        __cublasIzamax_v2 = GetProcAddress(handle, 'cublasIzamax_v2')

        global __cublasIamaxEx
        __cublasIamaxEx = GetProcAddress(handle, 'cublasIamaxEx')

        global __cublasIsamin_v2
        __cublasIsamin_v2 = GetProcAddress(handle, 'cublasIsamin_v2')

        global __cublasIdamin_v2
        __cublasIdamin_v2 = GetProcAddress(handle, 'cublasIdamin_v2')

        global __cublasIcamin_v2
        __cublasIcamin_v2 = GetProcAddress(handle, 'cublasIcamin_v2')

        global __cublasIzamin_v2
        __cublasIzamin_v2 = GetProcAddress(handle, 'cublasIzamin_v2')

        global __cublasIaminEx
        __cublasIaminEx = GetProcAddress(handle, 'cublasIaminEx')

        global __cublasAsumEx
        __cublasAsumEx = GetProcAddress(handle, 'cublasAsumEx')

        global __cublasSasum_v2
        __cublasSasum_v2 = GetProcAddress(handle, 'cublasSasum_v2')

        global __cublasDasum_v2
        __cublasDasum_v2 = GetProcAddress(handle, 'cublasDasum_v2')

        global __cublasScasum_v2
        __cublasScasum_v2 = GetProcAddress(handle, 'cublasScasum_v2')

        global __cublasDzasum_v2
        __cublasDzasum_v2 = GetProcAddress(handle, 'cublasDzasum_v2')

        global __cublasSrot_v2
        __cublasSrot_v2 = GetProcAddress(handle, 'cublasSrot_v2')

        global __cublasDrot_v2
        __cublasDrot_v2 = GetProcAddress(handle, 'cublasDrot_v2')

        global __cublasCrot_v2
        __cublasCrot_v2 = GetProcAddress(handle, 'cublasCrot_v2')

        global __cublasCsrot_v2
        __cublasCsrot_v2 = GetProcAddress(handle, 'cublasCsrot_v2')

        global __cublasZrot_v2
        __cublasZrot_v2 = GetProcAddress(handle, 'cublasZrot_v2')

        global __cublasZdrot_v2
        __cublasZdrot_v2 = GetProcAddress(handle, 'cublasZdrot_v2')

        global __cublasRotEx
        __cublasRotEx = GetProcAddress(handle, 'cublasRotEx')

        global __cublasSrotg_v2
        __cublasSrotg_v2 = GetProcAddress(handle, 'cublasSrotg_v2')

        global __cublasDrotg_v2
        __cublasDrotg_v2 = GetProcAddress(handle, 'cublasDrotg_v2')

        global __cublasCrotg_v2
        __cublasCrotg_v2 = GetProcAddress(handle, 'cublasCrotg_v2')

        global __cublasZrotg_v2
        __cublasZrotg_v2 = GetProcAddress(handle, 'cublasZrotg_v2')

        global __cublasRotgEx
        __cublasRotgEx = GetProcAddress(handle, 'cublasRotgEx')

        global __cublasSrotm_v2
        __cublasSrotm_v2 = GetProcAddress(handle, 'cublasSrotm_v2')

        global __cublasDrotm_v2
        __cublasDrotm_v2 = GetProcAddress(handle, 'cublasDrotm_v2')

        global __cublasRotmEx
        __cublasRotmEx = GetProcAddress(handle, 'cublasRotmEx')

        global __cublasSrotmg_v2
        __cublasSrotmg_v2 = GetProcAddress(handle, 'cublasSrotmg_v2')

        global __cublasDrotmg_v2
        __cublasDrotmg_v2 = GetProcAddress(handle, 'cublasDrotmg_v2')

        global __cublasRotmgEx
        __cublasRotmgEx = GetProcAddress(handle, 'cublasRotmgEx')

        global __cublasSgemv_v2
        __cublasSgemv_v2 = GetProcAddress(handle, 'cublasSgemv_v2')

        global __cublasDgemv_v2
        __cublasDgemv_v2 = GetProcAddress(handle, 'cublasDgemv_v2')

        global __cublasCgemv_v2
        __cublasCgemv_v2 = GetProcAddress(handle, 'cublasCgemv_v2')

        global __cublasZgemv_v2
        __cublasZgemv_v2 = GetProcAddress(handle, 'cublasZgemv_v2')

        global __cublasSgbmv_v2
        __cublasSgbmv_v2 = GetProcAddress(handle, 'cublasSgbmv_v2')

        global __cublasDgbmv_v2
        __cublasDgbmv_v2 = GetProcAddress(handle, 'cublasDgbmv_v2')

        global __cublasCgbmv_v2
        __cublasCgbmv_v2 = GetProcAddress(handle, 'cublasCgbmv_v2')

        global __cublasZgbmv_v2
        __cublasZgbmv_v2 = GetProcAddress(handle, 'cublasZgbmv_v2')

        global __cublasStrmv_v2
        __cublasStrmv_v2 = GetProcAddress(handle, 'cublasStrmv_v2')

        global __cublasDtrmv_v2
        __cublasDtrmv_v2 = GetProcAddress(handle, 'cublasDtrmv_v2')

        global __cublasCtrmv_v2
        __cublasCtrmv_v2 = GetProcAddress(handle, 'cublasCtrmv_v2')

        global __cublasZtrmv_v2
        __cublasZtrmv_v2 = GetProcAddress(handle, 'cublasZtrmv_v2')

        global __cublasStbmv_v2
        __cublasStbmv_v2 = GetProcAddress(handle, 'cublasStbmv_v2')

        global __cublasDtbmv_v2
        __cublasDtbmv_v2 = GetProcAddress(handle, 'cublasDtbmv_v2')

        global __cublasCtbmv_v2
        __cublasCtbmv_v2 = GetProcAddress(handle, 'cublasCtbmv_v2')

        global __cublasZtbmv_v2
        __cublasZtbmv_v2 = GetProcAddress(handle, 'cublasZtbmv_v2')

        global __cublasStpmv_v2
        __cublasStpmv_v2 = GetProcAddress(handle, 'cublasStpmv_v2')

        global __cublasDtpmv_v2
        __cublasDtpmv_v2 = GetProcAddress(handle, 'cublasDtpmv_v2')

        global __cublasCtpmv_v2
        __cublasCtpmv_v2 = GetProcAddress(handle, 'cublasCtpmv_v2')

        global __cublasZtpmv_v2
        __cublasZtpmv_v2 = GetProcAddress(handle, 'cublasZtpmv_v2')

        global __cublasStrsv_v2
        __cublasStrsv_v2 = GetProcAddress(handle, 'cublasStrsv_v2')

        global __cublasDtrsv_v2
        __cublasDtrsv_v2 = GetProcAddress(handle, 'cublasDtrsv_v2')

        global __cublasCtrsv_v2
        __cublasCtrsv_v2 = GetProcAddress(handle, 'cublasCtrsv_v2')

        global __cublasZtrsv_v2
        __cublasZtrsv_v2 = GetProcAddress(handle, 'cublasZtrsv_v2')

        global __cublasStpsv_v2
        __cublasStpsv_v2 = GetProcAddress(handle, 'cublasStpsv_v2')

        global __cublasDtpsv_v2
        __cublasDtpsv_v2 = GetProcAddress(handle, 'cublasDtpsv_v2')

        global __cublasCtpsv_v2
        __cublasCtpsv_v2 = GetProcAddress(handle, 'cublasCtpsv_v2')

        global __cublasZtpsv_v2
        __cublasZtpsv_v2 = GetProcAddress(handle, 'cublasZtpsv_v2')

        global __cublasStbsv_v2
        __cublasStbsv_v2 = GetProcAddress(handle, 'cublasStbsv_v2')

        global __cublasDtbsv_v2
        __cublasDtbsv_v2 = GetProcAddress(handle, 'cublasDtbsv_v2')

        global __cublasCtbsv_v2
        __cublasCtbsv_v2 = GetProcAddress(handle, 'cublasCtbsv_v2')

        global __cublasZtbsv_v2
        __cublasZtbsv_v2 = GetProcAddress(handle, 'cublasZtbsv_v2')

        global __cublasSsymv_v2
        __cublasSsymv_v2 = GetProcAddress(handle, 'cublasSsymv_v2')

        global __cublasDsymv_v2
        __cublasDsymv_v2 = GetProcAddress(handle, 'cublasDsymv_v2')

        global __cublasCsymv_v2
        __cublasCsymv_v2 = GetProcAddress(handle, 'cublasCsymv_v2')

        global __cublasZsymv_v2
        __cublasZsymv_v2 = GetProcAddress(handle, 'cublasZsymv_v2')

        global __cublasChemv_v2
        __cublasChemv_v2 = GetProcAddress(handle, 'cublasChemv_v2')

        global __cublasZhemv_v2
        __cublasZhemv_v2 = GetProcAddress(handle, 'cublasZhemv_v2')

        global __cublasSsbmv_v2
        __cublasSsbmv_v2 = GetProcAddress(handle, 'cublasSsbmv_v2')

        global __cublasDsbmv_v2
        __cublasDsbmv_v2 = GetProcAddress(handle, 'cublasDsbmv_v2')

        global __cublasChbmv_v2
        __cublasChbmv_v2 = GetProcAddress(handle, 'cublasChbmv_v2')

        global __cublasZhbmv_v2
        __cublasZhbmv_v2 = GetProcAddress(handle, 'cublasZhbmv_v2')

        global __cublasSspmv_v2
        __cublasSspmv_v2 = GetProcAddress(handle, 'cublasSspmv_v2')

        global __cublasDspmv_v2
        __cublasDspmv_v2 = GetProcAddress(handle, 'cublasDspmv_v2')

        global __cublasChpmv_v2
        __cublasChpmv_v2 = GetProcAddress(handle, 'cublasChpmv_v2')

        global __cublasZhpmv_v2
        __cublasZhpmv_v2 = GetProcAddress(handle, 'cublasZhpmv_v2')

        global __cublasSger_v2
        __cublasSger_v2 = GetProcAddress(handle, 'cublasSger_v2')

        global __cublasDger_v2
        __cublasDger_v2 = GetProcAddress(handle, 'cublasDger_v2')

        global __cublasCgeru_v2
        __cublasCgeru_v2 = GetProcAddress(handle, 'cublasCgeru_v2')

        global __cublasCgerc_v2
        __cublasCgerc_v2 = GetProcAddress(handle, 'cublasCgerc_v2')

        global __cublasZgeru_v2
        __cublasZgeru_v2 = GetProcAddress(handle, 'cublasZgeru_v2')

        global __cublasZgerc_v2
        __cublasZgerc_v2 = GetProcAddress(handle, 'cublasZgerc_v2')

        global __cublasSsyr_v2
        __cublasSsyr_v2 = GetProcAddress(handle, 'cublasSsyr_v2')

        global __cublasDsyr_v2
        __cublasDsyr_v2 = GetProcAddress(handle, 'cublasDsyr_v2')

        global __cublasCsyr_v2
        __cublasCsyr_v2 = GetProcAddress(handle, 'cublasCsyr_v2')

        global __cublasZsyr_v2
        __cublasZsyr_v2 = GetProcAddress(handle, 'cublasZsyr_v2')

        global __cublasCher_v2
        __cublasCher_v2 = GetProcAddress(handle, 'cublasCher_v2')

        global __cublasZher_v2
        __cublasZher_v2 = GetProcAddress(handle, 'cublasZher_v2')

        global __cublasSspr_v2
        __cublasSspr_v2 = GetProcAddress(handle, 'cublasSspr_v2')

        global __cublasDspr_v2
        __cublasDspr_v2 = GetProcAddress(handle, 'cublasDspr_v2')

        global __cublasChpr_v2
        __cublasChpr_v2 = GetProcAddress(handle, 'cublasChpr_v2')

        global __cublasZhpr_v2
        __cublasZhpr_v2 = GetProcAddress(handle, 'cublasZhpr_v2')

        global __cublasSsyr2_v2
        __cublasSsyr2_v2 = GetProcAddress(handle, 'cublasSsyr2_v2')

        global __cublasDsyr2_v2
        __cublasDsyr2_v2 = GetProcAddress(handle, 'cublasDsyr2_v2')

        global __cublasCsyr2_v2
        __cublasCsyr2_v2 = GetProcAddress(handle, 'cublasCsyr2_v2')

        global __cublasZsyr2_v2
        __cublasZsyr2_v2 = GetProcAddress(handle, 'cublasZsyr2_v2')

        global __cublasCher2_v2
        __cublasCher2_v2 = GetProcAddress(handle, 'cublasCher2_v2')

        global __cublasZher2_v2
        __cublasZher2_v2 = GetProcAddress(handle, 'cublasZher2_v2')

        global __cublasSspr2_v2
        __cublasSspr2_v2 = GetProcAddress(handle, 'cublasSspr2_v2')

        global __cublasDspr2_v2
        __cublasDspr2_v2 = GetProcAddress(handle, 'cublasDspr2_v2')

        global __cublasChpr2_v2
        __cublasChpr2_v2 = GetProcAddress(handle, 'cublasChpr2_v2')

        global __cublasZhpr2_v2
        __cublasZhpr2_v2 = GetProcAddress(handle, 'cublasZhpr2_v2')

        global __cublasSgemm_v2
        __cublasSgemm_v2 = GetProcAddress(handle, 'cublasSgemm_v2')

        global __cublasDgemm_v2
        __cublasDgemm_v2 = GetProcAddress(handle, 'cublasDgemm_v2')

        global __cublasCgemm_v2
        __cublasCgemm_v2 = GetProcAddress(handle, 'cublasCgemm_v2')

        global __cublasCgemm3m
        __cublasCgemm3m = GetProcAddress(handle, 'cublasCgemm3m')

        global __cublasCgemm3mEx
        __cublasCgemm3mEx = GetProcAddress(handle, 'cublasCgemm3mEx')

        global __cublasZgemm_v2
        __cublasZgemm_v2 = GetProcAddress(handle, 'cublasZgemm_v2')

        global __cublasZgemm3m
        __cublasZgemm3m = GetProcAddress(handle, 'cublasZgemm3m')

        global __cublasSgemmEx
        __cublasSgemmEx = GetProcAddress(handle, 'cublasSgemmEx')

        global __cublasGemmEx
        __cublasGemmEx = GetProcAddress(handle, 'cublasGemmEx')

        global __cublasCgemmEx
        __cublasCgemmEx = GetProcAddress(handle, 'cublasCgemmEx')

        global __cublasUint8gemmBias
        __cublasUint8gemmBias = GetProcAddress(handle, 'cublasUint8gemmBias')

        global __cublasSsyrk_v2
        __cublasSsyrk_v2 = GetProcAddress(handle, 'cublasSsyrk_v2')

        global __cublasDsyrk_v2
        __cublasDsyrk_v2 = GetProcAddress(handle, 'cublasDsyrk_v2')

        global __cublasCsyrk_v2
        __cublasCsyrk_v2 = GetProcAddress(handle, 'cublasCsyrk_v2')

        global __cublasZsyrk_v2
        __cublasZsyrk_v2 = GetProcAddress(handle, 'cublasZsyrk_v2')

        global __cublasCsyrkEx
        __cublasCsyrkEx = GetProcAddress(handle, 'cublasCsyrkEx')

        global __cublasCsyrk3mEx
        __cublasCsyrk3mEx = GetProcAddress(handle, 'cublasCsyrk3mEx')

        global __cublasCherk_v2
        __cublasCherk_v2 = GetProcAddress(handle, 'cublasCherk_v2')

        global __cublasZherk_v2
        __cublasZherk_v2 = GetProcAddress(handle, 'cublasZherk_v2')

        global __cublasCherkEx
        __cublasCherkEx = GetProcAddress(handle, 'cublasCherkEx')

        global __cublasCherk3mEx
        __cublasCherk3mEx = GetProcAddress(handle, 'cublasCherk3mEx')

        global __cublasSsyr2k_v2
        __cublasSsyr2k_v2 = GetProcAddress(handle, 'cublasSsyr2k_v2')

        global __cublasDsyr2k_v2
        __cublasDsyr2k_v2 = GetProcAddress(handle, 'cublasDsyr2k_v2')

        global __cublasCsyr2k_v2
        __cublasCsyr2k_v2 = GetProcAddress(handle, 'cublasCsyr2k_v2')

        global __cublasZsyr2k_v2
        __cublasZsyr2k_v2 = GetProcAddress(handle, 'cublasZsyr2k_v2')

        global __cublasCher2k_v2
        __cublasCher2k_v2 = GetProcAddress(handle, 'cublasCher2k_v2')

        global __cublasZher2k_v2
        __cublasZher2k_v2 = GetProcAddress(handle, 'cublasZher2k_v2')

        global __cublasSsyrkx
        __cublasSsyrkx = GetProcAddress(handle, 'cublasSsyrkx')

        global __cublasDsyrkx
        __cublasDsyrkx = GetProcAddress(handle, 'cublasDsyrkx')

        global __cublasCsyrkx
        __cublasCsyrkx = GetProcAddress(handle, 'cublasCsyrkx')

        global __cublasZsyrkx
        __cublasZsyrkx = GetProcAddress(handle, 'cublasZsyrkx')

        global __cublasCherkx
        __cublasCherkx = GetProcAddress(handle, 'cublasCherkx')

        global __cublasZherkx
        __cublasZherkx = GetProcAddress(handle, 'cublasZherkx')

        global __cublasSsymm_v2
        __cublasSsymm_v2 = GetProcAddress(handle, 'cublasSsymm_v2')

        global __cublasDsymm_v2
        __cublasDsymm_v2 = GetProcAddress(handle, 'cublasDsymm_v2')

        global __cublasCsymm_v2
        __cublasCsymm_v2 = GetProcAddress(handle, 'cublasCsymm_v2')

        global __cublasZsymm_v2
        __cublasZsymm_v2 = GetProcAddress(handle, 'cublasZsymm_v2')

        global __cublasChemm_v2
        __cublasChemm_v2 = GetProcAddress(handle, 'cublasChemm_v2')

        global __cublasZhemm_v2
        __cublasZhemm_v2 = GetProcAddress(handle, 'cublasZhemm_v2')

        global __cublasStrsm_v2
        __cublasStrsm_v2 = GetProcAddress(handle, 'cublasStrsm_v2')

        global __cublasDtrsm_v2
        __cublasDtrsm_v2 = GetProcAddress(handle, 'cublasDtrsm_v2')

        global __cublasCtrsm_v2
        __cublasCtrsm_v2 = GetProcAddress(handle, 'cublasCtrsm_v2')

        global __cublasZtrsm_v2
        __cublasZtrsm_v2 = GetProcAddress(handle, 'cublasZtrsm_v2')

        global __cublasStrmm_v2
        __cublasStrmm_v2 = GetProcAddress(handle, 'cublasStrmm_v2')

        global __cublasDtrmm_v2
        __cublasDtrmm_v2 = GetProcAddress(handle, 'cublasDtrmm_v2')

        global __cublasCtrmm_v2
        __cublasCtrmm_v2 = GetProcAddress(handle, 'cublasCtrmm_v2')

        global __cublasZtrmm_v2
        __cublasZtrmm_v2 = GetProcAddress(handle, 'cublasZtrmm_v2')

        global __cublasSgemmBatched
        __cublasSgemmBatched = GetProcAddress(handle, 'cublasSgemmBatched')

        global __cublasDgemmBatched
        __cublasDgemmBatched = GetProcAddress(handle, 'cublasDgemmBatched')

        global __cublasCgemmBatched
        __cublasCgemmBatched = GetProcAddress(handle, 'cublasCgemmBatched')

        global __cublasCgemm3mBatched
        __cublasCgemm3mBatched = GetProcAddress(handle, 'cublasCgemm3mBatched')

        global __cublasZgemmBatched
        __cublasZgemmBatched = GetProcAddress(handle, 'cublasZgemmBatched')

        global __cublasGemmBatchedEx
        __cublasGemmBatchedEx = GetProcAddress(handle, 'cublasGemmBatchedEx')

        global __cublasGemmStridedBatchedEx
        __cublasGemmStridedBatchedEx = GetProcAddress(handle, 'cublasGemmStridedBatchedEx')

        global __cublasSgemmStridedBatched
        __cublasSgemmStridedBatched = GetProcAddress(handle, 'cublasSgemmStridedBatched')

        global __cublasDgemmStridedBatched
        __cublasDgemmStridedBatched = GetProcAddress(handle, 'cublasDgemmStridedBatched')

        global __cublasCgemmStridedBatched
        __cublasCgemmStridedBatched = GetProcAddress(handle, 'cublasCgemmStridedBatched')

        global __cublasCgemm3mStridedBatched
        __cublasCgemm3mStridedBatched = GetProcAddress(handle, 'cublasCgemm3mStridedBatched')

        global __cublasZgemmStridedBatched
        __cublasZgemmStridedBatched = GetProcAddress(handle, 'cublasZgemmStridedBatched')

        global __cublasSgeam
        __cublasSgeam = GetProcAddress(handle, 'cublasSgeam')

        global __cublasDgeam
        __cublasDgeam = GetProcAddress(handle, 'cublasDgeam')

        global __cublasCgeam
        __cublasCgeam = GetProcAddress(handle, 'cublasCgeam')

        global __cublasZgeam
        __cublasZgeam = GetProcAddress(handle, 'cublasZgeam')

        global __cublasSgetrfBatched
        __cublasSgetrfBatched = GetProcAddress(handle, 'cublasSgetrfBatched')

        global __cublasDgetrfBatched
        __cublasDgetrfBatched = GetProcAddress(handle, 'cublasDgetrfBatched')

        global __cublasCgetrfBatched
        __cublasCgetrfBatched = GetProcAddress(handle, 'cublasCgetrfBatched')

        global __cublasZgetrfBatched
        __cublasZgetrfBatched = GetProcAddress(handle, 'cublasZgetrfBatched')

        global __cublasSgetriBatched
        __cublasSgetriBatched = GetProcAddress(handle, 'cublasSgetriBatched')

        global __cublasDgetriBatched
        __cublasDgetriBatched = GetProcAddress(handle, 'cublasDgetriBatched')

        global __cublasCgetriBatched
        __cublasCgetriBatched = GetProcAddress(handle, 'cublasCgetriBatched')

        global __cublasZgetriBatched
        __cublasZgetriBatched = GetProcAddress(handle, 'cublasZgetriBatched')

        global __cublasSgetrsBatched
        __cublasSgetrsBatched = GetProcAddress(handle, 'cublasSgetrsBatched')

        global __cublasDgetrsBatched
        __cublasDgetrsBatched = GetProcAddress(handle, 'cublasDgetrsBatched')

        global __cublasCgetrsBatched
        __cublasCgetrsBatched = GetProcAddress(handle, 'cublasCgetrsBatched')

        global __cublasZgetrsBatched
        __cublasZgetrsBatched = GetProcAddress(handle, 'cublasZgetrsBatched')

        global __cublasStrsmBatched
        __cublasStrsmBatched = GetProcAddress(handle, 'cublasStrsmBatched')

        global __cublasDtrsmBatched
        __cublasDtrsmBatched = GetProcAddress(handle, 'cublasDtrsmBatched')

        global __cublasCtrsmBatched
        __cublasCtrsmBatched = GetProcAddress(handle, 'cublasCtrsmBatched')

        global __cublasZtrsmBatched
        __cublasZtrsmBatched = GetProcAddress(handle, 'cublasZtrsmBatched')

        global __cublasSmatinvBatched
        __cublasSmatinvBatched = GetProcAddress(handle, 'cublasSmatinvBatched')

        global __cublasDmatinvBatched
        __cublasDmatinvBatched = GetProcAddress(handle, 'cublasDmatinvBatched')

        global __cublasCmatinvBatched
        __cublasCmatinvBatched = GetProcAddress(handle, 'cublasCmatinvBatched')

        global __cublasZmatinvBatched
        __cublasZmatinvBatched = GetProcAddress(handle, 'cublasZmatinvBatched')

        global __cublasSgeqrfBatched
        __cublasSgeqrfBatched = GetProcAddress(handle, 'cublasSgeqrfBatched')

        global __cublasDgeqrfBatched
        __cublasDgeqrfBatched = GetProcAddress(handle, 'cublasDgeqrfBatched')

        global __cublasCgeqrfBatched
        __cublasCgeqrfBatched = GetProcAddress(handle, 'cublasCgeqrfBatched')

        global __cublasZgeqrfBatched
        __cublasZgeqrfBatched = GetProcAddress(handle, 'cublasZgeqrfBatched')

        global __cublasSgelsBatched
        __cublasSgelsBatched = GetProcAddress(handle, 'cublasSgelsBatched')

        global __cublasDgelsBatched
        __cublasDgelsBatched = GetProcAddress(handle, 'cublasDgelsBatched')

        global __cublasCgelsBatched
        __cublasCgelsBatched = GetProcAddress(handle, 'cublasCgelsBatched')

        global __cublasZgelsBatched
        __cublasZgelsBatched = GetProcAddress(handle, 'cublasZgelsBatched')

        global __cublasSdgmm
        __cublasSdgmm = GetProcAddress(handle, 'cublasSdgmm')

        global __cublasDdgmm
        __cublasDdgmm = GetProcAddress(handle, 'cublasDdgmm')

        global __cublasCdgmm
        __cublasCdgmm = GetProcAddress(handle, 'cublasCdgmm')

        global __cublasZdgmm
        __cublasZdgmm = GetProcAddress(handle, 'cublasZdgmm')

        global __cublasStpttr
        __cublasStpttr = GetProcAddress(handle, 'cublasStpttr')

        global __cublasDtpttr
        __cublasDtpttr = GetProcAddress(handle, 'cublasDtpttr')

        global __cublasCtpttr
        __cublasCtpttr = GetProcAddress(handle, 'cublasCtpttr')

        global __cublasZtpttr
        __cublasZtpttr = GetProcAddress(handle, 'cublasZtpttr')

        global __cublasStrttp
        __cublasStrttp = GetProcAddress(handle, 'cublasStrttp')

        global __cublasDtrttp
        __cublasDtrttp = GetProcAddress(handle, 'cublasDtrttp')

        global __cublasCtrttp
        __cublasCtrttp = GetProcAddress(handle, 'cublasCtrttp')

        global __cublasZtrttp
        __cublasZtrttp = GetProcAddress(handle, 'cublasZtrttp')

        global __cublasGetSmCountTarget
        __cublasGetSmCountTarget = GetProcAddress(handle, 'cublasGetSmCountTarget')

        global __cublasSetSmCountTarget
        __cublasSetSmCountTarget = GetProcAddress(handle, 'cublasSetSmCountTarget')

        global __cublasGetStatusName
        __cublasGetStatusName = GetProcAddress(handle, 'cublasGetStatusName')

        global __cublasGetStatusString
        __cublasGetStatusString = GetProcAddress(handle, 'cublasGetStatusString')

        global __cublasSgemvBatched
        __cublasSgemvBatched = GetProcAddress(handle, 'cublasSgemvBatched')

        global __cublasDgemvBatched
        __cublasDgemvBatched = GetProcAddress(handle, 'cublasDgemvBatched')

        global __cublasCgemvBatched
        __cublasCgemvBatched = GetProcAddress(handle, 'cublasCgemvBatched')

        global __cublasZgemvBatched
        __cublasZgemvBatched = GetProcAddress(handle, 'cublasZgemvBatched')

        global __cublasSgemvStridedBatched
        __cublasSgemvStridedBatched = GetProcAddress(handle, 'cublasSgemvStridedBatched')

        global __cublasDgemvStridedBatched
        __cublasDgemvStridedBatched = GetProcAddress(handle, 'cublasDgemvStridedBatched')

        global __cublasCgemvStridedBatched
        __cublasCgemvStridedBatched = GetProcAddress(handle, 'cublasCgemvStridedBatched')

        global __cublasZgemvStridedBatched
        __cublasZgemvStridedBatched = GetProcAddress(handle, 'cublasZgemvStridedBatched')

        global __cublasSetVector_64
        __cublasSetVector_64 = GetProcAddress(handle, 'cublasSetVector_64')

        global __cublasGetVector_64
        __cublasGetVector_64 = GetProcAddress(handle, 'cublasGetVector_64')

        global __cublasSetMatrix_64
        __cublasSetMatrix_64 = GetProcAddress(handle, 'cublasSetMatrix_64')

        global __cublasGetMatrix_64
        __cublasGetMatrix_64 = GetProcAddress(handle, 'cublasGetMatrix_64')

        global __cublasSetVectorAsync_64
        __cublasSetVectorAsync_64 = GetProcAddress(handle, 'cublasSetVectorAsync_64')

        global __cublasGetVectorAsync_64
        __cublasGetVectorAsync_64 = GetProcAddress(handle, 'cublasGetVectorAsync_64')

        global __cublasSetMatrixAsync_64
        __cublasSetMatrixAsync_64 = GetProcAddress(handle, 'cublasSetMatrixAsync_64')

        global __cublasGetMatrixAsync_64
        __cublasGetMatrixAsync_64 = GetProcAddress(handle, 'cublasGetMatrixAsync_64')

        global __cublasNrm2Ex_64
        __cublasNrm2Ex_64 = GetProcAddress(handle, 'cublasNrm2Ex_64')

        global __cublasSnrm2_v2_64
        __cublasSnrm2_v2_64 = GetProcAddress(handle, 'cublasSnrm2_v2_64')

        global __cublasDnrm2_v2_64
        __cublasDnrm2_v2_64 = GetProcAddress(handle, 'cublasDnrm2_v2_64')

        global __cublasScnrm2_v2_64
        __cublasScnrm2_v2_64 = GetProcAddress(handle, 'cublasScnrm2_v2_64')

        global __cublasDznrm2_v2_64
        __cublasDznrm2_v2_64 = GetProcAddress(handle, 'cublasDznrm2_v2_64')

        global __cublasDotEx_64
        __cublasDotEx_64 = GetProcAddress(handle, 'cublasDotEx_64')

        global __cublasDotcEx_64
        __cublasDotcEx_64 = GetProcAddress(handle, 'cublasDotcEx_64')

        global __cublasSdot_v2_64
        __cublasSdot_v2_64 = GetProcAddress(handle, 'cublasSdot_v2_64')

        global __cublasDdot_v2_64
        __cublasDdot_v2_64 = GetProcAddress(handle, 'cublasDdot_v2_64')

        global __cublasCdotu_v2_64
        __cublasCdotu_v2_64 = GetProcAddress(handle, 'cublasCdotu_v2_64')

        global __cublasCdotc_v2_64
        __cublasCdotc_v2_64 = GetProcAddress(handle, 'cublasCdotc_v2_64')

        global __cublasZdotu_v2_64
        __cublasZdotu_v2_64 = GetProcAddress(handle, 'cublasZdotu_v2_64')

        global __cublasZdotc_v2_64
        __cublasZdotc_v2_64 = GetProcAddress(handle, 'cublasZdotc_v2_64')

        global __cublasScalEx_64
        __cublasScalEx_64 = GetProcAddress(handle, 'cublasScalEx_64')

        global __cublasSscal_v2_64
        __cublasSscal_v2_64 = GetProcAddress(handle, 'cublasSscal_v2_64')

        global __cublasDscal_v2_64
        __cublasDscal_v2_64 = GetProcAddress(handle, 'cublasDscal_v2_64')

        global __cublasCscal_v2_64
        __cublasCscal_v2_64 = GetProcAddress(handle, 'cublasCscal_v2_64')

        global __cublasCsscal_v2_64
        __cublasCsscal_v2_64 = GetProcAddress(handle, 'cublasCsscal_v2_64')

        global __cublasZscal_v2_64
        __cublasZscal_v2_64 = GetProcAddress(handle, 'cublasZscal_v2_64')

        global __cublasZdscal_v2_64
        __cublasZdscal_v2_64 = GetProcAddress(handle, 'cublasZdscal_v2_64')

        global __cublasAxpyEx_64
        __cublasAxpyEx_64 = GetProcAddress(handle, 'cublasAxpyEx_64')

        global __cublasSaxpy_v2_64
        __cublasSaxpy_v2_64 = GetProcAddress(handle, 'cublasSaxpy_v2_64')

        global __cublasDaxpy_v2_64
        __cublasDaxpy_v2_64 = GetProcAddress(handle, 'cublasDaxpy_v2_64')

        global __cublasCaxpy_v2_64
        __cublasCaxpy_v2_64 = GetProcAddress(handle, 'cublasCaxpy_v2_64')

        global __cublasZaxpy_v2_64
        __cublasZaxpy_v2_64 = GetProcAddress(handle, 'cublasZaxpy_v2_64')

        global __cublasCopyEx_64
        __cublasCopyEx_64 = GetProcAddress(handle, 'cublasCopyEx_64')

        global __cublasScopy_v2_64
        __cublasScopy_v2_64 = GetProcAddress(handle, 'cublasScopy_v2_64')

        global __cublasDcopy_v2_64
        __cublasDcopy_v2_64 = GetProcAddress(handle, 'cublasDcopy_v2_64')

        global __cublasCcopy_v2_64
        __cublasCcopy_v2_64 = GetProcAddress(handle, 'cublasCcopy_v2_64')

        global __cublasZcopy_v2_64
        __cublasZcopy_v2_64 = GetProcAddress(handle, 'cublasZcopy_v2_64')

        global __cublasSswap_v2_64
        __cublasSswap_v2_64 = GetProcAddress(handle, 'cublasSswap_v2_64')

        global __cublasDswap_v2_64
        __cublasDswap_v2_64 = GetProcAddress(handle, 'cublasDswap_v2_64')

        global __cublasCswap_v2_64
        __cublasCswap_v2_64 = GetProcAddress(handle, 'cublasCswap_v2_64')

        global __cublasZswap_v2_64
        __cublasZswap_v2_64 = GetProcAddress(handle, 'cublasZswap_v2_64')

        global __cublasSwapEx_64
        __cublasSwapEx_64 = GetProcAddress(handle, 'cublasSwapEx_64')

        global __cublasIsamax_v2_64
        __cublasIsamax_v2_64 = GetProcAddress(handle, 'cublasIsamax_v2_64')

        global __cublasIdamax_v2_64
        __cublasIdamax_v2_64 = GetProcAddress(handle, 'cublasIdamax_v2_64')

        global __cublasIcamax_v2_64
        __cublasIcamax_v2_64 = GetProcAddress(handle, 'cublasIcamax_v2_64')

        global __cublasIzamax_v2_64
        __cublasIzamax_v2_64 = GetProcAddress(handle, 'cublasIzamax_v2_64')

        global __cublasIamaxEx_64
        __cublasIamaxEx_64 = GetProcAddress(handle, 'cublasIamaxEx_64')

        global __cublasIsamin_v2_64
        __cublasIsamin_v2_64 = GetProcAddress(handle, 'cublasIsamin_v2_64')

        global __cublasIdamin_v2_64
        __cublasIdamin_v2_64 = GetProcAddress(handle, 'cublasIdamin_v2_64')

        global __cublasIcamin_v2_64
        __cublasIcamin_v2_64 = GetProcAddress(handle, 'cublasIcamin_v2_64')

        global __cublasIzamin_v2_64
        __cublasIzamin_v2_64 = GetProcAddress(handle, 'cublasIzamin_v2_64')

        global __cublasIaminEx_64
        __cublasIaminEx_64 = GetProcAddress(handle, 'cublasIaminEx_64')

        global __cublasAsumEx_64
        __cublasAsumEx_64 = GetProcAddress(handle, 'cublasAsumEx_64')

        global __cublasSasum_v2_64
        __cublasSasum_v2_64 = GetProcAddress(handle, 'cublasSasum_v2_64')

        global __cublasDasum_v2_64
        __cublasDasum_v2_64 = GetProcAddress(handle, 'cublasDasum_v2_64')

        global __cublasScasum_v2_64
        __cublasScasum_v2_64 = GetProcAddress(handle, 'cublasScasum_v2_64')

        global __cublasDzasum_v2_64
        __cublasDzasum_v2_64 = GetProcAddress(handle, 'cublasDzasum_v2_64')

        global __cublasSrot_v2_64
        __cublasSrot_v2_64 = GetProcAddress(handle, 'cublasSrot_v2_64')

        global __cublasDrot_v2_64
        __cublasDrot_v2_64 = GetProcAddress(handle, 'cublasDrot_v2_64')

        global __cublasCrot_v2_64
        __cublasCrot_v2_64 = GetProcAddress(handle, 'cublasCrot_v2_64')

        global __cublasCsrot_v2_64
        __cublasCsrot_v2_64 = GetProcAddress(handle, 'cublasCsrot_v2_64')

        global __cublasZrot_v2_64
        __cublasZrot_v2_64 = GetProcAddress(handle, 'cublasZrot_v2_64')

        global __cublasZdrot_v2_64
        __cublasZdrot_v2_64 = GetProcAddress(handle, 'cublasZdrot_v2_64')

        global __cublasRotEx_64
        __cublasRotEx_64 = GetProcAddress(handle, 'cublasRotEx_64')

        global __cublasSrotm_v2_64
        __cublasSrotm_v2_64 = GetProcAddress(handle, 'cublasSrotm_v2_64')

        global __cublasDrotm_v2_64
        __cublasDrotm_v2_64 = GetProcAddress(handle, 'cublasDrotm_v2_64')

        global __cublasRotmEx_64
        __cublasRotmEx_64 = GetProcAddress(handle, 'cublasRotmEx_64')

        global __cublasSgemv_v2_64
        __cublasSgemv_v2_64 = GetProcAddress(handle, 'cublasSgemv_v2_64')

        global __cublasDgemv_v2_64
        __cublasDgemv_v2_64 = GetProcAddress(handle, 'cublasDgemv_v2_64')

        global __cublasCgemv_v2_64
        __cublasCgemv_v2_64 = GetProcAddress(handle, 'cublasCgemv_v2_64')

        global __cublasZgemv_v2_64
        __cublasZgemv_v2_64 = GetProcAddress(handle, 'cublasZgemv_v2_64')

        global __cublasSgbmv_v2_64
        __cublasSgbmv_v2_64 = GetProcAddress(handle, 'cublasSgbmv_v2_64')

        global __cublasDgbmv_v2_64
        __cublasDgbmv_v2_64 = GetProcAddress(handle, 'cublasDgbmv_v2_64')

        global __cublasCgbmv_v2_64
        __cublasCgbmv_v2_64 = GetProcAddress(handle, 'cublasCgbmv_v2_64')

        global __cublasZgbmv_v2_64
        __cublasZgbmv_v2_64 = GetProcAddress(handle, 'cublasZgbmv_v2_64')

        global __cublasStrmv_v2_64
        __cublasStrmv_v2_64 = GetProcAddress(handle, 'cublasStrmv_v2_64')

        global __cublasDtrmv_v2_64
        __cublasDtrmv_v2_64 = GetProcAddress(handle, 'cublasDtrmv_v2_64')

        global __cublasCtrmv_v2_64
        __cublasCtrmv_v2_64 = GetProcAddress(handle, 'cublasCtrmv_v2_64')

        global __cublasZtrmv_v2_64
        __cublasZtrmv_v2_64 = GetProcAddress(handle, 'cublasZtrmv_v2_64')

        global __cublasStbmv_v2_64
        __cublasStbmv_v2_64 = GetProcAddress(handle, 'cublasStbmv_v2_64')

        global __cublasDtbmv_v2_64
        __cublasDtbmv_v2_64 = GetProcAddress(handle, 'cublasDtbmv_v2_64')

        global __cublasCtbmv_v2_64
        __cublasCtbmv_v2_64 = GetProcAddress(handle, 'cublasCtbmv_v2_64')

        global __cublasZtbmv_v2_64
        __cublasZtbmv_v2_64 = GetProcAddress(handle, 'cublasZtbmv_v2_64')

        global __cublasStpmv_v2_64
        __cublasStpmv_v2_64 = GetProcAddress(handle, 'cublasStpmv_v2_64')

        global __cublasDtpmv_v2_64
        __cublasDtpmv_v2_64 = GetProcAddress(handle, 'cublasDtpmv_v2_64')

        global __cublasCtpmv_v2_64
        __cublasCtpmv_v2_64 = GetProcAddress(handle, 'cublasCtpmv_v2_64')

        global __cublasZtpmv_v2_64
        __cublasZtpmv_v2_64 = GetProcAddress(handle, 'cublasZtpmv_v2_64')

        global __cublasStrsv_v2_64
        __cublasStrsv_v2_64 = GetProcAddress(handle, 'cublasStrsv_v2_64')

        global __cublasDtrsv_v2_64
        __cublasDtrsv_v2_64 = GetProcAddress(handle, 'cublasDtrsv_v2_64')

        global __cublasCtrsv_v2_64
        __cublasCtrsv_v2_64 = GetProcAddress(handle, 'cublasCtrsv_v2_64')

        global __cublasZtrsv_v2_64
        __cublasZtrsv_v2_64 = GetProcAddress(handle, 'cublasZtrsv_v2_64')

        global __cublasStpsv_v2_64
        __cublasStpsv_v2_64 = GetProcAddress(handle, 'cublasStpsv_v2_64')

        global __cublasDtpsv_v2_64
        __cublasDtpsv_v2_64 = GetProcAddress(handle, 'cublasDtpsv_v2_64')

        global __cublasCtpsv_v2_64
        __cublasCtpsv_v2_64 = GetProcAddress(handle, 'cublasCtpsv_v2_64')

        global __cublasZtpsv_v2_64
        __cublasZtpsv_v2_64 = GetProcAddress(handle, 'cublasZtpsv_v2_64')

        global __cublasStbsv_v2_64
        __cublasStbsv_v2_64 = GetProcAddress(handle, 'cublasStbsv_v2_64')

        global __cublasDtbsv_v2_64
        __cublasDtbsv_v2_64 = GetProcAddress(handle, 'cublasDtbsv_v2_64')

        global __cublasCtbsv_v2_64
        __cublasCtbsv_v2_64 = GetProcAddress(handle, 'cublasCtbsv_v2_64')

        global __cublasZtbsv_v2_64
        __cublasZtbsv_v2_64 = GetProcAddress(handle, 'cublasZtbsv_v2_64')

        global __cublasSsymv_v2_64
        __cublasSsymv_v2_64 = GetProcAddress(handle, 'cublasSsymv_v2_64')

        global __cublasDsymv_v2_64
        __cublasDsymv_v2_64 = GetProcAddress(handle, 'cublasDsymv_v2_64')

        global __cublasCsymv_v2_64
        __cublasCsymv_v2_64 = GetProcAddress(handle, 'cublasCsymv_v2_64')

        global __cublasZsymv_v2_64
        __cublasZsymv_v2_64 = GetProcAddress(handle, 'cublasZsymv_v2_64')

        global __cublasChemv_v2_64
        __cublasChemv_v2_64 = GetProcAddress(handle, 'cublasChemv_v2_64')

        global __cublasZhemv_v2_64
        __cublasZhemv_v2_64 = GetProcAddress(handle, 'cublasZhemv_v2_64')

        global __cublasSsbmv_v2_64
        __cublasSsbmv_v2_64 = GetProcAddress(handle, 'cublasSsbmv_v2_64')

        global __cublasDsbmv_v2_64
        __cublasDsbmv_v2_64 = GetProcAddress(handle, 'cublasDsbmv_v2_64')

        global __cublasChbmv_v2_64
        __cublasChbmv_v2_64 = GetProcAddress(handle, 'cublasChbmv_v2_64')

        global __cublasZhbmv_v2_64
        __cublasZhbmv_v2_64 = GetProcAddress(handle, 'cublasZhbmv_v2_64')

        global __cublasSspmv_v2_64
        __cublasSspmv_v2_64 = GetProcAddress(handle, 'cublasSspmv_v2_64')

        global __cublasDspmv_v2_64
        __cublasDspmv_v2_64 = GetProcAddress(handle, 'cublasDspmv_v2_64')

        global __cublasChpmv_v2_64
        __cublasChpmv_v2_64 = GetProcAddress(handle, 'cublasChpmv_v2_64')

        global __cublasZhpmv_v2_64
        __cublasZhpmv_v2_64 = GetProcAddress(handle, 'cublasZhpmv_v2_64')

        global __cublasSger_v2_64
        __cublasSger_v2_64 = GetProcAddress(handle, 'cublasSger_v2_64')

        global __cublasDger_v2_64
        __cublasDger_v2_64 = GetProcAddress(handle, 'cublasDger_v2_64')

        global __cublasCgeru_v2_64
        __cublasCgeru_v2_64 = GetProcAddress(handle, 'cublasCgeru_v2_64')

        global __cublasCgerc_v2_64
        __cublasCgerc_v2_64 = GetProcAddress(handle, 'cublasCgerc_v2_64')

        global __cublasZgeru_v2_64
        __cublasZgeru_v2_64 = GetProcAddress(handle, 'cublasZgeru_v2_64')

        global __cublasZgerc_v2_64
        __cublasZgerc_v2_64 = GetProcAddress(handle, 'cublasZgerc_v2_64')

        global __cublasSsyr_v2_64
        __cublasSsyr_v2_64 = GetProcAddress(handle, 'cublasSsyr_v2_64')

        global __cublasDsyr_v2_64
        __cublasDsyr_v2_64 = GetProcAddress(handle, 'cublasDsyr_v2_64')

        global __cublasCsyr_v2_64
        __cublasCsyr_v2_64 = GetProcAddress(handle, 'cublasCsyr_v2_64')

        global __cublasZsyr_v2_64
        __cublasZsyr_v2_64 = GetProcAddress(handle, 'cublasZsyr_v2_64')

        global __cublasCher_v2_64
        __cublasCher_v2_64 = GetProcAddress(handle, 'cublasCher_v2_64')

        global __cublasZher_v2_64
        __cublasZher_v2_64 = GetProcAddress(handle, 'cublasZher_v2_64')

        global __cublasSspr_v2_64
        __cublasSspr_v2_64 = GetProcAddress(handle, 'cublasSspr_v2_64')

        global __cublasDspr_v2_64
        __cublasDspr_v2_64 = GetProcAddress(handle, 'cublasDspr_v2_64')

        global __cublasChpr_v2_64
        __cublasChpr_v2_64 = GetProcAddress(handle, 'cublasChpr_v2_64')

        global __cublasZhpr_v2_64
        __cublasZhpr_v2_64 = GetProcAddress(handle, 'cublasZhpr_v2_64')

        global __cublasSsyr2_v2_64
        __cublasSsyr2_v2_64 = GetProcAddress(handle, 'cublasSsyr2_v2_64')

        global __cublasDsyr2_v2_64
        __cublasDsyr2_v2_64 = GetProcAddress(handle, 'cublasDsyr2_v2_64')

        global __cublasCsyr2_v2_64
        __cublasCsyr2_v2_64 = GetProcAddress(handle, 'cublasCsyr2_v2_64')

        global __cublasZsyr2_v2_64
        __cublasZsyr2_v2_64 = GetProcAddress(handle, 'cublasZsyr2_v2_64')

        global __cublasCher2_v2_64
        __cublasCher2_v2_64 = GetProcAddress(handle, 'cublasCher2_v2_64')

        global __cublasZher2_v2_64
        __cublasZher2_v2_64 = GetProcAddress(handle, 'cublasZher2_v2_64')

        global __cublasSspr2_v2_64
        __cublasSspr2_v2_64 = GetProcAddress(handle, 'cublasSspr2_v2_64')

        global __cublasDspr2_v2_64
        __cublasDspr2_v2_64 = GetProcAddress(handle, 'cublasDspr2_v2_64')

        global __cublasChpr2_v2_64
        __cublasChpr2_v2_64 = GetProcAddress(handle, 'cublasChpr2_v2_64')

        global __cublasZhpr2_v2_64
        __cublasZhpr2_v2_64 = GetProcAddress(handle, 'cublasZhpr2_v2_64')

        global __cublasSgemvBatched_64
        __cublasSgemvBatched_64 = GetProcAddress(handle, 'cublasSgemvBatched_64')

        global __cublasDgemvBatched_64
        __cublasDgemvBatched_64 = GetProcAddress(handle, 'cublasDgemvBatched_64')

        global __cublasCgemvBatched_64
        __cublasCgemvBatched_64 = GetProcAddress(handle, 'cublasCgemvBatched_64')

        global __cublasZgemvBatched_64
        __cublasZgemvBatched_64 = GetProcAddress(handle, 'cublasZgemvBatched_64')

        global __cublasSgemvStridedBatched_64
        __cublasSgemvStridedBatched_64 = GetProcAddress(handle, 'cublasSgemvStridedBatched_64')

        global __cublasDgemvStridedBatched_64
        __cublasDgemvStridedBatched_64 = GetProcAddress(handle, 'cublasDgemvStridedBatched_64')

        global __cublasCgemvStridedBatched_64
        __cublasCgemvStridedBatched_64 = GetProcAddress(handle, 'cublasCgemvStridedBatched_64')

        global __cublasZgemvStridedBatched_64
        __cublasZgemvStridedBatched_64 = GetProcAddress(handle, 'cublasZgemvStridedBatched_64')

        global __cublasSgemm_v2_64
        __cublasSgemm_v2_64 = GetProcAddress(handle, 'cublasSgemm_v2_64')

        global __cublasDgemm_v2_64
        __cublasDgemm_v2_64 = GetProcAddress(handle, 'cublasDgemm_v2_64')

        global __cublasCgemm_v2_64
        __cublasCgemm_v2_64 = GetProcAddress(handle, 'cublasCgemm_v2_64')

        global __cublasCgemm3m_64
        __cublasCgemm3m_64 = GetProcAddress(handle, 'cublasCgemm3m_64')

        global __cublasCgemm3mEx_64
        __cublasCgemm3mEx_64 = GetProcAddress(handle, 'cublasCgemm3mEx_64')

        global __cublasZgemm_v2_64
        __cublasZgemm_v2_64 = GetProcAddress(handle, 'cublasZgemm_v2_64')

        global __cublasZgemm3m_64
        __cublasZgemm3m_64 = GetProcAddress(handle, 'cublasZgemm3m_64')

        global __cublasSgemmEx_64
        __cublasSgemmEx_64 = GetProcAddress(handle, 'cublasSgemmEx_64')

        global __cublasGemmEx_64
        __cublasGemmEx_64 = GetProcAddress(handle, 'cublasGemmEx_64')

        global __cublasCgemmEx_64
        __cublasCgemmEx_64 = GetProcAddress(handle, 'cublasCgemmEx_64')

        global __cublasSsyrk_v2_64
        __cublasSsyrk_v2_64 = GetProcAddress(handle, 'cublasSsyrk_v2_64')

        global __cublasDsyrk_v2_64
        __cublasDsyrk_v2_64 = GetProcAddress(handle, 'cublasDsyrk_v2_64')

        global __cublasCsyrk_v2_64
        __cublasCsyrk_v2_64 = GetProcAddress(handle, 'cublasCsyrk_v2_64')

        global __cublasZsyrk_v2_64
        __cublasZsyrk_v2_64 = GetProcAddress(handle, 'cublasZsyrk_v2_64')

        global __cublasCsyrkEx_64
        __cublasCsyrkEx_64 = GetProcAddress(handle, 'cublasCsyrkEx_64')

        global __cublasCsyrk3mEx_64
        __cublasCsyrk3mEx_64 = GetProcAddress(handle, 'cublasCsyrk3mEx_64')

        global __cublasCherk_v2_64
        __cublasCherk_v2_64 = GetProcAddress(handle, 'cublasCherk_v2_64')

        global __cublasZherk_v2_64
        __cublasZherk_v2_64 = GetProcAddress(handle, 'cublasZherk_v2_64')

        global __cublasCherkEx_64
        __cublasCherkEx_64 = GetProcAddress(handle, 'cublasCherkEx_64')

        global __cublasCherk3mEx_64
        __cublasCherk3mEx_64 = GetProcAddress(handle, 'cublasCherk3mEx_64')

        global __cublasSsyr2k_v2_64
        __cublasSsyr2k_v2_64 = GetProcAddress(handle, 'cublasSsyr2k_v2_64')

        global __cublasDsyr2k_v2_64
        __cublasDsyr2k_v2_64 = GetProcAddress(handle, 'cublasDsyr2k_v2_64')

        global __cublasCsyr2k_v2_64
        __cublasCsyr2k_v2_64 = GetProcAddress(handle, 'cublasCsyr2k_v2_64')

        global __cublasZsyr2k_v2_64
        __cublasZsyr2k_v2_64 = GetProcAddress(handle, 'cublasZsyr2k_v2_64')

        global __cublasCher2k_v2_64
        __cublasCher2k_v2_64 = GetProcAddress(handle, 'cublasCher2k_v2_64')

        global __cublasZher2k_v2_64
        __cublasZher2k_v2_64 = GetProcAddress(handle, 'cublasZher2k_v2_64')

        global __cublasSsyrkx_64
        __cublasSsyrkx_64 = GetProcAddress(handle, 'cublasSsyrkx_64')

        global __cublasDsyrkx_64
        __cublasDsyrkx_64 = GetProcAddress(handle, 'cublasDsyrkx_64')

        global __cublasCsyrkx_64
        __cublasCsyrkx_64 = GetProcAddress(handle, 'cublasCsyrkx_64')

        global __cublasZsyrkx_64
        __cublasZsyrkx_64 = GetProcAddress(handle, 'cublasZsyrkx_64')

        global __cublasCherkx_64
        __cublasCherkx_64 = GetProcAddress(handle, 'cublasCherkx_64')

        global __cublasZherkx_64
        __cublasZherkx_64 = GetProcAddress(handle, 'cublasZherkx_64')

        global __cublasSsymm_v2_64
        __cublasSsymm_v2_64 = GetProcAddress(handle, 'cublasSsymm_v2_64')

        global __cublasDsymm_v2_64
        __cublasDsymm_v2_64 = GetProcAddress(handle, 'cublasDsymm_v2_64')

        global __cublasCsymm_v2_64
        __cublasCsymm_v2_64 = GetProcAddress(handle, 'cublasCsymm_v2_64')

        global __cublasZsymm_v2_64
        __cublasZsymm_v2_64 = GetProcAddress(handle, 'cublasZsymm_v2_64')

        global __cublasChemm_v2_64
        __cublasChemm_v2_64 = GetProcAddress(handle, 'cublasChemm_v2_64')

        global __cublasZhemm_v2_64
        __cublasZhemm_v2_64 = GetProcAddress(handle, 'cublasZhemm_v2_64')

        global __cublasStrsm_v2_64
        __cublasStrsm_v2_64 = GetProcAddress(handle, 'cublasStrsm_v2_64')

        global __cublasDtrsm_v2_64
        __cublasDtrsm_v2_64 = GetProcAddress(handle, 'cublasDtrsm_v2_64')

        global __cublasCtrsm_v2_64
        __cublasCtrsm_v2_64 = GetProcAddress(handle, 'cublasCtrsm_v2_64')

        global __cublasZtrsm_v2_64
        __cublasZtrsm_v2_64 = GetProcAddress(handle, 'cublasZtrsm_v2_64')

        global __cublasStrmm_v2_64
        __cublasStrmm_v2_64 = GetProcAddress(handle, 'cublasStrmm_v2_64')

        global __cublasDtrmm_v2_64
        __cublasDtrmm_v2_64 = GetProcAddress(handle, 'cublasDtrmm_v2_64')

        global __cublasCtrmm_v2_64
        __cublasCtrmm_v2_64 = GetProcAddress(handle, 'cublasCtrmm_v2_64')

        global __cublasZtrmm_v2_64
        __cublasZtrmm_v2_64 = GetProcAddress(handle, 'cublasZtrmm_v2_64')

        global __cublasSgemmBatched_64
        __cublasSgemmBatched_64 = GetProcAddress(handle, 'cublasSgemmBatched_64')

        global __cublasDgemmBatched_64
        __cublasDgemmBatched_64 = GetProcAddress(handle, 'cublasDgemmBatched_64')

        global __cublasCgemmBatched_64
        __cublasCgemmBatched_64 = GetProcAddress(handle, 'cublasCgemmBatched_64')

        global __cublasCgemm3mBatched_64
        __cublasCgemm3mBatched_64 = GetProcAddress(handle, 'cublasCgemm3mBatched_64')

        global __cublasZgemmBatched_64
        __cublasZgemmBatched_64 = GetProcAddress(handle, 'cublasZgemmBatched_64')

        global __cublasSgemmStridedBatched_64
        __cublasSgemmStridedBatched_64 = GetProcAddress(handle, 'cublasSgemmStridedBatched_64')

        global __cublasDgemmStridedBatched_64
        __cublasDgemmStridedBatched_64 = GetProcAddress(handle, 'cublasDgemmStridedBatched_64')

        global __cublasCgemmStridedBatched_64
        __cublasCgemmStridedBatched_64 = GetProcAddress(handle, 'cublasCgemmStridedBatched_64')

        global __cublasCgemm3mStridedBatched_64
        __cublasCgemm3mStridedBatched_64 = GetProcAddress(handle, 'cublasCgemm3mStridedBatched_64')

        global __cublasZgemmStridedBatched_64
        __cublasZgemmStridedBatched_64 = GetProcAddress(handle, 'cublasZgemmStridedBatched_64')

        global __cublasGemmBatchedEx_64
        __cublasGemmBatchedEx_64 = GetProcAddress(handle, 'cublasGemmBatchedEx_64')

        global __cublasGemmStridedBatchedEx_64
        __cublasGemmStridedBatchedEx_64 = GetProcAddress(handle, 'cublasGemmStridedBatchedEx_64')

        global __cublasSgeam_64
        __cublasSgeam_64 = GetProcAddress(handle, 'cublasSgeam_64')

        global __cublasDgeam_64
        __cublasDgeam_64 = GetProcAddress(handle, 'cublasDgeam_64')

        global __cublasCgeam_64
        __cublasCgeam_64 = GetProcAddress(handle, 'cublasCgeam_64')

        global __cublasZgeam_64
        __cublasZgeam_64 = GetProcAddress(handle, 'cublasZgeam_64')

        global __cublasStrsmBatched_64
        __cublasStrsmBatched_64 = GetProcAddress(handle, 'cublasStrsmBatched_64')

        global __cublasDtrsmBatched_64
        __cublasDtrsmBatched_64 = GetProcAddress(handle, 'cublasDtrsmBatched_64')

        global __cublasCtrsmBatched_64
        __cublasCtrsmBatched_64 = GetProcAddress(handle, 'cublasCtrsmBatched_64')

        global __cublasZtrsmBatched_64
        __cublasZtrsmBatched_64 = GetProcAddress(handle, 'cublasZtrsmBatched_64')

        global __cublasSdgmm_64
        __cublasSdgmm_64 = GetProcAddress(handle, 'cublasSdgmm_64')

        global __cublasDdgmm_64
        __cublasDdgmm_64 = GetProcAddress(handle, 'cublasDdgmm_64')

        global __cublasCdgmm_64
        __cublasCdgmm_64 = GetProcAddress(handle, 'cublasCdgmm_64')

        global __cublasZdgmm_64
        __cublasZdgmm_64 = GetProcAddress(handle, 'cublasZdgmm_64')

        global __cublasSgemmGroupedBatched
        __cublasSgemmGroupedBatched = GetProcAddress(handle, 'cublasSgemmGroupedBatched')

        global __cublasSgemmGroupedBatched_64
        __cublasSgemmGroupedBatched_64 = GetProcAddress(handle, 'cublasSgemmGroupedBatched_64')

        global __cublasDgemmGroupedBatched
        __cublasDgemmGroupedBatched = GetProcAddress(handle, 'cublasDgemmGroupedBatched')

        global __cublasDgemmGroupedBatched_64
        __cublasDgemmGroupedBatched_64 = GetProcAddress(handle, 'cublasDgemmGroupedBatched_64')

        global __cublasGemmGroupedBatchedEx
        __cublasGemmGroupedBatchedEx = GetProcAddress(handle, 'cublasGemmGroupedBatchedEx')

        global __cublasGemmGroupedBatchedEx_64
        __cublasGemmGroupedBatchedEx_64 = GetProcAddress(handle, 'cublasGemmGroupedBatchedEx_64')

        global __cublasGetEmulationStrategy
        __cublasGetEmulationStrategy = GetProcAddress(handle, 'cublasGetEmulationStrategy')

        global __cublasSetEmulationStrategy
        __cublasSetEmulationStrategy = GetProcAddress(handle, 'cublasSetEmulationStrategy')

        global __cublasGetEmulationSpecialValuesSupport
        __cublasGetEmulationSpecialValuesSupport = GetProcAddress(handle, 'cublasGetEmulationSpecialValuesSupport')

        global __cublasSetEmulationSpecialValuesSupport
        __cublasSetEmulationSpecialValuesSupport = GetProcAddress(handle, 'cublasSetEmulationSpecialValuesSupport')

        global __cublasGetFixedPointEmulationMantissaControl
        __cublasGetFixedPointEmulationMantissaControl = GetProcAddress(handle, 'cublasGetFixedPointEmulationMantissaControl')

        global __cublasSetFixedPointEmulationMantissaControl
        __cublasSetFixedPointEmulationMantissaControl = GetProcAddress(handle, 'cublasSetFixedPointEmulationMantissaControl')

        global __cublasGetFixedPointEmulationMaxMantissaBitCount
        __cublasGetFixedPointEmulationMaxMantissaBitCount = GetProcAddress(handle, 'cublasGetFixedPointEmulationMaxMantissaBitCount')

        global __cublasSetFixedPointEmulationMaxMantissaBitCount
        __cublasSetFixedPointEmulationMaxMantissaBitCount = GetProcAddress(handle, 'cublasSetFixedPointEmulationMaxMantissaBitCount')

        global __cublasGetFixedPointEmulationMantissaBitOffset
        __cublasGetFixedPointEmulationMantissaBitOffset = GetProcAddress(handle, 'cublasGetFixedPointEmulationMantissaBitOffset')

        global __cublasSetFixedPointEmulationMantissaBitOffset
        __cublasSetFixedPointEmulationMantissaBitOffset = GetProcAddress(handle, 'cublasSetFixedPointEmulationMantissaBitOffset')

        global __cublasGetFixedPointEmulationMantissaBitCountPointer
        __cublasGetFixedPointEmulationMantissaBitCountPointer = GetProcAddress(handle, 'cublasGetFixedPointEmulationMantissaBitCountPointer')

        global __cublasSetFixedPointEmulationMantissaBitCountPointer
        __cublasSetFixedPointEmulationMantissaBitCountPointer = GetProcAddress(handle, 'cublasSetFixedPointEmulationMantissaBitCountPointer')

        __py_cublas_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cublas()
    cdef dict data = {}

    global __cublasCreate_v2
    data["__cublasCreate_v2"] = <intptr_t>__cublasCreate_v2

    global __cublasDestroy_v2
    data["__cublasDestroy_v2"] = <intptr_t>__cublasDestroy_v2

    global __cublasGetVersion_v2
    data["__cublasGetVersion_v2"] = <intptr_t>__cublasGetVersion_v2

    global __cublasGetProperty
    data["__cublasGetProperty"] = <intptr_t>__cublasGetProperty

    global __cublasGetCudartVersion
    data["__cublasGetCudartVersion"] = <intptr_t>__cublasGetCudartVersion

    global __cublasSetWorkspace_v2
    data["__cublasSetWorkspace_v2"] = <intptr_t>__cublasSetWorkspace_v2

    global __cublasSetStream_v2
    data["__cublasSetStream_v2"] = <intptr_t>__cublasSetStream_v2

    global __cublasGetStream_v2
    data["__cublasGetStream_v2"] = <intptr_t>__cublasGetStream_v2

    global __cublasGetPointerMode_v2
    data["__cublasGetPointerMode_v2"] = <intptr_t>__cublasGetPointerMode_v2

    global __cublasSetPointerMode_v2
    data["__cublasSetPointerMode_v2"] = <intptr_t>__cublasSetPointerMode_v2

    global __cublasGetAtomicsMode
    data["__cublasGetAtomicsMode"] = <intptr_t>__cublasGetAtomicsMode

    global __cublasSetAtomicsMode
    data["__cublasSetAtomicsMode"] = <intptr_t>__cublasSetAtomicsMode

    global __cublasGetMathMode
    data["__cublasGetMathMode"] = <intptr_t>__cublasGetMathMode

    global __cublasSetMathMode
    data["__cublasSetMathMode"] = <intptr_t>__cublasSetMathMode

    global __cublasLoggerConfigure
    data["__cublasLoggerConfigure"] = <intptr_t>__cublasLoggerConfigure

    global __cublasSetLoggerCallback
    data["__cublasSetLoggerCallback"] = <intptr_t>__cublasSetLoggerCallback

    global __cublasGetLoggerCallback
    data["__cublasGetLoggerCallback"] = <intptr_t>__cublasGetLoggerCallback

    global __cublasSetVector
    data["__cublasSetVector"] = <intptr_t>__cublasSetVector

    global __cublasGetVector
    data["__cublasGetVector"] = <intptr_t>__cublasGetVector

    global __cublasSetMatrix
    data["__cublasSetMatrix"] = <intptr_t>__cublasSetMatrix

    global __cublasGetMatrix
    data["__cublasGetMatrix"] = <intptr_t>__cublasGetMatrix

    global __cublasSetVectorAsync
    data["__cublasSetVectorAsync"] = <intptr_t>__cublasSetVectorAsync

    global __cublasGetVectorAsync
    data["__cublasGetVectorAsync"] = <intptr_t>__cublasGetVectorAsync

    global __cublasSetMatrixAsync
    data["__cublasSetMatrixAsync"] = <intptr_t>__cublasSetMatrixAsync

    global __cublasGetMatrixAsync
    data["__cublasGetMatrixAsync"] = <intptr_t>__cublasGetMatrixAsync

    global __cublasNrm2Ex
    data["__cublasNrm2Ex"] = <intptr_t>__cublasNrm2Ex

    global __cublasSnrm2_v2
    data["__cublasSnrm2_v2"] = <intptr_t>__cublasSnrm2_v2

    global __cublasDnrm2_v2
    data["__cublasDnrm2_v2"] = <intptr_t>__cublasDnrm2_v2

    global __cublasScnrm2_v2
    data["__cublasScnrm2_v2"] = <intptr_t>__cublasScnrm2_v2

    global __cublasDznrm2_v2
    data["__cublasDznrm2_v2"] = <intptr_t>__cublasDznrm2_v2

    global __cublasDotEx
    data["__cublasDotEx"] = <intptr_t>__cublasDotEx

    global __cublasDotcEx
    data["__cublasDotcEx"] = <intptr_t>__cublasDotcEx

    global __cublasSdot_v2
    data["__cublasSdot_v2"] = <intptr_t>__cublasSdot_v2

    global __cublasDdot_v2
    data["__cublasDdot_v2"] = <intptr_t>__cublasDdot_v2

    global __cublasCdotu_v2
    data["__cublasCdotu_v2"] = <intptr_t>__cublasCdotu_v2

    global __cublasCdotc_v2
    data["__cublasCdotc_v2"] = <intptr_t>__cublasCdotc_v2

    global __cublasZdotu_v2
    data["__cublasZdotu_v2"] = <intptr_t>__cublasZdotu_v2

    global __cublasZdotc_v2
    data["__cublasZdotc_v2"] = <intptr_t>__cublasZdotc_v2

    global __cublasScalEx
    data["__cublasScalEx"] = <intptr_t>__cublasScalEx

    global __cublasSscal_v2
    data["__cublasSscal_v2"] = <intptr_t>__cublasSscal_v2

    global __cublasDscal_v2
    data["__cublasDscal_v2"] = <intptr_t>__cublasDscal_v2

    global __cublasCscal_v2
    data["__cublasCscal_v2"] = <intptr_t>__cublasCscal_v2

    global __cublasCsscal_v2
    data["__cublasCsscal_v2"] = <intptr_t>__cublasCsscal_v2

    global __cublasZscal_v2
    data["__cublasZscal_v2"] = <intptr_t>__cublasZscal_v2

    global __cublasZdscal_v2
    data["__cublasZdscal_v2"] = <intptr_t>__cublasZdscal_v2

    global __cublasAxpyEx
    data["__cublasAxpyEx"] = <intptr_t>__cublasAxpyEx

    global __cublasSaxpy_v2
    data["__cublasSaxpy_v2"] = <intptr_t>__cublasSaxpy_v2

    global __cublasDaxpy_v2
    data["__cublasDaxpy_v2"] = <intptr_t>__cublasDaxpy_v2

    global __cublasCaxpy_v2
    data["__cublasCaxpy_v2"] = <intptr_t>__cublasCaxpy_v2

    global __cublasZaxpy_v2
    data["__cublasZaxpy_v2"] = <intptr_t>__cublasZaxpy_v2

    global __cublasCopyEx
    data["__cublasCopyEx"] = <intptr_t>__cublasCopyEx

    global __cublasScopy_v2
    data["__cublasScopy_v2"] = <intptr_t>__cublasScopy_v2

    global __cublasDcopy_v2
    data["__cublasDcopy_v2"] = <intptr_t>__cublasDcopy_v2

    global __cublasCcopy_v2
    data["__cublasCcopy_v2"] = <intptr_t>__cublasCcopy_v2

    global __cublasZcopy_v2
    data["__cublasZcopy_v2"] = <intptr_t>__cublasZcopy_v2

    global __cublasSswap_v2
    data["__cublasSswap_v2"] = <intptr_t>__cublasSswap_v2

    global __cublasDswap_v2
    data["__cublasDswap_v2"] = <intptr_t>__cublasDswap_v2

    global __cublasCswap_v2
    data["__cublasCswap_v2"] = <intptr_t>__cublasCswap_v2

    global __cublasZswap_v2
    data["__cublasZswap_v2"] = <intptr_t>__cublasZswap_v2

    global __cublasSwapEx
    data["__cublasSwapEx"] = <intptr_t>__cublasSwapEx

    global __cublasIsamax_v2
    data["__cublasIsamax_v2"] = <intptr_t>__cublasIsamax_v2

    global __cublasIdamax_v2
    data["__cublasIdamax_v2"] = <intptr_t>__cublasIdamax_v2

    global __cublasIcamax_v2
    data["__cublasIcamax_v2"] = <intptr_t>__cublasIcamax_v2

    global __cublasIzamax_v2
    data["__cublasIzamax_v2"] = <intptr_t>__cublasIzamax_v2

    global __cublasIamaxEx
    data["__cublasIamaxEx"] = <intptr_t>__cublasIamaxEx

    global __cublasIsamin_v2
    data["__cublasIsamin_v2"] = <intptr_t>__cublasIsamin_v2

    global __cublasIdamin_v2
    data["__cublasIdamin_v2"] = <intptr_t>__cublasIdamin_v2

    global __cublasIcamin_v2
    data["__cublasIcamin_v2"] = <intptr_t>__cublasIcamin_v2

    global __cublasIzamin_v2
    data["__cublasIzamin_v2"] = <intptr_t>__cublasIzamin_v2

    global __cublasIaminEx
    data["__cublasIaminEx"] = <intptr_t>__cublasIaminEx

    global __cublasAsumEx
    data["__cublasAsumEx"] = <intptr_t>__cublasAsumEx

    global __cublasSasum_v2
    data["__cublasSasum_v2"] = <intptr_t>__cublasSasum_v2

    global __cublasDasum_v2
    data["__cublasDasum_v2"] = <intptr_t>__cublasDasum_v2

    global __cublasScasum_v2
    data["__cublasScasum_v2"] = <intptr_t>__cublasScasum_v2

    global __cublasDzasum_v2
    data["__cublasDzasum_v2"] = <intptr_t>__cublasDzasum_v2

    global __cublasSrot_v2
    data["__cublasSrot_v2"] = <intptr_t>__cublasSrot_v2

    global __cublasDrot_v2
    data["__cublasDrot_v2"] = <intptr_t>__cublasDrot_v2

    global __cublasCrot_v2
    data["__cublasCrot_v2"] = <intptr_t>__cublasCrot_v2

    global __cublasCsrot_v2
    data["__cublasCsrot_v2"] = <intptr_t>__cublasCsrot_v2

    global __cublasZrot_v2
    data["__cublasZrot_v2"] = <intptr_t>__cublasZrot_v2

    global __cublasZdrot_v2
    data["__cublasZdrot_v2"] = <intptr_t>__cublasZdrot_v2

    global __cublasRotEx
    data["__cublasRotEx"] = <intptr_t>__cublasRotEx

    global __cublasSrotg_v2
    data["__cublasSrotg_v2"] = <intptr_t>__cublasSrotg_v2

    global __cublasDrotg_v2
    data["__cublasDrotg_v2"] = <intptr_t>__cublasDrotg_v2

    global __cublasCrotg_v2
    data["__cublasCrotg_v2"] = <intptr_t>__cublasCrotg_v2

    global __cublasZrotg_v2
    data["__cublasZrotg_v2"] = <intptr_t>__cublasZrotg_v2

    global __cublasRotgEx
    data["__cublasRotgEx"] = <intptr_t>__cublasRotgEx

    global __cublasSrotm_v2
    data["__cublasSrotm_v2"] = <intptr_t>__cublasSrotm_v2

    global __cublasDrotm_v2
    data["__cublasDrotm_v2"] = <intptr_t>__cublasDrotm_v2

    global __cublasRotmEx
    data["__cublasRotmEx"] = <intptr_t>__cublasRotmEx

    global __cublasSrotmg_v2
    data["__cublasSrotmg_v2"] = <intptr_t>__cublasSrotmg_v2

    global __cublasDrotmg_v2
    data["__cublasDrotmg_v2"] = <intptr_t>__cublasDrotmg_v2

    global __cublasRotmgEx
    data["__cublasRotmgEx"] = <intptr_t>__cublasRotmgEx

    global __cublasSgemv_v2
    data["__cublasSgemv_v2"] = <intptr_t>__cublasSgemv_v2

    global __cublasDgemv_v2
    data["__cublasDgemv_v2"] = <intptr_t>__cublasDgemv_v2

    global __cublasCgemv_v2
    data["__cublasCgemv_v2"] = <intptr_t>__cublasCgemv_v2

    global __cublasZgemv_v2
    data["__cublasZgemv_v2"] = <intptr_t>__cublasZgemv_v2

    global __cublasSgbmv_v2
    data["__cublasSgbmv_v2"] = <intptr_t>__cublasSgbmv_v2

    global __cublasDgbmv_v2
    data["__cublasDgbmv_v2"] = <intptr_t>__cublasDgbmv_v2

    global __cublasCgbmv_v2
    data["__cublasCgbmv_v2"] = <intptr_t>__cublasCgbmv_v2

    global __cublasZgbmv_v2
    data["__cublasZgbmv_v2"] = <intptr_t>__cublasZgbmv_v2

    global __cublasStrmv_v2
    data["__cublasStrmv_v2"] = <intptr_t>__cublasStrmv_v2

    global __cublasDtrmv_v2
    data["__cublasDtrmv_v2"] = <intptr_t>__cublasDtrmv_v2

    global __cublasCtrmv_v2
    data["__cublasCtrmv_v2"] = <intptr_t>__cublasCtrmv_v2

    global __cublasZtrmv_v2
    data["__cublasZtrmv_v2"] = <intptr_t>__cublasZtrmv_v2

    global __cublasStbmv_v2
    data["__cublasStbmv_v2"] = <intptr_t>__cublasStbmv_v2

    global __cublasDtbmv_v2
    data["__cublasDtbmv_v2"] = <intptr_t>__cublasDtbmv_v2

    global __cublasCtbmv_v2
    data["__cublasCtbmv_v2"] = <intptr_t>__cublasCtbmv_v2

    global __cublasZtbmv_v2
    data["__cublasZtbmv_v2"] = <intptr_t>__cublasZtbmv_v2

    global __cublasStpmv_v2
    data["__cublasStpmv_v2"] = <intptr_t>__cublasStpmv_v2

    global __cublasDtpmv_v2
    data["__cublasDtpmv_v2"] = <intptr_t>__cublasDtpmv_v2

    global __cublasCtpmv_v2
    data["__cublasCtpmv_v2"] = <intptr_t>__cublasCtpmv_v2

    global __cublasZtpmv_v2
    data["__cublasZtpmv_v2"] = <intptr_t>__cublasZtpmv_v2

    global __cublasStrsv_v2
    data["__cublasStrsv_v2"] = <intptr_t>__cublasStrsv_v2

    global __cublasDtrsv_v2
    data["__cublasDtrsv_v2"] = <intptr_t>__cublasDtrsv_v2

    global __cublasCtrsv_v2
    data["__cublasCtrsv_v2"] = <intptr_t>__cublasCtrsv_v2

    global __cublasZtrsv_v2
    data["__cublasZtrsv_v2"] = <intptr_t>__cublasZtrsv_v2

    global __cublasStpsv_v2
    data["__cublasStpsv_v2"] = <intptr_t>__cublasStpsv_v2

    global __cublasDtpsv_v2
    data["__cublasDtpsv_v2"] = <intptr_t>__cublasDtpsv_v2

    global __cublasCtpsv_v2
    data["__cublasCtpsv_v2"] = <intptr_t>__cublasCtpsv_v2

    global __cublasZtpsv_v2
    data["__cublasZtpsv_v2"] = <intptr_t>__cublasZtpsv_v2

    global __cublasStbsv_v2
    data["__cublasStbsv_v2"] = <intptr_t>__cublasStbsv_v2

    global __cublasDtbsv_v2
    data["__cublasDtbsv_v2"] = <intptr_t>__cublasDtbsv_v2

    global __cublasCtbsv_v2
    data["__cublasCtbsv_v2"] = <intptr_t>__cublasCtbsv_v2

    global __cublasZtbsv_v2
    data["__cublasZtbsv_v2"] = <intptr_t>__cublasZtbsv_v2

    global __cublasSsymv_v2
    data["__cublasSsymv_v2"] = <intptr_t>__cublasSsymv_v2

    global __cublasDsymv_v2
    data["__cublasDsymv_v2"] = <intptr_t>__cublasDsymv_v2

    global __cublasCsymv_v2
    data["__cublasCsymv_v2"] = <intptr_t>__cublasCsymv_v2

    global __cublasZsymv_v2
    data["__cublasZsymv_v2"] = <intptr_t>__cublasZsymv_v2

    global __cublasChemv_v2
    data["__cublasChemv_v2"] = <intptr_t>__cublasChemv_v2

    global __cublasZhemv_v2
    data["__cublasZhemv_v2"] = <intptr_t>__cublasZhemv_v2

    global __cublasSsbmv_v2
    data["__cublasSsbmv_v2"] = <intptr_t>__cublasSsbmv_v2

    global __cublasDsbmv_v2
    data["__cublasDsbmv_v2"] = <intptr_t>__cublasDsbmv_v2

    global __cublasChbmv_v2
    data["__cublasChbmv_v2"] = <intptr_t>__cublasChbmv_v2

    global __cublasZhbmv_v2
    data["__cublasZhbmv_v2"] = <intptr_t>__cublasZhbmv_v2

    global __cublasSspmv_v2
    data["__cublasSspmv_v2"] = <intptr_t>__cublasSspmv_v2

    global __cublasDspmv_v2
    data["__cublasDspmv_v2"] = <intptr_t>__cublasDspmv_v2

    global __cublasChpmv_v2
    data["__cublasChpmv_v2"] = <intptr_t>__cublasChpmv_v2

    global __cublasZhpmv_v2
    data["__cublasZhpmv_v2"] = <intptr_t>__cublasZhpmv_v2

    global __cublasSger_v2
    data["__cublasSger_v2"] = <intptr_t>__cublasSger_v2

    global __cublasDger_v2
    data["__cublasDger_v2"] = <intptr_t>__cublasDger_v2

    global __cublasCgeru_v2
    data["__cublasCgeru_v2"] = <intptr_t>__cublasCgeru_v2

    global __cublasCgerc_v2
    data["__cublasCgerc_v2"] = <intptr_t>__cublasCgerc_v2

    global __cublasZgeru_v2
    data["__cublasZgeru_v2"] = <intptr_t>__cublasZgeru_v2

    global __cublasZgerc_v2
    data["__cublasZgerc_v2"] = <intptr_t>__cublasZgerc_v2

    global __cublasSsyr_v2
    data["__cublasSsyr_v2"] = <intptr_t>__cublasSsyr_v2

    global __cublasDsyr_v2
    data["__cublasDsyr_v2"] = <intptr_t>__cublasDsyr_v2

    global __cublasCsyr_v2
    data["__cublasCsyr_v2"] = <intptr_t>__cublasCsyr_v2

    global __cublasZsyr_v2
    data["__cublasZsyr_v2"] = <intptr_t>__cublasZsyr_v2

    global __cublasCher_v2
    data["__cublasCher_v2"] = <intptr_t>__cublasCher_v2

    global __cublasZher_v2
    data["__cublasZher_v2"] = <intptr_t>__cublasZher_v2

    global __cublasSspr_v2
    data["__cublasSspr_v2"] = <intptr_t>__cublasSspr_v2

    global __cublasDspr_v2
    data["__cublasDspr_v2"] = <intptr_t>__cublasDspr_v2

    global __cublasChpr_v2
    data["__cublasChpr_v2"] = <intptr_t>__cublasChpr_v2

    global __cublasZhpr_v2
    data["__cublasZhpr_v2"] = <intptr_t>__cublasZhpr_v2

    global __cublasSsyr2_v2
    data["__cublasSsyr2_v2"] = <intptr_t>__cublasSsyr2_v2

    global __cublasDsyr2_v2
    data["__cublasDsyr2_v2"] = <intptr_t>__cublasDsyr2_v2

    global __cublasCsyr2_v2
    data["__cublasCsyr2_v2"] = <intptr_t>__cublasCsyr2_v2

    global __cublasZsyr2_v2
    data["__cublasZsyr2_v2"] = <intptr_t>__cublasZsyr2_v2

    global __cublasCher2_v2
    data["__cublasCher2_v2"] = <intptr_t>__cublasCher2_v2

    global __cublasZher2_v2
    data["__cublasZher2_v2"] = <intptr_t>__cublasZher2_v2

    global __cublasSspr2_v2
    data["__cublasSspr2_v2"] = <intptr_t>__cublasSspr2_v2

    global __cublasDspr2_v2
    data["__cublasDspr2_v2"] = <intptr_t>__cublasDspr2_v2

    global __cublasChpr2_v2
    data["__cublasChpr2_v2"] = <intptr_t>__cublasChpr2_v2

    global __cublasZhpr2_v2
    data["__cublasZhpr2_v2"] = <intptr_t>__cublasZhpr2_v2

    global __cublasSgemm_v2
    data["__cublasSgemm_v2"] = <intptr_t>__cublasSgemm_v2

    global __cublasDgemm_v2
    data["__cublasDgemm_v2"] = <intptr_t>__cublasDgemm_v2

    global __cublasCgemm_v2
    data["__cublasCgemm_v2"] = <intptr_t>__cublasCgemm_v2

    global __cublasCgemm3m
    data["__cublasCgemm3m"] = <intptr_t>__cublasCgemm3m

    global __cublasCgemm3mEx
    data["__cublasCgemm3mEx"] = <intptr_t>__cublasCgemm3mEx

    global __cublasZgemm_v2
    data["__cublasZgemm_v2"] = <intptr_t>__cublasZgemm_v2

    global __cublasZgemm3m
    data["__cublasZgemm3m"] = <intptr_t>__cublasZgemm3m

    global __cublasSgemmEx
    data["__cublasSgemmEx"] = <intptr_t>__cublasSgemmEx

    global __cublasGemmEx
    data["__cublasGemmEx"] = <intptr_t>__cublasGemmEx

    global __cublasCgemmEx
    data["__cublasCgemmEx"] = <intptr_t>__cublasCgemmEx

    global __cublasUint8gemmBias
    data["__cublasUint8gemmBias"] = <intptr_t>__cublasUint8gemmBias

    global __cublasSsyrk_v2
    data["__cublasSsyrk_v2"] = <intptr_t>__cublasSsyrk_v2

    global __cublasDsyrk_v2
    data["__cublasDsyrk_v2"] = <intptr_t>__cublasDsyrk_v2

    global __cublasCsyrk_v2
    data["__cublasCsyrk_v2"] = <intptr_t>__cublasCsyrk_v2

    global __cublasZsyrk_v2
    data["__cublasZsyrk_v2"] = <intptr_t>__cublasZsyrk_v2

    global __cublasCsyrkEx
    data["__cublasCsyrkEx"] = <intptr_t>__cublasCsyrkEx

    global __cublasCsyrk3mEx
    data["__cublasCsyrk3mEx"] = <intptr_t>__cublasCsyrk3mEx

    global __cublasCherk_v2
    data["__cublasCherk_v2"] = <intptr_t>__cublasCherk_v2

    global __cublasZherk_v2
    data["__cublasZherk_v2"] = <intptr_t>__cublasZherk_v2

    global __cublasCherkEx
    data["__cublasCherkEx"] = <intptr_t>__cublasCherkEx

    global __cublasCherk3mEx
    data["__cublasCherk3mEx"] = <intptr_t>__cublasCherk3mEx

    global __cublasSsyr2k_v2
    data["__cublasSsyr2k_v2"] = <intptr_t>__cublasSsyr2k_v2

    global __cublasDsyr2k_v2
    data["__cublasDsyr2k_v2"] = <intptr_t>__cublasDsyr2k_v2

    global __cublasCsyr2k_v2
    data["__cublasCsyr2k_v2"] = <intptr_t>__cublasCsyr2k_v2

    global __cublasZsyr2k_v2
    data["__cublasZsyr2k_v2"] = <intptr_t>__cublasZsyr2k_v2

    global __cublasCher2k_v2
    data["__cublasCher2k_v2"] = <intptr_t>__cublasCher2k_v2

    global __cublasZher2k_v2
    data["__cublasZher2k_v2"] = <intptr_t>__cublasZher2k_v2

    global __cublasSsyrkx
    data["__cublasSsyrkx"] = <intptr_t>__cublasSsyrkx

    global __cublasDsyrkx
    data["__cublasDsyrkx"] = <intptr_t>__cublasDsyrkx

    global __cublasCsyrkx
    data["__cublasCsyrkx"] = <intptr_t>__cublasCsyrkx

    global __cublasZsyrkx
    data["__cublasZsyrkx"] = <intptr_t>__cublasZsyrkx

    global __cublasCherkx
    data["__cublasCherkx"] = <intptr_t>__cublasCherkx

    global __cublasZherkx
    data["__cublasZherkx"] = <intptr_t>__cublasZherkx

    global __cublasSsymm_v2
    data["__cublasSsymm_v2"] = <intptr_t>__cublasSsymm_v2

    global __cublasDsymm_v2
    data["__cublasDsymm_v2"] = <intptr_t>__cublasDsymm_v2

    global __cublasCsymm_v2
    data["__cublasCsymm_v2"] = <intptr_t>__cublasCsymm_v2

    global __cublasZsymm_v2
    data["__cublasZsymm_v2"] = <intptr_t>__cublasZsymm_v2

    global __cublasChemm_v2
    data["__cublasChemm_v2"] = <intptr_t>__cublasChemm_v2

    global __cublasZhemm_v2
    data["__cublasZhemm_v2"] = <intptr_t>__cublasZhemm_v2

    global __cublasStrsm_v2
    data["__cublasStrsm_v2"] = <intptr_t>__cublasStrsm_v2

    global __cublasDtrsm_v2
    data["__cublasDtrsm_v2"] = <intptr_t>__cublasDtrsm_v2

    global __cublasCtrsm_v2
    data["__cublasCtrsm_v2"] = <intptr_t>__cublasCtrsm_v2

    global __cublasZtrsm_v2
    data["__cublasZtrsm_v2"] = <intptr_t>__cublasZtrsm_v2

    global __cublasStrmm_v2
    data["__cublasStrmm_v2"] = <intptr_t>__cublasStrmm_v2

    global __cublasDtrmm_v2
    data["__cublasDtrmm_v2"] = <intptr_t>__cublasDtrmm_v2

    global __cublasCtrmm_v2
    data["__cublasCtrmm_v2"] = <intptr_t>__cublasCtrmm_v2

    global __cublasZtrmm_v2
    data["__cublasZtrmm_v2"] = <intptr_t>__cublasZtrmm_v2

    global __cublasSgemmBatched
    data["__cublasSgemmBatched"] = <intptr_t>__cublasSgemmBatched

    global __cublasDgemmBatched
    data["__cublasDgemmBatched"] = <intptr_t>__cublasDgemmBatched

    global __cublasCgemmBatched
    data["__cublasCgemmBatched"] = <intptr_t>__cublasCgemmBatched

    global __cublasCgemm3mBatched
    data["__cublasCgemm3mBatched"] = <intptr_t>__cublasCgemm3mBatched

    global __cublasZgemmBatched
    data["__cublasZgemmBatched"] = <intptr_t>__cublasZgemmBatched

    global __cublasGemmBatchedEx
    data["__cublasGemmBatchedEx"] = <intptr_t>__cublasGemmBatchedEx

    global __cublasGemmStridedBatchedEx
    data["__cublasGemmStridedBatchedEx"] = <intptr_t>__cublasGemmStridedBatchedEx

    global __cublasSgemmStridedBatched
    data["__cublasSgemmStridedBatched"] = <intptr_t>__cublasSgemmStridedBatched

    global __cublasDgemmStridedBatched
    data["__cublasDgemmStridedBatched"] = <intptr_t>__cublasDgemmStridedBatched

    global __cublasCgemmStridedBatched
    data["__cublasCgemmStridedBatched"] = <intptr_t>__cublasCgemmStridedBatched

    global __cublasCgemm3mStridedBatched
    data["__cublasCgemm3mStridedBatched"] = <intptr_t>__cublasCgemm3mStridedBatched

    global __cublasZgemmStridedBatched
    data["__cublasZgemmStridedBatched"] = <intptr_t>__cublasZgemmStridedBatched

    global __cublasSgeam
    data["__cublasSgeam"] = <intptr_t>__cublasSgeam

    global __cublasDgeam
    data["__cublasDgeam"] = <intptr_t>__cublasDgeam

    global __cublasCgeam
    data["__cublasCgeam"] = <intptr_t>__cublasCgeam

    global __cublasZgeam
    data["__cublasZgeam"] = <intptr_t>__cublasZgeam

    global __cublasSgetrfBatched
    data["__cublasSgetrfBatched"] = <intptr_t>__cublasSgetrfBatched

    global __cublasDgetrfBatched
    data["__cublasDgetrfBatched"] = <intptr_t>__cublasDgetrfBatched

    global __cublasCgetrfBatched
    data["__cublasCgetrfBatched"] = <intptr_t>__cublasCgetrfBatched

    global __cublasZgetrfBatched
    data["__cublasZgetrfBatched"] = <intptr_t>__cublasZgetrfBatched

    global __cublasSgetriBatched
    data["__cublasSgetriBatched"] = <intptr_t>__cublasSgetriBatched

    global __cublasDgetriBatched
    data["__cublasDgetriBatched"] = <intptr_t>__cublasDgetriBatched

    global __cublasCgetriBatched
    data["__cublasCgetriBatched"] = <intptr_t>__cublasCgetriBatched

    global __cublasZgetriBatched
    data["__cublasZgetriBatched"] = <intptr_t>__cublasZgetriBatched

    global __cublasSgetrsBatched
    data["__cublasSgetrsBatched"] = <intptr_t>__cublasSgetrsBatched

    global __cublasDgetrsBatched
    data["__cublasDgetrsBatched"] = <intptr_t>__cublasDgetrsBatched

    global __cublasCgetrsBatched
    data["__cublasCgetrsBatched"] = <intptr_t>__cublasCgetrsBatched

    global __cublasZgetrsBatched
    data["__cublasZgetrsBatched"] = <intptr_t>__cublasZgetrsBatched

    global __cublasStrsmBatched
    data["__cublasStrsmBatched"] = <intptr_t>__cublasStrsmBatched

    global __cublasDtrsmBatched
    data["__cublasDtrsmBatched"] = <intptr_t>__cublasDtrsmBatched

    global __cublasCtrsmBatched
    data["__cublasCtrsmBatched"] = <intptr_t>__cublasCtrsmBatched

    global __cublasZtrsmBatched
    data["__cublasZtrsmBatched"] = <intptr_t>__cublasZtrsmBatched

    global __cublasSmatinvBatched
    data["__cublasSmatinvBatched"] = <intptr_t>__cublasSmatinvBatched

    global __cublasDmatinvBatched
    data["__cublasDmatinvBatched"] = <intptr_t>__cublasDmatinvBatched

    global __cublasCmatinvBatched
    data["__cublasCmatinvBatched"] = <intptr_t>__cublasCmatinvBatched

    global __cublasZmatinvBatched
    data["__cublasZmatinvBatched"] = <intptr_t>__cublasZmatinvBatched

    global __cublasSgeqrfBatched
    data["__cublasSgeqrfBatched"] = <intptr_t>__cublasSgeqrfBatched

    global __cublasDgeqrfBatched
    data["__cublasDgeqrfBatched"] = <intptr_t>__cublasDgeqrfBatched

    global __cublasCgeqrfBatched
    data["__cublasCgeqrfBatched"] = <intptr_t>__cublasCgeqrfBatched

    global __cublasZgeqrfBatched
    data["__cublasZgeqrfBatched"] = <intptr_t>__cublasZgeqrfBatched

    global __cublasSgelsBatched
    data["__cublasSgelsBatched"] = <intptr_t>__cublasSgelsBatched

    global __cublasDgelsBatched
    data["__cublasDgelsBatched"] = <intptr_t>__cublasDgelsBatched

    global __cublasCgelsBatched
    data["__cublasCgelsBatched"] = <intptr_t>__cublasCgelsBatched

    global __cublasZgelsBatched
    data["__cublasZgelsBatched"] = <intptr_t>__cublasZgelsBatched

    global __cublasSdgmm
    data["__cublasSdgmm"] = <intptr_t>__cublasSdgmm

    global __cublasDdgmm
    data["__cublasDdgmm"] = <intptr_t>__cublasDdgmm

    global __cublasCdgmm
    data["__cublasCdgmm"] = <intptr_t>__cublasCdgmm

    global __cublasZdgmm
    data["__cublasZdgmm"] = <intptr_t>__cublasZdgmm

    global __cublasStpttr
    data["__cublasStpttr"] = <intptr_t>__cublasStpttr

    global __cublasDtpttr
    data["__cublasDtpttr"] = <intptr_t>__cublasDtpttr

    global __cublasCtpttr
    data["__cublasCtpttr"] = <intptr_t>__cublasCtpttr

    global __cublasZtpttr
    data["__cublasZtpttr"] = <intptr_t>__cublasZtpttr

    global __cublasStrttp
    data["__cublasStrttp"] = <intptr_t>__cublasStrttp

    global __cublasDtrttp
    data["__cublasDtrttp"] = <intptr_t>__cublasDtrttp

    global __cublasCtrttp
    data["__cublasCtrttp"] = <intptr_t>__cublasCtrttp

    global __cublasZtrttp
    data["__cublasZtrttp"] = <intptr_t>__cublasZtrttp

    global __cublasGetSmCountTarget
    data["__cublasGetSmCountTarget"] = <intptr_t>__cublasGetSmCountTarget

    global __cublasSetSmCountTarget
    data["__cublasSetSmCountTarget"] = <intptr_t>__cublasSetSmCountTarget

    global __cublasGetStatusName
    data["__cublasGetStatusName"] = <intptr_t>__cublasGetStatusName

    global __cublasGetStatusString
    data["__cublasGetStatusString"] = <intptr_t>__cublasGetStatusString

    global __cublasSgemvBatched
    data["__cublasSgemvBatched"] = <intptr_t>__cublasSgemvBatched

    global __cublasDgemvBatched
    data["__cublasDgemvBatched"] = <intptr_t>__cublasDgemvBatched

    global __cublasCgemvBatched
    data["__cublasCgemvBatched"] = <intptr_t>__cublasCgemvBatched

    global __cublasZgemvBatched
    data["__cublasZgemvBatched"] = <intptr_t>__cublasZgemvBatched

    global __cublasSgemvStridedBatched
    data["__cublasSgemvStridedBatched"] = <intptr_t>__cublasSgemvStridedBatched

    global __cublasDgemvStridedBatched
    data["__cublasDgemvStridedBatched"] = <intptr_t>__cublasDgemvStridedBatched

    global __cublasCgemvStridedBatched
    data["__cublasCgemvStridedBatched"] = <intptr_t>__cublasCgemvStridedBatched

    global __cublasZgemvStridedBatched
    data["__cublasZgemvStridedBatched"] = <intptr_t>__cublasZgemvStridedBatched

    global __cublasSetVector_64
    data["__cublasSetVector_64"] = <intptr_t>__cublasSetVector_64

    global __cublasGetVector_64
    data["__cublasGetVector_64"] = <intptr_t>__cublasGetVector_64

    global __cublasSetMatrix_64
    data["__cublasSetMatrix_64"] = <intptr_t>__cublasSetMatrix_64

    global __cublasGetMatrix_64
    data["__cublasGetMatrix_64"] = <intptr_t>__cublasGetMatrix_64

    global __cublasSetVectorAsync_64
    data["__cublasSetVectorAsync_64"] = <intptr_t>__cublasSetVectorAsync_64

    global __cublasGetVectorAsync_64
    data["__cublasGetVectorAsync_64"] = <intptr_t>__cublasGetVectorAsync_64

    global __cublasSetMatrixAsync_64
    data["__cublasSetMatrixAsync_64"] = <intptr_t>__cublasSetMatrixAsync_64

    global __cublasGetMatrixAsync_64
    data["__cublasGetMatrixAsync_64"] = <intptr_t>__cublasGetMatrixAsync_64

    global __cublasNrm2Ex_64
    data["__cublasNrm2Ex_64"] = <intptr_t>__cublasNrm2Ex_64

    global __cublasSnrm2_v2_64
    data["__cublasSnrm2_v2_64"] = <intptr_t>__cublasSnrm2_v2_64

    global __cublasDnrm2_v2_64
    data["__cublasDnrm2_v2_64"] = <intptr_t>__cublasDnrm2_v2_64

    global __cublasScnrm2_v2_64
    data["__cublasScnrm2_v2_64"] = <intptr_t>__cublasScnrm2_v2_64

    global __cublasDznrm2_v2_64
    data["__cublasDznrm2_v2_64"] = <intptr_t>__cublasDznrm2_v2_64

    global __cublasDotEx_64
    data["__cublasDotEx_64"] = <intptr_t>__cublasDotEx_64

    global __cublasDotcEx_64
    data["__cublasDotcEx_64"] = <intptr_t>__cublasDotcEx_64

    global __cublasSdot_v2_64
    data["__cublasSdot_v2_64"] = <intptr_t>__cublasSdot_v2_64

    global __cublasDdot_v2_64
    data["__cublasDdot_v2_64"] = <intptr_t>__cublasDdot_v2_64

    global __cublasCdotu_v2_64
    data["__cublasCdotu_v2_64"] = <intptr_t>__cublasCdotu_v2_64

    global __cublasCdotc_v2_64
    data["__cublasCdotc_v2_64"] = <intptr_t>__cublasCdotc_v2_64

    global __cublasZdotu_v2_64
    data["__cublasZdotu_v2_64"] = <intptr_t>__cublasZdotu_v2_64

    global __cublasZdotc_v2_64
    data["__cublasZdotc_v2_64"] = <intptr_t>__cublasZdotc_v2_64

    global __cublasScalEx_64
    data["__cublasScalEx_64"] = <intptr_t>__cublasScalEx_64

    global __cublasSscal_v2_64
    data["__cublasSscal_v2_64"] = <intptr_t>__cublasSscal_v2_64

    global __cublasDscal_v2_64
    data["__cublasDscal_v2_64"] = <intptr_t>__cublasDscal_v2_64

    global __cublasCscal_v2_64
    data["__cublasCscal_v2_64"] = <intptr_t>__cublasCscal_v2_64

    global __cublasCsscal_v2_64
    data["__cublasCsscal_v2_64"] = <intptr_t>__cublasCsscal_v2_64

    global __cublasZscal_v2_64
    data["__cublasZscal_v2_64"] = <intptr_t>__cublasZscal_v2_64

    global __cublasZdscal_v2_64
    data["__cublasZdscal_v2_64"] = <intptr_t>__cublasZdscal_v2_64

    global __cublasAxpyEx_64
    data["__cublasAxpyEx_64"] = <intptr_t>__cublasAxpyEx_64

    global __cublasSaxpy_v2_64
    data["__cublasSaxpy_v2_64"] = <intptr_t>__cublasSaxpy_v2_64

    global __cublasDaxpy_v2_64
    data["__cublasDaxpy_v2_64"] = <intptr_t>__cublasDaxpy_v2_64

    global __cublasCaxpy_v2_64
    data["__cublasCaxpy_v2_64"] = <intptr_t>__cublasCaxpy_v2_64

    global __cublasZaxpy_v2_64
    data["__cublasZaxpy_v2_64"] = <intptr_t>__cublasZaxpy_v2_64

    global __cublasCopyEx_64
    data["__cublasCopyEx_64"] = <intptr_t>__cublasCopyEx_64

    global __cublasScopy_v2_64
    data["__cublasScopy_v2_64"] = <intptr_t>__cublasScopy_v2_64

    global __cublasDcopy_v2_64
    data["__cublasDcopy_v2_64"] = <intptr_t>__cublasDcopy_v2_64

    global __cublasCcopy_v2_64
    data["__cublasCcopy_v2_64"] = <intptr_t>__cublasCcopy_v2_64

    global __cublasZcopy_v2_64
    data["__cublasZcopy_v2_64"] = <intptr_t>__cublasZcopy_v2_64

    global __cublasSswap_v2_64
    data["__cublasSswap_v2_64"] = <intptr_t>__cublasSswap_v2_64

    global __cublasDswap_v2_64
    data["__cublasDswap_v2_64"] = <intptr_t>__cublasDswap_v2_64

    global __cublasCswap_v2_64
    data["__cublasCswap_v2_64"] = <intptr_t>__cublasCswap_v2_64

    global __cublasZswap_v2_64
    data["__cublasZswap_v2_64"] = <intptr_t>__cublasZswap_v2_64

    global __cublasSwapEx_64
    data["__cublasSwapEx_64"] = <intptr_t>__cublasSwapEx_64

    global __cublasIsamax_v2_64
    data["__cublasIsamax_v2_64"] = <intptr_t>__cublasIsamax_v2_64

    global __cublasIdamax_v2_64
    data["__cublasIdamax_v2_64"] = <intptr_t>__cublasIdamax_v2_64

    global __cublasIcamax_v2_64
    data["__cublasIcamax_v2_64"] = <intptr_t>__cublasIcamax_v2_64

    global __cublasIzamax_v2_64
    data["__cublasIzamax_v2_64"] = <intptr_t>__cublasIzamax_v2_64

    global __cublasIamaxEx_64
    data["__cublasIamaxEx_64"] = <intptr_t>__cublasIamaxEx_64

    global __cublasIsamin_v2_64
    data["__cublasIsamin_v2_64"] = <intptr_t>__cublasIsamin_v2_64

    global __cublasIdamin_v2_64
    data["__cublasIdamin_v2_64"] = <intptr_t>__cublasIdamin_v2_64

    global __cublasIcamin_v2_64
    data["__cublasIcamin_v2_64"] = <intptr_t>__cublasIcamin_v2_64

    global __cublasIzamin_v2_64
    data["__cublasIzamin_v2_64"] = <intptr_t>__cublasIzamin_v2_64

    global __cublasIaminEx_64
    data["__cublasIaminEx_64"] = <intptr_t>__cublasIaminEx_64

    global __cublasAsumEx_64
    data["__cublasAsumEx_64"] = <intptr_t>__cublasAsumEx_64

    global __cublasSasum_v2_64
    data["__cublasSasum_v2_64"] = <intptr_t>__cublasSasum_v2_64

    global __cublasDasum_v2_64
    data["__cublasDasum_v2_64"] = <intptr_t>__cublasDasum_v2_64

    global __cublasScasum_v2_64
    data["__cublasScasum_v2_64"] = <intptr_t>__cublasScasum_v2_64

    global __cublasDzasum_v2_64
    data["__cublasDzasum_v2_64"] = <intptr_t>__cublasDzasum_v2_64

    global __cublasSrot_v2_64
    data["__cublasSrot_v2_64"] = <intptr_t>__cublasSrot_v2_64

    global __cublasDrot_v2_64
    data["__cublasDrot_v2_64"] = <intptr_t>__cublasDrot_v2_64

    global __cublasCrot_v2_64
    data["__cublasCrot_v2_64"] = <intptr_t>__cublasCrot_v2_64

    global __cublasCsrot_v2_64
    data["__cublasCsrot_v2_64"] = <intptr_t>__cublasCsrot_v2_64

    global __cublasZrot_v2_64
    data["__cublasZrot_v2_64"] = <intptr_t>__cublasZrot_v2_64

    global __cublasZdrot_v2_64
    data["__cublasZdrot_v2_64"] = <intptr_t>__cublasZdrot_v2_64

    global __cublasRotEx_64
    data["__cublasRotEx_64"] = <intptr_t>__cublasRotEx_64

    global __cublasSrotm_v2_64
    data["__cublasSrotm_v2_64"] = <intptr_t>__cublasSrotm_v2_64

    global __cublasDrotm_v2_64
    data["__cublasDrotm_v2_64"] = <intptr_t>__cublasDrotm_v2_64

    global __cublasRotmEx_64
    data["__cublasRotmEx_64"] = <intptr_t>__cublasRotmEx_64

    global __cublasSgemv_v2_64
    data["__cublasSgemv_v2_64"] = <intptr_t>__cublasSgemv_v2_64

    global __cublasDgemv_v2_64
    data["__cublasDgemv_v2_64"] = <intptr_t>__cublasDgemv_v2_64

    global __cublasCgemv_v2_64
    data["__cublasCgemv_v2_64"] = <intptr_t>__cublasCgemv_v2_64

    global __cublasZgemv_v2_64
    data["__cublasZgemv_v2_64"] = <intptr_t>__cublasZgemv_v2_64

    global __cublasSgbmv_v2_64
    data["__cublasSgbmv_v2_64"] = <intptr_t>__cublasSgbmv_v2_64

    global __cublasDgbmv_v2_64
    data["__cublasDgbmv_v2_64"] = <intptr_t>__cublasDgbmv_v2_64

    global __cublasCgbmv_v2_64
    data["__cublasCgbmv_v2_64"] = <intptr_t>__cublasCgbmv_v2_64

    global __cublasZgbmv_v2_64
    data["__cublasZgbmv_v2_64"] = <intptr_t>__cublasZgbmv_v2_64

    global __cublasStrmv_v2_64
    data["__cublasStrmv_v2_64"] = <intptr_t>__cublasStrmv_v2_64

    global __cublasDtrmv_v2_64
    data["__cublasDtrmv_v2_64"] = <intptr_t>__cublasDtrmv_v2_64

    global __cublasCtrmv_v2_64
    data["__cublasCtrmv_v2_64"] = <intptr_t>__cublasCtrmv_v2_64

    global __cublasZtrmv_v2_64
    data["__cublasZtrmv_v2_64"] = <intptr_t>__cublasZtrmv_v2_64

    global __cublasStbmv_v2_64
    data["__cublasStbmv_v2_64"] = <intptr_t>__cublasStbmv_v2_64

    global __cublasDtbmv_v2_64
    data["__cublasDtbmv_v2_64"] = <intptr_t>__cublasDtbmv_v2_64

    global __cublasCtbmv_v2_64
    data["__cublasCtbmv_v2_64"] = <intptr_t>__cublasCtbmv_v2_64

    global __cublasZtbmv_v2_64
    data["__cublasZtbmv_v2_64"] = <intptr_t>__cublasZtbmv_v2_64

    global __cublasStpmv_v2_64
    data["__cublasStpmv_v2_64"] = <intptr_t>__cublasStpmv_v2_64

    global __cublasDtpmv_v2_64
    data["__cublasDtpmv_v2_64"] = <intptr_t>__cublasDtpmv_v2_64

    global __cublasCtpmv_v2_64
    data["__cublasCtpmv_v2_64"] = <intptr_t>__cublasCtpmv_v2_64

    global __cublasZtpmv_v2_64
    data["__cublasZtpmv_v2_64"] = <intptr_t>__cublasZtpmv_v2_64

    global __cublasStrsv_v2_64
    data["__cublasStrsv_v2_64"] = <intptr_t>__cublasStrsv_v2_64

    global __cublasDtrsv_v2_64
    data["__cublasDtrsv_v2_64"] = <intptr_t>__cublasDtrsv_v2_64

    global __cublasCtrsv_v2_64
    data["__cublasCtrsv_v2_64"] = <intptr_t>__cublasCtrsv_v2_64

    global __cublasZtrsv_v2_64
    data["__cublasZtrsv_v2_64"] = <intptr_t>__cublasZtrsv_v2_64

    global __cublasStpsv_v2_64
    data["__cublasStpsv_v2_64"] = <intptr_t>__cublasStpsv_v2_64

    global __cublasDtpsv_v2_64
    data["__cublasDtpsv_v2_64"] = <intptr_t>__cublasDtpsv_v2_64

    global __cublasCtpsv_v2_64
    data["__cublasCtpsv_v2_64"] = <intptr_t>__cublasCtpsv_v2_64

    global __cublasZtpsv_v2_64
    data["__cublasZtpsv_v2_64"] = <intptr_t>__cublasZtpsv_v2_64

    global __cublasStbsv_v2_64
    data["__cublasStbsv_v2_64"] = <intptr_t>__cublasStbsv_v2_64

    global __cublasDtbsv_v2_64
    data["__cublasDtbsv_v2_64"] = <intptr_t>__cublasDtbsv_v2_64

    global __cublasCtbsv_v2_64
    data["__cublasCtbsv_v2_64"] = <intptr_t>__cublasCtbsv_v2_64

    global __cublasZtbsv_v2_64
    data["__cublasZtbsv_v2_64"] = <intptr_t>__cublasZtbsv_v2_64

    global __cublasSsymv_v2_64
    data["__cublasSsymv_v2_64"] = <intptr_t>__cublasSsymv_v2_64

    global __cublasDsymv_v2_64
    data["__cublasDsymv_v2_64"] = <intptr_t>__cublasDsymv_v2_64

    global __cublasCsymv_v2_64
    data["__cublasCsymv_v2_64"] = <intptr_t>__cublasCsymv_v2_64

    global __cublasZsymv_v2_64
    data["__cublasZsymv_v2_64"] = <intptr_t>__cublasZsymv_v2_64

    global __cublasChemv_v2_64
    data["__cublasChemv_v2_64"] = <intptr_t>__cublasChemv_v2_64

    global __cublasZhemv_v2_64
    data["__cublasZhemv_v2_64"] = <intptr_t>__cublasZhemv_v2_64

    global __cublasSsbmv_v2_64
    data["__cublasSsbmv_v2_64"] = <intptr_t>__cublasSsbmv_v2_64

    global __cublasDsbmv_v2_64
    data["__cublasDsbmv_v2_64"] = <intptr_t>__cublasDsbmv_v2_64

    global __cublasChbmv_v2_64
    data["__cublasChbmv_v2_64"] = <intptr_t>__cublasChbmv_v2_64

    global __cublasZhbmv_v2_64
    data["__cublasZhbmv_v2_64"] = <intptr_t>__cublasZhbmv_v2_64

    global __cublasSspmv_v2_64
    data["__cublasSspmv_v2_64"] = <intptr_t>__cublasSspmv_v2_64

    global __cublasDspmv_v2_64
    data["__cublasDspmv_v2_64"] = <intptr_t>__cublasDspmv_v2_64

    global __cublasChpmv_v2_64
    data["__cublasChpmv_v2_64"] = <intptr_t>__cublasChpmv_v2_64

    global __cublasZhpmv_v2_64
    data["__cublasZhpmv_v2_64"] = <intptr_t>__cublasZhpmv_v2_64

    global __cublasSger_v2_64
    data["__cublasSger_v2_64"] = <intptr_t>__cublasSger_v2_64

    global __cublasDger_v2_64
    data["__cublasDger_v2_64"] = <intptr_t>__cublasDger_v2_64

    global __cublasCgeru_v2_64
    data["__cublasCgeru_v2_64"] = <intptr_t>__cublasCgeru_v2_64

    global __cublasCgerc_v2_64
    data["__cublasCgerc_v2_64"] = <intptr_t>__cublasCgerc_v2_64

    global __cublasZgeru_v2_64
    data["__cublasZgeru_v2_64"] = <intptr_t>__cublasZgeru_v2_64

    global __cublasZgerc_v2_64
    data["__cublasZgerc_v2_64"] = <intptr_t>__cublasZgerc_v2_64

    global __cublasSsyr_v2_64
    data["__cublasSsyr_v2_64"] = <intptr_t>__cublasSsyr_v2_64

    global __cublasDsyr_v2_64
    data["__cublasDsyr_v2_64"] = <intptr_t>__cublasDsyr_v2_64

    global __cublasCsyr_v2_64
    data["__cublasCsyr_v2_64"] = <intptr_t>__cublasCsyr_v2_64

    global __cublasZsyr_v2_64
    data["__cublasZsyr_v2_64"] = <intptr_t>__cublasZsyr_v2_64

    global __cublasCher_v2_64
    data["__cublasCher_v2_64"] = <intptr_t>__cublasCher_v2_64

    global __cublasZher_v2_64
    data["__cublasZher_v2_64"] = <intptr_t>__cublasZher_v2_64

    global __cublasSspr_v2_64
    data["__cublasSspr_v2_64"] = <intptr_t>__cublasSspr_v2_64

    global __cublasDspr_v2_64
    data["__cublasDspr_v2_64"] = <intptr_t>__cublasDspr_v2_64

    global __cublasChpr_v2_64
    data["__cublasChpr_v2_64"] = <intptr_t>__cublasChpr_v2_64

    global __cublasZhpr_v2_64
    data["__cublasZhpr_v2_64"] = <intptr_t>__cublasZhpr_v2_64

    global __cublasSsyr2_v2_64
    data["__cublasSsyr2_v2_64"] = <intptr_t>__cublasSsyr2_v2_64

    global __cublasDsyr2_v2_64
    data["__cublasDsyr2_v2_64"] = <intptr_t>__cublasDsyr2_v2_64

    global __cublasCsyr2_v2_64
    data["__cublasCsyr2_v2_64"] = <intptr_t>__cublasCsyr2_v2_64

    global __cublasZsyr2_v2_64
    data["__cublasZsyr2_v2_64"] = <intptr_t>__cublasZsyr2_v2_64

    global __cublasCher2_v2_64
    data["__cublasCher2_v2_64"] = <intptr_t>__cublasCher2_v2_64

    global __cublasZher2_v2_64
    data["__cublasZher2_v2_64"] = <intptr_t>__cublasZher2_v2_64

    global __cublasSspr2_v2_64
    data["__cublasSspr2_v2_64"] = <intptr_t>__cublasSspr2_v2_64

    global __cublasDspr2_v2_64
    data["__cublasDspr2_v2_64"] = <intptr_t>__cublasDspr2_v2_64

    global __cublasChpr2_v2_64
    data["__cublasChpr2_v2_64"] = <intptr_t>__cublasChpr2_v2_64

    global __cublasZhpr2_v2_64
    data["__cublasZhpr2_v2_64"] = <intptr_t>__cublasZhpr2_v2_64

    global __cublasSgemvBatched_64
    data["__cublasSgemvBatched_64"] = <intptr_t>__cublasSgemvBatched_64

    global __cublasDgemvBatched_64
    data["__cublasDgemvBatched_64"] = <intptr_t>__cublasDgemvBatched_64

    global __cublasCgemvBatched_64
    data["__cublasCgemvBatched_64"] = <intptr_t>__cublasCgemvBatched_64

    global __cublasZgemvBatched_64
    data["__cublasZgemvBatched_64"] = <intptr_t>__cublasZgemvBatched_64

    global __cublasSgemvStridedBatched_64
    data["__cublasSgemvStridedBatched_64"] = <intptr_t>__cublasSgemvStridedBatched_64

    global __cublasDgemvStridedBatched_64
    data["__cublasDgemvStridedBatched_64"] = <intptr_t>__cublasDgemvStridedBatched_64

    global __cublasCgemvStridedBatched_64
    data["__cublasCgemvStridedBatched_64"] = <intptr_t>__cublasCgemvStridedBatched_64

    global __cublasZgemvStridedBatched_64
    data["__cublasZgemvStridedBatched_64"] = <intptr_t>__cublasZgemvStridedBatched_64

    global __cublasSgemm_v2_64
    data["__cublasSgemm_v2_64"] = <intptr_t>__cublasSgemm_v2_64

    global __cublasDgemm_v2_64
    data["__cublasDgemm_v2_64"] = <intptr_t>__cublasDgemm_v2_64

    global __cublasCgemm_v2_64
    data["__cublasCgemm_v2_64"] = <intptr_t>__cublasCgemm_v2_64

    global __cublasCgemm3m_64
    data["__cublasCgemm3m_64"] = <intptr_t>__cublasCgemm3m_64

    global __cublasCgemm3mEx_64
    data["__cublasCgemm3mEx_64"] = <intptr_t>__cublasCgemm3mEx_64

    global __cublasZgemm_v2_64
    data["__cublasZgemm_v2_64"] = <intptr_t>__cublasZgemm_v2_64

    global __cublasZgemm3m_64
    data["__cublasZgemm3m_64"] = <intptr_t>__cublasZgemm3m_64

    global __cublasSgemmEx_64
    data["__cublasSgemmEx_64"] = <intptr_t>__cublasSgemmEx_64

    global __cublasGemmEx_64
    data["__cublasGemmEx_64"] = <intptr_t>__cublasGemmEx_64

    global __cublasCgemmEx_64
    data["__cublasCgemmEx_64"] = <intptr_t>__cublasCgemmEx_64

    global __cublasSsyrk_v2_64
    data["__cublasSsyrk_v2_64"] = <intptr_t>__cublasSsyrk_v2_64

    global __cublasDsyrk_v2_64
    data["__cublasDsyrk_v2_64"] = <intptr_t>__cublasDsyrk_v2_64

    global __cublasCsyrk_v2_64
    data["__cublasCsyrk_v2_64"] = <intptr_t>__cublasCsyrk_v2_64

    global __cublasZsyrk_v2_64
    data["__cublasZsyrk_v2_64"] = <intptr_t>__cublasZsyrk_v2_64

    global __cublasCsyrkEx_64
    data["__cublasCsyrkEx_64"] = <intptr_t>__cublasCsyrkEx_64

    global __cublasCsyrk3mEx_64
    data["__cublasCsyrk3mEx_64"] = <intptr_t>__cublasCsyrk3mEx_64

    global __cublasCherk_v2_64
    data["__cublasCherk_v2_64"] = <intptr_t>__cublasCherk_v2_64

    global __cublasZherk_v2_64
    data["__cublasZherk_v2_64"] = <intptr_t>__cublasZherk_v2_64

    global __cublasCherkEx_64
    data["__cublasCherkEx_64"] = <intptr_t>__cublasCherkEx_64

    global __cublasCherk3mEx_64
    data["__cublasCherk3mEx_64"] = <intptr_t>__cublasCherk3mEx_64

    global __cublasSsyr2k_v2_64
    data["__cublasSsyr2k_v2_64"] = <intptr_t>__cublasSsyr2k_v2_64

    global __cublasDsyr2k_v2_64
    data["__cublasDsyr2k_v2_64"] = <intptr_t>__cublasDsyr2k_v2_64

    global __cublasCsyr2k_v2_64
    data["__cublasCsyr2k_v2_64"] = <intptr_t>__cublasCsyr2k_v2_64

    global __cublasZsyr2k_v2_64
    data["__cublasZsyr2k_v2_64"] = <intptr_t>__cublasZsyr2k_v2_64

    global __cublasCher2k_v2_64
    data["__cublasCher2k_v2_64"] = <intptr_t>__cublasCher2k_v2_64

    global __cublasZher2k_v2_64
    data["__cublasZher2k_v2_64"] = <intptr_t>__cublasZher2k_v2_64

    global __cublasSsyrkx_64
    data["__cublasSsyrkx_64"] = <intptr_t>__cublasSsyrkx_64

    global __cublasDsyrkx_64
    data["__cublasDsyrkx_64"] = <intptr_t>__cublasDsyrkx_64

    global __cublasCsyrkx_64
    data["__cublasCsyrkx_64"] = <intptr_t>__cublasCsyrkx_64

    global __cublasZsyrkx_64
    data["__cublasZsyrkx_64"] = <intptr_t>__cublasZsyrkx_64

    global __cublasCherkx_64
    data["__cublasCherkx_64"] = <intptr_t>__cublasCherkx_64

    global __cublasZherkx_64
    data["__cublasZherkx_64"] = <intptr_t>__cublasZherkx_64

    global __cublasSsymm_v2_64
    data["__cublasSsymm_v2_64"] = <intptr_t>__cublasSsymm_v2_64

    global __cublasDsymm_v2_64
    data["__cublasDsymm_v2_64"] = <intptr_t>__cublasDsymm_v2_64

    global __cublasCsymm_v2_64
    data["__cublasCsymm_v2_64"] = <intptr_t>__cublasCsymm_v2_64

    global __cublasZsymm_v2_64
    data["__cublasZsymm_v2_64"] = <intptr_t>__cublasZsymm_v2_64

    global __cublasChemm_v2_64
    data["__cublasChemm_v2_64"] = <intptr_t>__cublasChemm_v2_64

    global __cublasZhemm_v2_64
    data["__cublasZhemm_v2_64"] = <intptr_t>__cublasZhemm_v2_64

    global __cublasStrsm_v2_64
    data["__cublasStrsm_v2_64"] = <intptr_t>__cublasStrsm_v2_64

    global __cublasDtrsm_v2_64
    data["__cublasDtrsm_v2_64"] = <intptr_t>__cublasDtrsm_v2_64

    global __cublasCtrsm_v2_64
    data["__cublasCtrsm_v2_64"] = <intptr_t>__cublasCtrsm_v2_64

    global __cublasZtrsm_v2_64
    data["__cublasZtrsm_v2_64"] = <intptr_t>__cublasZtrsm_v2_64

    global __cublasStrmm_v2_64
    data["__cublasStrmm_v2_64"] = <intptr_t>__cublasStrmm_v2_64

    global __cublasDtrmm_v2_64
    data["__cublasDtrmm_v2_64"] = <intptr_t>__cublasDtrmm_v2_64

    global __cublasCtrmm_v2_64
    data["__cublasCtrmm_v2_64"] = <intptr_t>__cublasCtrmm_v2_64

    global __cublasZtrmm_v2_64
    data["__cublasZtrmm_v2_64"] = <intptr_t>__cublasZtrmm_v2_64

    global __cublasSgemmBatched_64
    data["__cublasSgemmBatched_64"] = <intptr_t>__cublasSgemmBatched_64

    global __cublasDgemmBatched_64
    data["__cublasDgemmBatched_64"] = <intptr_t>__cublasDgemmBatched_64

    global __cublasCgemmBatched_64
    data["__cublasCgemmBatched_64"] = <intptr_t>__cublasCgemmBatched_64

    global __cublasCgemm3mBatched_64
    data["__cublasCgemm3mBatched_64"] = <intptr_t>__cublasCgemm3mBatched_64

    global __cublasZgemmBatched_64
    data["__cublasZgemmBatched_64"] = <intptr_t>__cublasZgemmBatched_64

    global __cublasSgemmStridedBatched_64
    data["__cublasSgemmStridedBatched_64"] = <intptr_t>__cublasSgemmStridedBatched_64

    global __cublasDgemmStridedBatched_64
    data["__cublasDgemmStridedBatched_64"] = <intptr_t>__cublasDgemmStridedBatched_64

    global __cublasCgemmStridedBatched_64
    data["__cublasCgemmStridedBatched_64"] = <intptr_t>__cublasCgemmStridedBatched_64

    global __cublasCgemm3mStridedBatched_64
    data["__cublasCgemm3mStridedBatched_64"] = <intptr_t>__cublasCgemm3mStridedBatched_64

    global __cublasZgemmStridedBatched_64
    data["__cublasZgemmStridedBatched_64"] = <intptr_t>__cublasZgemmStridedBatched_64

    global __cublasGemmBatchedEx_64
    data["__cublasGemmBatchedEx_64"] = <intptr_t>__cublasGemmBatchedEx_64

    global __cublasGemmStridedBatchedEx_64
    data["__cublasGemmStridedBatchedEx_64"] = <intptr_t>__cublasGemmStridedBatchedEx_64

    global __cublasSgeam_64
    data["__cublasSgeam_64"] = <intptr_t>__cublasSgeam_64

    global __cublasDgeam_64
    data["__cublasDgeam_64"] = <intptr_t>__cublasDgeam_64

    global __cublasCgeam_64
    data["__cublasCgeam_64"] = <intptr_t>__cublasCgeam_64

    global __cublasZgeam_64
    data["__cublasZgeam_64"] = <intptr_t>__cublasZgeam_64

    global __cublasStrsmBatched_64
    data["__cublasStrsmBatched_64"] = <intptr_t>__cublasStrsmBatched_64

    global __cublasDtrsmBatched_64
    data["__cublasDtrsmBatched_64"] = <intptr_t>__cublasDtrsmBatched_64

    global __cublasCtrsmBatched_64
    data["__cublasCtrsmBatched_64"] = <intptr_t>__cublasCtrsmBatched_64

    global __cublasZtrsmBatched_64
    data["__cublasZtrsmBatched_64"] = <intptr_t>__cublasZtrsmBatched_64

    global __cublasSdgmm_64
    data["__cublasSdgmm_64"] = <intptr_t>__cublasSdgmm_64

    global __cublasDdgmm_64
    data["__cublasDdgmm_64"] = <intptr_t>__cublasDdgmm_64

    global __cublasCdgmm_64
    data["__cublasCdgmm_64"] = <intptr_t>__cublasCdgmm_64

    global __cublasZdgmm_64
    data["__cublasZdgmm_64"] = <intptr_t>__cublasZdgmm_64

    global __cublasSgemmGroupedBatched
    data["__cublasSgemmGroupedBatched"] = <intptr_t>__cublasSgemmGroupedBatched

    global __cublasSgemmGroupedBatched_64
    data["__cublasSgemmGroupedBatched_64"] = <intptr_t>__cublasSgemmGroupedBatched_64

    global __cublasDgemmGroupedBatched
    data["__cublasDgemmGroupedBatched"] = <intptr_t>__cublasDgemmGroupedBatched

    global __cublasDgemmGroupedBatched_64
    data["__cublasDgemmGroupedBatched_64"] = <intptr_t>__cublasDgemmGroupedBatched_64

    global __cublasGemmGroupedBatchedEx
    data["__cublasGemmGroupedBatchedEx"] = <intptr_t>__cublasGemmGroupedBatchedEx

    global __cublasGemmGroupedBatchedEx_64
    data["__cublasGemmGroupedBatchedEx_64"] = <intptr_t>__cublasGemmGroupedBatchedEx_64

    global __cublasGetEmulationStrategy
    data["__cublasGetEmulationStrategy"] = <intptr_t>__cublasGetEmulationStrategy

    global __cublasSetEmulationStrategy
    data["__cublasSetEmulationStrategy"] = <intptr_t>__cublasSetEmulationStrategy

    global __cublasGetEmulationSpecialValuesSupport
    data["__cublasGetEmulationSpecialValuesSupport"] = <intptr_t>__cublasGetEmulationSpecialValuesSupport

    global __cublasSetEmulationSpecialValuesSupport
    data["__cublasSetEmulationSpecialValuesSupport"] = <intptr_t>__cublasSetEmulationSpecialValuesSupport

    global __cublasGetFixedPointEmulationMantissaControl
    data["__cublasGetFixedPointEmulationMantissaControl"] = <intptr_t>__cublasGetFixedPointEmulationMantissaControl

    global __cublasSetFixedPointEmulationMantissaControl
    data["__cublasSetFixedPointEmulationMantissaControl"] = <intptr_t>__cublasSetFixedPointEmulationMantissaControl

    global __cublasGetFixedPointEmulationMaxMantissaBitCount
    data["__cublasGetFixedPointEmulationMaxMantissaBitCount"] = <intptr_t>__cublasGetFixedPointEmulationMaxMantissaBitCount

    global __cublasSetFixedPointEmulationMaxMantissaBitCount
    data["__cublasSetFixedPointEmulationMaxMantissaBitCount"] = <intptr_t>__cublasSetFixedPointEmulationMaxMantissaBitCount

    global __cublasGetFixedPointEmulationMantissaBitOffset
    data["__cublasGetFixedPointEmulationMantissaBitOffset"] = <intptr_t>__cublasGetFixedPointEmulationMantissaBitOffset

    global __cublasSetFixedPointEmulationMantissaBitOffset
    data["__cublasSetFixedPointEmulationMantissaBitOffset"] = <intptr_t>__cublasSetFixedPointEmulationMantissaBitOffset

    global __cublasGetFixedPointEmulationMantissaBitCountPointer
    data["__cublasGetFixedPointEmulationMantissaBitCountPointer"] = <intptr_t>__cublasGetFixedPointEmulationMantissaBitCountPointer

    global __cublasSetFixedPointEmulationMantissaBitCountPointer
    data["__cublasSetFixedPointEmulationMantissaBitCountPointer"] = <intptr_t>__cublasSetFixedPointEmulationMantissaBitCountPointer

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

cdef cublasStatus_t _cublasCreate(cublasHandle_t* handle) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCreate_v2
    _check_or_init_cublas()
    if __cublasCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCreate_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t*) noexcept nogil>__cublasCreate_v2)(
        handle)


cdef cublasStatus_t _cublasDestroy(cublasHandle_t handle) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDestroy_v2
    _check_or_init_cublas()
    if __cublasDestroy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDestroy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t) noexcept nogil>__cublasDestroy_v2)(
        handle)


cdef cublasStatus_t _cublasGetVersion(cublasHandle_t handle, int* version) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetVersion_v2
    _check_or_init_cublas()
    if __cublasGetVersion_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetVersion_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int*) noexcept nogil>__cublasGetVersion_v2)(
        handle, version)


cdef cublasStatus_t _cublasGetProperty(libraryPropertyType type, int* value) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetProperty
    _check_or_init_cublas()
    if __cublasGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetProperty is not found")
    return (<cublasStatus_t (*)(libraryPropertyType, int*) noexcept nogil>__cublasGetProperty)(
        type, value)


cdef size_t _cublasGetCudartVersion() except?0 nogil:
    global __cublasGetCudartVersion
    _check_or_init_cublas()
    if __cublasGetCudartVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetCudartVersion is not found")
    return (<size_t (*)() noexcept nogil>__cublasGetCudartVersion)(
        )


cdef cublasStatus_t _cublasSetWorkspace(cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetWorkspace_v2
    _check_or_init_cublas()
    if __cublasSetWorkspace_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetWorkspace_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, void*, size_t) noexcept nogil>__cublasSetWorkspace_v2)(
        handle, workspace, workspaceSizeInBytes)


cdef cublasStatus_t _cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetStream_v2
    _check_or_init_cublas()
    if __cublasSetStream_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetStream_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cudaStream_t) noexcept nogil>__cublasSetStream_v2)(
        handle, streamId)


cdef cublasStatus_t _cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetStream_v2
    _check_or_init_cublas()
    if __cublasGetStream_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetStream_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cudaStream_t*) noexcept nogil>__cublasGetStream_v2)(
        handle, streamId)


cdef cublasStatus_t _cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetPointerMode_v2
    _check_or_init_cublas()
    if __cublasGetPointerMode_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetPointerMode_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasPointerMode_t*) noexcept nogil>__cublasGetPointerMode_v2)(
        handle, mode)


cdef cublasStatus_t _cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetPointerMode_v2
    _check_or_init_cublas()
    if __cublasSetPointerMode_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetPointerMode_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasPointerMode_t) noexcept nogil>__cublasSetPointerMode_v2)(
        handle, mode)


cdef cublasStatus_t _cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetAtomicsMode
    _check_or_init_cublas()
    if __cublasGetAtomicsMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetAtomicsMode is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasAtomicsMode_t*) noexcept nogil>__cublasGetAtomicsMode)(
        handle, mode)


cdef cublasStatus_t _cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetAtomicsMode
    _check_or_init_cublas()
    if __cublasSetAtomicsMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetAtomicsMode is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasAtomicsMode_t) noexcept nogil>__cublasSetAtomicsMode)(
        handle, mode)


cdef cublasStatus_t _cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetMathMode
    _check_or_init_cublas()
    if __cublasGetMathMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetMathMode is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasMath_t*) noexcept nogil>__cublasGetMathMode)(
        handle, mode)


cdef cublasStatus_t _cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetMathMode
    _check_or_init_cublas()
    if __cublasSetMathMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetMathMode is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasMath_t) noexcept nogil>__cublasSetMathMode)(
        handle, mode)


cdef cublasStatus_t _cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFileName) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasLoggerConfigure
    _check_or_init_cublas()
    if __cublasLoggerConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLoggerConfigure is not found")
    return (<cublasStatus_t (*)(int, int, int, const char*) noexcept nogil>__cublasLoggerConfigure)(
        logIsOn, logToStdOut, logToStdErr, logFileName)


cdef cublasStatus_t _cublasSetLoggerCallback(cublasLogCallback userCallback) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetLoggerCallback
    _check_or_init_cublas()
    if __cublasSetLoggerCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetLoggerCallback is not found")
    return (<cublasStatus_t (*)(cublasLogCallback) noexcept nogil>__cublasSetLoggerCallback)(
        userCallback)


cdef cublasStatus_t _cublasGetLoggerCallback(cublasLogCallback* userCallback) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetLoggerCallback
    _check_or_init_cublas()
    if __cublasGetLoggerCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetLoggerCallback is not found")
    return (<cublasStatus_t (*)(cublasLogCallback*) noexcept nogil>__cublasGetLoggerCallback)(
        userCallback)


cdef cublasStatus_t _cublasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetVector
    _check_or_init_cublas()
    if __cublasSetVector == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetVector is not found")
    return (<cublasStatus_t (*)(int, int, const void*, int, void*, int) noexcept nogil>__cublasSetVector)(
        n, elemSize, x, incx, devicePtr, incy)


cdef cublasStatus_t _cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetVector
    _check_or_init_cublas()
    if __cublasGetVector == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetVector is not found")
    return (<cublasStatus_t (*)(int, int, const void*, int, void*, int) noexcept nogil>__cublasGetVector)(
        n, elemSize, x, incx, y, incy)


cdef cublasStatus_t _cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetMatrix
    _check_or_init_cublas()
    if __cublasSetMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetMatrix is not found")
    return (<cublasStatus_t (*)(int, int, int, const void*, int, void*, int) noexcept nogil>__cublasSetMatrix)(
        rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t _cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetMatrix
    _check_or_init_cublas()
    if __cublasGetMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetMatrix is not found")
    return (<cublasStatus_t (*)(int, int, int, const void*, int, void*, int) noexcept nogil>__cublasGetMatrix)(
        rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t _cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetVectorAsync
    _check_or_init_cublas()
    if __cublasSetVectorAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetVectorAsync is not found")
    return (<cublasStatus_t (*)(int, int, const void*, int, void*, int, cudaStream_t) noexcept nogil>__cublasSetVectorAsync)(
        n, elemSize, hostPtr, incx, devicePtr, incy, stream)


cdef cublasStatus_t _cublasGetVectorAsync(int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetVectorAsync
    _check_or_init_cublas()
    if __cublasGetVectorAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetVectorAsync is not found")
    return (<cublasStatus_t (*)(int, int, const void*, int, void*, int, cudaStream_t) noexcept nogil>__cublasGetVectorAsync)(
        n, elemSize, devicePtr, incx, hostPtr, incy, stream)


cdef cublasStatus_t _cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetMatrixAsync
    _check_or_init_cublas()
    if __cublasSetMatrixAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetMatrixAsync is not found")
    return (<cublasStatus_t (*)(int, int, int, const void*, int, void*, int, cudaStream_t) noexcept nogil>__cublasSetMatrixAsync)(
        rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t _cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetMatrixAsync
    _check_or_init_cublas()
    if __cublasGetMatrixAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetMatrixAsync is not found")
    return (<cublasStatus_t (*)(int, int, int, const void*, int, void*, int, cudaStream_t) noexcept nogil>__cublasGetMatrixAsync)(
        rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t _cublasNrm2Ex(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executionType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasNrm2Ex
    _check_or_init_cublas()
    if __cublasNrm2Ex == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasNrm2Ex is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, void*, cudaDataType, cudaDataType) noexcept nogil>__cublasNrm2Ex)(
        handle, n, x, xType, incx, result, resultType, executionType)


cdef cublasStatus_t _cublasSnrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSnrm2_v2
    _check_or_init_cublas()
    if __cublasSnrm2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSnrm2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, float*) noexcept nogil>__cublasSnrm2_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDnrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDnrm2_v2
    _check_or_init_cublas()
    if __cublasDnrm2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDnrm2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, double*) noexcept nogil>__cublasDnrm2_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasScnrm2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasScnrm2_v2
    _check_or_init_cublas()
    if __cublasScnrm2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScnrm2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, float*) noexcept nogil>__cublasScnrm2_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDznrm2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDznrm2_v2
    _check_or_init_cublas()
    if __cublasDznrm2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDznrm2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, double*) noexcept nogil>__cublasDznrm2_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDotEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDotEx
    _check_or_init_cublas()
    if __cublasDotEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDotEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, const void*, cudaDataType, int, void*, cudaDataType, cudaDataType) noexcept nogil>__cublasDotEx)(
        handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t _cublasDotcEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDotcEx
    _check_or_init_cublas()
    if __cublasDotcEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDotcEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, const void*, cudaDataType, int, void*, cudaDataType, cudaDataType) noexcept nogil>__cublasDotcEx)(
        handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t _cublasSdot(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSdot_v2
    _check_or_init_cublas()
    if __cublasSdot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSdot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, const float*, int, float*) noexcept nogil>__cublasSdot_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasDdot(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDdot_v2
    _check_or_init_cublas()
    if __cublasDdot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDdot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, const double*, int, double*) noexcept nogil>__cublasDdot_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasCdotu(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCdotu_v2
    _check_or_init_cublas()
    if __cublasCdotu_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdotu_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, const cuComplex*, int, cuComplex*) noexcept nogil>__cublasCdotu_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasCdotc(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCdotc_v2
    _check_or_init_cublas()
    if __cublasCdotc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdotc_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, const cuComplex*, int, cuComplex*) noexcept nogil>__cublasCdotc_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasZdotu(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZdotu_v2
    _check_or_init_cublas()
    if __cublasZdotu_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdotu_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*) noexcept nogil>__cublasZdotu_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasZdotc(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZdotc_v2
    _check_or_init_cublas()
    if __cublasZdotc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdotc_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*) noexcept nogil>__cublasZdotc_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasScalEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int incx, cudaDataType executionType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasScalEx
    _check_or_init_cublas()
    if __cublasScalEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScalEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, void*, cudaDataType, int, cudaDataType) noexcept nogil>__cublasScalEx)(
        handle, n, alpha, alphaType, x, xType, incx, executionType)


cdef cublasStatus_t _cublasSscal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSscal_v2
    _check_or_init_cublas()
    if __cublasSscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, float*, int) noexcept nogil>__cublasSscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasDscal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDscal_v2
    _check_or_init_cublas()
    if __cublasDscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, double*, int) noexcept nogil>__cublasDscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasCscal(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCscal_v2
    _check_or_init_cublas()
    if __cublasCscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasCsscal(cublasHandle_t handle, int n, const float* alpha, cuComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsscal_v2
    _check_or_init_cublas()
    if __cublasCsscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, cuComplex*, int) noexcept nogil>__cublasCsscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasZscal(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZscal_v2
    _check_or_init_cublas()
    if __cublasZscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasZdscal(cublasHandle_t handle, int n, const double* alpha, cuDoubleComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZdscal_v2
    _check_or_init_cublas()
    if __cublasZdscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, cuDoubleComplex*, int) noexcept nogil>__cublasZdscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasAxpyEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, cudaDataType executiontype) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasAxpyEx
    _check_or_init_cublas()
    if __cublasAxpyEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasAxpyEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, const void*, cudaDataType, int, void*, cudaDataType, int, cudaDataType) noexcept nogil>__cublasAxpyEx)(
        handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)


cdef cublasStatus_t _cublasSaxpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSaxpy_v2
    _check_or_init_cublas()
    if __cublasSaxpy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSaxpy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, const float*, int, float*, int) noexcept nogil>__cublasSaxpy_v2)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasDaxpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDaxpy_v2
    _check_or_init_cublas()
    if __cublasDaxpy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDaxpy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, const double*, int, double*, int) noexcept nogil>__cublasDaxpy_v2)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasCaxpy(cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCaxpy_v2
    _check_or_init_cublas()
    if __cublasCaxpy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCaxpy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCaxpy_v2)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasZaxpy(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZaxpy_v2
    _check_or_init_cublas()
    if __cublasZaxpy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZaxpy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZaxpy_v2)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasCopyEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCopyEx
    _check_or_init_cublas()
    if __cublasCopyEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCopyEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, void*, cudaDataType, int) noexcept nogil>__cublasCopyEx)(
        handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t _cublasScopy(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasScopy_v2
    _check_or_init_cublas()
    if __cublasScopy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScopy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, float*, int) noexcept nogil>__cublasScopy_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasDcopy(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDcopy_v2
    _check_or_init_cublas()
    if __cublasDcopy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDcopy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, double*, int) noexcept nogil>__cublasDcopy_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasCcopy(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCcopy_v2
    _check_or_init_cublas()
    if __cublasCcopy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCcopy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCcopy_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasZcopy(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZcopy_v2
    _check_or_init_cublas()
    if __cublasZcopy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZcopy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZcopy_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasSswap(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSswap_v2
    _check_or_init_cublas()
    if __cublasSswap_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSswap_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, float*, int, float*, int) noexcept nogil>__cublasSswap_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasDswap(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDswap_v2
    _check_or_init_cublas()
    if __cublasDswap_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDswap_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, double*, int, double*, int) noexcept nogil>__cublasDswap_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasCswap(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCswap_v2
    _check_or_init_cublas()
    if __cublasCswap_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCswap_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCswap_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZswap_v2
    _check_or_init_cublas()
    if __cublasZswap_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZswap_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZswap_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasSwapEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSwapEx
    _check_or_init_cublas()
    if __cublasSwapEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSwapEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, void*, cudaDataType, int, void*, cudaDataType, int) noexcept nogil>__cublasSwapEx)(
        handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t _cublasIsamax(cublasHandle_t handle, int n, const float* x, int incx, int* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIsamax_v2
    _check_or_init_cublas()
    if __cublasIsamax_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIsamax_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, int*) noexcept nogil>__cublasIsamax_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIdamax(cublasHandle_t handle, int n, const double* x, int incx, int* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIdamax_v2
    _check_or_init_cublas()
    if __cublasIdamax_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIdamax_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, int*) noexcept nogil>__cublasIdamax_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIcamax(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIcamax_v2
    _check_or_init_cublas()
    if __cublasIcamax_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIcamax_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, int*) noexcept nogil>__cublasIcamax_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIzamax(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIzamax_v2
    _check_or_init_cublas()
    if __cublasIzamax_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIzamax_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, int*) noexcept nogil>__cublasIzamax_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIamaxEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIamaxEx
    _check_or_init_cublas()
    if __cublasIamaxEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIamaxEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, int*) noexcept nogil>__cublasIamaxEx)(
        handle, n, x, xType, incx, result)


cdef cublasStatus_t _cublasIsamin(cublasHandle_t handle, int n, const float* x, int incx, int* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIsamin_v2
    _check_or_init_cublas()
    if __cublasIsamin_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIsamin_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, int*) noexcept nogil>__cublasIsamin_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIdamin(cublasHandle_t handle, int n, const double* x, int incx, int* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIdamin_v2
    _check_or_init_cublas()
    if __cublasIdamin_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIdamin_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, int*) noexcept nogil>__cublasIdamin_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIcamin(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIcamin_v2
    _check_or_init_cublas()
    if __cublasIcamin_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIcamin_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, int*) noexcept nogil>__cublasIcamin_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIzamin(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIzamin_v2
    _check_or_init_cublas()
    if __cublasIzamin_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIzamin_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, int*) noexcept nogil>__cublasIzamin_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIaminEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIaminEx
    _check_or_init_cublas()
    if __cublasIaminEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIaminEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, int*) noexcept nogil>__cublasIaminEx)(
        handle, n, x, xType, incx, result)


cdef cublasStatus_t _cublasAsumEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executiontype) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasAsumEx
    _check_or_init_cublas()
    if __cublasAsumEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasAsumEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, void*, cudaDataType, cudaDataType) noexcept nogil>__cublasAsumEx)(
        handle, n, x, xType, incx, result, resultType, executiontype)


cdef cublasStatus_t _cublasSasum(cublasHandle_t handle, int n, const float* x, int incx, float* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSasum_v2
    _check_or_init_cublas()
    if __cublasSasum_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSasum_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, float*) noexcept nogil>__cublasSasum_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDasum(cublasHandle_t handle, int n, const double* x, int incx, double* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDasum_v2
    _check_or_init_cublas()
    if __cublasDasum_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDasum_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, double*) noexcept nogil>__cublasDasum_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasScasum(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasScasum_v2
    _check_or_init_cublas()
    if __cublasScasum_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScasum_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, float*) noexcept nogil>__cublasScasum_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDzasum(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDzasum_v2
    _check_or_init_cublas()
    if __cublasDzasum_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDzasum_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, double*) noexcept nogil>__cublasDzasum_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasSrot(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSrot_v2
    _check_or_init_cublas()
    if __cublasSrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, float*, int, float*, int, const float*, const float*) noexcept nogil>__cublasSrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasDrot(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDrot_v2
    _check_or_init_cublas()
    if __cublasDrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, double*, int, double*, int, const double*, const double*) noexcept nogil>__cublasDrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasCrot(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCrot_v2
    _check_or_init_cublas()
    if __cublasCrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuComplex*, int, cuComplex*, int, const float*, const cuComplex*) noexcept nogil>__cublasCrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasCsrot(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsrot_v2
    _check_or_init_cublas()
    if __cublasCsrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuComplex*, int, cuComplex*, int, const float*, const float*) noexcept nogil>__cublasCsrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasZrot(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZrot_v2
    _check_or_init_cublas()
    if __cublasZrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, const double*, const cuDoubleComplex*) noexcept nogil>__cublasZrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasZdrot(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZdrot_v2
    _check_or_init_cublas()
    if __cublasZdrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, const double*, const double*) noexcept nogil>__cublasZdrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasRotEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasRotEx
    _check_or_init_cublas()
    if __cublasRotEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, void*, cudaDataType, int, void*, cudaDataType, int, const void*, const void*, cudaDataType, cudaDataType) noexcept nogil>__cublasRotEx)(
        handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)


cdef cublasStatus_t _cublasSrotg(cublasHandle_t handle, float* a, float* b, float* c, float* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSrotg_v2
    _check_or_init_cublas()
    if __cublasSrotg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrotg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, float*, float*, float*, float*) noexcept nogil>__cublasSrotg_v2)(
        handle, a, b, c, s)


cdef cublasStatus_t _cublasDrotg(cublasHandle_t handle, double* a, double* b, double* c, double* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDrotg_v2
    _check_or_init_cublas()
    if __cublasDrotg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrotg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, double*, double*, double*, double*) noexcept nogil>__cublasDrotg_v2)(
        handle, a, b, c, s)


cdef cublasStatus_t _cublasCrotg(cublasHandle_t handle, cuComplex* a, cuComplex* b, float* c, cuComplex* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCrotg_v2
    _check_or_init_cublas()
    if __cublasCrotg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCrotg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cuComplex*, cuComplex*, float*, cuComplex*) noexcept nogil>__cublasCrotg_v2)(
        handle, a, b, c, s)


cdef cublasStatus_t _cublasZrotg(cublasHandle_t handle, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZrotg_v2
    _check_or_init_cublas()
    if __cublasZrotg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZrotg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cuDoubleComplex*, cuDoubleComplex*, double*, cuDoubleComplex*) noexcept nogil>__cublasZrotg_v2)(
        handle, a, b, c, s)


cdef cublasStatus_t _cublasRotgEx(cublasHandle_t handle, void* a, void* b, cudaDataType abType, void* c, void* s, cudaDataType csType, cudaDataType executiontype) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasRotgEx
    _check_or_init_cublas()
    if __cublasRotgEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotgEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, void*, void*, cudaDataType, void*, void*, cudaDataType, cudaDataType) noexcept nogil>__cublasRotgEx)(
        handle, a, b, abType, c, s, csType, executiontype)


cdef cublasStatus_t _cublasSrotm(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSrotm_v2
    _check_or_init_cublas()
    if __cublasSrotm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrotm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, float*, int, float*, int, const float*) noexcept nogil>__cublasSrotm_v2)(
        handle, n, x, incx, y, incy, param)


cdef cublasStatus_t _cublasDrotm(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDrotm_v2
    _check_or_init_cublas()
    if __cublasDrotm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrotm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, double*, int, double*, int, const double*) noexcept nogil>__cublasDrotm_v2)(
        handle, n, x, incx, y, incy, param)


cdef cublasStatus_t _cublasRotmEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* param, cudaDataType paramType, cudaDataType executiontype) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasRotmEx
    _check_or_init_cublas()
    if __cublasRotmEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotmEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, void*, cudaDataType, int, void*, cudaDataType, int, const void*, cudaDataType, cudaDataType) noexcept nogil>__cublasRotmEx)(
        handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype)


cdef cublasStatus_t _cublasSrotmg(cublasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSrotmg_v2
    _check_or_init_cublas()
    if __cublasSrotmg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrotmg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, float*, float*, float*, const float*, float*) noexcept nogil>__cublasSrotmg_v2)(
        handle, d1, d2, x1, y1, param)


cdef cublasStatus_t _cublasDrotmg(cublasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDrotmg_v2
    _check_or_init_cublas()
    if __cublasDrotmg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrotmg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, double*, double*, double*, const double*, double*) noexcept nogil>__cublasDrotmg_v2)(
        handle, d1, d2, x1, y1, param)


cdef cublasStatus_t _cublasRotmgEx(cublasHandle_t handle, void* d1, cudaDataType d1Type, void* d2, cudaDataType d2Type, void* x1, cudaDataType x1Type, const void* y1, cudaDataType y1Type, void* param, cudaDataType paramType, cudaDataType executiontype) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasRotmgEx
    _check_or_init_cublas()
    if __cublasRotmgEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotmgEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, void*, cudaDataType, void*, cudaDataType, void*, cudaDataType, const void*, cudaDataType, void*, cudaDataType, cudaDataType) noexcept nogil>__cublasRotmgEx)(
        handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype)


cdef cublasStatus_t _cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemv_v2
    _check_or_init_cublas()
    if __cublasSgemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) noexcept nogil>__cublasSgemv_v2)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemv_v2
    _check_or_init_cublas()
    if __cublasDgemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) noexcept nogil>__cublasDgemv_v2)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemv_v2
    _check_or_init_cublas()
    if __cublasCgemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCgemv_v2)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemv_v2
    _check_or_init_cublas()
    if __cublasZgemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZgemv_v2)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgbmv_v2
    _check_or_init_cublas()
    if __cublasSgbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) noexcept nogil>__cublasSgbmv_v2)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgbmv_v2
    _check_or_init_cublas()
    if __cublasDgbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) noexcept nogil>__cublasDgbmv_v2)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgbmv_v2
    _check_or_init_cublas()
    if __cublasCgbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCgbmv_v2)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgbmv_v2
    _check_or_init_cublas()
    if __cublasZgbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZgbmv_v2)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrmv_v2
    _check_or_init_cublas()
    if __cublasStrmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float*, int, float*, int) noexcept nogil>__cublasStrmv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasDtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrmv_v2
    _check_or_init_cublas()
    if __cublasDtrmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double*, int, double*, int) noexcept nogil>__cublasDtrmv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasCtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrmv_v2
    _check_or_init_cublas()
    if __cublasCtrmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCtrmv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasZtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrmv_v2
    _check_or_init_cublas()
    if __cublasZtrmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZtrmv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasStbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStbmv_v2
    _check_or_init_cublas()
    if __cublasStbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, int, float*, int) noexcept nogil>__cublasStbmv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasDtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtbmv_v2
    _check_or_init_cublas()
    if __cublasDtbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, int, double*, int) noexcept nogil>__cublasDtbmv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasCtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtbmv_v2
    _check_or_init_cublas()
    if __cublasCtbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCtbmv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasZtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtbmv_v2
    _check_or_init_cublas()
    if __cublasZtbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZtbmv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasStpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStpmv_v2
    _check_or_init_cublas()
    if __cublasStpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float*, float*, int) noexcept nogil>__cublasStpmv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasDtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtpmv_v2
    _check_or_init_cublas()
    if __cublasDtpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double*, double*, int) noexcept nogil>__cublasDtpmv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasCtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtpmv_v2
    _check_or_init_cublas()
    if __cublasCtpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCtpmv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasZtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtpmv_v2
    _check_or_init_cublas()
    if __cublasZtpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZtpmv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasStrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrsv_v2
    _check_or_init_cublas()
    if __cublasStrsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float*, int, float*, int) noexcept nogil>__cublasStrsv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrsv_v2
    _check_or_init_cublas()
    if __cublasDtrsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double*, int, double*, int) noexcept nogil>__cublasDtrsv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasCtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrsv_v2
    _check_or_init_cublas()
    if __cublasCtrsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCtrsv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasZtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrsv_v2
    _check_or_init_cublas()
    if __cublasZtrsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZtrsv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasStpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStpsv_v2
    _check_or_init_cublas()
    if __cublasStpsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStpsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float*, float*, int) noexcept nogil>__cublasStpsv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasDtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtpsv_v2
    _check_or_init_cublas()
    if __cublasDtpsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtpsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double*, double*, int) noexcept nogil>__cublasDtpsv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasCtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtpsv_v2
    _check_or_init_cublas()
    if __cublasCtpsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtpsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCtpsv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasZtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtpsv_v2
    _check_or_init_cublas()
    if __cublasZtpsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtpsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZtpsv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasStbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStbsv_v2
    _check_or_init_cublas()
    if __cublasStbsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStbsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, int, float*, int) noexcept nogil>__cublasStbsv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasDtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtbsv_v2
    _check_or_init_cublas()
    if __cublasDtbsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtbsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, int, double*, int) noexcept nogil>__cublasDtbsv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasCtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtbsv_v2
    _check_or_init_cublas()
    if __cublasCtbsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtbsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCtbsv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasZtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtbsv_v2
    _check_or_init_cublas()
    if __cublasZtbsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtbsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZtbsv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsymv_v2
    _check_or_init_cublas()
    if __cublasSsymv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsymv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, int, const float*, int, const float*, float*, int) noexcept nogil>__cublasSsymv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsymv_v2
    _check_or_init_cublas()
    if __cublasDsymv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsymv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, int, const double*, int, const double*, double*, int) noexcept nogil>__cublasDsymv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsymv_v2
    _check_or_init_cublas()
    if __cublasCsymv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsymv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCsymv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsymv_v2
    _check_or_init_cublas()
    if __cublasZsymv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsymv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZsymv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChemv_v2
    _check_or_init_cublas()
    if __cublasChemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasChemv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhemv_v2
    _check_or_init_cublas()
    if __cublasZhemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZhemv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsbmv_v2
    _check_or_init_cublas()
    if __cublasSsbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) noexcept nogil>__cublasSsbmv_v2)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsbmv_v2
    _check_or_init_cublas()
    if __cublasDsbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) noexcept nogil>__cublasDsbmv_v2)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChbmv_v2
    _check_or_init_cublas()
    if __cublasChbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasChbmv_v2)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhbmv_v2
    _check_or_init_cublas()
    if __cublasZhbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZhbmv_v2)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSspmv_v2
    _check_or_init_cublas()
    if __cublasSspmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, const float*, int, const float*, float*, int) noexcept nogil>__cublasSspmv_v2)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDspmv_v2
    _check_or_init_cublas()
    if __cublasDspmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, const double*, int, const double*, double*, int) noexcept nogil>__cublasDspmv_v2)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChpmv_v2
    _check_or_init_cublas()
    if __cublasChpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasChpmv_v2)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhpmv_v2
    _check_or_init_cublas()
    if __cublasZhpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZhpmv_v2)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSger(cublasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSger_v2
    _check_or_init_cublas()
    if __cublasSger_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSger_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const float*, const float*, int, const float*, int, float*, int) noexcept nogil>__cublasSger_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasDger(cublasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDger_v2
    _check_or_init_cublas()
    if __cublasDger_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDger_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const double*, const double*, int, const double*, int, double*, int) noexcept nogil>__cublasDger_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCgeru(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgeru_v2
    _check_or_init_cublas()
    if __cublasCgeru_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgeru_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCgeru_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCgerc(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgerc_v2
    _check_or_init_cublas()
    if __cublasCgerc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgerc_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCgerc_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZgeru(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgeru_v2
    _check_or_init_cublas()
    if __cublasZgeru_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgeru_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZgeru_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZgerc(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgerc_v2
    _check_or_init_cublas()
    if __cublasZgerc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgerc_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZgerc_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsyr_v2
    _check_or_init_cublas()
    if __cublasSsyr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, int, float*, int) noexcept nogil>__cublasSsyr_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasDsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsyr_v2
    _check_or_init_cublas()
    if __cublasDsyr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, int, double*, int) noexcept nogil>__cublasDsyr_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasCsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyr_v2
    _check_or_init_cublas()
    if __cublasCsyr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCsyr_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasZsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsyr_v2
    _check_or_init_cublas()
    if __cublasZsyr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZsyr_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasCher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCher_v2
    _check_or_init_cublas()
    if __cublasCher_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCher_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasZher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZher_v2
    _check_or_init_cublas()
    if __cublasZher_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZher_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasSspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSspr_v2
    _check_or_init_cublas()
    if __cublasSspr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, int, float*) noexcept nogil>__cublasSspr_v2)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasDspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDspr_v2
    _check_or_init_cublas()
    if __cublasDspr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, int, double*) noexcept nogil>__cublasDspr_v2)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasChpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChpr_v2
    _check_or_init_cublas()
    if __cublasChpr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const cuComplex*, int, cuComplex*) noexcept nogil>__cublasChpr_v2)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasZhpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhpr_v2
    _check_or_init_cublas()
    if __cublasZhpr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const cuDoubleComplex*, int, cuDoubleComplex*) noexcept nogil>__cublasZhpr_v2)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasSsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsyr2_v2
    _check_or_init_cublas()
    if __cublasSsyr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, int, const float*, int, float*, int) noexcept nogil>__cublasSsyr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasDsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsyr2_v2
    _check_or_init_cublas()
    if __cublasDsyr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, int, const double*, int, double*, int) noexcept nogil>__cublasDsyr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyr2_v2
    _check_or_init_cublas()
    if __cublasCsyr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCsyr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsyr2_v2
    _check_or_init_cublas()
    if __cublasZsyr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZsyr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCher2_v2
    _check_or_init_cublas()
    if __cublasCher2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCher2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZher2_v2
    _check_or_init_cublas()
    if __cublasZher2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZher2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasSspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSspr2_v2
    _check_or_init_cublas()
    if __cublasSspr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, int, const float*, int, float*) noexcept nogil>__cublasSspr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasDspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDspr2_v2
    _check_or_init_cublas()
    if __cublasDspr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, int, const double*, int, double*) noexcept nogil>__cublasDspr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasChpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChpr2_v2
    _check_or_init_cublas()
    if __cublasChpr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*) noexcept nogil>__cublasChpr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasZhpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhpr2_v2
    _check_or_init_cublas()
    if __cublasZhpr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*) noexcept nogil>__cublasZhpr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemm_v2
    _check_or_init_cublas()
    if __cublasSgemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) noexcept nogil>__cublasSgemm_v2)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemm_v2
    _check_or_init_cublas()
    if __cublasDgemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) noexcept nogil>__cublasDgemm_v2)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemm_v2
    _check_or_init_cublas()
    if __cublasCgemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCgemm_v2)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemm3m
    _check_or_init_cublas()
    if __cublasCgemm3m == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3m is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCgemm3m)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemm3mEx
    _check_or_init_cublas()
    if __cublasCgemm3mEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const void*, cudaDataType, int, const void*, cudaDataType, int, const cuComplex*, void*, cudaDataType, int) noexcept nogil>__cublasCgemm3mEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasZgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemm_v2
    _check_or_init_cublas()
    if __cublasZgemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZgemm_v2)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemm3m
    _check_or_init_cublas()
    if __cublasZgemm3m == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemm3m is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZgemm3m)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const float* beta, void* C, cudaDataType Ctype, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemmEx
    _check_or_init_cublas()
    if __cublasSgemmEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const void*, cudaDataType, int, const void*, cudaDataType, int, const float*, void*, cudaDataType, int) noexcept nogil>__cublasSgemmEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGemmEx
    _check_or_init_cublas()
    if __cublasGemmEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType, int, const void*, cudaDataType, int, const void*, void*, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t) noexcept nogil>__cublasGemmEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)


cdef cublasStatus_t _cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemmEx
    _check_or_init_cublas()
    if __cublasCgemmEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const void*, cudaDataType, int, const void*, cudaDataType, int, const cuComplex*, void*, cudaDataType, int) noexcept nogil>__cublasCgemmEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasUint8gemmBias(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc, int m, int n, int k, const unsigned char* A, int A_bias, int lda, const unsigned char* B, int B_bias, int ldb, unsigned char* C, int C_bias, int ldc, int C_mult, int C_shift) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasUint8gemmBias
    _check_or_init_cublas()
    if __cublasUint8gemmBias == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasUint8gemmBias is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, cublasOperation_t, int, int, int, const unsigned char*, int, int, const unsigned char*, int, int, unsigned char*, int, int, int, int) noexcept nogil>__cublasUint8gemmBias)(
        handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift)


cdef cublasStatus_t _cublasSsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsyrk_v2
    _check_or_init_cublas()
    if __cublasSsyrk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyrk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, float*, int) noexcept nogil>__cublasSsyrk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasDsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsyrk_v2
    _check_or_init_cublas()
    if __cublasDsyrk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyrk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, double*, int) noexcept nogil>__cublasDsyrk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyrk_v2
    _check_or_init_cublas()
    if __cublasCsyrk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCsyrk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasZsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsyrk_v2
    _check_or_init_cublas()
    if __cublasZsyrk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyrk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZsyrk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyrkEx
    _check_or_init_cublas()
    if __cublasCsyrkEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrkEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const void*, cudaDataType, int, const cuComplex*, void*, cudaDataType, int) noexcept nogil>__cublasCsyrkEx)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyrk3mEx
    _check_or_init_cublas()
    if __cublasCsyrk3mEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrk3mEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const void*, cudaDataType, int, const cuComplex*, void*, cudaDataType, int) noexcept nogil>__cublasCsyrk3mEx)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const cuComplex* A, int lda, const float* beta, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCherk_v2
    _check_or_init_cublas()
    if __cublasCherk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const cuComplex*, int, const float*, cuComplex*, int) noexcept nogil>__cublasCherk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasZherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const cuDoubleComplex* A, int lda, const double* beta, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZherk_v2
    _check_or_init_cublas()
    if __cublasZherk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZherk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, const cuDoubleComplex*, int, const double*, cuDoubleComplex*, int) noexcept nogil>__cublasZherk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const float* beta, void* C, cudaDataType Ctype, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCherkEx
    _check_or_init_cublas()
    if __cublasCherkEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherkEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const void*, cudaDataType, int, const float*, void*, cudaDataType, int) noexcept nogil>__cublasCherkEx)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const float* beta, void* C, cudaDataType Ctype, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCherk3mEx
    _check_or_init_cublas()
    if __cublasCherk3mEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherk3mEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const void*, cudaDataType, int, const float*, void*, cudaDataType, int) noexcept nogil>__cublasCherk3mEx)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasSsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsyr2k_v2
    _check_or_init_cublas()
    if __cublasSsyr2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) noexcept nogil>__cublasSsyr2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsyr2k_v2
    _check_or_init_cublas()
    if __cublasDsyr2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) noexcept nogil>__cublasDsyr2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyr2k_v2
    _check_or_init_cublas()
    if __cublasCsyr2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCsyr2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsyr2k_v2
    _check_or_init_cublas()
    if __cublasZsyr2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZsyr2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCher2k_v2
    _check_or_init_cublas()
    if __cublasCher2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const float*, cuComplex*, int) noexcept nogil>__cublasCher2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZher2k_v2
    _check_or_init_cublas()
    if __cublasZher2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const double*, cuDoubleComplex*, int) noexcept nogil>__cublasZher2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsyrkx
    _check_or_init_cublas()
    if __cublasSsyrkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyrkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) noexcept nogil>__cublasSsyrkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsyrkx
    _check_or_init_cublas()
    if __cublasDsyrkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyrkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) noexcept nogil>__cublasDsyrkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyrkx
    _check_or_init_cublas()
    if __cublasCsyrkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCsyrkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsyrkx
    _check_or_init_cublas()
    if __cublasZsyrkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyrkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZsyrkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCherkx
    _check_or_init_cublas()
    if __cublasCherkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const float*, cuComplex*, int) noexcept nogil>__cublasCherkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZherkx
    _check_or_init_cublas()
    if __cublasZherkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZherkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const double*, cuDoubleComplex*, int) noexcept nogil>__cublasZherkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsymm_v2
    _check_or_init_cublas()
    if __cublasSsymm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsymm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) noexcept nogil>__cublasSsymm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsymm_v2
    _check_or_init_cublas()
    if __cublasDsymm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsymm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) noexcept nogil>__cublasDsymm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsymm_v2
    _check_or_init_cublas()
    if __cublasCsymm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsymm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCsymm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsymm_v2
    _check_or_init_cublas()
    if __cublasZsymm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsymm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZsymm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasChemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChemm_v2
    _check_or_init_cublas()
    if __cublasChemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasChemm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZhemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhemm_v2
    _check_or_init_cublas()
    if __cublasZhemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZhemm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasStrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrsm_v2
    _check_or_init_cublas()
    if __cublasStrsm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, const float*, int, float*, int) noexcept nogil>__cublasStrsm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasDtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrsm_v2
    _check_or_init_cublas()
    if __cublasDtrsm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, const double*, int, double*, int) noexcept nogil>__cublasDtrsm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasCtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrsm_v2
    _check_or_init_cublas()
    if __cublasCtrsm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCtrsm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasZtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrsm_v2
    _check_or_init_cublas()
    if __cublasZtrsm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZtrsm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasStrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrmm_v2
    _check_or_init_cublas()
    if __cublasStrmm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrmm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, const float*, int, const float*, int, float*, int) noexcept nogil>__cublasStrmm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasDtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, double* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrmm_v2
    _check_or_init_cublas()
    if __cublasDtrmm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrmm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, const double*, int, const double*, int, double*, int) noexcept nogil>__cublasDtrmm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasCtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrmm_v2
    _check_or_init_cublas()
    if __cublasCtrmm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrmm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCtrmm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasZtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrmm_v2
    _check_or_init_cublas()
    if __cublasZtrmm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrmm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZtrmm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemmBatched
    _check_or_init_cublas()
    if __cublasSgemmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float* const*, int, const float* const*, int, const float*, float* const*, int, int) noexcept nogil>__cublasSgemmBatched)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double* beta, double* const Carray[], int ldc, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemmBatched
    _check_or_init_cublas()
    if __cublasDgemmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double* const*, int, const double* const*, int, const double*, double* const*, int, int) noexcept nogil>__cublasDgemmBatched)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemmBatched
    _check_or_init_cublas()
    if __cublasCgemmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex* const*, int, const cuComplex* const*, int, const cuComplex*, cuComplex* const*, int, int) noexcept nogil>__cublasCgemmBatched)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasCgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemm3mBatched
    _check_or_init_cublas()
    if __cublasCgemm3mBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex* const*, int, const cuComplex* const*, int, const cuComplex*, cuComplex* const*, int, int) noexcept nogil>__cublasCgemm3mBatched)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int ldc, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemmBatched
    _check_or_init_cublas()
    if __cublasZgemmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, const cuDoubleComplex* const*, int, const cuDoubleComplex* const*, int, const cuDoubleComplex*, cuDoubleComplex* const*, int, int) noexcept nogil>__cublasZgemmBatched)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGemmBatchedEx
    _check_or_init_cublas()
    if __cublasGemmBatchedEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmBatchedEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void* const*, cudaDataType, int, const void* const*, cudaDataType, int, const void*, void* const*, cudaDataType, int, int, cublasComputeType_t, cublasGemmAlgo_t) noexcept nogil>__cublasGemmBatchedEx)(
        handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)


cdef cublasStatus_t _cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGemmStridedBatchedEx
    _check_or_init_cublas()
    if __cublasGemmStridedBatchedEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmStridedBatchedEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType, int, long long int, const void*, cudaDataType, int, long long int, const void*, void*, cudaDataType, int, long long int, int, cublasComputeType_t, cublasGemmAlgo_t) noexcept nogil>__cublasGemmStridedBatchedEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)


cdef cublasStatus_t _cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemmStridedBatched
    _check_or_init_cublas()
    if __cublasSgemmStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, long long int, const float*, int, long long int, const float*, float*, int, long long int, int) noexcept nogil>__cublasSgemmStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemmStridedBatched
    _check_or_init_cublas()
    if __cublasDgemmStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemmStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double*, int, long long int, const double*, int, long long int, const double*, double*, int, long long int, int) noexcept nogil>__cublasDgemmStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemmStridedBatched
    _check_or_init_cublas()
    if __cublasCgemmStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex*, int, long long int, const cuComplex*, int, long long int, const cuComplex*, cuComplex*, int, long long int, int) noexcept nogil>__cublasCgemmStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasCgemm3mStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemm3mStridedBatched
    _check_or_init_cublas()
    if __cublasCgemm3mStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex*, int, long long int, const cuComplex*, int, long long int, const cuComplex*, cuComplex*, int, long long int, int) noexcept nogil>__cublasCgemm3mStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemmStridedBatched
    _check_or_init_cublas()
    if __cublasZgemmStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemmStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, long long int, const cuDoubleComplex*, int, long long int, const cuDoubleComplex*, cuDoubleComplex*, int, long long int, int) noexcept nogil>__cublasZgemmStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgeam
    _check_or_init_cublas()
    if __cublasSgeam == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgeam is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, const float*, int, float*, int) noexcept nogil>__cublasSgeam)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgeam
    _check_or_init_cublas()
    if __cublasDgeam == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgeam is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, const double*, int, double*, int) noexcept nogil>__cublasDgeam)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, const cuComplex* B, int ldb, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgeam
    _check_or_init_cublas()
    if __cublasCgeam == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgeam is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCgeam)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgeam
    _check_or_init_cublas()
    if __cublasZgeam == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgeam is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZgeam)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasSgetrfBatched(cublasHandle_t handle, int n, float* const A[], int lda, int* P, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgetrfBatched
    _check_or_init_cublas()
    if __cublasSgetrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgetrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, float* const*, int, int*, int*, int) noexcept nogil>__cublasSgetrfBatched)(
        handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t _cublasDgetrfBatched(cublasHandle_t handle, int n, double* const A[], int lda, int* P, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgetrfBatched
    _check_or_init_cublas()
    if __cublasDgetrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgetrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, double* const*, int, int*, int*, int) noexcept nogil>__cublasDgetrfBatched)(
        handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t _cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex* const A[], int lda, int* P, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgetrfBatched
    _check_or_init_cublas()
    if __cublasCgetrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgetrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuComplex* const*, int, int*, int*, int) noexcept nogil>__cublasCgetrfBatched)(
        handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t _cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex* const A[], int lda, int* P, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgetrfBatched
    _check_or_init_cublas()
    if __cublasZgetrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgetrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex* const*, int, int*, int*, int) noexcept nogil>__cublasZgetrfBatched)(
        handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t _cublasSgetriBatched(cublasHandle_t handle, int n, const float* const A[], int lda, const int* P, float* const C[], int ldc, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgetriBatched
    _check_or_init_cublas()
    if __cublasSgetriBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgetriBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float* const*, int, const int*, float* const*, int, int*, int) noexcept nogil>__cublasSgetriBatched)(
        handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t _cublasDgetriBatched(cublasHandle_t handle, int n, const double* const A[], int lda, const int* P, double* const C[], int ldc, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgetriBatched
    _check_or_init_cublas()
    if __cublasDgetriBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgetriBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double* const*, int, const int*, double* const*, int, int*, int) noexcept nogil>__cublasDgetriBatched)(
        handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t _cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, const int* P, cuComplex* const C[], int ldc, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgetriBatched
    _check_or_init_cublas()
    if __cublasCgetriBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgetriBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex* const*, int, const int*, cuComplex* const*, int, int*, int) noexcept nogil>__cublasCgetriBatched)(
        handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t _cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, const int* P, cuDoubleComplex* const C[], int ldc, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgetriBatched
    _check_or_init_cublas()
    if __cublasZgetriBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgetriBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex* const*, int, const int*, cuDoubleComplex* const*, int, int*, int) noexcept nogil>__cublasZgetriBatched)(
        handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t _cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda, const int* devIpiv, float* const Barray[], int ldb, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgetrsBatched
    _check_or_init_cublas()
    if __cublasSgetrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgetrsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const float* const*, int, const int*, float* const*, int, int*, int) noexcept nogil>__cublasSgetrsBatched)(
        handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t _cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* const Aarray[], int lda, const int* devIpiv, double* const Barray[], int ldb, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgetrsBatched
    _check_or_init_cublas()
    if __cublasDgetrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgetrsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const double* const*, int, const int*, double* const*, int, int*, int) noexcept nogil>__cublasDgetrsBatched)(
        handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t _cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* const Aarray[], int lda, const int* devIpiv, cuComplex* const Barray[], int ldb, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgetrsBatched
    _check_or_init_cublas()
    if __cublasCgetrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgetrsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex* const*, int, const int*, cuComplex* const*, int, int*, int) noexcept nogil>__cublasCgetrsBatched)(
        handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t _cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int* devIpiv, cuDoubleComplex* const Barray[], int ldb, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgetrsBatched
    _check_or_init_cublas()
    if __cublasZgetrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgetrsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex* const*, int, const int*, cuDoubleComplex* const*, int, int*, int) noexcept nogil>__cublasZgetrsBatched)(
        handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t _cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrsmBatched
    _check_or_init_cublas()
    if __cublasStrsmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, const float* const*, int, float* const*, int, int) noexcept nogil>__cublasStrsmBatched)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrsmBatched
    _check_or_init_cublas()
    if __cublasDtrsmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, const double* const*, int, double* const*, int, int) noexcept nogil>__cublasDtrsmBatched)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrsmBatched
    _check_or_init_cublas()
    if __cublasCtrsmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*, const cuComplex* const*, int, cuComplex* const*, int, int) noexcept nogil>__cublasCtrsmBatched)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrsmBatched
    _check_or_init_cublas()
    if __cublasZtrsmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex*, const cuDoubleComplex* const*, int, cuDoubleComplex* const*, int, int) noexcept nogil>__cublasZtrsmBatched)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasSmatinvBatched(cublasHandle_t handle, int n, const float* const A[], int lda, float* const Ainv[], int lda_inv, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSmatinvBatched
    _check_or_init_cublas()
    if __cublasSmatinvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSmatinvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float* const*, int, float* const*, int, int*, int) noexcept nogil>__cublasSmatinvBatched)(
        handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t _cublasDmatinvBatched(cublasHandle_t handle, int n, const double* const A[], int lda, double* const Ainv[], int lda_inv, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDmatinvBatched
    _check_or_init_cublas()
    if __cublasDmatinvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDmatinvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double* const*, int, double* const*, int, int*, int) noexcept nogil>__cublasDmatinvBatched)(
        handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t _cublasCmatinvBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, cuComplex* const Ainv[], int lda_inv, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCmatinvBatched
    _check_or_init_cublas()
    if __cublasCmatinvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCmatinvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex* const*, int, cuComplex* const*, int, int*, int) noexcept nogil>__cublasCmatinvBatched)(
        handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t _cublasZmatinvBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const Ainv[], int lda_inv, int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZmatinvBatched
    _check_or_init_cublas()
    if __cublasZmatinvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZmatinvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex* const*, int, cuDoubleComplex* const*, int, int*, int) noexcept nogil>__cublasZmatinvBatched)(
        handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t _cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgeqrfBatched
    _check_or_init_cublas()
    if __cublasSgeqrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgeqrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, float* const*, int, float* const*, int*, int) noexcept nogil>__cublasSgeqrfBatched)(
        handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t _cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double* const Aarray[], int lda, double* const TauArray[], int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgeqrfBatched
    _check_or_init_cublas()
    if __cublasDgeqrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgeqrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, double* const*, int, double* const*, int*, int) noexcept nogil>__cublasDgeqrfBatched)(
        handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t _cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex* const Aarray[], int lda, cuComplex* const TauArray[], int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgeqrfBatched
    _check_or_init_cublas()
    if __cublasCgeqrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgeqrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, cuComplex* const*, int, cuComplex* const*, int*, int) noexcept nogil>__cublasCgeqrfBatched)(
        handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t _cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int* info, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgeqrfBatched
    _check_or_init_cublas()
    if __cublasZgeqrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgeqrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, cuDoubleComplex* const*, int, cuDoubleComplex* const*, int*, int) noexcept nogil>__cublasZgeqrfBatched)(
        handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t _cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda, float* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgelsBatched
    _check_or_init_cublas()
    if __cublasSgelsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgelsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, float* const*, int, float* const*, int, int*, int*, int) noexcept nogil>__cublasSgelsBatched)(
        handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t _cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double* const Aarray[], int lda, double* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgelsBatched
    _check_or_init_cublas()
    if __cublasDgelsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgelsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, double* const*, int, double* const*, int, int*, int*, int) noexcept nogil>__cublasDgelsBatched)(
        handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t _cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex* const Aarray[], int lda, cuComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgelsBatched
    _check_or_init_cublas()
    if __cublasCgelsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgelsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, cuComplex* const*, int, cuComplex* const*, int, int*, int*, int) noexcept nogil>__cublasCgelsBatched)(
        handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t _cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgelsBatched
    _check_or_init_cublas()
    if __cublasZgelsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgelsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, cuDoubleComplex* const*, int, cuDoubleComplex* const*, int, int*, int*, int) noexcept nogil>__cublasZgelsBatched)(
        handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t _cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSdgmm
    _check_or_init_cublas()
    if __cublasSdgmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSdgmm is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const float*, int, const float*, int, float*, int) noexcept nogil>__cublasSdgmm)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDdgmm
    _check_or_init_cublas()
    if __cublasDdgmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDdgmm is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const double*, int, const double*, int, double*, int) noexcept nogil>__cublasDdgmm)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCdgmm
    _check_or_init_cublas()
    if __cublasCdgmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdgmm is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) noexcept nogil>__cublasCdgmm)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C, int ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZdgmm
    _check_or_init_cublas()
    if __cublasZdgmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdgmm is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) noexcept nogil>__cublasZdgmm)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* AP, float* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStpttr
    _check_or_init_cublas()
    if __cublasStpttr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStpttr is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, float*, int) noexcept nogil>__cublasStpttr)(
        handle, uplo, n, AP, A, lda)


cdef cublasStatus_t _cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* AP, double* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtpttr
    _check_or_init_cublas()
    if __cublasDtpttr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtpttr is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, double*, int) noexcept nogil>__cublasDtpttr)(
        handle, uplo, n, AP, A, lda)


cdef cublasStatus_t _cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* AP, cuComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtpttr
    _check_or_init_cublas()
    if __cublasCtpttr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtpttr is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, cuComplex*, int) noexcept nogil>__cublasCtpttr)(
        handle, uplo, n, AP, A, lda)


cdef cublasStatus_t _cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* AP, cuDoubleComplex* A, int lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtpttr
    _check_or_init_cublas()
    if __cublasZtpttr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtpttr is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cublasZtpttr)(
        handle, uplo, n, AP, A, lda)


cdef cublasStatus_t _cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, float* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrttp
    _check_or_init_cublas()
    if __cublasStrttp == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrttp is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, int, float*) noexcept nogil>__cublasStrttp)(
        handle, uplo, n, A, lda, AP)


cdef cublasStatus_t _cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, double* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrttp
    _check_or_init_cublas()
    if __cublasDtrttp == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrttp is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, int, double*) noexcept nogil>__cublasDtrttp)(
        handle, uplo, n, A, lda, AP)


cdef cublasStatus_t _cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, cuComplex* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrttp
    _check_or_init_cublas()
    if __cublasCtrttp == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrttp is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, int, cuComplex*) noexcept nogil>__cublasCtrttp)(
        handle, uplo, n, A, lda, AP)


cdef cublasStatus_t _cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrttp
    _check_or_init_cublas()
    if __cublasZtrttp == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrttp is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, int, cuDoubleComplex*) noexcept nogil>__cublasZtrttp)(
        handle, uplo, n, A, lda, AP)


cdef cublasStatus_t _cublasGetSmCountTarget(cublasHandle_t handle, int* smCountTarget) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetSmCountTarget
    _check_or_init_cublas()
    if __cublasGetSmCountTarget == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetSmCountTarget is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int*) noexcept nogil>__cublasGetSmCountTarget)(
        handle, smCountTarget)


cdef cublasStatus_t _cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetSmCountTarget
    _check_or_init_cublas()
    if __cublasSetSmCountTarget == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetSmCountTarget is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int) noexcept nogil>__cublasSetSmCountTarget)(
        handle, smCountTarget)


cdef const char* _cublasGetStatusName(cublasStatus_t status) except?NULL nogil:
    global __cublasGetStatusName
    _check_or_init_cublas()
    if __cublasGetStatusName == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetStatusName is not found")
    return (<const char* (*)(cublasStatus_t) noexcept nogil>__cublasGetStatusName)(
        status)


cdef const char* _cublasGetStatusString(cublasStatus_t status) except?NULL nogil:
    global __cublasGetStatusString
    _check_or_init_cublas()
    if __cublasGetStatusString == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetStatusString is not found")
    return (<const char* (*)(cublasStatus_t) noexcept nogil>__cublasGetStatusString)(
        status)


cdef cublasStatus_t _cublasSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* const Aarray[], int lda, const float* const xarray[], int incx, const float* beta, float* const yarray[], int incy, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemvBatched
    _check_or_init_cublas()
    if __cublasSgemvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const float*, const float* const*, int, const float* const*, int, const float*, float* const*, int, int) noexcept nogil>__cublasSgemvBatched)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasDgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* const Aarray[], int lda, const double* const xarray[], int incx, const double* beta, double* const yarray[], int incy, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemvBatched
    _check_or_init_cublas()
    if __cublasDgemvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const double*, const double* const*, int, const double* const*, int, const double*, double* const*, int, int) noexcept nogil>__cublasDgemvBatched)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasCgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const xarray[], int incx, const cuComplex* beta, cuComplex* const yarray[], int incy, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemvBatched
    _check_or_init_cublas()
    if __cublasCgemvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex* const*, int, const cuComplex* const*, int, const cuComplex*, cuComplex* const*, int, int) noexcept nogil>__cublasCgemvBatched)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasZgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const xarray[], int incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int incy, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemvBatched
    _check_or_init_cublas()
    if __cublasZgemvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex* const*, int, const cuDoubleComplex* const*, int, const cuDoubleComplex*, cuDoubleComplex* const*, int, int) noexcept nogil>__cublasZgemvBatched)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, long long int strideA, const float* x, int incx, long long int stridex, const float* beta, float* y, int incy, long long int stridey, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemvStridedBatched
    _check_or_init_cublas()
    if __cublasSgemvStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemvStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const float*, const float*, int, long long int, const float*, int, long long int, const float*, float*, int, long long int, int) noexcept nogil>__cublasSgemvStridedBatched)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasDgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, long long int strideA, const double* x, int incx, long long int stridex, const double* beta, double* y, int incy, long long int stridey, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemvStridedBatched
    _check_or_init_cublas()
    if __cublasDgemvStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemvStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const double*, const double*, int, long long int, const double*, int, long long int, const double*, double*, int, long long int, int) noexcept nogil>__cublasDgemvStridedBatched)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasCgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* x, int incx, long long int stridex, const cuComplex* beta, cuComplex* y, int incy, long long int stridey, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemvStridedBatched
    _check_or_init_cublas()
    if __cublasCgemvStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemvStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, long long int, const cuComplex*, int, long long int, const cuComplex*, cuComplex*, int, long long int, int) noexcept nogil>__cublasCgemvStridedBatched)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasZgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* x, int incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy, long long int stridey, int batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemvStridedBatched
    _check_or_init_cublas()
    if __cublasZgemvStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemvStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, long long int, const cuDoubleComplex*, int, long long int, const cuDoubleComplex*, cuDoubleComplex*, int, long long int, int) noexcept nogil>__cublasZgemvStridedBatched)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasSetVector_64(int64_t n, int64_t elemSize, const void* x, int64_t incx, void* devicePtr, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetVector_64
    _check_or_init_cublas()
    if __cublasSetVector_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetVector_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, const void*, int64_t, void*, int64_t) noexcept nogil>__cublasSetVector_64)(
        n, elemSize, x, incx, devicePtr, incy)


cdef cublasStatus_t _cublasGetVector_64(int64_t n, int64_t elemSize, const void* x, int64_t incx, void* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetVector_64
    _check_or_init_cublas()
    if __cublasGetVector_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetVector_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, const void*, int64_t, void*, int64_t) noexcept nogil>__cublasGetVector_64)(
        n, elemSize, x, incx, y, incy)


cdef cublasStatus_t _cublasSetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetMatrix_64
    _check_or_init_cublas()
    if __cublasSetMatrix_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetMatrix_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, int64_t, const void*, int64_t, void*, int64_t) noexcept nogil>__cublasSetMatrix_64)(
        rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t _cublasGetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetMatrix_64
    _check_or_init_cublas()
    if __cublasGetMatrix_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetMatrix_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, int64_t, const void*, int64_t, void*, int64_t) noexcept nogil>__cublasGetMatrix_64)(
        rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t _cublasSetVectorAsync_64(int64_t n, int64_t elemSize, const void* hostPtr, int64_t incx, void* devicePtr, int64_t incy, cudaStream_t stream) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetVectorAsync_64
    _check_or_init_cublas()
    if __cublasSetVectorAsync_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetVectorAsync_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, const void*, int64_t, void*, int64_t, cudaStream_t) noexcept nogil>__cublasSetVectorAsync_64)(
        n, elemSize, hostPtr, incx, devicePtr, incy, stream)


cdef cublasStatus_t _cublasGetVectorAsync_64(int64_t n, int64_t elemSize, const void* devicePtr, int64_t incx, void* hostPtr, int64_t incy, cudaStream_t stream) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetVectorAsync_64
    _check_or_init_cublas()
    if __cublasGetVectorAsync_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetVectorAsync_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, const void*, int64_t, void*, int64_t, cudaStream_t) noexcept nogil>__cublasGetVectorAsync_64)(
        n, elemSize, devicePtr, incx, hostPtr, incy, stream)


cdef cublasStatus_t _cublasSetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb, cudaStream_t stream) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetMatrixAsync_64
    _check_or_init_cublas()
    if __cublasSetMatrixAsync_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetMatrixAsync_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, int64_t, const void*, int64_t, void*, int64_t, cudaStream_t) noexcept nogil>__cublasSetMatrixAsync_64)(
        rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t _cublasGetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb, cudaStream_t stream) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetMatrixAsync_64
    _check_or_init_cublas()
    if __cublasGetMatrixAsync_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetMatrixAsync_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, int64_t, const void*, int64_t, void*, int64_t, cudaStream_t) noexcept nogil>__cublasGetMatrixAsync_64)(
        rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t _cublasNrm2Ex_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* result, cudaDataType resultType, cudaDataType executionType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasNrm2Ex_64
    _check_or_init_cublas()
    if __cublasNrm2Ex_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasNrm2Ex_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, void*, cudaDataType, cudaDataType) noexcept nogil>__cublasNrm2Ex_64)(
        handle, n, x, xType, incx, result, resultType, executionType)


cdef cublasStatus_t _cublasSnrm2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSnrm2_v2_64
    _check_or_init_cublas()
    if __cublasSnrm2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSnrm2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, float*) noexcept nogil>__cublasSnrm2_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDnrm2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDnrm2_v2_64
    _check_or_init_cublas()
    if __cublasDnrm2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDnrm2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, double*) noexcept nogil>__cublasDnrm2_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasScnrm2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasScnrm2_v2_64
    _check_or_init_cublas()
    if __cublasScnrm2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScnrm2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, float*) noexcept nogil>__cublasScnrm2_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDznrm2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDznrm2_v2_64
    _check_or_init_cublas()
    if __cublasDznrm2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDznrm2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, double*) noexcept nogil>__cublasDznrm2_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDotEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDotEx_64
    _check_or_init_cublas()
    if __cublasDotEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDotEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, void*, cudaDataType, cudaDataType) noexcept nogil>__cublasDotEx_64)(
        handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t _cublasDotcEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDotcEx_64
    _check_or_init_cublas()
    if __cublasDotcEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDotcEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, void*, cudaDataType, cudaDataType) noexcept nogil>__cublasDotcEx_64)(
        handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t _cublasSdot_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSdot_v2_64
    _check_or_init_cublas()
    if __cublasSdot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSdot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, const float*, int64_t, float*) noexcept nogil>__cublasSdot_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasDdot_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDdot_v2_64
    _check_or_init_cublas()
    if __cublasDdot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDdot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, const double*, int64_t, double*) noexcept nogil>__cublasDdot_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasCdotu_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCdotu_v2_64
    _check_or_init_cublas()
    if __cublasCdotu_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdotu_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*) noexcept nogil>__cublasCdotu_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasCdotc_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCdotc_v2_64
    _check_or_init_cublas()
    if __cublasCdotc_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdotc_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*) noexcept nogil>__cublasCdotc_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasZdotu_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZdotu_v2_64
    _check_or_init_cublas()
    if __cublasZdotu_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdotu_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*) noexcept nogil>__cublasZdotu_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasZdotc_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZdotc_v2_64
    _check_or_init_cublas()
    if __cublasZdotc_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdotc_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*) noexcept nogil>__cublasZdotc_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasScalEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int64_t incx, cudaDataType executionType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasScalEx_64
    _check_or_init_cublas()
    if __cublasScalEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScalEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, void*, cudaDataType, int64_t, cudaDataType) noexcept nogil>__cublasScalEx_64)(
        handle, n, alpha, alphaType, x, xType, incx, executionType)


cdef cublasStatus_t _cublasSscal_64(cublasHandle_t handle, int64_t n, const float* alpha, float* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSscal_v2_64
    _check_or_init_cublas()
    if __cublasSscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasDscal_64(cublasHandle_t handle, int64_t n, const double* alpha, double* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDscal_v2_64
    _check_or_init_cublas()
    if __cublasDscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasCscal_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, cuComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCscal_v2_64
    _check_or_init_cublas()
    if __cublasCscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasCsscal_64(cublasHandle_t handle, int64_t n, const float* alpha, cuComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsscal_v2_64
    _check_or_init_cublas()
    if __cublasCsscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, cuComplex*, int64_t) noexcept nogil>__cublasCsscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasZscal_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZscal_v2_64
    _check_or_init_cublas()
    if __cublasZscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasZdscal_64(cublasHandle_t handle, int64_t n, const double* alpha, cuDoubleComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZdscal_v2_64
    _check_or_init_cublas()
    if __cublasZdscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZdscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasAxpyEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, cudaDataType executiontype) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasAxpyEx_64
    _check_or_init_cublas()
    if __cublasAxpyEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasAxpyEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, const void*, cudaDataType, int64_t, void*, cudaDataType, int64_t, cudaDataType) noexcept nogil>__cublasAxpyEx_64)(
        handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)


cdef cublasStatus_t _cublasSaxpy_64(cublasHandle_t handle, int64_t n, const float* alpha, const float* x, int64_t incx, float* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSaxpy_v2_64
    _check_or_init_cublas()
    if __cublasSaxpy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSaxpy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasSaxpy_v2_64)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasDaxpy_64(cublasHandle_t handle, int64_t n, const double* alpha, const double* x, int64_t incx, double* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDaxpy_v2_64
    _check_or_init_cublas()
    if __cublasDaxpy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDaxpy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDaxpy_v2_64)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasCaxpy_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCaxpy_v2_64
    _check_or_init_cublas()
    if __cublasCaxpy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCaxpy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCaxpy_v2_64)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasZaxpy_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZaxpy_v2_64
    _check_or_init_cublas()
    if __cublasZaxpy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZaxpy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZaxpy_v2_64)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasCopyEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCopyEx_64
    _check_or_init_cublas()
    if __cublasCopyEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCopyEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, void*, cudaDataType, int64_t) noexcept nogil>__cublasCopyEx_64)(
        handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t _cublasScopy_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasScopy_v2_64
    _check_or_init_cublas()
    if __cublasScopy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScopy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasScopy_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasDcopy_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDcopy_v2_64
    _check_or_init_cublas()
    if __cublasDcopy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDcopy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDcopy_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasCcopy_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCcopy_v2_64
    _check_or_init_cublas()
    if __cublasCcopy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCcopy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCcopy_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasZcopy_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZcopy_v2_64
    _check_or_init_cublas()
    if __cublasZcopy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZcopy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZcopy_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasSswap_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSswap_v2_64
    _check_or_init_cublas()
    if __cublasSswap_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSswap_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, float*, int64_t, float*, int64_t) noexcept nogil>__cublasSswap_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasDswap_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDswap_v2_64
    _check_or_init_cublas()
    if __cublasDswap_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDswap_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, double*, int64_t, double*, int64_t) noexcept nogil>__cublasDswap_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasCswap_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCswap_v2_64
    _check_or_init_cublas()
    if __cublasCswap_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCswap_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCswap_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasZswap_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZswap_v2_64
    _check_or_init_cublas()
    if __cublasZswap_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZswap_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZswap_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasSwapEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSwapEx_64
    _check_or_init_cublas()
    if __cublasSwapEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSwapEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, void*, cudaDataType, int64_t, void*, cudaDataType, int64_t) noexcept nogil>__cublasSwapEx_64)(
        handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t _cublasIsamax_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIsamax_v2_64
    _check_or_init_cublas()
    if __cublasIsamax_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIsamax_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, int64_t*) noexcept nogil>__cublasIsamax_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIdamax_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIdamax_v2_64
    _check_or_init_cublas()
    if __cublasIdamax_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIdamax_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, int64_t*) noexcept nogil>__cublasIdamax_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIcamax_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIcamax_v2_64
    _check_or_init_cublas()
    if __cublasIcamax_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIcamax_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, int64_t*) noexcept nogil>__cublasIcamax_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIzamax_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIzamax_v2_64
    _check_or_init_cublas()
    if __cublasIzamax_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIzamax_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, int64_t*) noexcept nogil>__cublasIzamax_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIamaxEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, int64_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIamaxEx_64
    _check_or_init_cublas()
    if __cublasIamaxEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIamaxEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, int64_t*) noexcept nogil>__cublasIamaxEx_64)(
        handle, n, x, xType, incx, result)


cdef cublasStatus_t _cublasIsamin_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIsamin_v2_64
    _check_or_init_cublas()
    if __cublasIsamin_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIsamin_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, int64_t*) noexcept nogil>__cublasIsamin_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIdamin_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIdamin_v2_64
    _check_or_init_cublas()
    if __cublasIdamin_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIdamin_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, int64_t*) noexcept nogil>__cublasIdamin_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIcamin_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIcamin_v2_64
    _check_or_init_cublas()
    if __cublasIcamin_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIcamin_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, int64_t*) noexcept nogil>__cublasIcamin_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIzamin_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIzamin_v2_64
    _check_or_init_cublas()
    if __cublasIzamin_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIzamin_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, int64_t*) noexcept nogil>__cublasIzamin_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIaminEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, int64_t* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasIaminEx_64
    _check_or_init_cublas()
    if __cublasIaminEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIaminEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, int64_t*) noexcept nogil>__cublasIaminEx_64)(
        handle, n, x, xType, incx, result)


cdef cublasStatus_t _cublasAsumEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* result, cudaDataType resultType, cudaDataType executiontype) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasAsumEx_64
    _check_or_init_cublas()
    if __cublasAsumEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasAsumEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, void*, cudaDataType, cudaDataType) noexcept nogil>__cublasAsumEx_64)(
        handle, n, x, xType, incx, result, resultType, executiontype)


cdef cublasStatus_t _cublasSasum_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSasum_v2_64
    _check_or_init_cublas()
    if __cublasSasum_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSasum_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, float*) noexcept nogil>__cublasSasum_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDasum_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDasum_v2_64
    _check_or_init_cublas()
    if __cublasDasum_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDasum_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, double*) noexcept nogil>__cublasDasum_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasScasum_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasScasum_v2_64
    _check_or_init_cublas()
    if __cublasScasum_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScasum_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, float*) noexcept nogil>__cublasScasum_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDzasum_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDzasum_v2_64
    _check_or_init_cublas()
    if __cublasDzasum_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDzasum_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, double*) noexcept nogil>__cublasDzasum_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasSrot_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* c, const float* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSrot_v2_64
    _check_or_init_cublas()
    if __cublasSrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, float*, int64_t, float*, int64_t, const float*, const float*) noexcept nogil>__cublasSrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasDrot_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* c, const double* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDrot_v2_64
    _check_or_init_cublas()
    if __cublasDrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, double*, int64_t, double*, int64_t, const double*, const double*) noexcept nogil>__cublasDrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasCrot_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const cuComplex* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCrot_v2_64
    _check_or_init_cublas()
    if __cublasCrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuComplex*, int64_t, cuComplex*, int64_t, const float*, const cuComplex*) noexcept nogil>__cublasCrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasCsrot_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const float* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsrot_v2_64
    _check_or_init_cublas()
    if __cublasCsrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuComplex*, int64_t, cuComplex*, int64_t, const float*, const float*) noexcept nogil>__cublasCsrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasZrot_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const cuDoubleComplex* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZrot_v2_64
    _check_or_init_cublas()
    if __cublasZrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t, const double*, const cuDoubleComplex*) noexcept nogil>__cublasZrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasZdrot_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const double* s) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZdrot_v2_64
    _check_or_init_cublas()
    if __cublasZdrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t, const double*, const double*) noexcept nogil>__cublasZdrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasRotEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasRotEx_64
    _check_or_init_cublas()
    if __cublasRotEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, void*, cudaDataType, int64_t, void*, cudaDataType, int64_t, const void*, const void*, cudaDataType, cudaDataType) noexcept nogil>__cublasRotEx_64)(
        handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)


cdef cublasStatus_t _cublasSrotm_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* param) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSrotm_v2_64
    _check_or_init_cublas()
    if __cublasSrotm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrotm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, float*, int64_t, float*, int64_t, const float*) noexcept nogil>__cublasSrotm_v2_64)(
        handle, n, x, incx, y, incy, param)


cdef cublasStatus_t _cublasDrotm_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* param) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDrotm_v2_64
    _check_or_init_cublas()
    if __cublasDrotm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrotm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, double*, int64_t, double*, int64_t, const double*) noexcept nogil>__cublasDrotm_v2_64)(
        handle, n, x, incx, y, incy, param)


cdef cublasStatus_t _cublasRotmEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, const void* param, cudaDataType paramType, cudaDataType executiontype) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasRotmEx_64
    _check_or_init_cublas()
    if __cublasRotmEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotmEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, void*, cudaDataType, int64_t, void*, cudaDataType, int64_t, const void*, cudaDataType, cudaDataType) noexcept nogil>__cublasRotmEx_64)(
        handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype)


cdef cublasStatus_t _cublasSgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemv_v2_64
    _check_or_init_cublas()
    if __cublasSgemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSgemv_v2_64)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemv_v2_64
    _check_or_init_cublas()
    if __cublasDgemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDgemv_v2_64)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemv_v2_64
    _check_or_init_cublas()
    if __cublasCgemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCgemv_v2_64)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemv_v2_64
    _check_or_init_cublas()
    if __cublasZgemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZgemv_v2_64)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgbmv_v2_64
    _check_or_init_cublas()
    if __cublasSgbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSgbmv_v2_64)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgbmv_v2_64
    _check_or_init_cublas()
    if __cublasDgbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDgbmv_v2_64)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgbmv_v2_64
    _check_or_init_cublas()
    if __cublasCgbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCgbmv_v2_64)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgbmv_v2_64
    _check_or_init_cublas()
    if __cublasZgbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZgbmv_v2_64)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasStrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrmv_v2_64
    _check_or_init_cublas()
    if __cublasStrmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasStrmv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasDtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrmv_v2_64
    _check_or_init_cublas()
    if __cublasDtrmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDtrmv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasCtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrmv_v2_64
    _check_or_init_cublas()
    if __cublasCtrmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCtrmv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasZtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrmv_v2_64
    _check_or_init_cublas()
    if __cublasZtrmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZtrmv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasStbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStbmv_v2_64
    _check_or_init_cublas()
    if __cublasStbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasStbmv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasDtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtbmv_v2_64
    _check_or_init_cublas()
    if __cublasDtbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDtbmv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasCtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtbmv_v2_64
    _check_or_init_cublas()
    if __cublasCtbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCtbmv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasZtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtbmv_v2_64
    _check_or_init_cublas()
    if __cublasZtbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZtbmv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasStpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStpmv_v2_64
    _check_or_init_cublas()
    if __cublasStpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasStpmv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasDtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtpmv_v2_64
    _check_or_init_cublas()
    if __cublasDtpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDtpmv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasCtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtpmv_v2_64
    _check_or_init_cublas()
    if __cublasCtpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCtpmv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasZtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtpmv_v2_64
    _check_or_init_cublas()
    if __cublasZtpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZtpmv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasStrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrsv_v2_64
    _check_or_init_cublas()
    if __cublasStrsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasStrsv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasDtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrsv_v2_64
    _check_or_init_cublas()
    if __cublasDtrsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDtrsv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasCtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrsv_v2_64
    _check_or_init_cublas()
    if __cublasCtrsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCtrsv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasZtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrsv_v2_64
    _check_or_init_cublas()
    if __cublasZtrsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZtrsv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasStpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStpsv_v2_64
    _check_or_init_cublas()
    if __cublasStpsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStpsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasStpsv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasDtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtpsv_v2_64
    _check_or_init_cublas()
    if __cublasDtpsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtpsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDtpsv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasCtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtpsv_v2_64
    _check_or_init_cublas()
    if __cublasCtpsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtpsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCtpsv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasZtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtpsv_v2_64
    _check_or_init_cublas()
    if __cublasZtpsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtpsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZtpsv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasStbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStbsv_v2_64
    _check_or_init_cublas()
    if __cublasStbsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStbsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasStbsv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasDtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtbsv_v2_64
    _check_or_init_cublas()
    if __cublasDtbsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtbsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDtbsv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasCtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtbsv_v2_64
    _check_or_init_cublas()
    if __cublasCtbsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtbsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCtbsv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasZtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtbsv_v2_64
    _check_or_init_cublas()
    if __cublasZtbsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtbsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZtbsv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasSsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsymv_v2_64
    _check_or_init_cublas()
    if __cublasSsymv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsymv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSsymv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsymv_v2_64
    _check_or_init_cublas()
    if __cublasDsymv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsymv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDsymv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsymv_v2_64
    _check_or_init_cublas()
    if __cublasCsymv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsymv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCsymv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsymv_v2_64
    _check_or_init_cublas()
    if __cublasZsymv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsymv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZsymv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChemv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChemv_v2_64
    _check_or_init_cublas()
    if __cublasChemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasChemv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhemv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhemv_v2_64
    _check_or_init_cublas()
    if __cublasZhemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZhemv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSsbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsbmv_v2_64
    _check_or_init_cublas()
    if __cublasSsbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSsbmv_v2_64)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDsbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsbmv_v2_64
    _check_or_init_cublas()
    if __cublasDsbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDsbmv_v2_64)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChbmv_v2_64
    _check_or_init_cublas()
    if __cublasChbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasChbmv_v2_64)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhbmv_v2_64
    _check_or_init_cublas()
    if __cublasZhbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZhbmv_v2_64)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSspmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* AP, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSspmv_v2_64
    _check_or_init_cublas()
    if __cublasSspmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, const float*, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSspmv_v2_64)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDspmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* AP, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDspmv_v2_64
    _check_or_init_cublas()
    if __cublasDspmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, const double*, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDspmv_v2_64)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChpmv_v2_64
    _check_or_init_cublas()
    if __cublasChpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasChpmv_v2_64)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhpmv_v2_64
    _check_or_init_cublas()
    if __cublasZhpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZhpmv_v2_64)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSger_64(cublasHandle_t handle, int64_t m, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSger_v2_64
    _check_or_init_cublas()
    if __cublasSger_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSger_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasSger_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasDger_64(cublasHandle_t handle, int64_t m, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDger_v2_64
    _check_or_init_cublas()
    if __cublasDger_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDger_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDger_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCgeru_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgeru_v2_64
    _check_or_init_cublas()
    if __cublasCgeru_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgeru_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCgeru_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCgerc_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgerc_v2_64
    _check_or_init_cublas()
    if __cublasCgerc_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgerc_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCgerc_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZgeru_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgeru_v2_64
    _check_or_init_cublas()
    if __cublasZgeru_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgeru_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZgeru_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZgerc_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgerc_v2_64
    _check_or_init_cublas()
    if __cublasZgerc_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgerc_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZgerc_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasSsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsyr_v2_64
    _check_or_init_cublas()
    if __cublasSsyr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasSsyr_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasDsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsyr_v2_64
    _check_or_init_cublas()
    if __cublasDsyr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDsyr_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasCsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyr_v2_64
    _check_or_init_cublas()
    if __cublasCsyr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCsyr_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasZsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsyr_v2_64
    _check_or_init_cublas()
    if __cublasZsyr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZsyr_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasCher_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCher_v2_64
    _check_or_init_cublas()
    if __cublasCher_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCher_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasZher_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZher_v2_64
    _check_or_init_cublas()
    if __cublasZher_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZher_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasSspr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSspr_v2_64
    _check_or_init_cublas()
    if __cublasSspr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, int64_t, float*) noexcept nogil>__cublasSspr_v2_64)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasDspr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDspr_v2_64
    _check_or_init_cublas()
    if __cublasDspr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, int64_t, double*) noexcept nogil>__cublasDspr_v2_64)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasChpr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const cuComplex* x, int64_t incx, cuComplex* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChpr_v2_64
    _check_or_init_cublas()
    if __cublasChpr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const cuComplex*, int64_t, cuComplex*) noexcept nogil>__cublasChpr_v2_64)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasZhpr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhpr_v2_64
    _check_or_init_cublas()
    if __cublasZhpr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const cuDoubleComplex*, int64_t, cuDoubleComplex*) noexcept nogil>__cublasZhpr_v2_64)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasSsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsyr2_v2_64
    _check_or_init_cublas()
    if __cublasSsyr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasSsyr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasDsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsyr2_v2_64
    _check_or_init_cublas()
    if __cublasDsyr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDsyr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyr2_v2_64
    _check_or_init_cublas()
    if __cublasCsyr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCsyr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsyr2_v2_64
    _check_or_init_cublas()
    if __cublasZsyr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZsyr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCher2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCher2_v2_64
    _check_or_init_cublas()
    if __cublasCher2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCher2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZher2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZher2_v2_64
    _check_or_init_cublas()
    if __cublasZher2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZher2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasSspr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSspr2_v2_64
    _check_or_init_cublas()
    if __cublasSspr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, float*) noexcept nogil>__cublasSspr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasDspr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDspr2_v2_64
    _check_or_init_cublas()
    if __cublasDspr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, double*) noexcept nogil>__cublasDspr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasChpr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChpr2_v2_64
    _check_or_init_cublas()
    if __cublasChpr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*) noexcept nogil>__cublasChpr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasZhpr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* AP) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhpr2_v2_64
    _check_or_init_cublas()
    if __cublasZhpr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*) noexcept nogil>__cublasZhpr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasSgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* const Aarray[], int64_t lda, const float* const xarray[], int64_t incx, const float* beta, float* const yarray[], int64_t incy, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemvBatched_64
    _check_or_init_cublas()
    if __cublasSgemvBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemvBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const float*, const float* const*, int64_t, const float* const*, int64_t, const float*, float* const*, int64_t, int64_t) noexcept nogil>__cublasSgemvBatched_64)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasDgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* const Aarray[], int64_t lda, const double* const xarray[], int64_t incx, const double* beta, double* const yarray[], int64_t incy, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemvBatched_64
    _check_or_init_cublas()
    if __cublasDgemvBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemvBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const double*, const double* const*, int64_t, const double* const*, int64_t, const double*, double* const*, int64_t, int64_t) noexcept nogil>__cublasDgemvBatched_64)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasCgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const xarray[], int64_t incx, const cuComplex* beta, cuComplex* const yarray[], int64_t incy, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemvBatched_64
    _check_or_init_cublas()
    if __cublasCgemvBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemvBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex* const*, int64_t, const cuComplex* const*, int64_t, const cuComplex*, cuComplex* const*, int64_t, int64_t) noexcept nogil>__cublasCgemvBatched_64)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasZgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const xarray[], int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int64_t incy, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemvBatched_64
    _check_or_init_cublas()
    if __cublasZgemvBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemvBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex* const*, int64_t, const cuDoubleComplex* const*, int64_t, const cuDoubleComplex*, cuDoubleComplex* const*, int64_t, int64_t) noexcept nogil>__cublasZgemvBatched_64)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasSgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* x, int64_t incx, long long int stridex, const float* beta, float* y, int64_t incy, long long int stridey, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemvStridedBatched_64
    _check_or_init_cublas()
    if __cublasSgemvStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemvStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, long long int, const float*, int64_t, long long int, const float*, float*, int64_t, long long int, int64_t) noexcept nogil>__cublasSgemvStridedBatched_64)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasDgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* x, int64_t incx, long long int stridex, const double* beta, double* y, int64_t incy, long long int stridey, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemvStridedBatched_64
    _check_or_init_cublas()
    if __cublasDgemvStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemvStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, long long int, const double*, int64_t, long long int, const double*, double*, int64_t, long long int, int64_t) noexcept nogil>__cublasDgemvStridedBatched_64)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasCgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* x, int64_t incx, long long int stridex, const cuComplex* beta, cuComplex* y, int64_t incy, long long int stridey, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemvStridedBatched_64
    _check_or_init_cublas()
    if __cublasCgemvStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemvStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, long long int, const cuComplex*, int64_t, long long int, const cuComplex*, cuComplex*, int64_t, long long int, int64_t) noexcept nogil>__cublasCgemvStridedBatched_64)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasZgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* x, int64_t incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy, long long int stridey, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemvStridedBatched_64
    _check_or_init_cublas()
    if __cublasZgemvStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemvStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, long long int, const cuDoubleComplex*, int64_t, long long int, const cuDoubleComplex*, cuDoubleComplex*, int64_t, long long int, int64_t) noexcept nogil>__cublasZgemvStridedBatched_64)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasSgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemm_v2_64
    _check_or_init_cublas()
    if __cublasSgemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSgemm_v2_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemm_v2_64
    _check_or_init_cublas()
    if __cublasDgemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDgemm_v2_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemm_v2_64
    _check_or_init_cublas()
    if __cublasCgemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCgemm_v2_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemm3m_64
    _check_or_init_cublas()
    if __cublasCgemm3m_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3m_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCgemm3m_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm3mEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemm3mEx_64
    _check_or_init_cublas()
    if __cublasCgemm3mEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, const cuComplex*, void*, cudaDataType, int64_t) noexcept nogil>__cublasCgemm3mEx_64)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasZgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemm_v2_64
    _check_or_init_cublas()
    if __cublasZgemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZgemm_v2_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemm3m_64
    _check_or_init_cublas()
    if __cublasZgemm3m_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemm3m_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZgemm3m_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemmEx_64
    _check_or_init_cublas()
    if __cublasSgemmEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const float*, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, const float*, void*, cudaDataType, int64_t) noexcept nogil>__cublasSgemmEx_64)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasGemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGemmEx_64
    _check_or_init_cublas()
    if __cublasGemmEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const void*, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, const void*, void*, cudaDataType, int64_t, cublasComputeType_t, cublasGemmAlgo_t) noexcept nogil>__cublasGemmEx_64)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)


cdef cublasStatus_t _cublasCgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemmEx_64
    _check_or_init_cublas()
    if __cublasCgemmEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, const cuComplex*, void*, cudaDataType, int64_t) noexcept nogil>__cublasCgemmEx_64)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasSsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* beta, float* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsyrk_v2_64
    _check_or_init_cublas()
    if __cublasSsyrk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyrk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSsyrk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasDsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* beta, double* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsyrk_v2_64
    _check_or_init_cublas()
    if __cublasDsyrk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyrk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDsyrk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyrk_v2_64
    _check_or_init_cublas()
    if __cublasCsyrk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCsyrk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasZsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsyrk_v2_64
    _check_or_init_cublas()
    if __cublasZsyrk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyrk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZsyrk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyrkEx_64
    _check_or_init_cublas()
    if __cublasCsyrkEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrkEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const void*, cudaDataType, int64_t, const cuComplex*, void*, cudaDataType, int64_t) noexcept nogil>__cublasCsyrkEx_64)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCsyrk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyrk3mEx_64
    _check_or_init_cublas()
    if __cublasCsyrk3mEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrk3mEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const void*, cudaDataType, int64_t, const cuComplex*, void*, cudaDataType, int64_t) noexcept nogil>__cublasCsyrk3mEx_64)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCherk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const cuComplex* A, int64_t lda, const float* beta, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCherk_v2_64
    _check_or_init_cublas()
    if __cublasCherk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const cuComplex*, int64_t, const float*, cuComplex*, int64_t) noexcept nogil>__cublasCherk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasZherk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const cuDoubleComplex* A, int64_t lda, const double* beta, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZherk_v2_64
    _check_or_init_cublas()
    if __cublasZherk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZherk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const double*, const cuDoubleComplex*, int64_t, const double*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZherk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCherkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCherkEx_64
    _check_or_init_cublas()
    if __cublasCherkEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherkEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const void*, cudaDataType, int64_t, const float*, void*, cudaDataType, int64_t) noexcept nogil>__cublasCherkEx_64)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCherk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCherk3mEx_64
    _check_or_init_cublas()
    if __cublasCherk3mEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherk3mEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const void*, cudaDataType, int64_t, const float*, void*, cudaDataType, int64_t) noexcept nogil>__cublasCherk3mEx_64)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasSsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsyr2k_v2_64
    _check_or_init_cublas()
    if __cublasSsyr2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSsyr2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsyr2k_v2_64
    _check_or_init_cublas()
    if __cublasDsyr2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDsyr2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyr2k_v2_64
    _check_or_init_cublas()
    if __cublasCsyr2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCsyr2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsyr2k_v2_64
    _check_or_init_cublas()
    if __cublasZsyr2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZsyr2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCher2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const float* beta, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCher2k_v2_64
    _check_or_init_cublas()
    if __cublasCher2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const float*, cuComplex*, int64_t) noexcept nogil>__cublasCher2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZher2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const double* beta, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZher2k_v2_64
    _check_or_init_cublas()
    if __cublasZher2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const double*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZher2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsyrkx_64
    _check_or_init_cublas()
    if __cublasSsyrkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyrkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSsyrkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsyrkx_64
    _check_or_init_cublas()
    if __cublasDsyrkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyrkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDsyrkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsyrkx_64
    _check_or_init_cublas()
    if __cublasCsyrkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCsyrkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsyrkx_64
    _check_or_init_cublas()
    if __cublasZsyrkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyrkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZsyrkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const float* beta, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCherkx_64
    _check_or_init_cublas()
    if __cublasCherkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const float*, cuComplex*, int64_t) noexcept nogil>__cublasCherkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const double* beta, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZherkx_64
    _check_or_init_cublas()
    if __cublasZherkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZherkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const double*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZherkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSsymm_v2_64
    _check_or_init_cublas()
    if __cublasSsymm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsymm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) noexcept nogil>__cublasSsymm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDsymm_v2_64
    _check_or_init_cublas()
    if __cublasDsymm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsymm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) noexcept nogil>__cublasDsymm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCsymm_v2_64
    _check_or_init_cublas()
    if __cublasCsymm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsymm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasCsymm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZsymm_v2_64
    _check_or_init_cublas()
    if __cublasZsymm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsymm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZsymm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasChemm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasChemm_v2_64
    _check_or_init_cublas()
    if __cublasChemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) noexcept nogil>__cublasChemm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZhemm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZhemm_v2_64
    _check_or_init_cublas()
    if __cublasZhemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZhemm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasStrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, float* B, int64_t ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrsm_v2_64
    _check_or_init_cublas()
    if __cublasStrsm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const float*, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasStrsm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasDtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, double* B, int64_t ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrsm_v2_64
    _check_or_init_cublas()
    if __cublasDtrsm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const double*, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDtrsm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasCtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, cuComplex* B, int64_t ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrsm_v2_64
    _check_or_init_cublas()
    if __cublasCtrsm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCtrsm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasZtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* B, int64_t ldb) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrsm_v2_64
    _check_or_init_cublas()
    if __cublasZtrsm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZtrsm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasStrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrmm_v2_64
    _check_or_init_cublas()
    if __cublasStrmm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrmm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasStrmm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasDtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, double* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrmm_v2_64
    _check_or_init_cublas()
    if __cublasDtrmm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrmm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDtrmm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasCtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrmm_v2_64
    _check_or_init_cublas()
    if __cublasCtrmm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrmm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCtrmm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasZtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrmm_v2_64
    _check_or_init_cublas()
    if __cublasZtrmm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrmm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZtrmm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasSgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* const Aarray[], int64_t lda, const float* const Barray[], int64_t ldb, const float* beta, float* const Carray[], int64_t ldc, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemmBatched_64
    _check_or_init_cublas()
    if __cublasSgemmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const float*, const float* const*, int64_t, const float* const*, int64_t, const float*, float* const*, int64_t, int64_t) noexcept nogil>__cublasSgemmBatched_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasDgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* const Aarray[], int64_t lda, const double* const Barray[], int64_t ldb, const double* beta, double* const Carray[], int64_t ldc, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemmBatched_64
    _check_or_init_cublas()
    if __cublasDgemmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const double*, const double* const*, int64_t, const double* const*, int64_t, const double*, double* const*, int64_t, int64_t) noexcept nogil>__cublasDgemmBatched_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasCgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex* beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemmBatched_64
    _check_or_init_cublas()
    if __cublasCgemmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex* const*, int64_t, const cuComplex* const*, int64_t, const cuComplex*, cuComplex* const*, int64_t, int64_t) noexcept nogil>__cublasCgemmBatched_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasCgemm3mBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex* beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemm3mBatched_64
    _check_or_init_cublas()
    if __cublasCgemm3mBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex* const*, int64_t, const cuComplex* const*, int64_t, const cuComplex*, cuComplex* const*, int64_t, int64_t) noexcept nogil>__cublasCgemm3mBatched_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasZgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const Barray[], int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int64_t ldc, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemmBatched_64
    _check_or_init_cublas()
    if __cublasZgemmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex* const*, int64_t, const cuDoubleComplex* const*, int64_t, const cuDoubleComplex*, cuDoubleComplex* const*, int64_t, int64_t) noexcept nogil>__cublasZgemmBatched_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasSgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* B, int64_t ldb, long long int strideB, const float* beta, float* C, int64_t ldc, long long int strideC, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemmStridedBatched_64
    _check_or_init_cublas()
    if __cublasSgemmStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const float*, const float*, int64_t, long long int, const float*, int64_t, long long int, const float*, float*, int64_t, long long int, int64_t) noexcept nogil>__cublasSgemmStridedBatched_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasDgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* B, int64_t ldb, long long int strideB, const double* beta, double* C, int64_t ldc, long long int strideC, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemmStridedBatched_64
    _check_or_init_cublas()
    if __cublasDgemmStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemmStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const double*, const double*, int64_t, long long int, const double*, int64_t, long long int, const double*, double*, int64_t, long long int, int64_t) noexcept nogil>__cublasDgemmStridedBatched_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasCgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* B, int64_t ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemmStridedBatched_64
    _check_or_init_cublas()
    if __cublasCgemmStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, long long int, const cuComplex*, int64_t, long long int, const cuComplex*, cuComplex*, int64_t, long long int, int64_t) noexcept nogil>__cublasCgemmStridedBatched_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasCgemm3mStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* B, int64_t ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgemm3mStridedBatched_64
    _check_or_init_cublas()
    if __cublasCgemm3mStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, long long int, const cuComplex*, int64_t, long long int, const cuComplex*, cuComplex*, int64_t, long long int, int64_t) noexcept nogil>__cublasCgemm3mStridedBatched_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasZgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* B, int64_t ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgemmStridedBatched_64
    _check_or_init_cublas()
    if __cublasZgemmStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemmStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, long long int, const cuDoubleComplex*, int64_t, long long int, const cuDoubleComplex*, cuDoubleComplex*, int64_t, long long int, int64_t) noexcept nogil>__cublasZgemmStridedBatched_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasGemmBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int64_t lda, const void* const Barray[], cudaDataType Btype, int64_t ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int64_t ldc, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGemmBatchedEx_64
    _check_or_init_cublas()
    if __cublasGemmBatchedEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmBatchedEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const void*, const void* const*, cudaDataType, int64_t, const void* const*, cudaDataType, int64_t, const void*, void* const*, cudaDataType, int64_t, int64_t, cublasComputeType_t, cublasGemmAlgo_t) noexcept nogil>__cublasGemmBatchedEx_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)


cdef cublasStatus_t _cublasGemmStridedBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, long long int strideA, const void* B, cudaDataType Btype, int64_t ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, long long int strideC, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGemmStridedBatchedEx_64
    _check_or_init_cublas()
    if __cublasGemmStridedBatchedEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmStridedBatchedEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const void*, const void*, cudaDataType, int64_t, long long int, const void*, cudaDataType, int64_t, long long int, const void*, void*, cudaDataType, int64_t, long long int, int64_t, cublasComputeType_t, cublasGemmAlgo_t) noexcept nogil>__cublasGemmStridedBatchedEx_64)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)


cdef cublasStatus_t _cublasSgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* beta, const float* B, int64_t ldb, float* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgeam_64
    _check_or_init_cublas()
    if __cublasSgeam_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgeam_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasSgeam_64)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasDgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* beta, const double* B, int64_t ldb, double* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgeam_64
    _check_or_init_cublas()
    if __cublasDgeam_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgeam_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDgeam_64)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasCgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, const cuComplex* B, int64_t ldb, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCgeam_64
    _check_or_init_cublas()
    if __cublasCgeam_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgeam_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCgeam_64)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasZgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZgeam_64
    _check_or_init_cublas()
    if __cublasZgeam_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgeam_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZgeam_64)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasStrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* const A[], int64_t lda, float* const B[], int64_t ldb, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasStrsmBatched_64
    _check_or_init_cublas()
    if __cublasStrsmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const float*, const float* const*, int64_t, float* const*, int64_t, int64_t) noexcept nogil>__cublasStrsmBatched_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasDtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* const A[], int64_t lda, double* const B[], int64_t ldb, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDtrsmBatched_64
    _check_or_init_cublas()
    if __cublasDtrsmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const double*, const double* const*, int64_t, double* const*, int64_t, int64_t) noexcept nogil>__cublasDtrsmBatched_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasCtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const A[], int64_t lda, cuComplex* const B[], int64_t ldb, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCtrsmBatched_64
    _check_or_init_cublas()
    if __cublasCtrsmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuComplex*, const cuComplex* const*, int64_t, cuComplex* const*, int64_t, int64_t) noexcept nogil>__cublasCtrsmBatched_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasZtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int64_t lda, cuDoubleComplex* const B[], int64_t ldb, int64_t batchCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZtrsmBatched_64
    _check_or_init_cublas()
    if __cublasZtrsmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex* const*, int64_t, cuDoubleComplex* const*, int64_t, int64_t) noexcept nogil>__cublasZtrsmBatched_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasSdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const float* A, int64_t lda, const float* x, int64_t incx, float* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSdgmm_64
    _check_or_init_cublas()
    if __cublasSdgmm_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSdgmm_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int64_t, int64_t, const float*, int64_t, const float*, int64_t, float*, int64_t) noexcept nogil>__cublasSdgmm_64)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasDdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const double* A, int64_t lda, const double* x, int64_t incx, double* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDdgmm_64
    _check_or_init_cublas()
    if __cublasDdgmm_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDdgmm_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int64_t, int64_t, const double*, int64_t, const double*, int64_t, double*, int64_t) noexcept nogil>__cublasDdgmm_64)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasCdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, cuComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasCdgmm_64
    _check_or_init_cublas()
    if __cublasCdgmm_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdgmm_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int64_t, int64_t, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) noexcept nogil>__cublasCdgmm_64)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasZdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* C, int64_t ldc) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasZdgmm_64
    _check_or_init_cublas()
    if __cublasZdgmm_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdgmm_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int64_t, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) noexcept nogil>__cublasZdgmm_64)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasSgemmGroupedBatched(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const float alpha_array[], const float* const Aarray[], const int lda_array[], const float* const Barray[], const int ldb_array[], const float beta_array[], float* const Carray[], const int ldc_array[], int group_count, const int group_size[]) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemmGroupedBatched
    _check_or_init_cublas()
    if __cublasSgemmGroupedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmGroupedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, const cublasOperation_t*, const cublasOperation_t*, const int*, const int*, const int*, const float*, const float* const*, const int*, const float* const*, const int*, const float*, float* const*, const int*, int, const int*) noexcept nogil>__cublasSgemmGroupedBatched)(
        handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)


cdef cublasStatus_t _cublasSgemmGroupedBatched_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const float alpha_array[], const float* const Aarray[], const int64_t lda_array[], const float* const Barray[], const int64_t ldb_array[], const float beta_array[], float* const Carray[], const int64_t ldc_array[], int64_t group_count, const int64_t group_size[]) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSgemmGroupedBatched_64
    _check_or_init_cublas()
    if __cublasSgemmGroupedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmGroupedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, const cublasOperation_t*, const cublasOperation_t*, const int64_t*, const int64_t*, const int64_t*, const float*, const float* const*, const int64_t*, const float* const*, const int64_t*, const float*, float* const*, const int64_t*, int64_t, const int64_t*) noexcept nogil>__cublasSgemmGroupedBatched_64)(
        handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)


cdef cublasStatus_t _cublasDgemmGroupedBatched(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const double alpha_array[], const double* const Aarray[], const int lda_array[], const double* const Barray[], const int ldb_array[], const double beta_array[], double* const Carray[], const int ldc_array[], int group_count, const int group_size[]) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemmGroupedBatched
    _check_or_init_cublas()
    if __cublasDgemmGroupedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemmGroupedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, const cublasOperation_t*, const cublasOperation_t*, const int*, const int*, const int*, const double*, const double* const*, const int*, const double* const*, const int*, const double*, double* const*, const int*, int, const int*) noexcept nogil>__cublasDgemmGroupedBatched)(
        handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)


cdef cublasStatus_t _cublasDgemmGroupedBatched_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const double alpha_array[], const double* const Aarray[], const int64_t lda_array[], const double* const Barray[], const int64_t ldb_array[], const double beta_array[], double* const Carray[], const int64_t ldc_array[], int64_t group_count, const int64_t group_size[]) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasDgemmGroupedBatched_64
    _check_or_init_cublas()
    if __cublasDgemmGroupedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemmGroupedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, const cublasOperation_t*, const cublasOperation_t*, const int64_t*, const int64_t*, const int64_t*, const double*, const double* const*, const int64_t*, const double* const*, const int64_t*, const double*, double* const*, const int64_t*, int64_t, const int64_t*) noexcept nogil>__cublasDgemmGroupedBatched_64)(
        handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)


cdef cublasStatus_t _cublasGemmGroupedBatchedEx(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const void* alpha_array, const void* const Aarray[], cudaDataType_t Atype, const int lda_array[], const void* const Barray[], cudaDataType_t Btype, const int ldb_array[], const void* beta_array, void* const Carray[], cudaDataType_t Ctype, const int ldc_array[], int group_count, const int group_size[], cublasComputeType_t computeType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGemmGroupedBatchedEx
    _check_or_init_cublas()
    if __cublasGemmGroupedBatchedEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmGroupedBatchedEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, const cublasOperation_t*, const cublasOperation_t*, const int*, const int*, const int*, const void*, const void* const*, cudaDataType_t, const int*, const void* const*, cudaDataType_t, const int*, const void*, void* const*, cudaDataType_t, const int*, int, const int*, cublasComputeType_t) noexcept nogil>__cublasGemmGroupedBatchedEx)(
        handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, Atype, lda_array, Barray, Btype, ldb_array, beta_array, Carray, Ctype, ldc_array, group_count, group_size, computeType)


cdef cublasStatus_t _cublasGemmGroupedBatchedEx_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const void* alpha_array, const void* const Aarray[], cudaDataType_t Atype, const int64_t lda_array[], const void* const Barray[], cudaDataType_t Btype, const int64_t ldb_array[], const void* beta_array, void* const Carray[], cudaDataType_t Ctype, const int64_t ldc_array[], int64_t group_count, const int64_t group_size[], cublasComputeType_t computeType) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGemmGroupedBatchedEx_64
    _check_or_init_cublas()
    if __cublasGemmGroupedBatchedEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmGroupedBatchedEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, const cublasOperation_t*, const cublasOperation_t*, const int64_t*, const int64_t*, const int64_t*, const void*, const void* const*, cudaDataType_t, const int64_t*, const void* const*, cudaDataType_t, const int64_t*, const void*, void* const*, cudaDataType_t, const int64_t*, int64_t, const int64_t*, cublasComputeType_t) noexcept nogil>__cublasGemmGroupedBatchedEx_64)(
        handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, Atype, lda_array, Barray, Btype, ldb_array, beta_array, Carray, Ctype, ldc_array, group_count, group_size, computeType)


cdef cublasStatus_t _cublasGetEmulationStrategy(cublasHandle_t handle, cublasEmulationStrategy_t* emulationStrategy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetEmulationStrategy
    _check_or_init_cublas()
    if __cublasGetEmulationStrategy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetEmulationStrategy is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasEmulationStrategy_t*) noexcept nogil>__cublasGetEmulationStrategy)(
        handle, emulationStrategy)


cdef cublasStatus_t _cublasSetEmulationStrategy(cublasHandle_t handle, cublasEmulationStrategy_t emulationStrategy) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetEmulationStrategy
    _check_or_init_cublas()
    if __cublasSetEmulationStrategy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetEmulationStrategy is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasEmulationStrategy_t) noexcept nogil>__cublasSetEmulationStrategy)(
        handle, emulationStrategy)


cdef cublasStatus_t _cublasGetEmulationSpecialValuesSupport(cublasHandle_t handle, cudaEmulationSpecialValuesSupport* mask) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetEmulationSpecialValuesSupport
    _check_or_init_cublas()
    if __cublasGetEmulationSpecialValuesSupport == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetEmulationSpecialValuesSupport is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cudaEmulationSpecialValuesSupport*) noexcept nogil>__cublasGetEmulationSpecialValuesSupport)(
        handle, mask)


cdef cublasStatus_t _cublasSetEmulationSpecialValuesSupport(cublasHandle_t handle, cudaEmulationSpecialValuesSupport mask) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetEmulationSpecialValuesSupport
    _check_or_init_cublas()
    if __cublasSetEmulationSpecialValuesSupport == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetEmulationSpecialValuesSupport is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cudaEmulationSpecialValuesSupport) noexcept nogil>__cublasSetEmulationSpecialValuesSupport)(
        handle, mask)


cdef cublasStatus_t _cublasGetFixedPointEmulationMantissaControl(cublasHandle_t handle, cudaEmulationMantissaControl* mantissaControl) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetFixedPointEmulationMantissaControl
    _check_or_init_cublas()
    if __cublasGetFixedPointEmulationMantissaControl == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetFixedPointEmulationMantissaControl is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cudaEmulationMantissaControl*) noexcept nogil>__cublasGetFixedPointEmulationMantissaControl)(
        handle, mantissaControl)


cdef cublasStatus_t _cublasSetFixedPointEmulationMantissaControl(cublasHandle_t handle, cudaEmulationMantissaControl mantissaControl) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetFixedPointEmulationMantissaControl
    _check_or_init_cublas()
    if __cublasSetFixedPointEmulationMantissaControl == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetFixedPointEmulationMantissaControl is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cudaEmulationMantissaControl) noexcept nogil>__cublasSetFixedPointEmulationMantissaControl)(
        handle, mantissaControl)


cdef cublasStatus_t _cublasGetFixedPointEmulationMaxMantissaBitCount(cublasHandle_t handle, int* maxMantissaBitCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetFixedPointEmulationMaxMantissaBitCount
    _check_or_init_cublas()
    if __cublasGetFixedPointEmulationMaxMantissaBitCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetFixedPointEmulationMaxMantissaBitCount is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int*) noexcept nogil>__cublasGetFixedPointEmulationMaxMantissaBitCount)(
        handle, maxMantissaBitCount)


cdef cublasStatus_t _cublasSetFixedPointEmulationMaxMantissaBitCount(cublasHandle_t handle, int maxMantissaBitCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetFixedPointEmulationMaxMantissaBitCount
    _check_or_init_cublas()
    if __cublasSetFixedPointEmulationMaxMantissaBitCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetFixedPointEmulationMaxMantissaBitCount is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int) noexcept nogil>__cublasSetFixedPointEmulationMaxMantissaBitCount)(
        handle, maxMantissaBitCount)


cdef cublasStatus_t _cublasGetFixedPointEmulationMantissaBitOffset(cublasHandle_t handle, int* mantissaBitOffset) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetFixedPointEmulationMantissaBitOffset
    _check_or_init_cublas()
    if __cublasGetFixedPointEmulationMantissaBitOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetFixedPointEmulationMantissaBitOffset is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int*) noexcept nogil>__cublasGetFixedPointEmulationMantissaBitOffset)(
        handle, mantissaBitOffset)


cdef cublasStatus_t _cublasSetFixedPointEmulationMantissaBitOffset(cublasHandle_t handle, int mantissaBitOffset) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetFixedPointEmulationMantissaBitOffset
    _check_or_init_cublas()
    if __cublasSetFixedPointEmulationMantissaBitOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetFixedPointEmulationMantissaBitOffset is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int) noexcept nogil>__cublasSetFixedPointEmulationMantissaBitOffset)(
        handle, mantissaBitOffset)


cdef cublasStatus_t _cublasGetFixedPointEmulationMantissaBitCountPointer(cublasHandle_t handle, int** mantissaBitCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasGetFixedPointEmulationMantissaBitCountPointer
    _check_or_init_cublas()
    if __cublasGetFixedPointEmulationMantissaBitCountPointer == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetFixedPointEmulationMantissaBitCountPointer is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int**) noexcept nogil>__cublasGetFixedPointEmulationMantissaBitCountPointer)(
        handle, mantissaBitCount)


cdef cublasStatus_t _cublasSetFixedPointEmulationMantissaBitCountPointer(cublasHandle_t handle, int* mantissaBitCount) except?_CUBLASSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasSetFixedPointEmulationMantissaBitCountPointer
    _check_or_init_cublas()
    if __cublasSetFixedPointEmulationMantissaBitCountPointer == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetFixedPointEmulationMantissaBitCountPointer is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int*) noexcept nogil>__cublasSetFixedPointEmulationMantissaBitCountPointer)(
        handle, mantissaBitCount)
