# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_cusparse_dso_version_suffix

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
cdef bint __py_cusparse_init = False
cdef void* __cuDriverGetVersion = NULL

cdef void* __cusparseCreate = NULL
cdef void* __cusparseDestroy = NULL
cdef void* __cusparseGetVersion = NULL
cdef void* __cusparseGetProperty = NULL
cdef void* __cusparseGetErrorName = NULL
cdef void* __cusparseGetErrorString = NULL
cdef void* __cusparseSetStream = NULL
cdef void* __cusparseGetStream = NULL
cdef void* __cusparseGetPointerMode = NULL
cdef void* __cusparseSetPointerMode = NULL
cdef void* __cusparseCreateMatDescr = NULL
cdef void* __cusparseDestroyMatDescr = NULL
cdef void* __cusparseSetMatType = NULL
cdef void* __cusparseGetMatType = NULL
cdef void* __cusparseSetMatFillMode = NULL
cdef void* __cusparseGetMatFillMode = NULL
cdef void* __cusparseSetMatDiagType = NULL
cdef void* __cusparseGetMatDiagType = NULL
cdef void* __cusparseSetMatIndexBase = NULL
cdef void* __cusparseGetMatIndexBase = NULL
cdef void* __cusparseSgemvi = NULL
cdef void* __cusparseSgemvi_bufferSize = NULL
cdef void* __cusparseDgemvi = NULL
cdef void* __cusparseDgemvi_bufferSize = NULL
cdef void* __cusparseCgemvi = NULL
cdef void* __cusparseCgemvi_bufferSize = NULL
cdef void* __cusparseZgemvi = NULL
cdef void* __cusparseZgemvi_bufferSize = NULL
cdef void* __cusparseSbsrmv = NULL
cdef void* __cusparseDbsrmv = NULL
cdef void* __cusparseCbsrmv = NULL
cdef void* __cusparseZbsrmv = NULL
cdef void* __cusparseSbsrmm = NULL
cdef void* __cusparseDbsrmm = NULL
cdef void* __cusparseCbsrmm = NULL
cdef void* __cusparseZbsrmm = NULL
cdef void* __cusparseSgtsv2_bufferSizeExt = NULL
cdef void* __cusparseDgtsv2_bufferSizeExt = NULL
cdef void* __cusparseCgtsv2_bufferSizeExt = NULL
cdef void* __cusparseZgtsv2_bufferSizeExt = NULL
cdef void* __cusparseSgtsv2 = NULL
cdef void* __cusparseDgtsv2 = NULL
cdef void* __cusparseCgtsv2 = NULL
cdef void* __cusparseZgtsv2 = NULL
cdef void* __cusparseSgtsv2_nopivot_bufferSizeExt = NULL
cdef void* __cusparseDgtsv2_nopivot_bufferSizeExt = NULL
cdef void* __cusparseCgtsv2_nopivot_bufferSizeExt = NULL
cdef void* __cusparseZgtsv2_nopivot_bufferSizeExt = NULL
cdef void* __cusparseSgtsv2_nopivot = NULL
cdef void* __cusparseDgtsv2_nopivot = NULL
cdef void* __cusparseCgtsv2_nopivot = NULL
cdef void* __cusparseZgtsv2_nopivot = NULL
cdef void* __cusparseSgtsv2StridedBatch_bufferSizeExt = NULL
cdef void* __cusparseDgtsv2StridedBatch_bufferSizeExt = NULL
cdef void* __cusparseCgtsv2StridedBatch_bufferSizeExt = NULL
cdef void* __cusparseZgtsv2StridedBatch_bufferSizeExt = NULL
cdef void* __cusparseSgtsv2StridedBatch = NULL
cdef void* __cusparseDgtsv2StridedBatch = NULL
cdef void* __cusparseCgtsv2StridedBatch = NULL
cdef void* __cusparseZgtsv2StridedBatch = NULL
cdef void* __cusparseSgtsvInterleavedBatch_bufferSizeExt = NULL
cdef void* __cusparseDgtsvInterleavedBatch_bufferSizeExt = NULL
cdef void* __cusparseCgtsvInterleavedBatch_bufferSizeExt = NULL
cdef void* __cusparseZgtsvInterleavedBatch_bufferSizeExt = NULL
cdef void* __cusparseSgtsvInterleavedBatch = NULL
cdef void* __cusparseDgtsvInterleavedBatch = NULL
cdef void* __cusparseCgtsvInterleavedBatch = NULL
cdef void* __cusparseZgtsvInterleavedBatch = NULL
cdef void* __cusparseSgpsvInterleavedBatch_bufferSizeExt = NULL
cdef void* __cusparseDgpsvInterleavedBatch_bufferSizeExt = NULL
cdef void* __cusparseCgpsvInterleavedBatch_bufferSizeExt = NULL
cdef void* __cusparseZgpsvInterleavedBatch_bufferSizeExt = NULL
cdef void* __cusparseSgpsvInterleavedBatch = NULL
cdef void* __cusparseDgpsvInterleavedBatch = NULL
cdef void* __cusparseCgpsvInterleavedBatch = NULL
cdef void* __cusparseZgpsvInterleavedBatch = NULL
cdef void* __cusparseScsrgeam2_bufferSizeExt = NULL
cdef void* __cusparseDcsrgeam2_bufferSizeExt = NULL
cdef void* __cusparseCcsrgeam2_bufferSizeExt = NULL
cdef void* __cusparseZcsrgeam2_bufferSizeExt = NULL
cdef void* __cusparseXcsrgeam2Nnz = NULL
cdef void* __cusparseScsrgeam2 = NULL
cdef void* __cusparseDcsrgeam2 = NULL
cdef void* __cusparseCcsrgeam2 = NULL
cdef void* __cusparseZcsrgeam2 = NULL
cdef void* __cusparseSnnz = NULL
cdef void* __cusparseDnnz = NULL
cdef void* __cusparseCnnz = NULL
cdef void* __cusparseZnnz = NULL
cdef void* __cusparseXcoo2csr = NULL
cdef void* __cusparseXcsr2coo = NULL
cdef void* __cusparseSbsr2csr = NULL
cdef void* __cusparseDbsr2csr = NULL
cdef void* __cusparseCbsr2csr = NULL
cdef void* __cusparseZbsr2csr = NULL
cdef void* __cusparseSgebsr2gebsc_bufferSize = NULL
cdef void* __cusparseDgebsr2gebsc_bufferSize = NULL
cdef void* __cusparseCgebsr2gebsc_bufferSize = NULL
cdef void* __cusparseZgebsr2gebsc_bufferSize = NULL
cdef void* __cusparseSgebsr2gebsc_bufferSizeExt = NULL
cdef void* __cusparseDgebsr2gebsc_bufferSizeExt = NULL
cdef void* __cusparseCgebsr2gebsc_bufferSizeExt = NULL
cdef void* __cusparseZgebsr2gebsc_bufferSizeExt = NULL
cdef void* __cusparseSgebsr2gebsc = NULL
cdef void* __cusparseDgebsr2gebsc = NULL
cdef void* __cusparseCgebsr2gebsc = NULL
cdef void* __cusparseZgebsr2gebsc = NULL
cdef void* __cusparseScsr2gebsr_bufferSize = NULL
cdef void* __cusparseDcsr2gebsr_bufferSize = NULL
cdef void* __cusparseCcsr2gebsr_bufferSize = NULL
cdef void* __cusparseZcsr2gebsr_bufferSize = NULL
cdef void* __cusparseScsr2gebsr_bufferSizeExt = NULL
cdef void* __cusparseDcsr2gebsr_bufferSizeExt = NULL
cdef void* __cusparseCcsr2gebsr_bufferSizeExt = NULL
cdef void* __cusparseZcsr2gebsr_bufferSizeExt = NULL
cdef void* __cusparseXcsr2gebsrNnz = NULL
cdef void* __cusparseScsr2gebsr = NULL
cdef void* __cusparseDcsr2gebsr = NULL
cdef void* __cusparseCcsr2gebsr = NULL
cdef void* __cusparseZcsr2gebsr = NULL
cdef void* __cusparseSgebsr2gebsr_bufferSize = NULL
cdef void* __cusparseDgebsr2gebsr_bufferSize = NULL
cdef void* __cusparseCgebsr2gebsr_bufferSize = NULL
cdef void* __cusparseZgebsr2gebsr_bufferSize = NULL
cdef void* __cusparseSgebsr2gebsr_bufferSizeExt = NULL
cdef void* __cusparseDgebsr2gebsr_bufferSizeExt = NULL
cdef void* __cusparseCgebsr2gebsr_bufferSizeExt = NULL
cdef void* __cusparseZgebsr2gebsr_bufferSizeExt = NULL
cdef void* __cusparseXgebsr2gebsrNnz = NULL
cdef void* __cusparseSgebsr2gebsr = NULL
cdef void* __cusparseDgebsr2gebsr = NULL
cdef void* __cusparseCgebsr2gebsr = NULL
cdef void* __cusparseZgebsr2gebsr = NULL
cdef void* __cusparseXcoosort_bufferSizeExt = NULL
cdef void* __cusparseXcoosortByRow = NULL
cdef void* __cusparseXcoosortByColumn = NULL
cdef void* __cusparseXcsrsort_bufferSizeExt = NULL
cdef void* __cusparseXcsrsort = NULL
cdef void* __cusparseXcscsort_bufferSizeExt = NULL
cdef void* __cusparseXcscsort = NULL
cdef void* __cusparseCsr2cscEx2 = NULL
cdef void* __cusparseCsr2cscEx2_bufferSize = NULL
cdef void* __cusparseCreateSpVec = NULL
cdef void* __cusparseDestroySpVec = NULL
cdef void* __cusparseSpVecGet = NULL
cdef void* __cusparseSpVecGetIndexBase = NULL
cdef void* __cusparseSpVecGetValues = NULL
cdef void* __cusparseSpVecSetValues = NULL
cdef void* __cusparseCreateDnVec = NULL
cdef void* __cusparseDestroyDnVec = NULL
cdef void* __cusparseDnVecGet = NULL
cdef void* __cusparseDnVecGetValues = NULL
cdef void* __cusparseDnVecSetValues = NULL
cdef void* __cusparseDestroySpMat = NULL
cdef void* __cusparseSpMatGetFormat = NULL
cdef void* __cusparseSpMatGetIndexBase = NULL
cdef void* __cusparseSpMatGetValues = NULL
cdef void* __cusparseSpMatSetValues = NULL
cdef void* __cusparseSpMatGetSize = NULL
cdef void* __cusparseSpMatGetStridedBatch = NULL
cdef void* __cusparseCooSetStridedBatch = NULL
cdef void* __cusparseCsrSetStridedBatch = NULL
cdef void* __cusparseCreateCsr = NULL
cdef void* __cusparseCsrGet = NULL
cdef void* __cusparseCsrSetPointers = NULL
cdef void* __cusparseCreateCoo = NULL
cdef void* __cusparseCooGet = NULL
cdef void* __cusparseCreateDnMat = NULL
cdef void* __cusparseDestroyDnMat = NULL
cdef void* __cusparseDnMatGet = NULL
cdef void* __cusparseDnMatGetValues = NULL
cdef void* __cusparseDnMatSetValues = NULL
cdef void* __cusparseDnMatSetStridedBatch = NULL
cdef void* __cusparseDnMatGetStridedBatch = NULL
cdef void* __cusparseAxpby = NULL
cdef void* __cusparseGather = NULL
cdef void* __cusparseScatter = NULL
cdef void* __cusparseSpVV_bufferSize = NULL
cdef void* __cusparseSpVV = NULL
cdef void* __cusparseSpMV = NULL
cdef void* __cusparseSpMV_bufferSize = NULL
cdef void* __cusparseSpMM = NULL
cdef void* __cusparseSpMM_bufferSize = NULL
cdef void* __cusparseSpGEMM_createDescr = NULL
cdef void* __cusparseSpGEMM_destroyDescr = NULL
cdef void* __cusparseSpGEMM_workEstimation = NULL
cdef void* __cusparseSpGEMM_compute = NULL
cdef void* __cusparseSpGEMM_copy = NULL
cdef void* __cusparseCreateCsc = NULL
cdef void* __cusparseCscSetPointers = NULL
cdef void* __cusparseCooSetPointers = NULL
cdef void* __cusparseSparseToDense_bufferSize = NULL
cdef void* __cusparseSparseToDense = NULL
cdef void* __cusparseDenseToSparse_bufferSize = NULL
cdef void* __cusparseDenseToSparse_analysis = NULL
cdef void* __cusparseDenseToSparse_convert = NULL
cdef void* __cusparseCreateBlockedEll = NULL
cdef void* __cusparseBlockedEllGet = NULL
cdef void* __cusparseSpMM_preprocess = NULL
cdef void* __cusparseSDDMM_bufferSize = NULL
cdef void* __cusparseSDDMM_preprocess = NULL
cdef void* __cusparseSDDMM = NULL
cdef void* __cusparseSpMatGetAttribute = NULL
cdef void* __cusparseSpMatSetAttribute = NULL
cdef void* __cusparseSpSV_createDescr = NULL
cdef void* __cusparseSpSV_destroyDescr = NULL
cdef void* __cusparseSpSV_bufferSize = NULL
cdef void* __cusparseSpSV_analysis = NULL
cdef void* __cusparseSpSV_solve = NULL
cdef void* __cusparseSpSM_createDescr = NULL
cdef void* __cusparseSpSM_destroyDescr = NULL
cdef void* __cusparseSpSM_bufferSize = NULL
cdef void* __cusparseSpSM_analysis = NULL
cdef void* __cusparseSpSM_solve = NULL
cdef void* __cusparseSpGEMMreuse_workEstimation = NULL
cdef void* __cusparseSpGEMMreuse_nnz = NULL
cdef void* __cusparseSpGEMMreuse_copy = NULL
cdef void* __cusparseSpGEMMreuse_compute = NULL
cdef void* __cusparseLoggerSetCallback = NULL
cdef void* __cusparseLoggerSetFile = NULL
cdef void* __cusparseLoggerOpenFile = NULL
cdef void* __cusparseLoggerSetLevel = NULL
cdef void* __cusparseLoggerSetMask = NULL
cdef void* __cusparseLoggerForceDisable = NULL
cdef void* __cusparseSpMMOp_createPlan = NULL
cdef void* __cusparseSpMMOp = NULL
cdef void* __cusparseSpMMOp_destroyPlan = NULL
cdef void* __cusparseCscGet = NULL
cdef void* __cusparseCreateConstSpVec = NULL
cdef void* __cusparseConstSpVecGet = NULL
cdef void* __cusparseConstSpVecGetValues = NULL
cdef void* __cusparseCreateConstDnVec = NULL
cdef void* __cusparseConstDnVecGet = NULL
cdef void* __cusparseConstDnVecGetValues = NULL
cdef void* __cusparseConstSpMatGetValues = NULL
cdef void* __cusparseCreateConstCsr = NULL
cdef void* __cusparseCreateConstCsc = NULL
cdef void* __cusparseConstCsrGet = NULL
cdef void* __cusparseConstCscGet = NULL
cdef void* __cusparseCreateConstCoo = NULL
cdef void* __cusparseConstCooGet = NULL
cdef void* __cusparseCreateConstBlockedEll = NULL
cdef void* __cusparseConstBlockedEllGet = NULL
cdef void* __cusparseCreateConstDnMat = NULL
cdef void* __cusparseConstDnMatGet = NULL
cdef void* __cusparseConstDnMatGetValues = NULL
cdef void* __cusparseSpGEMM_getNumProducts = NULL
cdef void* __cusparseSpGEMM_estimateMemory = NULL
cdef void* __cusparseBsrSetStridedBatch = NULL
cdef void* __cusparseCreateBsr = NULL
cdef void* __cusparseCreateConstBsr = NULL
cdef void* __cusparseCreateSlicedEll = NULL
cdef void* __cusparseCreateConstSlicedEll = NULL
cdef void* __cusparseSpSV_updateMatrix = NULL
cdef void* __cusparseSpMV_preprocess = NULL
cdef void* __cusparseSpSM_updateMatrix = NULL


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef void* load_library(const int driver_ver) except* with gil:
    handle = 0

    for suffix in get_cusparse_dso_version_suffix(driver_ver):
        if len(suffix) == 0:
            continue
        dll_name = f"cusparse64_{suffix}.dll"

        # First check if the DLL has been loaded by 3rd parties
        try:
            handle = win32api.GetModuleHandle(dll_name)
        except:
            pass
        else:
            break

        # Next, check if DLLs are installed via pip
        for sp in get_site_packages():
            mod_path = os.path.join(sp, "nvidia", "cusparse", "bin")
            if not os.path.isdir(mod_path):
                continue
            os.add_dll_directory(mod_path)
            # cuSPARSE also requires additional dependencies...
            mod_path_jit = mod_path.replace("cusparse", "nvjitlink")
            if os.path.isdir(mod_path_jit):
                os.add_dll_directory(mod_path_jit)
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
        raise RuntimeError('Failed to load cusparse')

    assert handle != 0
    return <void*><intptr_t>handle


cdef int _check_or_init_cusparse() except -1 nogil:
    global __py_cusparse_init
    if __py_cusparse_init:
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
        err = (<int (*)(int*) noexcept nogil>__cuDriverGetVersion)(&driver_ver)
        if err != 0:
            raise RuntimeError('something went wrong')

        # Load library
        handle = <intptr_t>load_library(driver_ver)

        # Load function
        global __cusparseCreate
        try:
            __cusparseCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreate')
        except:
            pass

        global __cusparseDestroy
        try:
            __cusparseDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDestroy')
        except:
            pass

        global __cusparseGetVersion
        try:
            __cusparseGetVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGetVersion')
        except:
            pass

        global __cusparseGetProperty
        try:
            __cusparseGetProperty = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGetProperty')
        except:
            pass

        global __cusparseGetErrorName
        try:
            __cusparseGetErrorName = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGetErrorName')
        except:
            pass

        global __cusparseGetErrorString
        try:
            __cusparseGetErrorString = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGetErrorString')
        except:
            pass

        global __cusparseSetStream
        try:
            __cusparseSetStream = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSetStream')
        except:
            pass

        global __cusparseGetStream
        try:
            __cusparseGetStream = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGetStream')
        except:
            pass

        global __cusparseGetPointerMode
        try:
            __cusparseGetPointerMode = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGetPointerMode')
        except:
            pass

        global __cusparseSetPointerMode
        try:
            __cusparseSetPointerMode = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSetPointerMode')
        except:
            pass

        global __cusparseCreateMatDescr
        try:
            __cusparseCreateMatDescr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateMatDescr')
        except:
            pass

        global __cusparseDestroyMatDescr
        try:
            __cusparseDestroyMatDescr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDestroyMatDescr')
        except:
            pass

        global __cusparseSetMatType
        try:
            __cusparseSetMatType = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSetMatType')
        except:
            pass

        global __cusparseGetMatType
        try:
            __cusparseGetMatType = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGetMatType')
        except:
            pass

        global __cusparseSetMatFillMode
        try:
            __cusparseSetMatFillMode = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSetMatFillMode')
        except:
            pass

        global __cusparseGetMatFillMode
        try:
            __cusparseGetMatFillMode = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGetMatFillMode')
        except:
            pass

        global __cusparseSetMatDiagType
        try:
            __cusparseSetMatDiagType = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSetMatDiagType')
        except:
            pass

        global __cusparseGetMatDiagType
        try:
            __cusparseGetMatDiagType = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGetMatDiagType')
        except:
            pass

        global __cusparseSetMatIndexBase
        try:
            __cusparseSetMatIndexBase = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSetMatIndexBase')
        except:
            pass

        global __cusparseGetMatIndexBase
        try:
            __cusparseGetMatIndexBase = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGetMatIndexBase')
        except:
            pass

        global __cusparseSgemvi
        try:
            __cusparseSgemvi = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgemvi')
        except:
            pass

        global __cusparseSgemvi_bufferSize
        try:
            __cusparseSgemvi_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgemvi_bufferSize')
        except:
            pass

        global __cusparseDgemvi
        try:
            __cusparseDgemvi = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgemvi')
        except:
            pass

        global __cusparseDgemvi_bufferSize
        try:
            __cusparseDgemvi_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgemvi_bufferSize')
        except:
            pass

        global __cusparseCgemvi
        try:
            __cusparseCgemvi = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgemvi')
        except:
            pass

        global __cusparseCgemvi_bufferSize
        try:
            __cusparseCgemvi_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgemvi_bufferSize')
        except:
            pass

        global __cusparseZgemvi
        try:
            __cusparseZgemvi = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgemvi')
        except:
            pass

        global __cusparseZgemvi_bufferSize
        try:
            __cusparseZgemvi_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgemvi_bufferSize')
        except:
            pass

        global __cusparseSbsrmv
        try:
            __cusparseSbsrmv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSbsrmv')
        except:
            pass

        global __cusparseDbsrmv
        try:
            __cusparseDbsrmv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDbsrmv')
        except:
            pass

        global __cusparseCbsrmv
        try:
            __cusparseCbsrmv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCbsrmv')
        except:
            pass

        global __cusparseZbsrmv
        try:
            __cusparseZbsrmv = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZbsrmv')
        except:
            pass

        global __cusparseSbsrmm
        try:
            __cusparseSbsrmm = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSbsrmm')
        except:
            pass

        global __cusparseDbsrmm
        try:
            __cusparseDbsrmm = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDbsrmm')
        except:
            pass

        global __cusparseCbsrmm
        try:
            __cusparseCbsrmm = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCbsrmm')
        except:
            pass

        global __cusparseZbsrmm
        try:
            __cusparseZbsrmm = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZbsrmm')
        except:
            pass

        global __cusparseSgtsv2_bufferSizeExt
        try:
            __cusparseSgtsv2_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgtsv2_bufferSizeExt')
        except:
            pass

        global __cusparseDgtsv2_bufferSizeExt
        try:
            __cusparseDgtsv2_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgtsv2_bufferSizeExt')
        except:
            pass

        global __cusparseCgtsv2_bufferSizeExt
        try:
            __cusparseCgtsv2_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgtsv2_bufferSizeExt')
        except:
            pass

        global __cusparseZgtsv2_bufferSizeExt
        try:
            __cusparseZgtsv2_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgtsv2_bufferSizeExt')
        except:
            pass

        global __cusparseSgtsv2
        try:
            __cusparseSgtsv2 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgtsv2')
        except:
            pass

        global __cusparseDgtsv2
        try:
            __cusparseDgtsv2 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgtsv2')
        except:
            pass

        global __cusparseCgtsv2
        try:
            __cusparseCgtsv2 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgtsv2')
        except:
            pass

        global __cusparseZgtsv2
        try:
            __cusparseZgtsv2 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgtsv2')
        except:
            pass

        global __cusparseSgtsv2_nopivot_bufferSizeExt
        try:
            __cusparseSgtsv2_nopivot_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgtsv2_nopivot_bufferSizeExt')
        except:
            pass

        global __cusparseDgtsv2_nopivot_bufferSizeExt
        try:
            __cusparseDgtsv2_nopivot_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgtsv2_nopivot_bufferSizeExt')
        except:
            pass

        global __cusparseCgtsv2_nopivot_bufferSizeExt
        try:
            __cusparseCgtsv2_nopivot_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgtsv2_nopivot_bufferSizeExt')
        except:
            pass

        global __cusparseZgtsv2_nopivot_bufferSizeExt
        try:
            __cusparseZgtsv2_nopivot_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgtsv2_nopivot_bufferSizeExt')
        except:
            pass

        global __cusparseSgtsv2_nopivot
        try:
            __cusparseSgtsv2_nopivot = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgtsv2_nopivot')
        except:
            pass

        global __cusparseDgtsv2_nopivot
        try:
            __cusparseDgtsv2_nopivot = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgtsv2_nopivot')
        except:
            pass

        global __cusparseCgtsv2_nopivot
        try:
            __cusparseCgtsv2_nopivot = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgtsv2_nopivot')
        except:
            pass

        global __cusparseZgtsv2_nopivot
        try:
            __cusparseZgtsv2_nopivot = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgtsv2_nopivot')
        except:
            pass

        global __cusparseSgtsv2StridedBatch_bufferSizeExt
        try:
            __cusparseSgtsv2StridedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgtsv2StridedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseDgtsv2StridedBatch_bufferSizeExt
        try:
            __cusparseDgtsv2StridedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgtsv2StridedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseCgtsv2StridedBatch_bufferSizeExt
        try:
            __cusparseCgtsv2StridedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgtsv2StridedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseZgtsv2StridedBatch_bufferSizeExt
        try:
            __cusparseZgtsv2StridedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgtsv2StridedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseSgtsv2StridedBatch
        try:
            __cusparseSgtsv2StridedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgtsv2StridedBatch')
        except:
            pass

        global __cusparseDgtsv2StridedBatch
        try:
            __cusparseDgtsv2StridedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgtsv2StridedBatch')
        except:
            pass

        global __cusparseCgtsv2StridedBatch
        try:
            __cusparseCgtsv2StridedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgtsv2StridedBatch')
        except:
            pass

        global __cusparseZgtsv2StridedBatch
        try:
            __cusparseZgtsv2StridedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgtsv2StridedBatch')
        except:
            pass

        global __cusparseSgtsvInterleavedBatch_bufferSizeExt
        try:
            __cusparseSgtsvInterleavedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgtsvInterleavedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseDgtsvInterleavedBatch_bufferSizeExt
        try:
            __cusparseDgtsvInterleavedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgtsvInterleavedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseCgtsvInterleavedBatch_bufferSizeExt
        try:
            __cusparseCgtsvInterleavedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgtsvInterleavedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseZgtsvInterleavedBatch_bufferSizeExt
        try:
            __cusparseZgtsvInterleavedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgtsvInterleavedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseSgtsvInterleavedBatch
        try:
            __cusparseSgtsvInterleavedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgtsvInterleavedBatch')
        except:
            pass

        global __cusparseDgtsvInterleavedBatch
        try:
            __cusparseDgtsvInterleavedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgtsvInterleavedBatch')
        except:
            pass

        global __cusparseCgtsvInterleavedBatch
        try:
            __cusparseCgtsvInterleavedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgtsvInterleavedBatch')
        except:
            pass

        global __cusparseZgtsvInterleavedBatch
        try:
            __cusparseZgtsvInterleavedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgtsvInterleavedBatch')
        except:
            pass

        global __cusparseSgpsvInterleavedBatch_bufferSizeExt
        try:
            __cusparseSgpsvInterleavedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgpsvInterleavedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseDgpsvInterleavedBatch_bufferSizeExt
        try:
            __cusparseDgpsvInterleavedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgpsvInterleavedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseCgpsvInterleavedBatch_bufferSizeExt
        try:
            __cusparseCgpsvInterleavedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgpsvInterleavedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseZgpsvInterleavedBatch_bufferSizeExt
        try:
            __cusparseZgpsvInterleavedBatch_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgpsvInterleavedBatch_bufferSizeExt')
        except:
            pass

        global __cusparseSgpsvInterleavedBatch
        try:
            __cusparseSgpsvInterleavedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgpsvInterleavedBatch')
        except:
            pass

        global __cusparseDgpsvInterleavedBatch
        try:
            __cusparseDgpsvInterleavedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgpsvInterleavedBatch')
        except:
            pass

        global __cusparseCgpsvInterleavedBatch
        try:
            __cusparseCgpsvInterleavedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgpsvInterleavedBatch')
        except:
            pass

        global __cusparseZgpsvInterleavedBatch
        try:
            __cusparseZgpsvInterleavedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgpsvInterleavedBatch')
        except:
            pass

        global __cusparseScsrgeam2_bufferSizeExt
        try:
            __cusparseScsrgeam2_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseScsrgeam2_bufferSizeExt')
        except:
            pass

        global __cusparseDcsrgeam2_bufferSizeExt
        try:
            __cusparseDcsrgeam2_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDcsrgeam2_bufferSizeExt')
        except:
            pass

        global __cusparseCcsrgeam2_bufferSizeExt
        try:
            __cusparseCcsrgeam2_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCcsrgeam2_bufferSizeExt')
        except:
            pass

        global __cusparseZcsrgeam2_bufferSizeExt
        try:
            __cusparseZcsrgeam2_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZcsrgeam2_bufferSizeExt')
        except:
            pass

        global __cusparseXcsrgeam2Nnz
        try:
            __cusparseXcsrgeam2Nnz = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcsrgeam2Nnz')
        except:
            pass

        global __cusparseScsrgeam2
        try:
            __cusparseScsrgeam2 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseScsrgeam2')
        except:
            pass

        global __cusparseDcsrgeam2
        try:
            __cusparseDcsrgeam2 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDcsrgeam2')
        except:
            pass

        global __cusparseCcsrgeam2
        try:
            __cusparseCcsrgeam2 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCcsrgeam2')
        except:
            pass

        global __cusparseZcsrgeam2
        try:
            __cusparseZcsrgeam2 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZcsrgeam2')
        except:
            pass

        global __cusparseSnnz
        try:
            __cusparseSnnz = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSnnz')
        except:
            pass

        global __cusparseDnnz
        try:
            __cusparseDnnz = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDnnz')
        except:
            pass

        global __cusparseCnnz
        try:
            __cusparseCnnz = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCnnz')
        except:
            pass

        global __cusparseZnnz
        try:
            __cusparseZnnz = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZnnz')
        except:
            pass

        global __cusparseXcoo2csr
        try:
            __cusparseXcoo2csr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcoo2csr')
        except:
            pass

        global __cusparseXcsr2coo
        try:
            __cusparseXcsr2coo = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcsr2coo')
        except:
            pass

        global __cusparseSbsr2csr
        try:
            __cusparseSbsr2csr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSbsr2csr')
        except:
            pass

        global __cusparseDbsr2csr
        try:
            __cusparseDbsr2csr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDbsr2csr')
        except:
            pass

        global __cusparseCbsr2csr
        try:
            __cusparseCbsr2csr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCbsr2csr')
        except:
            pass

        global __cusparseZbsr2csr
        try:
            __cusparseZbsr2csr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZbsr2csr')
        except:
            pass

        global __cusparseSgebsr2gebsc_bufferSize
        try:
            __cusparseSgebsr2gebsc_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgebsr2gebsc_bufferSize')
        except:
            pass

        global __cusparseDgebsr2gebsc_bufferSize
        try:
            __cusparseDgebsr2gebsc_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgebsr2gebsc_bufferSize')
        except:
            pass

        global __cusparseCgebsr2gebsc_bufferSize
        try:
            __cusparseCgebsr2gebsc_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgebsr2gebsc_bufferSize')
        except:
            pass

        global __cusparseZgebsr2gebsc_bufferSize
        try:
            __cusparseZgebsr2gebsc_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgebsr2gebsc_bufferSize')
        except:
            pass

        global __cusparseSgebsr2gebsc_bufferSizeExt
        try:
            __cusparseSgebsr2gebsc_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgebsr2gebsc_bufferSizeExt')
        except:
            pass

        global __cusparseDgebsr2gebsc_bufferSizeExt
        try:
            __cusparseDgebsr2gebsc_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgebsr2gebsc_bufferSizeExt')
        except:
            pass

        global __cusparseCgebsr2gebsc_bufferSizeExt
        try:
            __cusparseCgebsr2gebsc_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgebsr2gebsc_bufferSizeExt')
        except:
            pass

        global __cusparseZgebsr2gebsc_bufferSizeExt
        try:
            __cusparseZgebsr2gebsc_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgebsr2gebsc_bufferSizeExt')
        except:
            pass

        global __cusparseSgebsr2gebsc
        try:
            __cusparseSgebsr2gebsc = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgebsr2gebsc')
        except:
            pass

        global __cusparseDgebsr2gebsc
        try:
            __cusparseDgebsr2gebsc = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgebsr2gebsc')
        except:
            pass

        global __cusparseCgebsr2gebsc
        try:
            __cusparseCgebsr2gebsc = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgebsr2gebsc')
        except:
            pass

        global __cusparseZgebsr2gebsc
        try:
            __cusparseZgebsr2gebsc = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgebsr2gebsc')
        except:
            pass

        global __cusparseScsr2gebsr_bufferSize
        try:
            __cusparseScsr2gebsr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseScsr2gebsr_bufferSize')
        except:
            pass

        global __cusparseDcsr2gebsr_bufferSize
        try:
            __cusparseDcsr2gebsr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDcsr2gebsr_bufferSize')
        except:
            pass

        global __cusparseCcsr2gebsr_bufferSize
        try:
            __cusparseCcsr2gebsr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCcsr2gebsr_bufferSize')
        except:
            pass

        global __cusparseZcsr2gebsr_bufferSize
        try:
            __cusparseZcsr2gebsr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZcsr2gebsr_bufferSize')
        except:
            pass

        global __cusparseScsr2gebsr_bufferSizeExt
        try:
            __cusparseScsr2gebsr_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseScsr2gebsr_bufferSizeExt')
        except:
            pass

        global __cusparseDcsr2gebsr_bufferSizeExt
        try:
            __cusparseDcsr2gebsr_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDcsr2gebsr_bufferSizeExt')
        except:
            pass

        global __cusparseCcsr2gebsr_bufferSizeExt
        try:
            __cusparseCcsr2gebsr_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCcsr2gebsr_bufferSizeExt')
        except:
            pass

        global __cusparseZcsr2gebsr_bufferSizeExt
        try:
            __cusparseZcsr2gebsr_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZcsr2gebsr_bufferSizeExt')
        except:
            pass

        global __cusparseXcsr2gebsrNnz
        try:
            __cusparseXcsr2gebsrNnz = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcsr2gebsrNnz')
        except:
            pass

        global __cusparseScsr2gebsr
        try:
            __cusparseScsr2gebsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseScsr2gebsr')
        except:
            pass

        global __cusparseDcsr2gebsr
        try:
            __cusparseDcsr2gebsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDcsr2gebsr')
        except:
            pass

        global __cusparseCcsr2gebsr
        try:
            __cusparseCcsr2gebsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCcsr2gebsr')
        except:
            pass

        global __cusparseZcsr2gebsr
        try:
            __cusparseZcsr2gebsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZcsr2gebsr')
        except:
            pass

        global __cusparseSgebsr2gebsr_bufferSize
        try:
            __cusparseSgebsr2gebsr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgebsr2gebsr_bufferSize')
        except:
            pass

        global __cusparseDgebsr2gebsr_bufferSize
        try:
            __cusparseDgebsr2gebsr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgebsr2gebsr_bufferSize')
        except:
            pass

        global __cusparseCgebsr2gebsr_bufferSize
        try:
            __cusparseCgebsr2gebsr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgebsr2gebsr_bufferSize')
        except:
            pass

        global __cusparseZgebsr2gebsr_bufferSize
        try:
            __cusparseZgebsr2gebsr_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgebsr2gebsr_bufferSize')
        except:
            pass

        global __cusparseSgebsr2gebsr_bufferSizeExt
        try:
            __cusparseSgebsr2gebsr_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgebsr2gebsr_bufferSizeExt')
        except:
            pass

        global __cusparseDgebsr2gebsr_bufferSizeExt
        try:
            __cusparseDgebsr2gebsr_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgebsr2gebsr_bufferSizeExt')
        except:
            pass

        global __cusparseCgebsr2gebsr_bufferSizeExt
        try:
            __cusparseCgebsr2gebsr_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgebsr2gebsr_bufferSizeExt')
        except:
            pass

        global __cusparseZgebsr2gebsr_bufferSizeExt
        try:
            __cusparseZgebsr2gebsr_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgebsr2gebsr_bufferSizeExt')
        except:
            pass

        global __cusparseXgebsr2gebsrNnz
        try:
            __cusparseXgebsr2gebsrNnz = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXgebsr2gebsrNnz')
        except:
            pass

        global __cusparseSgebsr2gebsr
        try:
            __cusparseSgebsr2gebsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSgebsr2gebsr')
        except:
            pass

        global __cusparseDgebsr2gebsr
        try:
            __cusparseDgebsr2gebsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDgebsr2gebsr')
        except:
            pass

        global __cusparseCgebsr2gebsr
        try:
            __cusparseCgebsr2gebsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCgebsr2gebsr')
        except:
            pass

        global __cusparseZgebsr2gebsr
        try:
            __cusparseZgebsr2gebsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseZgebsr2gebsr')
        except:
            pass

        global __cusparseXcoosort_bufferSizeExt
        try:
            __cusparseXcoosort_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcoosort_bufferSizeExt')
        except:
            pass

        global __cusparseXcoosortByRow
        try:
            __cusparseXcoosortByRow = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcoosortByRow')
        except:
            pass

        global __cusparseXcoosortByColumn
        try:
            __cusparseXcoosortByColumn = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcoosortByColumn')
        except:
            pass

        global __cusparseXcsrsort_bufferSizeExt
        try:
            __cusparseXcsrsort_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcsrsort_bufferSizeExt')
        except:
            pass

        global __cusparseXcsrsort
        try:
            __cusparseXcsrsort = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcsrsort')
        except:
            pass

        global __cusparseXcscsort_bufferSizeExt
        try:
            __cusparseXcscsort_bufferSizeExt = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcscsort_bufferSizeExt')
        except:
            pass

        global __cusparseXcscsort
        try:
            __cusparseXcscsort = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseXcscsort')
        except:
            pass

        global __cusparseCsr2cscEx2
        try:
            __cusparseCsr2cscEx2 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCsr2cscEx2')
        except:
            pass

        global __cusparseCsr2cscEx2_bufferSize
        try:
            __cusparseCsr2cscEx2_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCsr2cscEx2_bufferSize')
        except:
            pass

        global __cusparseCreateSpVec
        try:
            __cusparseCreateSpVec = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateSpVec')
        except:
            pass

        global __cusparseDestroySpVec
        try:
            __cusparseDestroySpVec = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDestroySpVec')
        except:
            pass

        global __cusparseSpVecGet
        try:
            __cusparseSpVecGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpVecGet')
        except:
            pass

        global __cusparseSpVecGetIndexBase
        try:
            __cusparseSpVecGetIndexBase = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpVecGetIndexBase')
        except:
            pass

        global __cusparseSpVecGetValues
        try:
            __cusparseSpVecGetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpVecGetValues')
        except:
            pass

        global __cusparseSpVecSetValues
        try:
            __cusparseSpVecSetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpVecSetValues')
        except:
            pass

        global __cusparseCreateDnVec
        try:
            __cusparseCreateDnVec = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateDnVec')
        except:
            pass

        global __cusparseDestroyDnVec
        try:
            __cusparseDestroyDnVec = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDestroyDnVec')
        except:
            pass

        global __cusparseDnVecGet
        try:
            __cusparseDnVecGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDnVecGet')
        except:
            pass

        global __cusparseDnVecGetValues
        try:
            __cusparseDnVecGetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDnVecGetValues')
        except:
            pass

        global __cusparseDnVecSetValues
        try:
            __cusparseDnVecSetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDnVecSetValues')
        except:
            pass

        global __cusparseDestroySpMat
        try:
            __cusparseDestroySpMat = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDestroySpMat')
        except:
            pass

        global __cusparseSpMatGetFormat
        try:
            __cusparseSpMatGetFormat = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMatGetFormat')
        except:
            pass

        global __cusparseSpMatGetIndexBase
        try:
            __cusparseSpMatGetIndexBase = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMatGetIndexBase')
        except:
            pass

        global __cusparseSpMatGetValues
        try:
            __cusparseSpMatGetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMatGetValues')
        except:
            pass

        global __cusparseSpMatSetValues
        try:
            __cusparseSpMatSetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMatSetValues')
        except:
            pass

        global __cusparseSpMatGetSize
        try:
            __cusparseSpMatGetSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMatGetSize')
        except:
            pass

        global __cusparseSpMatGetStridedBatch
        try:
            __cusparseSpMatGetStridedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMatGetStridedBatch')
        except:
            pass

        global __cusparseCooSetStridedBatch
        try:
            __cusparseCooSetStridedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCooSetStridedBatch')
        except:
            pass

        global __cusparseCsrSetStridedBatch
        try:
            __cusparseCsrSetStridedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCsrSetStridedBatch')
        except:
            pass

        global __cusparseCreateCsr
        try:
            __cusparseCreateCsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateCsr')
        except:
            pass

        global __cusparseCsrGet
        try:
            __cusparseCsrGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCsrGet')
        except:
            pass

        global __cusparseCsrSetPointers
        try:
            __cusparseCsrSetPointers = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCsrSetPointers')
        except:
            pass

        global __cusparseCreateCoo
        try:
            __cusparseCreateCoo = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateCoo')
        except:
            pass

        global __cusparseCooGet
        try:
            __cusparseCooGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCooGet')
        except:
            pass

        global __cusparseCreateDnMat
        try:
            __cusparseCreateDnMat = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateDnMat')
        except:
            pass

        global __cusparseDestroyDnMat
        try:
            __cusparseDestroyDnMat = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDestroyDnMat')
        except:
            pass

        global __cusparseDnMatGet
        try:
            __cusparseDnMatGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDnMatGet')
        except:
            pass

        global __cusparseDnMatGetValues
        try:
            __cusparseDnMatGetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDnMatGetValues')
        except:
            pass

        global __cusparseDnMatSetValues
        try:
            __cusparseDnMatSetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDnMatSetValues')
        except:
            pass

        global __cusparseDnMatSetStridedBatch
        try:
            __cusparseDnMatSetStridedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDnMatSetStridedBatch')
        except:
            pass

        global __cusparseDnMatGetStridedBatch
        try:
            __cusparseDnMatGetStridedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDnMatGetStridedBatch')
        except:
            pass

        global __cusparseAxpby
        try:
            __cusparseAxpby = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseAxpby')
        except:
            pass

        global __cusparseGather
        try:
            __cusparseGather = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseGather')
        except:
            pass

        global __cusparseScatter
        try:
            __cusparseScatter = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseScatter')
        except:
            pass

        global __cusparseSpVV_bufferSize
        try:
            __cusparseSpVV_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpVV_bufferSize')
        except:
            pass

        global __cusparseSpVV
        try:
            __cusparseSpVV = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpVV')
        except:
            pass

        global __cusparseSpMV
        try:
            __cusparseSpMV = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMV')
        except:
            pass

        global __cusparseSpMV_bufferSize
        try:
            __cusparseSpMV_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMV_bufferSize')
        except:
            pass

        global __cusparseSpMM
        try:
            __cusparseSpMM = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMM')
        except:
            pass

        global __cusparseSpMM_bufferSize
        try:
            __cusparseSpMM_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMM_bufferSize')
        except:
            pass

        global __cusparseSpGEMM_createDescr
        try:
            __cusparseSpGEMM_createDescr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMM_createDescr')
        except:
            pass

        global __cusparseSpGEMM_destroyDescr
        try:
            __cusparseSpGEMM_destroyDescr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMM_destroyDescr')
        except:
            pass

        global __cusparseSpGEMM_workEstimation
        try:
            __cusparseSpGEMM_workEstimation = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMM_workEstimation')
        except:
            pass

        global __cusparseSpGEMM_compute
        try:
            __cusparseSpGEMM_compute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMM_compute')
        except:
            pass

        global __cusparseSpGEMM_copy
        try:
            __cusparseSpGEMM_copy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMM_copy')
        except:
            pass

        global __cusparseCreateCsc
        try:
            __cusparseCreateCsc = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateCsc')
        except:
            pass

        global __cusparseCscSetPointers
        try:
            __cusparseCscSetPointers = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCscSetPointers')
        except:
            pass

        global __cusparseCooSetPointers
        try:
            __cusparseCooSetPointers = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCooSetPointers')
        except:
            pass

        global __cusparseSparseToDense_bufferSize
        try:
            __cusparseSparseToDense_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSparseToDense_bufferSize')
        except:
            pass

        global __cusparseSparseToDense
        try:
            __cusparseSparseToDense = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSparseToDense')
        except:
            pass

        global __cusparseDenseToSparse_bufferSize
        try:
            __cusparseDenseToSparse_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDenseToSparse_bufferSize')
        except:
            pass

        global __cusparseDenseToSparse_analysis
        try:
            __cusparseDenseToSparse_analysis = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDenseToSparse_analysis')
        except:
            pass

        global __cusparseDenseToSparse_convert
        try:
            __cusparseDenseToSparse_convert = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseDenseToSparse_convert')
        except:
            pass

        global __cusparseCreateBlockedEll
        try:
            __cusparseCreateBlockedEll = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateBlockedEll')
        except:
            pass

        global __cusparseBlockedEllGet
        try:
            __cusparseBlockedEllGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseBlockedEllGet')
        except:
            pass

        global __cusparseSpMM_preprocess
        try:
            __cusparseSpMM_preprocess = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMM_preprocess')
        except:
            pass

        global __cusparseSDDMM_bufferSize
        try:
            __cusparseSDDMM_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSDDMM_bufferSize')
        except:
            pass

        global __cusparseSDDMM_preprocess
        try:
            __cusparseSDDMM_preprocess = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSDDMM_preprocess')
        except:
            pass

        global __cusparseSDDMM
        try:
            __cusparseSDDMM = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSDDMM')
        except:
            pass

        global __cusparseSpMatGetAttribute
        try:
            __cusparseSpMatGetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMatGetAttribute')
        except:
            pass

        global __cusparseSpMatSetAttribute
        try:
            __cusparseSpMatSetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMatSetAttribute')
        except:
            pass

        global __cusparseSpSV_createDescr
        try:
            __cusparseSpSV_createDescr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSV_createDescr')
        except:
            pass

        global __cusparseSpSV_destroyDescr
        try:
            __cusparseSpSV_destroyDescr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSV_destroyDescr')
        except:
            pass

        global __cusparseSpSV_bufferSize
        try:
            __cusparseSpSV_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSV_bufferSize')
        except:
            pass

        global __cusparseSpSV_analysis
        try:
            __cusparseSpSV_analysis = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSV_analysis')
        except:
            pass

        global __cusparseSpSV_solve
        try:
            __cusparseSpSV_solve = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSV_solve')
        except:
            pass

        global __cusparseSpSM_createDescr
        try:
            __cusparseSpSM_createDescr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSM_createDescr')
        except:
            pass

        global __cusparseSpSM_destroyDescr
        try:
            __cusparseSpSM_destroyDescr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSM_destroyDescr')
        except:
            pass

        global __cusparseSpSM_bufferSize
        try:
            __cusparseSpSM_bufferSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSM_bufferSize')
        except:
            pass

        global __cusparseSpSM_analysis
        try:
            __cusparseSpSM_analysis = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSM_analysis')
        except:
            pass

        global __cusparseSpSM_solve
        try:
            __cusparseSpSM_solve = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSM_solve')
        except:
            pass

        global __cusparseSpGEMMreuse_workEstimation
        try:
            __cusparseSpGEMMreuse_workEstimation = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMMreuse_workEstimation')
        except:
            pass

        global __cusparseSpGEMMreuse_nnz
        try:
            __cusparseSpGEMMreuse_nnz = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMMreuse_nnz')
        except:
            pass

        global __cusparseSpGEMMreuse_copy
        try:
            __cusparseSpGEMMreuse_copy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMMreuse_copy')
        except:
            pass

        global __cusparseSpGEMMreuse_compute
        try:
            __cusparseSpGEMMreuse_compute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMMreuse_compute')
        except:
            pass

        global __cusparseLoggerSetCallback
        try:
            __cusparseLoggerSetCallback = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseLoggerSetCallback')
        except:
            pass

        global __cusparseLoggerSetFile
        try:
            __cusparseLoggerSetFile = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseLoggerSetFile')
        except:
            pass

        global __cusparseLoggerOpenFile
        try:
            __cusparseLoggerOpenFile = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseLoggerOpenFile')
        except:
            pass

        global __cusparseLoggerSetLevel
        try:
            __cusparseLoggerSetLevel = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseLoggerSetLevel')
        except:
            pass

        global __cusparseLoggerSetMask
        try:
            __cusparseLoggerSetMask = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseLoggerSetMask')
        except:
            pass

        global __cusparseLoggerForceDisable
        try:
            __cusparseLoggerForceDisable = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseLoggerForceDisable')
        except:
            pass

        global __cusparseSpMMOp_createPlan
        try:
            __cusparseSpMMOp_createPlan = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMMOp_createPlan')
        except:
            pass

        global __cusparseSpMMOp
        try:
            __cusparseSpMMOp = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMMOp')
        except:
            pass

        global __cusparseSpMMOp_destroyPlan
        try:
            __cusparseSpMMOp_destroyPlan = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMMOp_destroyPlan')
        except:
            pass

        global __cusparseCscGet
        try:
            __cusparseCscGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCscGet')
        except:
            pass

        global __cusparseCreateConstSpVec
        try:
            __cusparseCreateConstSpVec = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateConstSpVec')
        except:
            pass

        global __cusparseConstSpVecGet
        try:
            __cusparseConstSpVecGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstSpVecGet')
        except:
            pass

        global __cusparseConstSpVecGetValues
        try:
            __cusparseConstSpVecGetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstSpVecGetValues')
        except:
            pass

        global __cusparseCreateConstDnVec
        try:
            __cusparseCreateConstDnVec = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateConstDnVec')
        except:
            pass

        global __cusparseConstDnVecGet
        try:
            __cusparseConstDnVecGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstDnVecGet')
        except:
            pass

        global __cusparseConstDnVecGetValues
        try:
            __cusparseConstDnVecGetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstDnVecGetValues')
        except:
            pass

        global __cusparseConstSpMatGetValues
        try:
            __cusparseConstSpMatGetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstSpMatGetValues')
        except:
            pass

        global __cusparseCreateConstCsr
        try:
            __cusparseCreateConstCsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateConstCsr')
        except:
            pass

        global __cusparseCreateConstCsc
        try:
            __cusparseCreateConstCsc = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateConstCsc')
        except:
            pass

        global __cusparseConstCsrGet
        try:
            __cusparseConstCsrGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstCsrGet')
        except:
            pass

        global __cusparseConstCscGet
        try:
            __cusparseConstCscGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstCscGet')
        except:
            pass

        global __cusparseCreateConstCoo
        try:
            __cusparseCreateConstCoo = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateConstCoo')
        except:
            pass

        global __cusparseConstCooGet
        try:
            __cusparseConstCooGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstCooGet')
        except:
            pass

        global __cusparseCreateConstBlockedEll
        try:
            __cusparseCreateConstBlockedEll = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateConstBlockedEll')
        except:
            pass

        global __cusparseConstBlockedEllGet
        try:
            __cusparseConstBlockedEllGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstBlockedEllGet')
        except:
            pass

        global __cusparseCreateConstDnMat
        try:
            __cusparseCreateConstDnMat = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateConstDnMat')
        except:
            pass

        global __cusparseConstDnMatGet
        try:
            __cusparseConstDnMatGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstDnMatGet')
        except:
            pass

        global __cusparseConstDnMatGetValues
        try:
            __cusparseConstDnMatGetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseConstDnMatGetValues')
        except:
            pass

        global __cusparseSpGEMM_getNumProducts
        try:
            __cusparseSpGEMM_getNumProducts = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMM_getNumProducts')
        except:
            pass

        global __cusparseSpGEMM_estimateMemory
        try:
            __cusparseSpGEMM_estimateMemory = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpGEMM_estimateMemory')
        except:
            pass

        global __cusparseBsrSetStridedBatch
        try:
            __cusparseBsrSetStridedBatch = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseBsrSetStridedBatch')
        except:
            pass

        global __cusparseCreateBsr
        try:
            __cusparseCreateBsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateBsr')
        except:
            pass

        global __cusparseCreateConstBsr
        try:
            __cusparseCreateConstBsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateConstBsr')
        except:
            pass

        global __cusparseCreateSlicedEll
        try:
            __cusparseCreateSlicedEll = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateSlicedEll')
        except:
            pass

        global __cusparseCreateConstSlicedEll
        try:
            __cusparseCreateConstSlicedEll = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseCreateConstSlicedEll')
        except:
            pass

        global __cusparseSpSV_updateMatrix
        try:
            __cusparseSpSV_updateMatrix = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSV_updateMatrix')
        except:
            pass

        global __cusparseSpMV_preprocess
        try:
            __cusparseSpMV_preprocess = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpMV_preprocess')
        except:
            pass

        global __cusparseSpSM_updateMatrix
        try:
            __cusparseSpSM_updateMatrix = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusparseSpSM_updateMatrix')
        except:
            pass

    __py_cusparse_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cusparse()
    cdef dict data = {}

    global __cusparseCreate
    data["__cusparseCreate"] = <intptr_t>__cusparseCreate

    global __cusparseDestroy
    data["__cusparseDestroy"] = <intptr_t>__cusparseDestroy

    global __cusparseGetVersion
    data["__cusparseGetVersion"] = <intptr_t>__cusparseGetVersion

    global __cusparseGetProperty
    data["__cusparseGetProperty"] = <intptr_t>__cusparseGetProperty

    global __cusparseGetErrorName
    data["__cusparseGetErrorName"] = <intptr_t>__cusparseGetErrorName

    global __cusparseGetErrorString
    data["__cusparseGetErrorString"] = <intptr_t>__cusparseGetErrorString

    global __cusparseSetStream
    data["__cusparseSetStream"] = <intptr_t>__cusparseSetStream

    global __cusparseGetStream
    data["__cusparseGetStream"] = <intptr_t>__cusparseGetStream

    global __cusparseGetPointerMode
    data["__cusparseGetPointerMode"] = <intptr_t>__cusparseGetPointerMode

    global __cusparseSetPointerMode
    data["__cusparseSetPointerMode"] = <intptr_t>__cusparseSetPointerMode

    global __cusparseCreateMatDescr
    data["__cusparseCreateMatDescr"] = <intptr_t>__cusparseCreateMatDescr

    global __cusparseDestroyMatDescr
    data["__cusparseDestroyMatDescr"] = <intptr_t>__cusparseDestroyMatDescr

    global __cusparseSetMatType
    data["__cusparseSetMatType"] = <intptr_t>__cusparseSetMatType

    global __cusparseGetMatType
    data["__cusparseGetMatType"] = <intptr_t>__cusparseGetMatType

    global __cusparseSetMatFillMode
    data["__cusparseSetMatFillMode"] = <intptr_t>__cusparseSetMatFillMode

    global __cusparseGetMatFillMode
    data["__cusparseGetMatFillMode"] = <intptr_t>__cusparseGetMatFillMode

    global __cusparseSetMatDiagType
    data["__cusparseSetMatDiagType"] = <intptr_t>__cusparseSetMatDiagType

    global __cusparseGetMatDiagType
    data["__cusparseGetMatDiagType"] = <intptr_t>__cusparseGetMatDiagType

    global __cusparseSetMatIndexBase
    data["__cusparseSetMatIndexBase"] = <intptr_t>__cusparseSetMatIndexBase

    global __cusparseGetMatIndexBase
    data["__cusparseGetMatIndexBase"] = <intptr_t>__cusparseGetMatIndexBase

    global __cusparseSgemvi
    data["__cusparseSgemvi"] = <intptr_t>__cusparseSgemvi

    global __cusparseSgemvi_bufferSize
    data["__cusparseSgemvi_bufferSize"] = <intptr_t>__cusparseSgemvi_bufferSize

    global __cusparseDgemvi
    data["__cusparseDgemvi"] = <intptr_t>__cusparseDgemvi

    global __cusparseDgemvi_bufferSize
    data["__cusparseDgemvi_bufferSize"] = <intptr_t>__cusparseDgemvi_bufferSize

    global __cusparseCgemvi
    data["__cusparseCgemvi"] = <intptr_t>__cusparseCgemvi

    global __cusparseCgemvi_bufferSize
    data["__cusparseCgemvi_bufferSize"] = <intptr_t>__cusparseCgemvi_bufferSize

    global __cusparseZgemvi
    data["__cusparseZgemvi"] = <intptr_t>__cusparseZgemvi

    global __cusparseZgemvi_bufferSize
    data["__cusparseZgemvi_bufferSize"] = <intptr_t>__cusparseZgemvi_bufferSize

    global __cusparseSbsrmv
    data["__cusparseSbsrmv"] = <intptr_t>__cusparseSbsrmv

    global __cusparseDbsrmv
    data["__cusparseDbsrmv"] = <intptr_t>__cusparseDbsrmv

    global __cusparseCbsrmv
    data["__cusparseCbsrmv"] = <intptr_t>__cusparseCbsrmv

    global __cusparseZbsrmv
    data["__cusparseZbsrmv"] = <intptr_t>__cusparseZbsrmv

    global __cusparseSbsrmm
    data["__cusparseSbsrmm"] = <intptr_t>__cusparseSbsrmm

    global __cusparseDbsrmm
    data["__cusparseDbsrmm"] = <intptr_t>__cusparseDbsrmm

    global __cusparseCbsrmm
    data["__cusparseCbsrmm"] = <intptr_t>__cusparseCbsrmm

    global __cusparseZbsrmm
    data["__cusparseZbsrmm"] = <intptr_t>__cusparseZbsrmm

    global __cusparseSgtsv2_bufferSizeExt
    data["__cusparseSgtsv2_bufferSizeExt"] = <intptr_t>__cusparseSgtsv2_bufferSizeExt

    global __cusparseDgtsv2_bufferSizeExt
    data["__cusparseDgtsv2_bufferSizeExt"] = <intptr_t>__cusparseDgtsv2_bufferSizeExt

    global __cusparseCgtsv2_bufferSizeExt
    data["__cusparseCgtsv2_bufferSizeExt"] = <intptr_t>__cusparseCgtsv2_bufferSizeExt

    global __cusparseZgtsv2_bufferSizeExt
    data["__cusparseZgtsv2_bufferSizeExt"] = <intptr_t>__cusparseZgtsv2_bufferSizeExt

    global __cusparseSgtsv2
    data["__cusparseSgtsv2"] = <intptr_t>__cusparseSgtsv2

    global __cusparseDgtsv2
    data["__cusparseDgtsv2"] = <intptr_t>__cusparseDgtsv2

    global __cusparseCgtsv2
    data["__cusparseCgtsv2"] = <intptr_t>__cusparseCgtsv2

    global __cusparseZgtsv2
    data["__cusparseZgtsv2"] = <intptr_t>__cusparseZgtsv2

    global __cusparseSgtsv2_nopivot_bufferSizeExt
    data["__cusparseSgtsv2_nopivot_bufferSizeExt"] = <intptr_t>__cusparseSgtsv2_nopivot_bufferSizeExt

    global __cusparseDgtsv2_nopivot_bufferSizeExt
    data["__cusparseDgtsv2_nopivot_bufferSizeExt"] = <intptr_t>__cusparseDgtsv2_nopivot_bufferSizeExt

    global __cusparseCgtsv2_nopivot_bufferSizeExt
    data["__cusparseCgtsv2_nopivot_bufferSizeExt"] = <intptr_t>__cusparseCgtsv2_nopivot_bufferSizeExt

    global __cusparseZgtsv2_nopivot_bufferSizeExt
    data["__cusparseZgtsv2_nopivot_bufferSizeExt"] = <intptr_t>__cusparseZgtsv2_nopivot_bufferSizeExt

    global __cusparseSgtsv2_nopivot
    data["__cusparseSgtsv2_nopivot"] = <intptr_t>__cusparseSgtsv2_nopivot

    global __cusparseDgtsv2_nopivot
    data["__cusparseDgtsv2_nopivot"] = <intptr_t>__cusparseDgtsv2_nopivot

    global __cusparseCgtsv2_nopivot
    data["__cusparseCgtsv2_nopivot"] = <intptr_t>__cusparseCgtsv2_nopivot

    global __cusparseZgtsv2_nopivot
    data["__cusparseZgtsv2_nopivot"] = <intptr_t>__cusparseZgtsv2_nopivot

    global __cusparseSgtsv2StridedBatch_bufferSizeExt
    data["__cusparseSgtsv2StridedBatch_bufferSizeExt"] = <intptr_t>__cusparseSgtsv2StridedBatch_bufferSizeExt

    global __cusparseDgtsv2StridedBatch_bufferSizeExt
    data["__cusparseDgtsv2StridedBatch_bufferSizeExt"] = <intptr_t>__cusparseDgtsv2StridedBatch_bufferSizeExt

    global __cusparseCgtsv2StridedBatch_bufferSizeExt
    data["__cusparseCgtsv2StridedBatch_bufferSizeExt"] = <intptr_t>__cusparseCgtsv2StridedBatch_bufferSizeExt

    global __cusparseZgtsv2StridedBatch_bufferSizeExt
    data["__cusparseZgtsv2StridedBatch_bufferSizeExt"] = <intptr_t>__cusparseZgtsv2StridedBatch_bufferSizeExt

    global __cusparseSgtsv2StridedBatch
    data["__cusparseSgtsv2StridedBatch"] = <intptr_t>__cusparseSgtsv2StridedBatch

    global __cusparseDgtsv2StridedBatch
    data["__cusparseDgtsv2StridedBatch"] = <intptr_t>__cusparseDgtsv2StridedBatch

    global __cusparseCgtsv2StridedBatch
    data["__cusparseCgtsv2StridedBatch"] = <intptr_t>__cusparseCgtsv2StridedBatch

    global __cusparseZgtsv2StridedBatch
    data["__cusparseZgtsv2StridedBatch"] = <intptr_t>__cusparseZgtsv2StridedBatch

    global __cusparseSgtsvInterleavedBatch_bufferSizeExt
    data["__cusparseSgtsvInterleavedBatch_bufferSizeExt"] = <intptr_t>__cusparseSgtsvInterleavedBatch_bufferSizeExt

    global __cusparseDgtsvInterleavedBatch_bufferSizeExt
    data["__cusparseDgtsvInterleavedBatch_bufferSizeExt"] = <intptr_t>__cusparseDgtsvInterleavedBatch_bufferSizeExt

    global __cusparseCgtsvInterleavedBatch_bufferSizeExt
    data["__cusparseCgtsvInterleavedBatch_bufferSizeExt"] = <intptr_t>__cusparseCgtsvInterleavedBatch_bufferSizeExt

    global __cusparseZgtsvInterleavedBatch_bufferSizeExt
    data["__cusparseZgtsvInterleavedBatch_bufferSizeExt"] = <intptr_t>__cusparseZgtsvInterleavedBatch_bufferSizeExt

    global __cusparseSgtsvInterleavedBatch
    data["__cusparseSgtsvInterleavedBatch"] = <intptr_t>__cusparseSgtsvInterleavedBatch

    global __cusparseDgtsvInterleavedBatch
    data["__cusparseDgtsvInterleavedBatch"] = <intptr_t>__cusparseDgtsvInterleavedBatch

    global __cusparseCgtsvInterleavedBatch
    data["__cusparseCgtsvInterleavedBatch"] = <intptr_t>__cusparseCgtsvInterleavedBatch

    global __cusparseZgtsvInterleavedBatch
    data["__cusparseZgtsvInterleavedBatch"] = <intptr_t>__cusparseZgtsvInterleavedBatch

    global __cusparseSgpsvInterleavedBatch_bufferSizeExt
    data["__cusparseSgpsvInterleavedBatch_bufferSizeExt"] = <intptr_t>__cusparseSgpsvInterleavedBatch_bufferSizeExt

    global __cusparseDgpsvInterleavedBatch_bufferSizeExt
    data["__cusparseDgpsvInterleavedBatch_bufferSizeExt"] = <intptr_t>__cusparseDgpsvInterleavedBatch_bufferSizeExt

    global __cusparseCgpsvInterleavedBatch_bufferSizeExt
    data["__cusparseCgpsvInterleavedBatch_bufferSizeExt"] = <intptr_t>__cusparseCgpsvInterleavedBatch_bufferSizeExt

    global __cusparseZgpsvInterleavedBatch_bufferSizeExt
    data["__cusparseZgpsvInterleavedBatch_bufferSizeExt"] = <intptr_t>__cusparseZgpsvInterleavedBatch_bufferSizeExt

    global __cusparseSgpsvInterleavedBatch
    data["__cusparseSgpsvInterleavedBatch"] = <intptr_t>__cusparseSgpsvInterleavedBatch

    global __cusparseDgpsvInterleavedBatch
    data["__cusparseDgpsvInterleavedBatch"] = <intptr_t>__cusparseDgpsvInterleavedBatch

    global __cusparseCgpsvInterleavedBatch
    data["__cusparseCgpsvInterleavedBatch"] = <intptr_t>__cusparseCgpsvInterleavedBatch

    global __cusparseZgpsvInterleavedBatch
    data["__cusparseZgpsvInterleavedBatch"] = <intptr_t>__cusparseZgpsvInterleavedBatch

    global __cusparseScsrgeam2_bufferSizeExt
    data["__cusparseScsrgeam2_bufferSizeExt"] = <intptr_t>__cusparseScsrgeam2_bufferSizeExt

    global __cusparseDcsrgeam2_bufferSizeExt
    data["__cusparseDcsrgeam2_bufferSizeExt"] = <intptr_t>__cusparseDcsrgeam2_bufferSizeExt

    global __cusparseCcsrgeam2_bufferSizeExt
    data["__cusparseCcsrgeam2_bufferSizeExt"] = <intptr_t>__cusparseCcsrgeam2_bufferSizeExt

    global __cusparseZcsrgeam2_bufferSizeExt
    data["__cusparseZcsrgeam2_bufferSizeExt"] = <intptr_t>__cusparseZcsrgeam2_bufferSizeExt

    global __cusparseXcsrgeam2Nnz
    data["__cusparseXcsrgeam2Nnz"] = <intptr_t>__cusparseXcsrgeam2Nnz

    global __cusparseScsrgeam2
    data["__cusparseScsrgeam2"] = <intptr_t>__cusparseScsrgeam2

    global __cusparseDcsrgeam2
    data["__cusparseDcsrgeam2"] = <intptr_t>__cusparseDcsrgeam2

    global __cusparseCcsrgeam2
    data["__cusparseCcsrgeam2"] = <intptr_t>__cusparseCcsrgeam2

    global __cusparseZcsrgeam2
    data["__cusparseZcsrgeam2"] = <intptr_t>__cusparseZcsrgeam2

    global __cusparseSnnz
    data["__cusparseSnnz"] = <intptr_t>__cusparseSnnz

    global __cusparseDnnz
    data["__cusparseDnnz"] = <intptr_t>__cusparseDnnz

    global __cusparseCnnz
    data["__cusparseCnnz"] = <intptr_t>__cusparseCnnz

    global __cusparseZnnz
    data["__cusparseZnnz"] = <intptr_t>__cusparseZnnz

    global __cusparseXcoo2csr
    data["__cusparseXcoo2csr"] = <intptr_t>__cusparseXcoo2csr

    global __cusparseXcsr2coo
    data["__cusparseXcsr2coo"] = <intptr_t>__cusparseXcsr2coo

    global __cusparseSbsr2csr
    data["__cusparseSbsr2csr"] = <intptr_t>__cusparseSbsr2csr

    global __cusparseDbsr2csr
    data["__cusparseDbsr2csr"] = <intptr_t>__cusparseDbsr2csr

    global __cusparseCbsr2csr
    data["__cusparseCbsr2csr"] = <intptr_t>__cusparseCbsr2csr

    global __cusparseZbsr2csr
    data["__cusparseZbsr2csr"] = <intptr_t>__cusparseZbsr2csr

    global __cusparseSgebsr2gebsc_bufferSize
    data["__cusparseSgebsr2gebsc_bufferSize"] = <intptr_t>__cusparseSgebsr2gebsc_bufferSize

    global __cusparseDgebsr2gebsc_bufferSize
    data["__cusparseDgebsr2gebsc_bufferSize"] = <intptr_t>__cusparseDgebsr2gebsc_bufferSize

    global __cusparseCgebsr2gebsc_bufferSize
    data["__cusparseCgebsr2gebsc_bufferSize"] = <intptr_t>__cusparseCgebsr2gebsc_bufferSize

    global __cusparseZgebsr2gebsc_bufferSize
    data["__cusparseZgebsr2gebsc_bufferSize"] = <intptr_t>__cusparseZgebsr2gebsc_bufferSize

    global __cusparseSgebsr2gebsc_bufferSizeExt
    data["__cusparseSgebsr2gebsc_bufferSizeExt"] = <intptr_t>__cusparseSgebsr2gebsc_bufferSizeExt

    global __cusparseDgebsr2gebsc_bufferSizeExt
    data["__cusparseDgebsr2gebsc_bufferSizeExt"] = <intptr_t>__cusparseDgebsr2gebsc_bufferSizeExt

    global __cusparseCgebsr2gebsc_bufferSizeExt
    data["__cusparseCgebsr2gebsc_bufferSizeExt"] = <intptr_t>__cusparseCgebsr2gebsc_bufferSizeExt

    global __cusparseZgebsr2gebsc_bufferSizeExt
    data["__cusparseZgebsr2gebsc_bufferSizeExt"] = <intptr_t>__cusparseZgebsr2gebsc_bufferSizeExt

    global __cusparseSgebsr2gebsc
    data["__cusparseSgebsr2gebsc"] = <intptr_t>__cusparseSgebsr2gebsc

    global __cusparseDgebsr2gebsc
    data["__cusparseDgebsr2gebsc"] = <intptr_t>__cusparseDgebsr2gebsc

    global __cusparseCgebsr2gebsc
    data["__cusparseCgebsr2gebsc"] = <intptr_t>__cusparseCgebsr2gebsc

    global __cusparseZgebsr2gebsc
    data["__cusparseZgebsr2gebsc"] = <intptr_t>__cusparseZgebsr2gebsc

    global __cusparseScsr2gebsr_bufferSize
    data["__cusparseScsr2gebsr_bufferSize"] = <intptr_t>__cusparseScsr2gebsr_bufferSize

    global __cusparseDcsr2gebsr_bufferSize
    data["__cusparseDcsr2gebsr_bufferSize"] = <intptr_t>__cusparseDcsr2gebsr_bufferSize

    global __cusparseCcsr2gebsr_bufferSize
    data["__cusparseCcsr2gebsr_bufferSize"] = <intptr_t>__cusparseCcsr2gebsr_bufferSize

    global __cusparseZcsr2gebsr_bufferSize
    data["__cusparseZcsr2gebsr_bufferSize"] = <intptr_t>__cusparseZcsr2gebsr_bufferSize

    global __cusparseScsr2gebsr_bufferSizeExt
    data["__cusparseScsr2gebsr_bufferSizeExt"] = <intptr_t>__cusparseScsr2gebsr_bufferSizeExt

    global __cusparseDcsr2gebsr_bufferSizeExt
    data["__cusparseDcsr2gebsr_bufferSizeExt"] = <intptr_t>__cusparseDcsr2gebsr_bufferSizeExt

    global __cusparseCcsr2gebsr_bufferSizeExt
    data["__cusparseCcsr2gebsr_bufferSizeExt"] = <intptr_t>__cusparseCcsr2gebsr_bufferSizeExt

    global __cusparseZcsr2gebsr_bufferSizeExt
    data["__cusparseZcsr2gebsr_bufferSizeExt"] = <intptr_t>__cusparseZcsr2gebsr_bufferSizeExt

    global __cusparseXcsr2gebsrNnz
    data["__cusparseXcsr2gebsrNnz"] = <intptr_t>__cusparseXcsr2gebsrNnz

    global __cusparseScsr2gebsr
    data["__cusparseScsr2gebsr"] = <intptr_t>__cusparseScsr2gebsr

    global __cusparseDcsr2gebsr
    data["__cusparseDcsr2gebsr"] = <intptr_t>__cusparseDcsr2gebsr

    global __cusparseCcsr2gebsr
    data["__cusparseCcsr2gebsr"] = <intptr_t>__cusparseCcsr2gebsr

    global __cusparseZcsr2gebsr
    data["__cusparseZcsr2gebsr"] = <intptr_t>__cusparseZcsr2gebsr

    global __cusparseSgebsr2gebsr_bufferSize
    data["__cusparseSgebsr2gebsr_bufferSize"] = <intptr_t>__cusparseSgebsr2gebsr_bufferSize

    global __cusparseDgebsr2gebsr_bufferSize
    data["__cusparseDgebsr2gebsr_bufferSize"] = <intptr_t>__cusparseDgebsr2gebsr_bufferSize

    global __cusparseCgebsr2gebsr_bufferSize
    data["__cusparseCgebsr2gebsr_bufferSize"] = <intptr_t>__cusparseCgebsr2gebsr_bufferSize

    global __cusparseZgebsr2gebsr_bufferSize
    data["__cusparseZgebsr2gebsr_bufferSize"] = <intptr_t>__cusparseZgebsr2gebsr_bufferSize

    global __cusparseSgebsr2gebsr_bufferSizeExt
    data["__cusparseSgebsr2gebsr_bufferSizeExt"] = <intptr_t>__cusparseSgebsr2gebsr_bufferSizeExt

    global __cusparseDgebsr2gebsr_bufferSizeExt
    data["__cusparseDgebsr2gebsr_bufferSizeExt"] = <intptr_t>__cusparseDgebsr2gebsr_bufferSizeExt

    global __cusparseCgebsr2gebsr_bufferSizeExt
    data["__cusparseCgebsr2gebsr_bufferSizeExt"] = <intptr_t>__cusparseCgebsr2gebsr_bufferSizeExt

    global __cusparseZgebsr2gebsr_bufferSizeExt
    data["__cusparseZgebsr2gebsr_bufferSizeExt"] = <intptr_t>__cusparseZgebsr2gebsr_bufferSizeExt

    global __cusparseXgebsr2gebsrNnz
    data["__cusparseXgebsr2gebsrNnz"] = <intptr_t>__cusparseXgebsr2gebsrNnz

    global __cusparseSgebsr2gebsr
    data["__cusparseSgebsr2gebsr"] = <intptr_t>__cusparseSgebsr2gebsr

    global __cusparseDgebsr2gebsr
    data["__cusparseDgebsr2gebsr"] = <intptr_t>__cusparseDgebsr2gebsr

    global __cusparseCgebsr2gebsr
    data["__cusparseCgebsr2gebsr"] = <intptr_t>__cusparseCgebsr2gebsr

    global __cusparseZgebsr2gebsr
    data["__cusparseZgebsr2gebsr"] = <intptr_t>__cusparseZgebsr2gebsr

    global __cusparseXcoosort_bufferSizeExt
    data["__cusparseXcoosort_bufferSizeExt"] = <intptr_t>__cusparseXcoosort_bufferSizeExt

    global __cusparseXcoosortByRow
    data["__cusparseXcoosortByRow"] = <intptr_t>__cusparseXcoosortByRow

    global __cusparseXcoosortByColumn
    data["__cusparseXcoosortByColumn"] = <intptr_t>__cusparseXcoosortByColumn

    global __cusparseXcsrsort_bufferSizeExt
    data["__cusparseXcsrsort_bufferSizeExt"] = <intptr_t>__cusparseXcsrsort_bufferSizeExt

    global __cusparseXcsrsort
    data["__cusparseXcsrsort"] = <intptr_t>__cusparseXcsrsort

    global __cusparseXcscsort_bufferSizeExt
    data["__cusparseXcscsort_bufferSizeExt"] = <intptr_t>__cusparseXcscsort_bufferSizeExt

    global __cusparseXcscsort
    data["__cusparseXcscsort"] = <intptr_t>__cusparseXcscsort

    global __cusparseCsr2cscEx2
    data["__cusparseCsr2cscEx2"] = <intptr_t>__cusparseCsr2cscEx2

    global __cusparseCsr2cscEx2_bufferSize
    data["__cusparseCsr2cscEx2_bufferSize"] = <intptr_t>__cusparseCsr2cscEx2_bufferSize

    global __cusparseCreateSpVec
    data["__cusparseCreateSpVec"] = <intptr_t>__cusparseCreateSpVec

    global __cusparseDestroySpVec
    data["__cusparseDestroySpVec"] = <intptr_t>__cusparseDestroySpVec

    global __cusparseSpVecGet
    data["__cusparseSpVecGet"] = <intptr_t>__cusparseSpVecGet

    global __cusparseSpVecGetIndexBase
    data["__cusparseSpVecGetIndexBase"] = <intptr_t>__cusparseSpVecGetIndexBase

    global __cusparseSpVecGetValues
    data["__cusparseSpVecGetValues"] = <intptr_t>__cusparseSpVecGetValues

    global __cusparseSpVecSetValues
    data["__cusparseSpVecSetValues"] = <intptr_t>__cusparseSpVecSetValues

    global __cusparseCreateDnVec
    data["__cusparseCreateDnVec"] = <intptr_t>__cusparseCreateDnVec

    global __cusparseDestroyDnVec
    data["__cusparseDestroyDnVec"] = <intptr_t>__cusparseDestroyDnVec

    global __cusparseDnVecGet
    data["__cusparseDnVecGet"] = <intptr_t>__cusparseDnVecGet

    global __cusparseDnVecGetValues
    data["__cusparseDnVecGetValues"] = <intptr_t>__cusparseDnVecGetValues

    global __cusparseDnVecSetValues
    data["__cusparseDnVecSetValues"] = <intptr_t>__cusparseDnVecSetValues

    global __cusparseDestroySpMat
    data["__cusparseDestroySpMat"] = <intptr_t>__cusparseDestroySpMat

    global __cusparseSpMatGetFormat
    data["__cusparseSpMatGetFormat"] = <intptr_t>__cusparseSpMatGetFormat

    global __cusparseSpMatGetIndexBase
    data["__cusparseSpMatGetIndexBase"] = <intptr_t>__cusparseSpMatGetIndexBase

    global __cusparseSpMatGetValues
    data["__cusparseSpMatGetValues"] = <intptr_t>__cusparseSpMatGetValues

    global __cusparseSpMatSetValues
    data["__cusparseSpMatSetValues"] = <intptr_t>__cusparseSpMatSetValues

    global __cusparseSpMatGetSize
    data["__cusparseSpMatGetSize"] = <intptr_t>__cusparseSpMatGetSize

    global __cusparseSpMatGetStridedBatch
    data["__cusparseSpMatGetStridedBatch"] = <intptr_t>__cusparseSpMatGetStridedBatch

    global __cusparseCooSetStridedBatch
    data["__cusparseCooSetStridedBatch"] = <intptr_t>__cusparseCooSetStridedBatch

    global __cusparseCsrSetStridedBatch
    data["__cusparseCsrSetStridedBatch"] = <intptr_t>__cusparseCsrSetStridedBatch

    global __cusparseCreateCsr
    data["__cusparseCreateCsr"] = <intptr_t>__cusparseCreateCsr

    global __cusparseCsrGet
    data["__cusparseCsrGet"] = <intptr_t>__cusparseCsrGet

    global __cusparseCsrSetPointers
    data["__cusparseCsrSetPointers"] = <intptr_t>__cusparseCsrSetPointers

    global __cusparseCreateCoo
    data["__cusparseCreateCoo"] = <intptr_t>__cusparseCreateCoo

    global __cusparseCooGet
    data["__cusparseCooGet"] = <intptr_t>__cusparseCooGet

    global __cusparseCreateDnMat
    data["__cusparseCreateDnMat"] = <intptr_t>__cusparseCreateDnMat

    global __cusparseDestroyDnMat
    data["__cusparseDestroyDnMat"] = <intptr_t>__cusparseDestroyDnMat

    global __cusparseDnMatGet
    data["__cusparseDnMatGet"] = <intptr_t>__cusparseDnMatGet

    global __cusparseDnMatGetValues
    data["__cusparseDnMatGetValues"] = <intptr_t>__cusparseDnMatGetValues

    global __cusparseDnMatSetValues
    data["__cusparseDnMatSetValues"] = <intptr_t>__cusparseDnMatSetValues

    global __cusparseDnMatSetStridedBatch
    data["__cusparseDnMatSetStridedBatch"] = <intptr_t>__cusparseDnMatSetStridedBatch

    global __cusparseDnMatGetStridedBatch
    data["__cusparseDnMatGetStridedBatch"] = <intptr_t>__cusparseDnMatGetStridedBatch

    global __cusparseAxpby
    data["__cusparseAxpby"] = <intptr_t>__cusparseAxpby

    global __cusparseGather
    data["__cusparseGather"] = <intptr_t>__cusparseGather

    global __cusparseScatter
    data["__cusparseScatter"] = <intptr_t>__cusparseScatter

    global __cusparseSpVV_bufferSize
    data["__cusparseSpVV_bufferSize"] = <intptr_t>__cusparseSpVV_bufferSize

    global __cusparseSpVV
    data["__cusparseSpVV"] = <intptr_t>__cusparseSpVV

    global __cusparseSpMV
    data["__cusparseSpMV"] = <intptr_t>__cusparseSpMV

    global __cusparseSpMV_bufferSize
    data["__cusparseSpMV_bufferSize"] = <intptr_t>__cusparseSpMV_bufferSize

    global __cusparseSpMM
    data["__cusparseSpMM"] = <intptr_t>__cusparseSpMM

    global __cusparseSpMM_bufferSize
    data["__cusparseSpMM_bufferSize"] = <intptr_t>__cusparseSpMM_bufferSize

    global __cusparseSpGEMM_createDescr
    data["__cusparseSpGEMM_createDescr"] = <intptr_t>__cusparseSpGEMM_createDescr

    global __cusparseSpGEMM_destroyDescr
    data["__cusparseSpGEMM_destroyDescr"] = <intptr_t>__cusparseSpGEMM_destroyDescr

    global __cusparseSpGEMM_workEstimation
    data["__cusparseSpGEMM_workEstimation"] = <intptr_t>__cusparseSpGEMM_workEstimation

    global __cusparseSpGEMM_compute
    data["__cusparseSpGEMM_compute"] = <intptr_t>__cusparseSpGEMM_compute

    global __cusparseSpGEMM_copy
    data["__cusparseSpGEMM_copy"] = <intptr_t>__cusparseSpGEMM_copy

    global __cusparseCreateCsc
    data["__cusparseCreateCsc"] = <intptr_t>__cusparseCreateCsc

    global __cusparseCscSetPointers
    data["__cusparseCscSetPointers"] = <intptr_t>__cusparseCscSetPointers

    global __cusparseCooSetPointers
    data["__cusparseCooSetPointers"] = <intptr_t>__cusparseCooSetPointers

    global __cusparseSparseToDense_bufferSize
    data["__cusparseSparseToDense_bufferSize"] = <intptr_t>__cusparseSparseToDense_bufferSize

    global __cusparseSparseToDense
    data["__cusparseSparseToDense"] = <intptr_t>__cusparseSparseToDense

    global __cusparseDenseToSparse_bufferSize
    data["__cusparseDenseToSparse_bufferSize"] = <intptr_t>__cusparseDenseToSparse_bufferSize

    global __cusparseDenseToSparse_analysis
    data["__cusparseDenseToSparse_analysis"] = <intptr_t>__cusparseDenseToSparse_analysis

    global __cusparseDenseToSparse_convert
    data["__cusparseDenseToSparse_convert"] = <intptr_t>__cusparseDenseToSparse_convert

    global __cusparseCreateBlockedEll
    data["__cusparseCreateBlockedEll"] = <intptr_t>__cusparseCreateBlockedEll

    global __cusparseBlockedEllGet
    data["__cusparseBlockedEllGet"] = <intptr_t>__cusparseBlockedEllGet

    global __cusparseSpMM_preprocess
    data["__cusparseSpMM_preprocess"] = <intptr_t>__cusparseSpMM_preprocess

    global __cusparseSDDMM_bufferSize
    data["__cusparseSDDMM_bufferSize"] = <intptr_t>__cusparseSDDMM_bufferSize

    global __cusparseSDDMM_preprocess
    data["__cusparseSDDMM_preprocess"] = <intptr_t>__cusparseSDDMM_preprocess

    global __cusparseSDDMM
    data["__cusparseSDDMM"] = <intptr_t>__cusparseSDDMM

    global __cusparseSpMatGetAttribute
    data["__cusparseSpMatGetAttribute"] = <intptr_t>__cusparseSpMatGetAttribute

    global __cusparseSpMatSetAttribute
    data["__cusparseSpMatSetAttribute"] = <intptr_t>__cusparseSpMatSetAttribute

    global __cusparseSpSV_createDescr
    data["__cusparseSpSV_createDescr"] = <intptr_t>__cusparseSpSV_createDescr

    global __cusparseSpSV_destroyDescr
    data["__cusparseSpSV_destroyDescr"] = <intptr_t>__cusparseSpSV_destroyDescr

    global __cusparseSpSV_bufferSize
    data["__cusparseSpSV_bufferSize"] = <intptr_t>__cusparseSpSV_bufferSize

    global __cusparseSpSV_analysis
    data["__cusparseSpSV_analysis"] = <intptr_t>__cusparseSpSV_analysis

    global __cusparseSpSV_solve
    data["__cusparseSpSV_solve"] = <intptr_t>__cusparseSpSV_solve

    global __cusparseSpSM_createDescr
    data["__cusparseSpSM_createDescr"] = <intptr_t>__cusparseSpSM_createDescr

    global __cusparseSpSM_destroyDescr
    data["__cusparseSpSM_destroyDescr"] = <intptr_t>__cusparseSpSM_destroyDescr

    global __cusparseSpSM_bufferSize
    data["__cusparseSpSM_bufferSize"] = <intptr_t>__cusparseSpSM_bufferSize

    global __cusparseSpSM_analysis
    data["__cusparseSpSM_analysis"] = <intptr_t>__cusparseSpSM_analysis

    global __cusparseSpSM_solve
    data["__cusparseSpSM_solve"] = <intptr_t>__cusparseSpSM_solve

    global __cusparseSpGEMMreuse_workEstimation
    data["__cusparseSpGEMMreuse_workEstimation"] = <intptr_t>__cusparseSpGEMMreuse_workEstimation

    global __cusparseSpGEMMreuse_nnz
    data["__cusparseSpGEMMreuse_nnz"] = <intptr_t>__cusparseSpGEMMreuse_nnz

    global __cusparseSpGEMMreuse_copy
    data["__cusparseSpGEMMreuse_copy"] = <intptr_t>__cusparseSpGEMMreuse_copy

    global __cusparseSpGEMMreuse_compute
    data["__cusparseSpGEMMreuse_compute"] = <intptr_t>__cusparseSpGEMMreuse_compute

    global __cusparseLoggerSetCallback
    data["__cusparseLoggerSetCallback"] = <intptr_t>__cusparseLoggerSetCallback

    global __cusparseLoggerSetFile
    data["__cusparseLoggerSetFile"] = <intptr_t>__cusparseLoggerSetFile

    global __cusparseLoggerOpenFile
    data["__cusparseLoggerOpenFile"] = <intptr_t>__cusparseLoggerOpenFile

    global __cusparseLoggerSetLevel
    data["__cusparseLoggerSetLevel"] = <intptr_t>__cusparseLoggerSetLevel

    global __cusparseLoggerSetMask
    data["__cusparseLoggerSetMask"] = <intptr_t>__cusparseLoggerSetMask

    global __cusparseLoggerForceDisable
    data["__cusparseLoggerForceDisable"] = <intptr_t>__cusparseLoggerForceDisable

    global __cusparseSpMMOp_createPlan
    data["__cusparseSpMMOp_createPlan"] = <intptr_t>__cusparseSpMMOp_createPlan

    global __cusparseSpMMOp
    data["__cusparseSpMMOp"] = <intptr_t>__cusparseSpMMOp

    global __cusparseSpMMOp_destroyPlan
    data["__cusparseSpMMOp_destroyPlan"] = <intptr_t>__cusparseSpMMOp_destroyPlan

    global __cusparseCscGet
    data["__cusparseCscGet"] = <intptr_t>__cusparseCscGet

    global __cusparseCreateConstSpVec
    data["__cusparseCreateConstSpVec"] = <intptr_t>__cusparseCreateConstSpVec

    global __cusparseConstSpVecGet
    data["__cusparseConstSpVecGet"] = <intptr_t>__cusparseConstSpVecGet

    global __cusparseConstSpVecGetValues
    data["__cusparseConstSpVecGetValues"] = <intptr_t>__cusparseConstSpVecGetValues

    global __cusparseCreateConstDnVec
    data["__cusparseCreateConstDnVec"] = <intptr_t>__cusparseCreateConstDnVec

    global __cusparseConstDnVecGet
    data["__cusparseConstDnVecGet"] = <intptr_t>__cusparseConstDnVecGet

    global __cusparseConstDnVecGetValues
    data["__cusparseConstDnVecGetValues"] = <intptr_t>__cusparseConstDnVecGetValues

    global __cusparseConstSpMatGetValues
    data["__cusparseConstSpMatGetValues"] = <intptr_t>__cusparseConstSpMatGetValues

    global __cusparseCreateConstCsr
    data["__cusparseCreateConstCsr"] = <intptr_t>__cusparseCreateConstCsr

    global __cusparseCreateConstCsc
    data["__cusparseCreateConstCsc"] = <intptr_t>__cusparseCreateConstCsc

    global __cusparseConstCsrGet
    data["__cusparseConstCsrGet"] = <intptr_t>__cusparseConstCsrGet

    global __cusparseConstCscGet
    data["__cusparseConstCscGet"] = <intptr_t>__cusparseConstCscGet

    global __cusparseCreateConstCoo
    data["__cusparseCreateConstCoo"] = <intptr_t>__cusparseCreateConstCoo

    global __cusparseConstCooGet
    data["__cusparseConstCooGet"] = <intptr_t>__cusparseConstCooGet

    global __cusparseCreateConstBlockedEll
    data["__cusparseCreateConstBlockedEll"] = <intptr_t>__cusparseCreateConstBlockedEll

    global __cusparseConstBlockedEllGet
    data["__cusparseConstBlockedEllGet"] = <intptr_t>__cusparseConstBlockedEllGet

    global __cusparseCreateConstDnMat
    data["__cusparseCreateConstDnMat"] = <intptr_t>__cusparseCreateConstDnMat

    global __cusparseConstDnMatGet
    data["__cusparseConstDnMatGet"] = <intptr_t>__cusparseConstDnMatGet

    global __cusparseConstDnMatGetValues
    data["__cusparseConstDnMatGetValues"] = <intptr_t>__cusparseConstDnMatGetValues

    global __cusparseSpGEMM_getNumProducts
    data["__cusparseSpGEMM_getNumProducts"] = <intptr_t>__cusparseSpGEMM_getNumProducts

    global __cusparseSpGEMM_estimateMemory
    data["__cusparseSpGEMM_estimateMemory"] = <intptr_t>__cusparseSpGEMM_estimateMemory

    global __cusparseBsrSetStridedBatch
    data["__cusparseBsrSetStridedBatch"] = <intptr_t>__cusparseBsrSetStridedBatch

    global __cusparseCreateBsr
    data["__cusparseCreateBsr"] = <intptr_t>__cusparseCreateBsr

    global __cusparseCreateConstBsr
    data["__cusparseCreateConstBsr"] = <intptr_t>__cusparseCreateConstBsr

    global __cusparseCreateSlicedEll
    data["__cusparseCreateSlicedEll"] = <intptr_t>__cusparseCreateSlicedEll

    global __cusparseCreateConstSlicedEll
    data["__cusparseCreateConstSlicedEll"] = <intptr_t>__cusparseCreateConstSlicedEll

    global __cusparseSpSV_updateMatrix
    data["__cusparseSpSV_updateMatrix"] = <intptr_t>__cusparseSpSV_updateMatrix

    global __cusparseSpMV_preprocess
    data["__cusparseSpMV_preprocess"] = <intptr_t>__cusparseSpMV_preprocess

    global __cusparseSpSM_updateMatrix
    data["__cusparseSpSM_updateMatrix"] = <intptr_t>__cusparseSpSM_updateMatrix

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

cdef cusparseStatus_t _cusparseCreate(cusparseHandle_t* handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreate
    _check_or_init_cusparse()
    if __cusparseCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreate is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t*) noexcept nogil>__cusparseCreate)(
        handle)


cdef cusparseStatus_t _cusparseDestroy(cusparseHandle_t handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDestroy
    _check_or_init_cusparse()
    if __cusparseDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDestroy is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t) noexcept nogil>__cusparseDestroy)(
        handle)


cdef cusparseStatus_t _cusparseGetVersion(cusparseHandle_t handle, int* version) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseGetVersion
    _check_or_init_cusparse()
    if __cusparseGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGetVersion is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int*) noexcept nogil>__cusparseGetVersion)(
        handle, version)


cdef cusparseStatus_t _cusparseGetProperty(libraryPropertyType type, int* value) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseGetProperty
    _check_or_init_cusparse()
    if __cusparseGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGetProperty is not found")
    return (<cusparseStatus_t (*)(libraryPropertyType, int*) noexcept nogil>__cusparseGetProperty)(
        type, value)


cdef const char* _cusparseGetErrorName(cusparseStatus_t status) except?NULL nogil:
    global __cusparseGetErrorName
    _check_or_init_cusparse()
    if __cusparseGetErrorName == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGetErrorName is not found")
    return (<const char* (*)(cusparseStatus_t) noexcept nogil>__cusparseGetErrorName)(
        status)


cdef const char* _cusparseGetErrorString(cusparseStatus_t status) except?NULL nogil:
    global __cusparseGetErrorString
    _check_or_init_cusparse()
    if __cusparseGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGetErrorString is not found")
    return (<const char* (*)(cusparseStatus_t) noexcept nogil>__cusparseGetErrorString)(
        status)


cdef cusparseStatus_t _cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSetStream
    _check_or_init_cusparse()
    if __cusparseSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSetStream is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cudaStream_t) noexcept nogil>__cusparseSetStream)(
        handle, streamId)


cdef cusparseStatus_t _cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseGetStream
    _check_or_init_cusparse()
    if __cusparseGetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGetStream is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cudaStream_t*) noexcept nogil>__cusparseGetStream)(
        handle, streamId)


cdef cusparseStatus_t _cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t* mode) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseGetPointerMode
    _check_or_init_cusparse()
    if __cusparseGetPointerMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGetPointerMode is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparsePointerMode_t*) noexcept nogil>__cusparseGetPointerMode)(
        handle, mode)


cdef cusparseStatus_t _cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSetPointerMode
    _check_or_init_cusparse()
    if __cusparseSetPointerMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSetPointerMode is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparsePointerMode_t) noexcept nogil>__cusparseSetPointerMode)(
        handle, mode)


cdef cusparseStatus_t _cusparseCreateMatDescr(cusparseMatDescr_t* descrA) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateMatDescr
    _check_or_init_cusparse()
    if __cusparseCreateMatDescr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateMatDescr is not found")
    return (<cusparseStatus_t (*)(cusparseMatDescr_t*) noexcept nogil>__cusparseCreateMatDescr)(
        descrA)


cdef cusparseStatus_t _cusparseDestroyMatDescr(cusparseMatDescr_t descrA) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDestroyMatDescr
    _check_or_init_cusparse()
    if __cusparseDestroyMatDescr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDestroyMatDescr is not found")
    return (<cusparseStatus_t (*)(cusparseMatDescr_t) noexcept nogil>__cusparseDestroyMatDescr)(
        descrA)


cdef cusparseStatus_t _cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSetMatType
    _check_or_init_cusparse()
    if __cusparseSetMatType == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSetMatType is not found")
    return (<cusparseStatus_t (*)(cusparseMatDescr_t, cusparseMatrixType_t) noexcept nogil>__cusparseSetMatType)(
        descrA, type)


cdef cusparseMatrixType_t _cusparseGetMatType(const cusparseMatDescr_t descrA) except?_CUSPARSEMATRIXTYPE_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseGetMatType
    _check_or_init_cusparse()
    if __cusparseGetMatType == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGetMatType is not found")
    return (<cusparseMatrixType_t (*)(const cusparseMatDescr_t) noexcept nogil>__cusparseGetMatType)(
        descrA)


cdef cusparseStatus_t _cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSetMatFillMode
    _check_or_init_cusparse()
    if __cusparseSetMatFillMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSetMatFillMode is not found")
    return (<cusparseStatus_t (*)(cusparseMatDescr_t, cusparseFillMode_t) noexcept nogil>__cusparseSetMatFillMode)(
        descrA, fillMode)


cdef cusparseFillMode_t _cusparseGetMatFillMode(const cusparseMatDescr_t descrA) except?_CUSPARSEFILLMODE_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseGetMatFillMode
    _check_or_init_cusparse()
    if __cusparseGetMatFillMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGetMatFillMode is not found")
    return (<cusparseFillMode_t (*)(const cusparseMatDescr_t) noexcept nogil>__cusparseGetMatFillMode)(
        descrA)


cdef cusparseStatus_t _cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSetMatDiagType
    _check_or_init_cusparse()
    if __cusparseSetMatDiagType == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSetMatDiagType is not found")
    return (<cusparseStatus_t (*)(cusparseMatDescr_t, cusparseDiagType_t) noexcept nogil>__cusparseSetMatDiagType)(
        descrA, diagType)


cdef cusparseDiagType_t _cusparseGetMatDiagType(const cusparseMatDescr_t descrA) except?_CUSPARSEDIAGTYPE_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseGetMatDiagType
    _check_or_init_cusparse()
    if __cusparseGetMatDiagType == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGetMatDiagType is not found")
    return (<cusparseDiagType_t (*)(const cusparseMatDescr_t) noexcept nogil>__cusparseGetMatDiagType)(
        descrA)


cdef cusparseStatus_t _cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSetMatIndexBase
    _check_or_init_cusparse()
    if __cusparseSetMatIndexBase == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSetMatIndexBase is not found")
    return (<cusparseStatus_t (*)(cusparseMatDescr_t, cusparseIndexBase_t) noexcept nogil>__cusparseSetMatIndexBase)(
        descrA, base)


cdef cusparseIndexBase_t _cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) except?_CUSPARSEINDEXBASE_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseGetMatIndexBase
    _check_or_init_cusparse()
    if __cusparseGetMatIndexBase == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGetMatIndexBase is not found")
    return (<cusparseIndexBase_t (*)(const cusparseMatDescr_t) noexcept nogil>__cusparseGetMatIndexBase)(
        descrA)


cdef cusparseStatus_t _cusparseSgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const float* alpha, const float* A, int lda, int nnz, const float* xVal, const int* xInd, const float* beta, float* y, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgemvi
    _check_or_init_cusparse()
    if __cusparseSgemvi == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgemvi is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const float*, const float*, int, int, const float*, const int*, const float*, float*, cusparseIndexBase_t, void*) noexcept nogil>__cusparseSgemvi)(
        handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)


cdef cusparseStatus_t _cusparseSgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgemvi_bufferSize
    _check_or_init_cusparse()
    if __cusparseSgemvi_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgemvi_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, int, int*) noexcept nogil>__cusparseSgemvi_bufferSize)(
        handle, transA, m, n, nnz, pBufferSize)


cdef cusparseStatus_t _cusparseDgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const double* alpha, const double* A, int lda, int nnz, const double* xVal, const int* xInd, const double* beta, double* y, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgemvi
    _check_or_init_cusparse()
    if __cusparseDgemvi == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgemvi is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const double*, const double*, int, int, const double*, const int*, const double*, double*, cusparseIndexBase_t, void*) noexcept nogil>__cusparseDgemvi)(
        handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)


cdef cusparseStatus_t _cusparseDgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgemvi_bufferSize
    _check_or_init_cusparse()
    if __cusparseDgemvi_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgemvi_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, int, int*) noexcept nogil>__cusparseDgemvi_bufferSize)(
        handle, transA, m, n, nnz, pBufferSize)


cdef cusparseStatus_t _cusparseCgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, int nnz, const cuComplex* xVal, const int* xInd, const cuComplex* beta, cuComplex* y, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgemvi
    _check_or_init_cusparse()
    if __cusparseCgemvi == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgemvi is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cuComplex*, const cuComplex*, int, int, const cuComplex*, const int*, const cuComplex*, cuComplex*, cusparseIndexBase_t, void*) noexcept nogil>__cusparseCgemvi)(
        handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)


cdef cusparseStatus_t _cusparseCgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgemvi_bufferSize
    _check_or_init_cusparse()
    if __cusparseCgemvi_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgemvi_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, int, int*) noexcept nogil>__cusparseCgemvi_bufferSize)(
        handle, transA, m, n, nnz, pBufferSize)


cdef cusparseStatus_t _cusparseZgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, int nnz, const cuDoubleComplex* xVal, const int* xInd, const cuDoubleComplex* beta, cuDoubleComplex* y, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgemvi
    _check_or_init_cusparse()
    if __cusparseZgemvi == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgemvi is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, int, const cuDoubleComplex*, const int*, const cuDoubleComplex*, cuDoubleComplex*, cusparseIndexBase_t, void*) noexcept nogil>__cusparseZgemvi)(
        handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)


cdef cusparseStatus_t _cusparseZgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgemvi_bufferSize
    _check_or_init_cusparse()
    if __cusparseZgemvi_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgemvi_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, int, int*) noexcept nogil>__cusparseZgemvi_bufferSize)(
        handle, transA, m, n, nnz, pBufferSize)


cdef cusparseStatus_t _cusparseSbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const float* x, const float* beta, float* y) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSbsrmv
    _check_or_init_cusparse()
    if __cusparseSbsrmv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSbsrmv is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int, const float*, const cusparseMatDescr_t, const float*, const int*, const int*, int, const float*, const float*, float*) noexcept nogil>__cusparseSbsrmv)(
        handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)


cdef cusparseStatus_t _cusparseDbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const double* x, const double* beta, double* y) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDbsrmv
    _check_or_init_cusparse()
    if __cusparseDbsrmv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDbsrmv is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int, const double*, const cusparseMatDescr_t, const double*, const int*, const int*, int, const double*, const double*, double*) noexcept nogil>__cusparseDbsrmv)(
        handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)


cdef cusparseStatus_t _cusparseCbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cuComplex* x, const cuComplex* beta, cuComplex* y) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCbsrmv
    _check_or_init_cusparse()
    if __cusparseCbsrmv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCbsrmv is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int, const cuComplex*, const cusparseMatDescr_t, const cuComplex*, const int*, const int*, int, const cuComplex*, const cuComplex*, cuComplex*) noexcept nogil>__cusparseCbsrmv)(
        handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)


cdef cusparseStatus_t _cusparseZbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZbsrmv
    _check_or_init_cusparse()
    if __cusparseZbsrmv == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZbsrmv is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int, const cuDoubleComplex*, const cusparseMatDescr_t, const cuDoubleComplex*, const int*, const int*, int, const cuDoubleComplex*, const cuDoubleComplex*, cuDoubleComplex*) noexcept nogil>__cusparseZbsrmv)(
        handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)


cdef cusparseStatus_t _cusparseSbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const float* B, const int ldb, const float* beta, float* C, int ldc) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSbsrmm
    _check_or_init_cusparse()
    if __cusparseSbsrmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSbsrmm is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int, int, int, const float*, const cusparseMatDescr_t, const float*, const int*, const int*, const int, const float*, const int, const float*, float*, int) noexcept nogil>__cusparseSbsrmm)(
        handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)


cdef cusparseStatus_t _cusparseDbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const double* B, const int ldb, const double* beta, double* C, int ldc) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDbsrmm
    _check_or_init_cusparse()
    if __cusparseDbsrmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDbsrmm is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int, int, int, const double*, const cusparseMatDescr_t, const double*, const int*, const int*, const int, const double*, const int, const double*, double*, int) noexcept nogil>__cusparseDbsrmm)(
        handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)


cdef cusparseStatus_t _cusparseCbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const cuComplex* B, const int ldb, const cuComplex* beta, cuComplex* C, int ldc) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCbsrmm
    _check_or_init_cusparse()
    if __cusparseCbsrmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCbsrmm is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int, int, int, const cuComplex*, const cusparseMatDescr_t, const cuComplex*, const int*, const int*, const int, const cuComplex*, const int, const cuComplex*, cuComplex*, int) noexcept nogil>__cusparseCbsrmm)(
        handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)


cdef cusparseStatus_t _cusparseZbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const cuDoubleComplex* B, const int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZbsrmm
    _check_or_init_cusparse()
    if __cusparseZbsrmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZbsrmm is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int, int, int, const cuDoubleComplex*, const cusparseMatDescr_t, const cuDoubleComplex*, const int*, const int*, const int, const cuDoubleComplex*, const int, const cuDoubleComplex*, cuDoubleComplex*, int) noexcept nogil>__cusparseZbsrmm)(
        handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)


cdef cusparseStatus_t _cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgtsv2_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseSgtsv2_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgtsv2_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const float*, const float*, const float*, const float*, int, size_t*) noexcept nogil>__cusparseSgtsv2_bufferSizeExt)(
        handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgtsv2_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseDgtsv2_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgtsv2_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const double*, const double*, const double*, const double*, int, size_t*) noexcept nogil>__cusparseDgtsv2_bufferSizeExt)(
        handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgtsv2_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseCgtsv2_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgtsv2_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex*, const cuComplex*, const cuComplex*, const cuComplex*, int, size_t*) noexcept nogil>__cusparseCgtsv2_bufferSizeExt)(
        handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseZgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgtsv2_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseZgtsv2_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgtsv2_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, int, size_t*) noexcept nogil>__cusparseZgtsv2_bufferSizeExt)(
        handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseSgtsv2(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgtsv2
    _check_or_init_cusparse()
    if __cusparseSgtsv2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgtsv2 is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const float*, const float*, const float*, float*, int, void*) noexcept nogil>__cusparseSgtsv2)(
        handle, m, n, dl, d, du, B, ldb, pBuffer)


cdef cusparseStatus_t _cusparseDgtsv2(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgtsv2
    _check_or_init_cusparse()
    if __cusparseDgtsv2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgtsv2 is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const double*, const double*, const double*, double*, int, void*) noexcept nogil>__cusparseDgtsv2)(
        handle, m, n, dl, d, du, B, ldb, pBuffer)


cdef cusparseStatus_t _cusparseCgtsv2(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgtsv2
    _check_or_init_cusparse()
    if __cusparseCgtsv2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgtsv2 is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex*, const cuComplex*, const cuComplex*, cuComplex*, int, void*) noexcept nogil>__cusparseCgtsv2)(
        handle, m, n, dl, d, du, B, ldb, pBuffer)


cdef cusparseStatus_t _cusparseZgtsv2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgtsv2
    _check_or_init_cusparse()
    if __cusparseZgtsv2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgtsv2 is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, cuDoubleComplex*, int, void*) noexcept nogil>__cusparseZgtsv2)(
        handle, m, n, dl, d, du, B, ldb, pBuffer)


cdef cusparseStatus_t _cusparseSgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgtsv2_nopivot_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseSgtsv2_nopivot_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgtsv2_nopivot_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const float*, const float*, const float*, const float*, int, size_t*) noexcept nogil>__cusparseSgtsv2_nopivot_bufferSizeExt)(
        handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgtsv2_nopivot_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseDgtsv2_nopivot_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgtsv2_nopivot_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const double*, const double*, const double*, const double*, int, size_t*) noexcept nogil>__cusparseDgtsv2_nopivot_bufferSizeExt)(
        handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgtsv2_nopivot_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseCgtsv2_nopivot_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgtsv2_nopivot_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex*, const cuComplex*, const cuComplex*, const cuComplex*, int, size_t*) noexcept nogil>__cusparseCgtsv2_nopivot_bufferSizeExt)(
        handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseZgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* B, int ldb, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgtsv2_nopivot_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseZgtsv2_nopivot_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgtsv2_nopivot_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, int, size_t*) noexcept nogil>__cusparseZgtsv2_nopivot_bufferSizeExt)(
        handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseSgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgtsv2_nopivot
    _check_or_init_cusparse()
    if __cusparseSgtsv2_nopivot == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgtsv2_nopivot is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const float*, const float*, const float*, float*, int, void*) noexcept nogil>__cusparseSgtsv2_nopivot)(
        handle, m, n, dl, d, du, B, ldb, pBuffer)


cdef cusparseStatus_t _cusparseDgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgtsv2_nopivot
    _check_or_init_cusparse()
    if __cusparseDgtsv2_nopivot == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgtsv2_nopivot is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const double*, const double*, const double*, double*, int, void*) noexcept nogil>__cusparseDgtsv2_nopivot)(
        handle, m, n, dl, d, du, B, ldb, pBuffer)


cdef cusparseStatus_t _cusparseCgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgtsv2_nopivot
    _check_or_init_cusparse()
    if __cusparseCgtsv2_nopivot == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgtsv2_nopivot is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex*, const cuComplex*, const cuComplex*, cuComplex*, int, void*) noexcept nogil>__cusparseCgtsv2_nopivot)(
        handle, m, n, dl, d, du, B, ldb, pBuffer)


cdef cusparseStatus_t _cusparseZgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* B, int ldb, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgtsv2_nopivot
    _check_or_init_cusparse()
    if __cusparseZgtsv2_nopivot == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgtsv2_nopivot is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, cuDoubleComplex*, int, void*) noexcept nogil>__cusparseZgtsv2_nopivot)(
        handle, m, n, dl, d, du, B, ldb, pBuffer)


cdef cusparseStatus_t _cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, int batchStride, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgtsv2StridedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseSgtsv2StridedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgtsv2StridedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, const float*, const float*, const float*, const float*, int, int, size_t*) noexcept nogil>__cusparseSgtsv2StridedBatch_bufferSizeExt)(
        handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, int batchStride, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgtsv2StridedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseDgtsv2StridedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgtsv2StridedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, const double*, const double*, const double*, const double*, int, int, size_t*) noexcept nogil>__cusparseDgtsv2StridedBatch_bufferSizeExt)(
        handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* x, int batchCount, int batchStride, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgtsv2StridedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseCgtsv2StridedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgtsv2StridedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, const cuComplex*, const cuComplex*, const cuComplex*, const cuComplex*, int, int, size_t*) noexcept nogil>__cusparseCgtsv2StridedBatch_bufferSizeExt)(
        handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseZgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* x, int batchCount, int batchStride, size_t* bufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgtsv2StridedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseZgtsv2StridedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgtsv2StridedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, int, int, size_t*) noexcept nogil>__cusparseZgtsv2StridedBatch_bufferSizeExt)(
        handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)


cdef cusparseStatus_t _cusparseSgtsv2StridedBatch(cusparseHandle_t handle, int m, const float* dl, const float* d, const float* du, float* x, int batchCount, int batchStride, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgtsv2StridedBatch
    _check_or_init_cusparse()
    if __cusparseSgtsv2StridedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgtsv2StridedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, const float*, const float*, const float*, float*, int, int, void*) noexcept nogil>__cusparseSgtsv2StridedBatch)(
        handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)


cdef cusparseStatus_t _cusparseDgtsv2StridedBatch(cusparseHandle_t handle, int m, const double* dl, const double* d, const double* du, double* x, int batchCount, int batchStride, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgtsv2StridedBatch
    _check_or_init_cusparse()
    if __cusparseDgtsv2StridedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgtsv2StridedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, const double*, const double*, const double*, double*, int, int, void*) noexcept nogil>__cusparseDgtsv2StridedBatch)(
        handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)


cdef cusparseStatus_t _cusparseCgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* x, int batchCount, int batchStride, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgtsv2StridedBatch
    _check_or_init_cusparse()
    if __cusparseCgtsv2StridedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgtsv2StridedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, const cuComplex*, const cuComplex*, const cuComplex*, cuComplex*, int, int, void*) noexcept nogil>__cusparseCgtsv2StridedBatch)(
        handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)


cdef cusparseStatus_t _cusparseZgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* x, int batchCount, int batchStride, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgtsv2StridedBatch
    _check_or_init_cusparse()
    if __cusparseZgtsv2StridedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgtsv2StridedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, cuDoubleComplex*, int, int, void*) noexcept nogil>__cusparseZgtsv2StridedBatch)(
        handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)


cdef cusparseStatus_t _cusparseSgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgtsvInterleavedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseSgtsvInterleavedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgtsvInterleavedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const float*, const float*, const float*, const float*, int, size_t*) noexcept nogil>__cusparseSgtsvInterleavedBatch_bufferSizeExt)(
        handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseDgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgtsvInterleavedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseDgtsvInterleavedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgtsvInterleavedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const double*, const double*, const double*, const double*, int, size_t*) noexcept nogil>__cusparseDgtsvInterleavedBatch_bufferSizeExt)(
        handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseCgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgtsvInterleavedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseCgtsvInterleavedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgtsvInterleavedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex*, const cuComplex*, const cuComplex*, const cuComplex*, int, size_t*) noexcept nogil>__cusparseCgtsvInterleavedBatch_bufferSizeExt)(
        handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseZgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgtsvInterleavedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseZgtsvInterleavedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgtsvInterleavedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, int, size_t*) noexcept nogil>__cusparseZgtsvInterleavedBatch_bufferSizeExt)(
        handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseSgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float* dl, float* d, float* du, float* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgtsvInterleavedBatch
    _check_or_init_cusparse()
    if __cusparseSgtsvInterleavedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgtsvInterleavedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, float*, float*, float*, float*, int, void*) noexcept nogil>__cusparseSgtsvInterleavedBatch)(
        handle, algo, m, dl, d, du, x, batchCount, pBuffer)


cdef cusparseStatus_t _cusparseDgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double* dl, double* d, double* du, double* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgtsvInterleavedBatch
    _check_or_init_cusparse()
    if __cusparseDgtsvInterleavedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgtsvInterleavedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, double*, double*, double*, double*, int, void*) noexcept nogil>__cusparseDgtsvInterleavedBatch)(
        handle, algo, m, dl, d, du, x, batchCount, pBuffer)


cdef cusparseStatus_t _cusparseCgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgtsvInterleavedBatch
    _check_or_init_cusparse()
    if __cusparseCgtsvInterleavedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgtsvInterleavedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, cuComplex*, cuComplex*, cuComplex*, cuComplex*, int, void*) noexcept nogil>__cusparseCgtsvInterleavedBatch)(
        handle, algo, m, dl, d, du, x, batchCount, pBuffer)


cdef cusparseStatus_t _cusparseZgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuDoubleComplex* dl, cuDoubleComplex* d, cuDoubleComplex* du, cuDoubleComplex* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgtsvInterleavedBatch
    _check_or_init_cusparse()
    if __cusparseZgtsvInterleavedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgtsvInterleavedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, int, void*) noexcept nogil>__cusparseZgtsvInterleavedBatch)(
        handle, algo, m, dl, d, du, x, batchCount, pBuffer)


cdef cusparseStatus_t _cusparseSgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const float* ds, const float* dl, const float* d, const float* du, const float* dw, const float* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgpsvInterleavedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseSgpsvInterleavedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgpsvInterleavedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const float*, const float*, const float*, const float*, const float*, const float*, int, size_t*) noexcept nogil>__cusparseSgpsvInterleavedBatch_bufferSizeExt)(
        handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseDgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const double* ds, const double* dl, const double* d, const double* du, const double* dw, const double* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgpsvInterleavedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseDgpsvInterleavedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgpsvInterleavedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const double*, const double*, const double*, const double*, const double*, const double*, int, size_t*) noexcept nogil>__cusparseDgpsvInterleavedBatch_bufferSizeExt)(
        handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseCgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuComplex* ds, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* dw, const cuComplex* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgpsvInterleavedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseCgpsvInterleavedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgpsvInterleavedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex*, const cuComplex*, const cuComplex*, const cuComplex*, const cuComplex*, const cuComplex*, int, size_t*) noexcept nogil>__cusparseCgpsvInterleavedBatch_bufferSizeExt)(
        handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseZgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuDoubleComplex* ds, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* dw, const cuDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgpsvInterleavedBatch_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseZgpsvInterleavedBatch_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgpsvInterleavedBatch_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, int, size_t*) noexcept nogil>__cusparseZgpsvInterleavedBatch_bufferSizeExt)(
        handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseSgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float* ds, float* dl, float* d, float* du, float* dw, float* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgpsvInterleavedBatch
    _check_or_init_cusparse()
    if __cusparseSgpsvInterleavedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgpsvInterleavedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, float*, float*, float*, float*, float*, float*, int, void*) noexcept nogil>__cusparseSgpsvInterleavedBatch)(
        handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)


cdef cusparseStatus_t _cusparseDgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double* ds, double* dl, double* d, double* du, double* dw, double* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgpsvInterleavedBatch
    _check_or_init_cusparse()
    if __cusparseDgpsvInterleavedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgpsvInterleavedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, double*, double*, double*, double*, double*, double*, int, void*) noexcept nogil>__cusparseDgpsvInterleavedBatch)(
        handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)


cdef cusparseStatus_t _cusparseCgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex* ds, cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* dw, cuComplex* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgpsvInterleavedBatch
    _check_or_init_cusparse()
    if __cusparseCgpsvInterleavedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgpsvInterleavedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, cuComplex*, cuComplex*, cuComplex*, cuComplex*, cuComplex*, cuComplex*, int, void*) noexcept nogil>__cusparseCgpsvInterleavedBatch)(
        handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)


cdef cusparseStatus_t _cusparseZgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuDoubleComplex* ds, cuDoubleComplex* dl, cuDoubleComplex* d, cuDoubleComplex* du, cuDoubleComplex* dw, cuDoubleComplex* x, int batchCount, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgpsvInterleavedBatch
    _check_or_init_cusparse()
    if __cusparseZgpsvInterleavedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgpsvInterleavedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, int, void*) noexcept nogil>__cusparseZgpsvInterleavedBatch)(
        handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)


cdef cusparseStatus_t _cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseScsrgeam2_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseScsrgeam2_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseScsrgeam2_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const float*, const cusparseMatDescr_t, int, const float*, const int*, const int*, const float*, const cusparseMatDescr_t, int, const float*, const int*, const int*, const cusparseMatDescr_t, const float*, const int*, const int*, size_t*) noexcept nogil>__cusparseScsrgeam2_bufferSizeExt)(
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDcsrgeam2_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseDcsrgeam2_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDcsrgeam2_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const double*, const cusparseMatDescr_t, int, const double*, const int*, const int*, const double*, const cusparseMatDescr_t, int, const double*, const int*, const int*, const cusparseMatDescr_t, const double*, const int*, const int*, size_t*) noexcept nogil>__cusparseDcsrgeam2_bufferSizeExt)(
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseCcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const cuComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCcsrgeam2_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseCcsrgeam2_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCcsrgeam2_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex*, const cusparseMatDescr_t, int, const cuComplex*, const int*, const int*, const cuComplex*, const cusparseMatDescr_t, int, const cuComplex*, const int*, const int*, const cusparseMatDescr_t, const cuComplex*, const int*, const int*, size_t*) noexcept nogil>__cusparseCcsrgeam2_bufferSizeExt)(
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseZcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZcsrgeam2_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseZcsrgeam2_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZcsrgeam2_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex*, const cusparseMatDescr_t, int, const cuDoubleComplex*, const int*, const int*, const cuDoubleComplex*, const cusparseMatDescr_t, int, const cuDoubleComplex*, const int*, const int*, const cusparseMatDescr_t, const cuDoubleComplex*, const int*, const int*, size_t*) noexcept nogil>__cusparseZcsrgeam2_bufferSizeExt)(
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcsrgeam2Nnz
    _check_or_init_cusparse()
    if __cusparseXcsrgeam2Nnz == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcsrgeam2Nnz is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, int, const int*, const int*, const cusparseMatDescr_t, int, const int*, const int*, const cusparseMatDescr_t, int*, int*, void*) noexcept nogil>__cusparseXcsrgeam2Nnz)(
        handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace)


cdef cusparseStatus_t _cusparseScsrgeam2(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseScsrgeam2
    _check_or_init_cusparse()
    if __cusparseScsrgeam2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseScsrgeam2 is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const float*, const cusparseMatDescr_t, int, const float*, const int*, const int*, const float*, const cusparseMatDescr_t, int, const float*, const int*, const int*, const cusparseMatDescr_t, float*, int*, int*, void*) noexcept nogil>__cusparseScsrgeam2)(
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)


cdef cusparseStatus_t _cusparseDcsrgeam2(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDcsrgeam2
    _check_or_init_cusparse()
    if __cusparseDcsrgeam2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDcsrgeam2 is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const double*, const cusparseMatDescr_t, int, const double*, const int*, const int*, const double*, const cusparseMatDescr_t, int, const double*, const int*, const int*, const cusparseMatDescr_t, double*, int*, int*, void*) noexcept nogil>__cusparseDcsrgeam2)(
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)


cdef cusparseStatus_t _cusparseCcsrgeam2(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCcsrgeam2
    _check_or_init_cusparse()
    if __cusparseCcsrgeam2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCcsrgeam2 is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex*, const cusparseMatDescr_t, int, const cuComplex*, const int*, const int*, const cuComplex*, const cusparseMatDescr_t, int, const cuComplex*, const int*, const int*, const cusparseMatDescr_t, cuComplex*, int*, int*, void*) noexcept nogil>__cusparseCcsrgeam2)(
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)


cdef cusparseStatus_t _cusparseZcsrgeam2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZcsrgeam2
    _check_or_init_cusparse()
    if __cusparseZcsrgeam2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZcsrgeam2 is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex*, const cusparseMatDescr_t, int, const cuDoubleComplex*, const int*, const int*, const cuDoubleComplex*, const cusparseMatDescr_t, int, const cuDoubleComplex*, const int*, const int*, const cusparseMatDescr_t, cuDoubleComplex*, int*, int*, void*) noexcept nogil>__cusparseZcsrgeam2)(
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)


cdef cusparseStatus_t _cusparseSnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSnnz
    _check_or_init_cusparse()
    if __cusparseSnnz == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSnnz is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const float*, int, int*, int*) noexcept nogil>__cusparseSnnz)(
        handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)


cdef cusparseStatus_t _cusparseDnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDnnz
    _check_or_init_cusparse()
    if __cusparseDnnz == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDnnz is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const double*, int, int*, int*) noexcept nogil>__cusparseDnnz)(
        handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)


cdef cusparseStatus_t _cusparseCnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCnnz
    _check_or_init_cusparse()
    if __cusparseCnnz == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCnnz is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const cuComplex*, int, int*, int*) noexcept nogil>__cusparseCnnz)(
        handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)


cdef cusparseStatus_t _cusparseZnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZnnz
    _check_or_init_cusparse()
    if __cusparseZnnz == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZnnz is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex*, int, int*, int*) noexcept nogil>__cusparseZnnz)(
        handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)


cdef cusparseStatus_t _cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz, int m, int* csrSortedRowPtr, cusparseIndexBase_t idxBase) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcoo2csr
    _check_or_init_cusparse()
    if __cusparseXcoo2csr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcoo2csr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, const int*, int, int, int*, cusparseIndexBase_t) noexcept nogil>__cusparseXcoo2csr)(
        handle, cooRowInd, nnz, m, csrSortedRowPtr, idxBase)


cdef cusparseStatus_t _cusparseXcsr2coo(cusparseHandle_t handle, const int* csrSortedRowPtr, int nnz, int m, int* cooRowInd, cusparseIndexBase_t idxBase) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcsr2coo
    _check_or_init_cusparse()
    if __cusparseXcsr2coo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcsr2coo is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, const int*, int, int, int*, cusparseIndexBase_t) noexcept nogil>__cusparseXcsr2coo)(
        handle, csrSortedRowPtr, nnz, m, cooRowInd, idxBase)


cdef cusparseStatus_t _cusparseSbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSbsr2csr
    _check_or_init_cusparse()
    if __cusparseSbsr2csr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSbsr2csr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const float*, const int*, const int*, int, const cusparseMatDescr_t, float*, int*, int*) noexcept nogil>__cusparseSbsr2csr)(
        handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)


cdef cusparseStatus_t _cusparseDbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDbsr2csr
    _check_or_init_cusparse()
    if __cusparseDbsr2csr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDbsr2csr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const double*, const int*, const int*, int, const cusparseMatDescr_t, double*, int*, int*) noexcept nogil>__cusparseDbsr2csr)(
        handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)


cdef cusparseStatus_t _cusparseCbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCbsr2csr
    _check_or_init_cusparse()
    if __cusparseCbsr2csr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCbsr2csr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const cuComplex*, const int*, const int*, int, const cusparseMatDescr_t, cuComplex*, int*, int*) noexcept nogil>__cusparseCbsr2csr)(
        handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)


cdef cusparseStatus_t _cusparseZbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZbsr2csr
    _check_or_init_cusparse()
    if __cusparseZbsr2csr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZbsr2csr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex*, const int*, const int*, int, const cusparseMatDescr_t, cuDoubleComplex*, int*, int*) noexcept nogil>__cusparseZbsr2csr)(
        handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)


cdef cusparseStatus_t _cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgebsr2gebsc_bufferSize
    _check_or_init_cusparse()
    if __cusparseSgebsr2gebsc_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgebsr2gebsc_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const float*, const int*, const int*, int, int, int*) noexcept nogil>__cusparseSgebsr2gebsc_bufferSize)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgebsr2gebsc_bufferSize
    _check_or_init_cusparse()
    if __cusparseDgebsr2gebsc_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgebsr2gebsc_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const double*, const int*, const int*, int, int, int*) noexcept nogil>__cusparseDgebsr2gebsc_bufferSize)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgebsr2gebsc_bufferSize
    _check_or_init_cusparse()
    if __cusparseCgebsr2gebsc_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgebsr2gebsc_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuComplex*, const int*, const int*, int, int, int*) noexcept nogil>__cusparseCgebsr2gebsc_bufferSize)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgebsr2gebsc_bufferSize
    _check_or_init_cusparse()
    if __cusparseZgebsr2gebsc_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgebsr2gebsc_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuDoubleComplex*, const int*, const int*, int, int, int*) noexcept nogil>__cusparseZgebsr2gebsc_bufferSize)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseSgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgebsr2gebsc_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseSgebsr2gebsc_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgebsr2gebsc_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const float*, const int*, const int*, int, int, size_t*) noexcept nogil>__cusparseSgebsr2gebsc_bufferSizeExt)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSize)


cdef cusparseStatus_t _cusparseDgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgebsr2gebsc_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseDgebsr2gebsc_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgebsr2gebsc_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const double*, const int*, const int*, int, int, size_t*) noexcept nogil>__cusparseDgebsr2gebsc_bufferSizeExt)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSize)


cdef cusparseStatus_t _cusparseCgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgebsr2gebsc_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseCgebsr2gebsc_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgebsr2gebsc_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuComplex*, const int*, const int*, int, int, size_t*) noexcept nogil>__cusparseCgebsr2gebsc_bufferSizeExt)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSize)


cdef cusparseStatus_t _cusparseZgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgebsr2gebsc_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseZgebsr2gebsc_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgebsr2gebsc_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuDoubleComplex*, const int*, const int*, int, int, size_t*) noexcept nogil>__cusparseZgebsr2gebsc_bufferSizeExt)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSize)


cdef cusparseStatus_t _cusparseSgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, float* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgebsr2gebsc
    _check_or_init_cusparse()
    if __cusparseSgebsr2gebsc == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgebsr2gebsc is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const float*, const int*, const int*, int, int, float*, int*, int*, cusparseAction_t, cusparseIndexBase_t, void*) noexcept nogil>__cusparseSgebsr2gebsc)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)


cdef cusparseStatus_t _cusparseDgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, double* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgebsr2gebsc
    _check_or_init_cusparse()
    if __cusparseDgebsr2gebsc == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgebsr2gebsc is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const double*, const int*, const int*, int, int, double*, int*, int*, cusparseAction_t, cusparseIndexBase_t, void*) noexcept nogil>__cusparseDgebsr2gebsc)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)


cdef cusparseStatus_t _cusparseCgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, cuComplex* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgebsr2gebsc
    _check_or_init_cusparse()
    if __cusparseCgebsr2gebsc == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgebsr2gebsc is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuComplex*, const int*, const int*, int, int, cuComplex*, int*, int*, cusparseAction_t, cusparseIndexBase_t, void*) noexcept nogil>__cusparseCgebsr2gebsc)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)


cdef cusparseStatus_t _cusparseZgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, cuDoubleComplex* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgebsr2gebsc
    _check_or_init_cusparse()
    if __cusparseZgebsr2gebsc == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgebsr2gebsc is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuDoubleComplex*, const int*, const int*, int, int, cuDoubleComplex*, int*, int*, cusparseAction_t, cusparseIndexBase_t, void*) noexcept nogil>__cusparseZgebsr2gebsc)(
        handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)


cdef cusparseStatus_t _cusparseScsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseScsr2gebsr_bufferSize
    _check_or_init_cusparse()
    if __cusparseScsr2gebsr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseScsr2gebsr_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const float*, const int*, const int*, int, int, int*) noexcept nogil>__cusparseScsr2gebsr_bufferSize)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseDcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDcsr2gebsr_bufferSize
    _check_or_init_cusparse()
    if __cusparseDcsr2gebsr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDcsr2gebsr_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const double*, const int*, const int*, int, int, int*) noexcept nogil>__cusparseDcsr2gebsr_bufferSize)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseCcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCcsr2gebsr_bufferSize
    _check_or_init_cusparse()
    if __cusparseCcsr2gebsr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCcsr2gebsr_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const cuComplex*, const int*, const int*, int, int, int*) noexcept nogil>__cusparseCcsr2gebsr_bufferSize)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseZcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZcsr2gebsr_bufferSize
    _check_or_init_cusparse()
    if __cusparseZcsr2gebsr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZcsr2gebsr_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex*, const int*, const int*, int, int, int*) noexcept nogil>__cusparseZcsr2gebsr_bufferSize)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseScsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseScsr2gebsr_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseScsr2gebsr_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseScsr2gebsr_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const float*, const int*, const int*, int, int, size_t*) noexcept nogil>__cusparseScsr2gebsr_bufferSizeExt)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize)


cdef cusparseStatus_t _cusparseDcsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDcsr2gebsr_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseDcsr2gebsr_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDcsr2gebsr_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const double*, const int*, const int*, int, int, size_t*) noexcept nogil>__cusparseDcsr2gebsr_bufferSizeExt)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize)


cdef cusparseStatus_t _cusparseCcsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCcsr2gebsr_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseCcsr2gebsr_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCcsr2gebsr_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const cuComplex*, const int*, const int*, int, int, size_t*) noexcept nogil>__cusparseCcsr2gebsr_bufferSizeExt)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize)


cdef cusparseStatus_t _cusparseZcsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZcsr2gebsr_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseZcsr2gebsr_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZcsr2gebsr_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex*, const int*, const int*, int, int, size_t*) noexcept nogil>__cusparseZcsr2gebsr_bufferSizeExt)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize)


cdef cusparseStatus_t _cusparseXcsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, int* bsrSortedRowPtrC, int rowBlockDim, int colBlockDim, int* nnzTotalDevHostPtr, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcsr2gebsrNnz
    _check_or_init_cusparse()
    if __cusparseXcsr2gebsrNnz == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcsr2gebsrNnz is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const int*, const int*, const cusparseMatDescr_t, int*, int, int, int*, void*) noexcept nogil>__cusparseXcsr2gebsrNnz)(
        handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedRowPtrC, rowBlockDim, colBlockDim, nnzTotalDevHostPtr, pBuffer)


cdef cusparseStatus_t _cusparseScsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseScsr2gebsr
    _check_or_init_cusparse()
    if __cusparseScsr2gebsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseScsr2gebsr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const float*, const int*, const int*, const cusparseMatDescr_t, float*, int*, int*, int, int, void*) noexcept nogil>__cusparseScsr2gebsr)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)


cdef cusparseStatus_t _cusparseDcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDcsr2gebsr
    _check_or_init_cusparse()
    if __cusparseDcsr2gebsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDcsr2gebsr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const double*, const int*, const int*, const cusparseMatDescr_t, double*, int*, int*, int, int, void*) noexcept nogil>__cusparseDcsr2gebsr)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)


cdef cusparseStatus_t _cusparseCcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, cuComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCcsr2gebsr
    _check_or_init_cusparse()
    if __cusparseCcsr2gebsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCcsr2gebsr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const cuComplex*, const int*, const int*, const cusparseMatDescr_t, cuComplex*, int*, int*, int, int, void*) noexcept nogil>__cusparseCcsr2gebsr)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)


cdef cusparseStatus_t _cusparseZcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZcsr2gebsr
    _check_or_init_cusparse()
    if __cusparseZcsr2gebsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZcsr2gebsr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex*, const int*, const int*, const cusparseMatDescr_t, cuDoubleComplex*, int*, int*, int, int, void*) noexcept nogil>__cusparseZcsr2gebsr)(
        handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)


cdef cusparseStatus_t _cusparseSgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgebsr2gebsr_bufferSize
    _check_or_init_cusparse()
    if __cusparseSgebsr2gebsr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgebsr2gebsr_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const float*, const int*, const int*, int, int, int, int, int*) noexcept nogil>__cusparseSgebsr2gebsr_bufferSize)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseDgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgebsr2gebsr_bufferSize
    _check_or_init_cusparse()
    if __cusparseDgebsr2gebsr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgebsr2gebsr_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const double*, const int*, const int*, int, int, int, int, int*) noexcept nogil>__cusparseDgebsr2gebsr_bufferSize)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseCgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgebsr2gebsr_bufferSize
    _check_or_init_cusparse()
    if __cusparseCgebsr2gebsr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgebsr2gebsr_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const cuComplex*, const int*, const int*, int, int, int, int, int*) noexcept nogil>__cusparseCgebsr2gebsr_bufferSize)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseZgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgebsr2gebsr_bufferSize
    _check_or_init_cusparse()
    if __cusparseZgebsr2gebsr_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgebsr2gebsr_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const cuDoubleComplex*, const int*, const int*, int, int, int, int, int*) noexcept nogil>__cusparseZgebsr2gebsr_bufferSize)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseSgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgebsr2gebsr_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseSgebsr2gebsr_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgebsr2gebsr_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const float*, const int*, const int*, int, int, int, int, size_t*) noexcept nogil>__cusparseSgebsr2gebsr_bufferSizeExt)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize)


cdef cusparseStatus_t _cusparseDgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgebsr2gebsr_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseDgebsr2gebsr_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgebsr2gebsr_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const double*, const int*, const int*, int, int, int, int, size_t*) noexcept nogil>__cusparseDgebsr2gebsr_bufferSizeExt)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize)


cdef cusparseStatus_t _cusparseCgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgebsr2gebsr_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseCgebsr2gebsr_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgebsr2gebsr_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const cuComplex*, const int*, const int*, int, int, int, int, size_t*) noexcept nogil>__cusparseCgebsr2gebsr_bufferSizeExt)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize)


cdef cusparseStatus_t _cusparseZgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t* pBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgebsr2gebsr_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseZgebsr2gebsr_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgebsr2gebsr_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const cuDoubleComplex*, const int*, const int*, int, int, int, int, size_t*) noexcept nogil>__cusparseZgebsr2gebsr_bufferSizeExt)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize)


cdef cusparseStatus_t _cusparseXgebsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, int* bsrSortedRowPtrC, int rowBlockDimC, int colBlockDimC, int* nnzTotalDevHostPtr, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXgebsr2gebsrNnz
    _check_or_init_cusparse()
    if __cusparseXgebsr2gebsrNnz == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXgebsr2gebsrNnz is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const int*, const int*, int, int, const cusparseMatDescr_t, int*, int, int, int*, void*) noexcept nogil>__cusparseXgebsr2gebsrNnz)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedRowPtrC, rowBlockDimC, colBlockDimC, nnzTotalDevHostPtr, pBuffer)


cdef cusparseStatus_t _cusparseSgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSgebsr2gebsr
    _check_or_init_cusparse()
    if __cusparseSgebsr2gebsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSgebsr2gebsr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const float*, const int*, const int*, int, int, const cusparseMatDescr_t, float*, int*, int*, int, int, void*) noexcept nogil>__cusparseSgebsr2gebsr)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)


cdef cusparseStatus_t _cusparseDgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDgebsr2gebsr
    _check_or_init_cusparse()
    if __cusparseDgebsr2gebsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDgebsr2gebsr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const double*, const int*, const int*, int, int, const cusparseMatDescr_t, double*, int*, int*, int, int, void*) noexcept nogil>__cusparseDgebsr2gebsr)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)


cdef cusparseStatus_t _cusparseCgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, cuComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCgebsr2gebsr
    _check_or_init_cusparse()
    if __cusparseCgebsr2gebsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCgebsr2gebsr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const cuComplex*, const int*, const int*, int, int, const cusparseMatDescr_t, cuComplex*, int*, int*, int, int, void*) noexcept nogil>__cusparseCgebsr2gebsr)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)


cdef cusparseStatus_t _cusparseZgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseZgebsr2gebsr
    _check_or_init_cusparse()
    if __cusparseZgebsr2gebsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseZgebsr2gebsr is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const cuDoubleComplex*, const int*, const int*, int, int, const cusparseMatDescr_t, cuDoubleComplex*, int*, int*, int, int, void*) noexcept nogil>__cusparseZgebsr2gebsr)(
        handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)


cdef cusparseStatus_t _cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* cooRowsA, const int* cooColsA, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcoosort_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseXcoosort_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcoosort_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const int*, const int*, size_t*) noexcept nogil>__cusparseXcoosort_bufferSizeExt)(
        handle, m, n, nnz, cooRowsA, cooColsA, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA, int* P, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcoosortByRow
    _check_or_init_cusparse()
    if __cusparseXcoosortByRow == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcoosortByRow is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, int*, int*, int*, void*) noexcept nogil>__cusparseXcoosortByRow)(
        handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer)


cdef cusparseStatus_t _cusparseXcoosortByColumn(cusparseHandle_t handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA, int* P, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcoosortByColumn
    _check_or_init_cusparse()
    if __cusparseXcoosortByColumn == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcoosortByColumn is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, int*, int*, int*, void*) noexcept nogil>__cusparseXcoosortByColumn)(
        handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer)


cdef cusparseStatus_t _cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* csrRowPtrA, const int* csrColIndA, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcsrsort_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseXcsrsort_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcsrsort_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const int*, const int*, size_t*) noexcept nogil>__cusparseXcsrsort_bufferSizeExt)(
        handle, m, n, nnz, csrRowPtrA, csrColIndA, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const int* csrRowPtrA, int* csrColIndA, int* P, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcsrsort
    _check_or_init_cusparse()
    if __cusparseXcsrsort == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcsrsort is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const int*, int*, int*, void*) noexcept nogil>__cusparseXcsrsort)(
        handle, m, n, nnz, descrA, csrRowPtrA, csrColIndA, P, pBuffer)


cdef cusparseStatus_t _cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* cscColPtrA, const int* cscRowIndA, size_t* pBufferSizeInBytes) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcscsort_bufferSizeExt
    _check_or_init_cusparse()
    if __cusparseXcscsort_bufferSizeExt == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcscsort_bufferSizeExt is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const int*, const int*, size_t*) noexcept nogil>__cusparseXcscsort_bufferSizeExt)(
        handle, m, n, nnz, cscColPtrA, cscRowIndA, pBufferSizeInBytes)


cdef cusparseStatus_t _cusparseXcscsort(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const int* cscColPtrA, int* cscRowIndA, int* P, void* pBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseXcscsort
    _check_or_init_cusparse()
    if __cusparseXcscsort == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseXcscsort is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const int*, int*, int*, void*) noexcept nogil>__cusparseXcscsort)(
        handle, m, n, nnz, descrA, cscColPtrA, cscRowIndA, P, pBuffer)


cdef cusparseStatus_t _cusparseCsr2cscEx2(cusparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, cudaDataType valType, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, void* buffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCsr2cscEx2
    _check_or_init_cusparse()
    if __cusparseCsr2cscEx2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCsr2cscEx2 is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const void*, const int*, const int*, void*, int*, int*, cudaDataType, cusparseAction_t, cusparseIndexBase_t, cusparseCsr2CscAlg_t, void*) noexcept nogil>__cusparseCsr2cscEx2)(
        handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, buffer)


cdef cusparseStatus_t _cusparseCsr2cscEx2_bufferSize(cusparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, cudaDataType valType, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCsr2cscEx2_bufferSize
    _check_or_init_cusparse()
    if __cusparseCsr2cscEx2_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCsr2cscEx2_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const void*, const int*, const int*, void*, int*, int*, cudaDataType, cusparseAction_t, cusparseIndexBase_t, cusparseCsr2CscAlg_t, size_t*) noexcept nogil>__cusparseCsr2cscEx2_bufferSize)(
        handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, bufferSize)


cdef cusparseStatus_t _cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices, void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateSpVec
    _check_or_init_cusparse()
    if __cusparseCreateSpVec == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateSpVec is not found")
    return (<cusparseStatus_t (*)(cusparseSpVecDescr_t*, int64_t, int64_t, void*, void*, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateSpVec)(
        spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseDestroySpVec(cusparseConstSpVecDescr_t spVecDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDestroySpVec
    _check_or_init_cusparse()
    if __cusparseDestroySpVec == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDestroySpVec is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpVecDescr_t) noexcept nogil>__cusparseDestroySpVec)(
        spVecDescr)


cdef cusparseStatus_t _cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpVecGet
    _check_or_init_cusparse()
    if __cusparseSpVecGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpVecGet is not found")
    return (<cusparseStatus_t (*)(cusparseSpVecDescr_t, int64_t*, int64_t*, void**, void**, cusparseIndexType_t*, cusparseIndexBase_t*, cudaDataType*) noexcept nogil>__cusparseSpVecGet)(
        spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseSpVecGetIndexBase(cusparseConstSpVecDescr_t spVecDescr, cusparseIndexBase_t* idxBase) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpVecGetIndexBase
    _check_or_init_cusparse()
    if __cusparseSpVecGetIndexBase == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpVecGetIndexBase is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpVecDescr_t, cusparseIndexBase_t*) noexcept nogil>__cusparseSpVecGetIndexBase)(
        spVecDescr, idxBase)


cdef cusparseStatus_t _cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr, void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpVecGetValues
    _check_or_init_cusparse()
    if __cusparseSpVecGetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpVecGetValues is not found")
    return (<cusparseStatus_t (*)(cusparseSpVecDescr_t, void**) noexcept nogil>__cusparseSpVecGetValues)(
        spVecDescr, values)


cdef cusparseStatus_t _cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void* values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpVecSetValues
    _check_or_init_cusparse()
    if __cusparseSpVecSetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpVecSetValues is not found")
    return (<cusparseStatus_t (*)(cusparseSpVecDescr_t, void*) noexcept nogil>__cusparseSpVecSetValues)(
        spVecDescr, values)


cdef cusparseStatus_t _cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateDnVec
    _check_or_init_cusparse()
    if __cusparseCreateDnVec == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateDnVec is not found")
    return (<cusparseStatus_t (*)(cusparseDnVecDescr_t*, int64_t, void*, cudaDataType) noexcept nogil>__cusparseCreateDnVec)(
        dnVecDescr, size, values, valueType)


cdef cusparseStatus_t _cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDestroyDnVec
    _check_or_init_cusparse()
    if __cusparseDestroyDnVec == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDestroyDnVec is not found")
    return (<cusparseStatus_t (*)(cusparseConstDnVecDescr_t) noexcept nogil>__cusparseDestroyDnVec)(
        dnVecDescr)


cdef cusparseStatus_t _cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr, int64_t* size, void** values, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDnVecGet
    _check_or_init_cusparse()
    if __cusparseDnVecGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDnVecGet is not found")
    return (<cusparseStatus_t (*)(cusparseDnVecDescr_t, int64_t*, void**, cudaDataType*) noexcept nogil>__cusparseDnVecGet)(
        dnVecDescr, size, values, valueType)


cdef cusparseStatus_t _cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr, void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDnVecGetValues
    _check_or_init_cusparse()
    if __cusparseDnVecGetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDnVecGetValues is not found")
    return (<cusparseStatus_t (*)(cusparseDnVecDescr_t, void**) noexcept nogil>__cusparseDnVecGetValues)(
        dnVecDescr, values)


cdef cusparseStatus_t _cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void* values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDnVecSetValues
    _check_or_init_cusparse()
    if __cusparseDnVecSetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDnVecSetValues is not found")
    return (<cusparseStatus_t (*)(cusparseDnVecDescr_t, void*) noexcept nogil>__cusparseDnVecSetValues)(
        dnVecDescr, values)


cdef cusparseStatus_t _cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDestroySpMat
    _check_or_init_cusparse()
    if __cusparseDestroySpMat == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDestroySpMat is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t) noexcept nogil>__cusparseDestroySpMat)(
        spMatDescr)


cdef cusparseStatus_t _cusparseSpMatGetFormat(cusparseConstSpMatDescr_t spMatDescr, cusparseFormat_t* format) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMatGetFormat
    _check_or_init_cusparse()
    if __cusparseSpMatGetFormat == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMatGetFormat is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t, cusparseFormat_t*) noexcept nogil>__cusparseSpMatGetFormat)(
        spMatDescr, format)


cdef cusparseStatus_t _cusparseSpMatGetIndexBase(cusparseConstSpMatDescr_t spMatDescr, cusparseIndexBase_t* idxBase) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMatGetIndexBase
    _check_or_init_cusparse()
    if __cusparseSpMatGetIndexBase == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMatGetIndexBase is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t, cusparseIndexBase_t*) noexcept nogil>__cusparseSpMatGetIndexBase)(
        spMatDescr, idxBase)


cdef cusparseStatus_t _cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr, void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMatGetValues
    _check_or_init_cusparse()
    if __cusparseSpMatGetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMatGetValues is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, void**) noexcept nogil>__cusparseSpMatGetValues)(
        spMatDescr, values)


cdef cusparseStatus_t _cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void* values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMatSetValues
    _check_or_init_cusparse()
    if __cusparseSpMatSetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMatSetValues is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, void*) noexcept nogil>__cusparseSpMatSetValues)(
        spMatDescr, values)


cdef cusparseStatus_t _cusparseSpMatGetSize(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMatGetSize
    _check_or_init_cusparse()
    if __cusparseSpMatGetSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMatGetSize is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t, int64_t*, int64_t*, int64_t*) noexcept nogil>__cusparseSpMatGetSize)(
        spMatDescr, rows, cols, nnz)


cdef cusparseStatus_t _cusparseSpMatGetStridedBatch(cusparseConstSpMatDescr_t spMatDescr, int* batchCount) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMatGetStridedBatch
    _check_or_init_cusparse()
    if __cusparseSpMatGetStridedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMatGetStridedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t, int*) noexcept nogil>__cusparseSpMatGetStridedBatch)(
        spMatDescr, batchCount)


cdef cusparseStatus_t _cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCooSetStridedBatch
    _check_or_init_cusparse()
    if __cusparseCooSetStridedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCooSetStridedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, int, int64_t) noexcept nogil>__cusparseCooSetStridedBatch)(
        spMatDescr, batchCount, batchStride)


cdef cusparseStatus_t _cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride, int64_t columnsValuesBatchStride) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCsrSetStridedBatch
    _check_or_init_cusparse()
    if __cusparseCsrSetStridedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCsrSetStridedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, int, int64_t, int64_t) noexcept nogil>__cusparseCsrSetStridedBatch)(
        spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride)


cdef cusparseStatus_t _cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets, void* csrColInd, void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateCsr
    _check_or_init_cusparse()
    if __cusparseCreateCsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateCsr is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t*, int64_t, int64_t, int64_t, void*, void*, void*, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateCsr)(
        spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCsrGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** csrRowOffsets, void** csrColInd, void** csrValues, cusparseIndexType_t* csrRowOffsetsType, cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCsrGet
    _check_or_init_cusparse()
    if __cusparseCsrGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCsrGet is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, int64_t*, int64_t*, int64_t*, void**, void**, void**, cusparseIndexType_t*, cusparseIndexType_t*, cusparseIndexBase_t*, cudaDataType*) noexcept nogil>__cusparseCsrGet)(
        spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd, void* csrValues) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCsrSetPointers
    _check_or_init_cusparse()
    if __cusparseCsrSetPointers == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCsrSetPointers is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, void*, void*, void*) noexcept nogil>__cusparseCsrSetPointers)(
        spMatDescr, csrRowOffsets, csrColInd, csrValues)


cdef cusparseStatus_t _cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooRowInd, void* cooColInd, void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateCoo
    _check_or_init_cusparse()
    if __cusparseCreateCoo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateCoo is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t*, int64_t, int64_t, int64_t, void*, void*, void*, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateCoo)(
        spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd, void** cooColInd, void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCooGet
    _check_or_init_cusparse()
    if __cusparseCooGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCooGet is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, int64_t*, int64_t*, int64_t*, void**, void**, void**, cusparseIndexType_t*, cusparseIndexBase_t*, cudaDataType*) noexcept nogil>__cusparseCooGet)(
        spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateDnMat
    _check_or_init_cusparse()
    if __cusparseCreateDnMat == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateDnMat is not found")
    return (<cusparseStatus_t (*)(cusparseDnMatDescr_t*, int64_t, int64_t, int64_t, void*, cudaDataType, cusparseOrder_t) noexcept nogil>__cusparseCreateDnMat)(
        dnMatDescr, rows, cols, ld, values, valueType, order)


cdef cusparseStatus_t _cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDestroyDnMat
    _check_or_init_cusparse()
    if __cusparseDestroyDnMat == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDestroyDnMat is not found")
    return (<cusparseStatus_t (*)(cusparseConstDnMatDescr_t) noexcept nogil>__cusparseDestroyDnMat)(
        dnMatDescr)


cdef cusparseStatus_t _cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, cudaDataType* type, cusparseOrder_t* order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDnMatGet
    _check_or_init_cusparse()
    if __cusparseDnMatGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDnMatGet is not found")
    return (<cusparseStatus_t (*)(cusparseDnMatDescr_t, int64_t*, int64_t*, int64_t*, void**, cudaDataType*, cusparseOrder_t*) noexcept nogil>__cusparseDnMatGet)(
        dnMatDescr, rows, cols, ld, values, type, order)


cdef cusparseStatus_t _cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr, void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDnMatGetValues
    _check_or_init_cusparse()
    if __cusparseDnMatGetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDnMatGetValues is not found")
    return (<cusparseStatus_t (*)(cusparseDnMatDescr_t, void**) noexcept nogil>__cusparseDnMatGetValues)(
        dnMatDescr, values)


cdef cusparseStatus_t _cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void* values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDnMatSetValues
    _check_or_init_cusparse()
    if __cusparseDnMatSetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDnMatSetValues is not found")
    return (<cusparseStatus_t (*)(cusparseDnMatDescr_t, void*) noexcept nogil>__cusparseDnMatSetValues)(
        dnMatDescr, values)


cdef cusparseStatus_t _cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDnMatSetStridedBatch
    _check_or_init_cusparse()
    if __cusparseDnMatSetStridedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDnMatSetStridedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseDnMatDescr_t, int, int64_t) noexcept nogil>__cusparseDnMatSetStridedBatch)(
        dnMatDescr, batchCount, batchStride)


cdef cusparseStatus_t _cusparseDnMatGetStridedBatch(cusparseConstDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDnMatGetStridedBatch
    _check_or_init_cusparse()
    if __cusparseDnMatGetStridedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDnMatGetStridedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseConstDnMatDescr_t, int*, int64_t*) noexcept nogil>__cusparseDnMatGetStridedBatch)(
        dnMatDescr, batchCount, batchStride)


cdef cusparseStatus_t _cusparseAxpby(cusparseHandle_t handle, const void* alpha, cusparseConstSpVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseAxpby
    _check_or_init_cusparse()
    if __cusparseAxpby == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseAxpby is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, const void*, cusparseConstSpVecDescr_t, const void*, cusparseDnVecDescr_t) noexcept nogil>__cusparseAxpby)(
        handle, alpha, vecX, beta, vecY)


cdef cusparseStatus_t _cusparseGather(cusparseHandle_t handle, cusparseConstDnVecDescr_t vecY, cusparseSpVecDescr_t vecX) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseGather
    _check_or_init_cusparse()
    if __cusparseGather == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseGather is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseConstDnVecDescr_t, cusparseSpVecDescr_t) noexcept nogil>__cusparseGather)(
        handle, vecY, vecX)


cdef cusparseStatus_t _cusparseScatter(cusparseHandle_t handle, cusparseConstSpVecDescr_t vecX, cusparseDnVecDescr_t vecY) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseScatter
    _check_or_init_cusparse()
    if __cusparseScatter == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseScatter is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseConstSpVecDescr_t, cusparseDnVecDescr_t) noexcept nogil>__cusparseScatter)(
        handle, vecX, vecY)


cdef cusparseStatus_t _cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX, cusparseConstSpVecDescr_t vecX, cusparseConstDnVecDescr_t vecY, const void* result, cudaDataType computeType, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpVV_bufferSize
    _check_or_init_cusparse()
    if __cusparseSpVV_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpVV_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseConstSpVecDescr_t, cusparseConstDnVecDescr_t, const void*, cudaDataType, size_t*) noexcept nogil>__cusparseSpVV_bufferSize)(
        handle, opX, vecX, vecY, result, computeType, bufferSize)


cdef cusparseStatus_t _cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX, cusparseConstSpVecDescr_t vecX, cusparseConstDnVecDescr_t vecY, void* result, cudaDataType computeType, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpVV
    _check_or_init_cusparse()
    if __cusparseSpVV == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpVV is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseConstSpVecDescr_t, cusparseConstDnVecDescr_t, void*, cudaDataType, void*) noexcept nogil>__cusparseSpVV)(
        handle, opX, vecX, vecY, result, computeType, externalBuffer)


cdef cusparseStatus_t _cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMV
    _check_or_init_cusparse()
    if __cusparseSpMV == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMV is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, const void*, cusparseDnVecDescr_t, cudaDataType, cusparseSpMVAlg_t, void*) noexcept nogil>__cusparseSpMV)(
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer)


cdef cusparseStatus_t _cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMV_bufferSize
    _check_or_init_cusparse()
    if __cusparseSpMV_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMV_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, const void*, cusparseDnVecDescr_t, cudaDataType, cusparseSpMVAlg_t, size_t*) noexcept nogil>__cusparseSpMV_bufferSize)(
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize)


cdef cusparseStatus_t _cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMM
    _check_or_init_cusparse()
    if __cusparseSpMM == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMM is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, const void*, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, void*) noexcept nogil>__cusparseSpMM)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)


cdef cusparseStatus_t _cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMM_bufferSize
    _check_or_init_cusparse()
    if __cusparseSpMM_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMM_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, const void*, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, size_t*) noexcept nogil>__cusparseSpMM_bufferSize)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize)


cdef cusparseStatus_t _cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMM_createDescr
    _check_or_init_cusparse()
    if __cusparseSpGEMM_createDescr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMM_createDescr is not found")
    return (<cusparseStatus_t (*)(cusparseSpGEMMDescr_t*) noexcept nogil>__cusparseSpGEMM_createDescr)(
        descr)


cdef cusparseStatus_t _cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMM_destroyDescr
    _check_or_init_cusparse()
    if __cusparseSpGEMM_destroyDescr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMM_destroyDescr is not found")
    return (<cusparseStatus_t (*)(cusparseSpGEMMDescr_t) noexcept nogil>__cusparseSpGEMM_destroyDescr)(
        descr)


cdef cusparseStatus_t _cusparseSpGEMM_workEstimation(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMM_workEstimation
    _check_or_init_cusparse()
    if __cusparseSpGEMM_workEstimation == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMM_workEstimation is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, size_t*, void*) noexcept nogil>__cusparseSpGEMM_workEstimation)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize1, externalBuffer1)


cdef cusparseStatus_t _cusparseSpGEMM_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMM_compute
    _check_or_init_cusparse()
    if __cusparseSpGEMM_compute == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMM_compute is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, size_t*, void*) noexcept nogil>__cusparseSpGEMM_compute)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize2, externalBuffer2)


cdef cusparseStatus_t _cusparseSpGEMM_copy(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMM_copy
    _check_or_init_cusparse()
    if __cusparseSpGEMM_copy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMM_copy is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t) noexcept nogil>__cusparseSpGEMM_copy)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr)


cdef cusparseStatus_t _cusparseCreateCsc(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cscColOffsets, void* cscRowInd, void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateCsc
    _check_or_init_cusparse()
    if __cusparseCreateCsc == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateCsc is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t*, int64_t, int64_t, int64_t, void*, void*, void*, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateCsc)(
        spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCscSetPointers(cusparseSpMatDescr_t spMatDescr, void* cscColOffsets, void* cscRowInd, void* cscValues) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCscSetPointers
    _check_or_init_cusparse()
    if __cusparseCscSetPointers == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCscSetPointers is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, void*, void*, void*) noexcept nogil>__cusparseCscSetPointers)(
        spMatDescr, cscColOffsets, cscRowInd, cscValues)


cdef cusparseStatus_t _cusparseCooSetPointers(cusparseSpMatDescr_t spMatDescr, void* cooRows, void* cooColumns, void* cooValues) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCooSetPointers
    _check_or_init_cusparse()
    if __cusparseCooSetPointers == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCooSetPointers is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, void*, void*, void*) noexcept nogil>__cusparseCooSetPointers)(
        spMatDescr, cooRows, cooColumns, cooValues)


cdef cusparseStatus_t _cusparseSparseToDense_bufferSize(cusparseHandle_t handle, cusparseConstSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSparseToDense_bufferSize
    _check_or_init_cusparse()
    if __cusparseSparseToDense_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSparseToDense_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseConstSpMatDescr_t, cusparseDnMatDescr_t, cusparseSparseToDenseAlg_t, size_t*) noexcept nogil>__cusparseSparseToDense_bufferSize)(
        handle, matA, matB, alg, bufferSize)


cdef cusparseStatus_t _cusparseSparseToDense(cusparseHandle_t handle, cusparseConstSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSparseToDense
    _check_or_init_cusparse()
    if __cusparseSparseToDense == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSparseToDense is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseConstSpMatDescr_t, cusparseDnMatDescr_t, cusparseSparseToDenseAlg_t, void*) noexcept nogil>__cusparseSparseToDense)(
        handle, matA, matB, alg, externalBuffer)


cdef cusparseStatus_t _cusparseDenseToSparse_bufferSize(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDenseToSparse_bufferSize
    _check_or_init_cusparse()
    if __cusparseDenseToSparse_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDenseToSparse_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseConstDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, size_t*) noexcept nogil>__cusparseDenseToSparse_bufferSize)(
        handle, matA, matB, alg, bufferSize)


cdef cusparseStatus_t _cusparseDenseToSparse_analysis(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDenseToSparse_analysis
    _check_or_init_cusparse()
    if __cusparseDenseToSparse_analysis == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDenseToSparse_analysis is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseConstDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, void*) noexcept nogil>__cusparseDenseToSparse_analysis)(
        handle, matA, matB, alg, externalBuffer)


cdef cusparseStatus_t _cusparseDenseToSparse_convert(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseDenseToSparse_convert
    _check_or_init_cusparse()
    if __cusparseDenseToSparse_convert == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseDenseToSparse_convert is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseConstDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, void*) noexcept nogil>__cusparseDenseToSparse_convert)(
        handle, matA, matB, alg, externalBuffer)


cdef cusparseStatus_t _cusparseCreateBlockedEll(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateBlockedEll
    _check_or_init_cusparse()
    if __cusparseCreateBlockedEll == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateBlockedEll is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t*, int64_t, int64_t, int64_t, int64_t, void*, void*, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateBlockedEll)(
        spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseBlockedEllGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, void** ellColInd, void** ellValue, cusparseIndexType_t* ellIdxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseBlockedEllGet
    _check_or_init_cusparse()
    if __cusparseBlockedEllGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseBlockedEllGet is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, int64_t*, int64_t*, int64_t*, int64_t*, void**, void**, cusparseIndexType_t*, cusparseIndexBase_t*, cudaDataType*) noexcept nogil>__cusparseBlockedEllGet)(
        spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseSpMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMM_preprocess
    _check_or_init_cusparse()
    if __cusparseSpMM_preprocess == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMM_preprocess is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, const void*, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, void*) noexcept nogil>__cusparseSpMM_preprocess)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)


cdef cusparseStatus_t _cusparseSDDMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSDDMM_bufferSize
    _check_or_init_cusparse()
    if __cusparseSDDMM_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSDDMM_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstDnMatDescr_t, cusparseConstDnMatDescr_t, const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, size_t*) noexcept nogil>__cusparseSDDMM_bufferSize)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize)


cdef cusparseStatus_t _cusparseSDDMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSDDMM_preprocess
    _check_or_init_cusparse()
    if __cusparseSDDMM_preprocess == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSDDMM_preprocess is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstDnMatDescr_t, cusparseConstDnMatDescr_t, const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, void*) noexcept nogil>__cusparseSDDMM_preprocess)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)


cdef cusparseStatus_t _cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSDDMM
    _check_or_init_cusparse()
    if __cusparseSDDMM == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSDDMM is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstDnMatDescr_t, cusparseConstDnMatDescr_t, const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, void*) noexcept nogil>__cusparseSDDMM)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)


cdef cusparseStatus_t _cusparseSpMatGetAttribute(cusparseConstSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMatGetAttribute
    _check_or_init_cusparse()
    if __cusparseSpMatGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMatGetAttribute is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t, cusparseSpMatAttribute_t, void*, size_t) noexcept nogil>__cusparseSpMatGetAttribute)(
        spMatDescr, attribute, data, dataSize)


cdef cusparseStatus_t _cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMatSetAttribute
    _check_or_init_cusparse()
    if __cusparseSpMatSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMatSetAttribute is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, cusparseSpMatAttribute_t, void*, size_t) noexcept nogil>__cusparseSpMatSetAttribute)(
        spMatDescr, attribute, data, dataSize)


cdef cusparseStatus_t _cusparseSpSV_createDescr(cusparseSpSVDescr_t* descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSV_createDescr
    _check_or_init_cusparse()
    if __cusparseSpSV_createDescr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSV_createDescr is not found")
    return (<cusparseStatus_t (*)(cusparseSpSVDescr_t*) noexcept nogil>__cusparseSpSV_createDescr)(
        descr)


cdef cusparseStatus_t _cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSV_destroyDescr
    _check_or_init_cusparse()
    if __cusparseSpSV_destroyDescr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSV_destroyDescr is not found")
    return (<cusparseStatus_t (*)(cusparseSpSVDescr_t) noexcept nogil>__cusparseSpSV_destroyDescr)(
        descr)


cdef cusparseStatus_t _cusparseSpSV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSV_bufferSize
    _check_or_init_cusparse()
    if __cusparseSpSV_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSV_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t, cusparseSpSVDescr_t, size_t*) noexcept nogil>__cusparseSpSV_bufferSize)(
        handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, bufferSize)


cdef cusparseStatus_t _cusparseSpSV_analysis(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSV_analysis
    _check_or_init_cusparse()
    if __cusparseSpSV_analysis == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSV_analysis is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t, cusparseSpSVDescr_t, void*) noexcept nogil>__cusparseSpSV_analysis)(
        handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, externalBuffer)


cdef cusparseStatus_t _cusparseSpSV_solve(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSV_solve
    _check_or_init_cusparse()
    if __cusparseSpSV_solve == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSV_solve is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t, cusparseSpSVDescr_t) noexcept nogil>__cusparseSpSV_solve)(
        handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr)


cdef cusparseStatus_t _cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSM_createDescr
    _check_or_init_cusparse()
    if __cusparseSpSM_createDescr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSM_createDescr is not found")
    return (<cusparseStatus_t (*)(cusparseSpSMDescr_t*) noexcept nogil>__cusparseSpSM_createDescr)(
        descr)


cdef cusparseStatus_t _cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSM_destroyDescr
    _check_or_init_cusparse()
    if __cusparseSpSM_destroyDescr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSM_destroyDescr is not found")
    return (<cusparseStatus_t (*)(cusparseSpSMDescr_t) noexcept nogil>__cusparseSpSM_destroyDescr)(
        descr)


cdef cusparseStatus_t _cusparseSpSM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr, size_t* bufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSM_bufferSize
    _check_or_init_cusparse()
    if __cusparseSpSM_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSM_bufferSize is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t, size_t*) noexcept nogil>__cusparseSpSM_bufferSize)(
        handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, bufferSize)


cdef cusparseStatus_t _cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSM_analysis
    _check_or_init_cusparse()
    if __cusparseSpSM_analysis == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSM_analysis is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t, void*) noexcept nogil>__cusparseSpSM_analysis)(
        handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, externalBuffer)


cdef cusparseStatus_t _cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSM_solve
    _check_or_init_cusparse()
    if __cusparseSpSM_solve == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSM_solve is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t) noexcept nogil>__cusparseSpSM_solve)(
        handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr)


cdef cusparseStatus_t _cusparseSpGEMMreuse_workEstimation(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMMreuse_workEstimation
    _check_or_init_cusparse()
    if __cusparseSpGEMMreuse_workEstimation == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMMreuse_workEstimation is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, size_t*, void*) noexcept nogil>__cusparseSpGEMMreuse_workEstimation)(
        handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize1, externalBuffer1)


cdef cusparseStatus_t _cusparseSpGEMMreuse_nnz(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2, size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize4, void* externalBuffer4) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMMreuse_nnz
    _check_or_init_cusparse()
    if __cusparseSpGEMMreuse_nnz == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMMreuse_nnz is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, size_t*, void*, size_t*, void*, size_t*, void*) noexcept nogil>__cusparseSpGEMMreuse_nnz)(
        handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize2, externalBuffer2, bufferSize3, externalBuffer3, bufferSize4, externalBuffer4)


cdef cusparseStatus_t _cusparseSpGEMMreuse_copy(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize5, void* externalBuffer5) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMMreuse_copy
    _check_or_init_cusparse()
    if __cusparseSpGEMMreuse_copy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMMreuse_copy is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, size_t*, void*) noexcept nogil>__cusparseSpGEMMreuse_copy)(
        handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize5, externalBuffer5)


cdef cusparseStatus_t _cusparseSpGEMMreuse_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMMreuse_compute
    _check_or_init_cusparse()
    if __cusparseSpGEMMreuse_compute == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMMreuse_compute is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t) noexcept nogil>__cusparseSpGEMMreuse_compute)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr)


cdef cusparseStatus_t _cusparseLoggerSetCallback(cusparseLoggerCallback_t callback) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLoggerSetCallback
    _check_or_init_cusparse()
    if __cusparseLoggerSetCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLoggerSetCallback is not found")
    return (<cusparseStatus_t (*)(cusparseLoggerCallback_t) noexcept nogil>__cusparseLoggerSetCallback)(
        callback)


cdef cusparseStatus_t _cusparseLoggerSetFile(FILE* file) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLoggerSetFile
    _check_or_init_cusparse()
    if __cusparseLoggerSetFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLoggerSetFile is not found")
    return (<cusparseStatus_t (*)(FILE*) noexcept nogil>__cusparseLoggerSetFile)(
        file)


cdef cusparseStatus_t _cusparseLoggerOpenFile(const char* logFile) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLoggerOpenFile
    _check_or_init_cusparse()
    if __cusparseLoggerOpenFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLoggerOpenFile is not found")
    return (<cusparseStatus_t (*)(const char*) noexcept nogil>__cusparseLoggerOpenFile)(
        logFile)


cdef cusparseStatus_t _cusparseLoggerSetLevel(int level) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLoggerSetLevel
    _check_or_init_cusparse()
    if __cusparseLoggerSetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLoggerSetLevel is not found")
    return (<cusparseStatus_t (*)(int) noexcept nogil>__cusparseLoggerSetLevel)(
        level)


cdef cusparseStatus_t _cusparseLoggerSetMask(int mask) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLoggerSetMask
    _check_or_init_cusparse()
    if __cusparseLoggerSetMask == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLoggerSetMask is not found")
    return (<cusparseStatus_t (*)(int) noexcept nogil>__cusparseLoggerSetMask)(
        mask)


cdef cusparseStatus_t _cusparseLoggerForceDisable() except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLoggerForceDisable
    _check_or_init_cusparse()
    if __cusparseLoggerForceDisable == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLoggerForceDisable is not found")
    return (<cusparseStatus_t (*)() noexcept nogil>__cusparseLoggerForceDisable)(
        )


cdef cusparseStatus_t _cusparseSpMMOp_createPlan(cusparseHandle_t handle, cusparseSpMMOpPlan_t* plan, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMOpAlg_t alg, const void* addOperationNvvmBuffer, size_t addOperationBufferSize, const void* mulOperationNvvmBuffer, size_t mulOperationBufferSize, const void* epilogueNvvmBuffer, size_t epilogueBufferSize, size_t* SpMMWorkspaceSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMMOp_createPlan
    _check_or_init_cusparse()
    if __cusparseSpMMOp_createPlan == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMMOp_createPlan is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseSpMMOpPlan_t*, cusparseOperation_t, cusparseOperation_t, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMOpAlg_t, const void*, size_t, const void*, size_t, const void*, size_t, size_t*) noexcept nogil>__cusparseSpMMOp_createPlan)(
        handle, plan, opA, opB, matA, matB, matC, computeType, alg, addOperationNvvmBuffer, addOperationBufferSize, mulOperationNvvmBuffer, mulOperationBufferSize, epilogueNvvmBuffer, epilogueBufferSize, SpMMWorkspaceSize)


cdef cusparseStatus_t _cusparseSpMMOp(cusparseSpMMOpPlan_t plan, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMMOp
    _check_or_init_cusparse()
    if __cusparseSpMMOp == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMMOp is not found")
    return (<cusparseStatus_t (*)(cusparseSpMMOpPlan_t, void*) noexcept nogil>__cusparseSpMMOp)(
        plan, externalBuffer)


cdef cusparseStatus_t _cusparseSpMMOp_destroyPlan(cusparseSpMMOpPlan_t plan) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMMOp_destroyPlan
    _check_or_init_cusparse()
    if __cusparseSpMMOp_destroyPlan == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMMOp_destroyPlan is not found")
    return (<cusparseStatus_t (*)(cusparseSpMMOpPlan_t) noexcept nogil>__cusparseSpMMOp_destroyPlan)(
        plan)


cdef cusparseStatus_t _cusparseCscGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cscColOffsets, void** cscRowInd, void** cscValues, cusparseIndexType_t* cscColOffsetsType, cusparseIndexType_t* cscRowIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCscGet
    _check_or_init_cusparse()
    if __cusparseCscGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCscGet is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, int64_t*, int64_t*, int64_t*, void**, void**, void**, cusparseIndexType_t*, cusparseIndexType_t*, cusparseIndexBase_t*, cudaDataType*) noexcept nogil>__cusparseCscGet)(
        spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCreateConstSpVec(cusparseConstSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, const void* indices, const void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateConstSpVec
    _check_or_init_cusparse()
    if __cusparseCreateConstSpVec == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateConstSpVec is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpVecDescr_t*, int64_t, int64_t, const void*, const void*, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateConstSpVec)(
        spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseConstSpVecGet(cusparseConstSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, const void** indices, const void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstSpVecGet
    _check_or_init_cusparse()
    if __cusparseConstSpVecGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstSpVecGet is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpVecDescr_t, int64_t*, int64_t*, const void**, const void**, cusparseIndexType_t*, cusparseIndexBase_t*, cudaDataType*) noexcept nogil>__cusparseConstSpVecGet)(
        spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseConstSpVecGetValues(cusparseConstSpVecDescr_t spVecDescr, const void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstSpVecGetValues
    _check_or_init_cusparse()
    if __cusparseConstSpVecGetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstSpVecGetValues is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpVecDescr_t, const void**) noexcept nogil>__cusparseConstSpVecGetValues)(
        spVecDescr, values)


cdef cusparseStatus_t _cusparseCreateConstDnVec(cusparseConstDnVecDescr_t* dnVecDescr, int64_t size, const void* values, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateConstDnVec
    _check_or_init_cusparse()
    if __cusparseCreateConstDnVec == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateConstDnVec is not found")
    return (<cusparseStatus_t (*)(cusparseConstDnVecDescr_t*, int64_t, const void*, cudaDataType) noexcept nogil>__cusparseCreateConstDnVec)(
        dnVecDescr, size, values, valueType)


cdef cusparseStatus_t _cusparseConstDnVecGet(cusparseConstDnVecDescr_t dnVecDescr, int64_t* size, const void** values, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstDnVecGet
    _check_or_init_cusparse()
    if __cusparseConstDnVecGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstDnVecGet is not found")
    return (<cusparseStatus_t (*)(cusparseConstDnVecDescr_t, int64_t*, const void**, cudaDataType*) noexcept nogil>__cusparseConstDnVecGet)(
        dnVecDescr, size, values, valueType)


cdef cusparseStatus_t _cusparseConstDnVecGetValues(cusparseConstDnVecDescr_t dnVecDescr, const void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstDnVecGetValues
    _check_or_init_cusparse()
    if __cusparseConstDnVecGetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstDnVecGetValues is not found")
    return (<cusparseStatus_t (*)(cusparseConstDnVecDescr_t, const void**) noexcept nogil>__cusparseConstDnVecGetValues)(
        dnVecDescr, values)


cdef cusparseStatus_t _cusparseConstSpMatGetValues(cusparseConstSpMatDescr_t spMatDescr, const void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstSpMatGetValues
    _check_or_init_cusparse()
    if __cusparseConstSpMatGetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstSpMatGetValues is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t, const void**) noexcept nogil>__cusparseConstSpMatGetValues)(
        spMatDescr, values)


cdef cusparseStatus_t _cusparseCreateConstCsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* csrRowOffsets, const void* csrColInd, const void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateConstCsr
    _check_or_init_cusparse()
    if __cusparseCreateConstCsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateConstCsr is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t*, int64_t, int64_t, int64_t, const void*, const void*, const void*, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateConstCsr)(
        spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCreateConstCsc(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cscColOffsets, const void* cscRowInd, const void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateConstCsc
    _check_or_init_cusparse()
    if __cusparseCreateConstCsc == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateConstCsc is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t*, int64_t, int64_t, int64_t, const void*, const void*, const void*, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateConstCsc)(
        spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)


cdef cusparseStatus_t _cusparseConstCsrGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** csrRowOffsets, const void** csrColInd, const void** csrValues, cusparseIndexType_t* csrRowOffsetsType, cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstCsrGet
    _check_or_init_cusparse()
    if __cusparseConstCsrGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstCsrGet is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t, int64_t*, int64_t*, int64_t*, const void**, const void**, const void**, cusparseIndexType_t*, cusparseIndexType_t*, cusparseIndexBase_t*, cudaDataType*) noexcept nogil>__cusparseConstCsrGet)(
        spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)


cdef cusparseStatus_t _cusparseConstCscGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** cscColOffsets, const void** cscRowInd, const void** cscValues, cusparseIndexType_t* cscColOffsetsType, cusparseIndexType_t* cscRowIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstCscGet
    _check_or_init_cusparse()
    if __cusparseConstCscGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstCscGet is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t, int64_t*, int64_t*, int64_t*, const void**, const void**, const void**, cusparseIndexType_t*, cusparseIndexType_t*, cusparseIndexBase_t*, cudaDataType*) noexcept nogil>__cusparseConstCscGet)(
        spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCreateConstCoo(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cooRowInd, const void* cooColInd, const void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateConstCoo
    _check_or_init_cusparse()
    if __cusparseCreateConstCoo == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateConstCoo is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t*, int64_t, int64_t, int64_t, const void*, const void*, const void*, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateConstCoo)(
        spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseConstCooGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** cooRowInd, const void** cooColInd, const void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstCooGet
    _check_or_init_cusparse()
    if __cusparseConstCooGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstCooGet is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t, int64_t*, int64_t*, int64_t*, const void**, const void**, const void**, cusparseIndexType_t*, cusparseIndexBase_t*, cudaDataType*) noexcept nogil>__cusparseConstCooGet)(
        spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCreateConstBlockedEll(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, const void* ellColInd, const void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateConstBlockedEll
    _check_or_init_cusparse()
    if __cusparseCreateConstBlockedEll == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateConstBlockedEll is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t*, int64_t, int64_t, int64_t, int64_t, const void*, const void*, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateConstBlockedEll)(
        spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseConstBlockedEllGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, const void** ellColInd, const void** ellValue, cusparseIndexType_t* ellIdxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstBlockedEllGet
    _check_or_init_cusparse()
    if __cusparseConstBlockedEllGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstBlockedEllGet is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t, int64_t*, int64_t*, int64_t*, int64_t*, const void**, const void**, cusparseIndexType_t*, cusparseIndexBase_t*, cudaDataType*) noexcept nogil>__cusparseConstBlockedEllGet)(
        spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCreateConstDnMat(cusparseConstDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, const void* values, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateConstDnMat
    _check_or_init_cusparse()
    if __cusparseCreateConstDnMat == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateConstDnMat is not found")
    return (<cusparseStatus_t (*)(cusparseConstDnMatDescr_t*, int64_t, int64_t, int64_t, const void*, cudaDataType, cusparseOrder_t) noexcept nogil>__cusparseCreateConstDnMat)(
        dnMatDescr, rows, cols, ld, values, valueType, order)


cdef cusparseStatus_t _cusparseConstDnMatGet(cusparseConstDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, const void** values, cudaDataType* type, cusparseOrder_t* order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstDnMatGet
    _check_or_init_cusparse()
    if __cusparseConstDnMatGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstDnMatGet is not found")
    return (<cusparseStatus_t (*)(cusparseConstDnMatDescr_t, int64_t*, int64_t*, int64_t*, const void**, cudaDataType*, cusparseOrder_t*) noexcept nogil>__cusparseConstDnMatGet)(
        dnMatDescr, rows, cols, ld, values, type, order)


cdef cusparseStatus_t _cusparseConstDnMatGetValues(cusparseConstDnMatDescr_t dnMatDescr, const void** values) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseConstDnMatGetValues
    _check_or_init_cusparse()
    if __cusparseConstDnMatGetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseConstDnMatGetValues is not found")
    return (<cusparseStatus_t (*)(cusparseConstDnMatDescr_t, const void**) noexcept nogil>__cusparseConstDnMatGetValues)(
        dnMatDescr, values)


cdef cusparseStatus_t _cusparseSpGEMM_getNumProducts(cusparseSpGEMMDescr_t spgemmDescr, int64_t* num_prods) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMM_getNumProducts
    _check_or_init_cusparse()
    if __cusparseSpGEMM_getNumProducts == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMM_getNumProducts is not found")
    return (<cusparseStatus_t (*)(cusparseSpGEMMDescr_t, int64_t*) noexcept nogil>__cusparseSpGEMM_getNumProducts)(
        spgemmDescr, num_prods)


cdef cusparseStatus_t _cusparseSpGEMM_estimateMemory(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, float chunk_fraction, size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize2) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpGEMM_estimateMemory
    _check_or_init_cusparse()
    if __cusparseSpGEMM_estimateMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpGEMM_estimateMemory is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, const void*, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, float, size_t*, void*, size_t*) noexcept nogil>__cusparseSpGEMM_estimateMemory)(
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, chunk_fraction, bufferSize3, externalBuffer3, bufferSize2)


cdef cusparseStatus_t _cusparseBsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride, int64_t columnsBatchStride, int64_t ValuesBatchStride) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseBsrSetStridedBatch
    _check_or_init_cusparse()
    if __cusparseBsrSetStridedBatch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseBsrSetStridedBatch is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t, int, int64_t, int64_t, int64_t) noexcept nogil>__cusparseBsrSetStridedBatch)(
        spMatDescr, batchCount, offsetsBatchStride, columnsBatchStride, ValuesBatchStride)


cdef cusparseStatus_t _cusparseCreateBsr(cusparseSpMatDescr_t* spMatDescr, int64_t brows, int64_t bcols, int64_t bnnz, int64_t rowBlockSize, int64_t colBlockSize, void* bsrRowOffsets, void* bsrColInd, void* bsrValues, cusparseIndexType_t bsrRowOffsetsType, cusparseIndexType_t bsrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateBsr
    _check_or_init_cusparse()
    if __cusparseCreateBsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateBsr is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t*, int64_t, int64_t, int64_t, int64_t, int64_t, void*, void*, void*, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType, cusparseOrder_t) noexcept nogil>__cusparseCreateBsr)(
        spMatDescr, brows, bcols, bnnz, rowBlockSize, colBlockSize, bsrRowOffsets, bsrColInd, bsrValues, bsrRowOffsetsType, bsrColIndType, idxBase, valueType, order)


cdef cusparseStatus_t _cusparseCreateConstBsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t brows, int64_t bcols, int64_t bnnz, int64_t rowBlockDim, int64_t colBlockDim, const void* bsrRowOffsets, const void* bsrColInd, const void* bsrValues, cusparseIndexType_t bsrRowOffsetsType, cusparseIndexType_t bsrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateConstBsr
    _check_or_init_cusparse()
    if __cusparseCreateConstBsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateConstBsr is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t*, int64_t, int64_t, int64_t, int64_t, int64_t, const void*, const void*, const void*, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType, cusparseOrder_t) noexcept nogil>__cusparseCreateConstBsr)(
        spMatDescr, brows, bcols, bnnz, rowBlockDim, colBlockDim, bsrRowOffsets, bsrColInd, bsrValues, bsrRowOffsetsType, bsrColIndType, idxBase, valueType, order)


cdef cusparseStatus_t _cusparseCreateSlicedEll(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, int64_t sellValuesSize, int64_t sliceSize, void* sellSliceOffsets, void* sellColInd, void* sellValues, cusparseIndexType_t sellSliceOffsetsType, cusparseIndexType_t sellColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateSlicedEll
    _check_or_init_cusparse()
    if __cusparseCreateSlicedEll == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateSlicedEll is not found")
    return (<cusparseStatus_t (*)(cusparseSpMatDescr_t*, int64_t, int64_t, int64_t, int64_t, int64_t, void*, void*, void*, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateSlicedEll)(
        spMatDescr, rows, cols, nnz, sellValuesSize, sliceSize, sellSliceOffsets, sellColInd, sellValues, sellSliceOffsetsType, sellColIndType, idxBase, valueType)


cdef cusparseStatus_t _cusparseCreateConstSlicedEll(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, int64_t sellValuesSize, int64_t sliceSize, const void* sellSliceOffsets, const void* sellColInd, const void* sellValues, cusparseIndexType_t sellSliceOffsetsType, cusparseIndexType_t sellColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseCreateConstSlicedEll
    _check_or_init_cusparse()
    if __cusparseCreateConstSlicedEll == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseCreateConstSlicedEll is not found")
    return (<cusparseStatus_t (*)(cusparseConstSpMatDescr_t*, int64_t, int64_t, int64_t, int64_t, int64_t, const void*, const void*, const void*, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) noexcept nogil>__cusparseCreateConstSlicedEll)(
        spMatDescr, rows, cols, nnz, sellValuesSize, sliceSize, sellSliceOffsets, sellColInd, sellValues, sellSliceOffsetsType, sellColIndType, idxBase, valueType)


cdef cusparseStatus_t _cusparseSpSV_updateMatrix(cusparseHandle_t handle, cusparseSpSVDescr_t spsvDescr, void* newValues, cusparseSpSVUpdate_t updatePart) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSV_updateMatrix
    _check_or_init_cusparse()
    if __cusparseSpSV_updateMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSV_updateMatrix is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseSpSVDescr_t, void*, cusparseSpSVUpdate_t) noexcept nogil>__cusparseSpSV_updateMatrix)(
        handle, spsvDescr, newValues, updatePart)


cdef cusparseStatus_t _cusparseSpMV_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpMV_preprocess
    _check_or_init_cusparse()
    if __cusparseSpMV_preprocess == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpMV_preprocess is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void*, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, const void*, cusparseDnVecDescr_t, cudaDataType, cusparseSpMVAlg_t, void*) noexcept nogil>__cusparseSpMV_preprocess)(
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer)


cdef cusparseStatus_t _cusparseSpSM_updateMatrix(cusparseHandle_t handle, cusparseSpSMDescr_t spsmDescr, void* newValues, cusparseSpSMUpdate_t updatePart) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseSpSM_updateMatrix
    _check_or_init_cusparse()
    if __cusparseSpSM_updateMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseSpSM_updateMatrix is not found")
    return (<cusparseStatus_t (*)(cusparseHandle_t, cusparseSpSMDescr_t, void*, cusparseSpSMUpdate_t) noexcept nogil>__cusparseSpSM_updateMatrix)(
        handle, spsmDescr, newValues, updatePart)
