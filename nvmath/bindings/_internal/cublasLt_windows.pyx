# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_cublas_dso_version_suffix

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
cdef bint __py_cublasLt_init = False
cdef void* __cuDriverGetVersion = NULL

cdef void* __cublasLtCreate = NULL
cdef void* __cublasLtDestroy = NULL
cdef void* __cublasLtGetVersion = NULL
cdef void* __cublasLtGetCudartVersion = NULL
cdef void* __cublasLtGetProperty = NULL
cdef void* __cublasLtMatmul = NULL
cdef void* __cublasLtMatrixTransform = NULL
cdef void* __cublasLtMatrixLayoutCreate = NULL
cdef void* __cublasLtMatrixLayoutDestroy = NULL
cdef void* __cublasLtMatrixLayoutSetAttribute = NULL
cdef void* __cublasLtMatrixLayoutGetAttribute = NULL
cdef void* __cublasLtMatmulDescCreate = NULL
cdef void* __cublasLtMatmulDescDestroy = NULL
cdef void* __cublasLtMatmulDescSetAttribute = NULL
cdef void* __cublasLtMatmulDescGetAttribute = NULL
cdef void* __cublasLtMatrixTransformDescCreate = NULL
cdef void* __cublasLtMatrixTransformDescDestroy = NULL
cdef void* __cublasLtMatrixTransformDescSetAttribute = NULL
cdef void* __cublasLtMatrixTransformDescGetAttribute = NULL
cdef void* __cublasLtMatmulPreferenceCreate = NULL
cdef void* __cublasLtMatmulPreferenceDestroy = NULL
cdef void* __cublasLtMatmulPreferenceSetAttribute = NULL
cdef void* __cublasLtMatmulPreferenceGetAttribute = NULL
cdef void* __cublasLtMatmulAlgoGetHeuristic = NULL
cdef void* __cublasLtMatmulAlgoGetIds = NULL
cdef void* __cublasLtMatmulAlgoInit = NULL
cdef void* __cublasLtMatmulAlgoCheck = NULL
cdef void* __cublasLtMatmulAlgoCapGetAttribute = NULL
cdef void* __cublasLtMatmulAlgoConfigSetAttribute = NULL
cdef void* __cublasLtMatmulAlgoConfigGetAttribute = NULL
cdef void* __cublasLtLoggerSetCallback = NULL
cdef void* __cublasLtLoggerSetFile = NULL
cdef void* __cublasLtLoggerOpenFile = NULL
cdef void* __cublasLtLoggerSetLevel = NULL
cdef void* __cublasLtLoggerSetMask = NULL
cdef void* __cublasLtLoggerForceDisable = NULL
cdef void* __cublasLtGetStatusName = NULL
cdef void* __cublasLtGetStatusString = NULL
cdef void* __cublasLtHeuristicsCacheGetCapacity = NULL
cdef void* __cublasLtHeuristicsCacheSetCapacity = NULL
cdef void* __cublasLtDisableCpuInstructionsSetMask = NULL


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef load_library(const int driver_ver):
    handle = 0

    for suffix in get_cublas_dso_version_suffix(driver_ver):
        if len(suffix) == 0:
            continue
        dll_name = f"cublasLt64_{suffix}.dll"

        # First check if the DLL has been loaded by 3rd parties
        try:
            handle = win32api.GetModuleHandle(dll_name)
        except:
            pass
        else:
            break

        # Next, check if DLLs are installed via pip
        for sp in get_site_packages():
            mod_path = os.path.join(sp, "nvidia", "cublas", "bin")
            if not os.path.isdir(mod_path):
                continue
            os.add_dll_directory(mod_path)
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
        raise RuntimeError('Failed to load cublasLt')

    assert handle != 0
    return handle


cdef int _check_or_init_cublasLt() except -1 nogil:
    global __py_cublasLt_init
    if __py_cublasLt_init:
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
        global __cublasLtCreate
        try:
            __cublasLtCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtCreate')
        except:
            pass

        global __cublasLtDestroy
        try:
            __cublasLtDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtDestroy')
        except:
            pass

        global __cublasLtGetVersion
        try:
            __cublasLtGetVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtGetVersion')
        except:
            pass

        global __cublasLtGetCudartVersion
        try:
            __cublasLtGetCudartVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtGetCudartVersion')
        except:
            pass

        global __cublasLtGetProperty
        try:
            __cublasLtGetProperty = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtGetProperty')
        except:
            pass

        global __cublasLtMatmul
        try:
            __cublasLtMatmul = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmul')
        except:
            pass

        global __cublasLtMatrixTransform
        try:
            __cublasLtMatrixTransform = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatrixTransform')
        except:
            pass

        global __cublasLtMatrixLayoutCreate
        try:
            __cublasLtMatrixLayoutCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatrixLayoutCreate')
        except:
            pass

        global __cublasLtMatrixLayoutDestroy
        try:
            __cublasLtMatrixLayoutDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatrixLayoutDestroy')
        except:
            pass

        global __cublasLtMatrixLayoutSetAttribute
        try:
            __cublasLtMatrixLayoutSetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatrixLayoutSetAttribute')
        except:
            pass

        global __cublasLtMatrixLayoutGetAttribute
        try:
            __cublasLtMatrixLayoutGetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatrixLayoutGetAttribute')
        except:
            pass

        global __cublasLtMatmulDescCreate
        try:
            __cublasLtMatmulDescCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulDescCreate')
        except:
            pass

        global __cublasLtMatmulDescDestroy
        try:
            __cublasLtMatmulDescDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulDescDestroy')
        except:
            pass

        global __cublasLtMatmulDescSetAttribute
        try:
            __cublasLtMatmulDescSetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulDescSetAttribute')
        except:
            pass

        global __cublasLtMatmulDescGetAttribute
        try:
            __cublasLtMatmulDescGetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulDescGetAttribute')
        except:
            pass

        global __cublasLtMatrixTransformDescCreate
        try:
            __cublasLtMatrixTransformDescCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatrixTransformDescCreate')
        except:
            pass

        global __cublasLtMatrixTransformDescDestroy
        try:
            __cublasLtMatrixTransformDescDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatrixTransformDescDestroy')
        except:
            pass

        global __cublasLtMatrixTransformDescSetAttribute
        try:
            __cublasLtMatrixTransformDescSetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatrixTransformDescSetAttribute')
        except:
            pass

        global __cublasLtMatrixTransformDescGetAttribute
        try:
            __cublasLtMatrixTransformDescGetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatrixTransformDescGetAttribute')
        except:
            pass

        global __cublasLtMatmulPreferenceCreate
        try:
            __cublasLtMatmulPreferenceCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulPreferenceCreate')
        except:
            pass

        global __cublasLtMatmulPreferenceDestroy
        try:
            __cublasLtMatmulPreferenceDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulPreferenceDestroy')
        except:
            pass

        global __cublasLtMatmulPreferenceSetAttribute
        try:
            __cublasLtMatmulPreferenceSetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulPreferenceSetAttribute')
        except:
            pass

        global __cublasLtMatmulPreferenceGetAttribute
        try:
            __cublasLtMatmulPreferenceGetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulPreferenceGetAttribute')
        except:
            pass

        global __cublasLtMatmulAlgoGetHeuristic
        try:
            __cublasLtMatmulAlgoGetHeuristic = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulAlgoGetHeuristic')
        except:
            pass

        global __cublasLtMatmulAlgoGetIds
        try:
            __cublasLtMatmulAlgoGetIds = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulAlgoGetIds')
        except:
            pass

        global __cublasLtMatmulAlgoInit
        try:
            __cublasLtMatmulAlgoInit = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulAlgoInit')
        except:
            pass

        global __cublasLtMatmulAlgoCheck
        try:
            __cublasLtMatmulAlgoCheck = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulAlgoCheck')
        except:
            pass

        global __cublasLtMatmulAlgoCapGetAttribute
        try:
            __cublasLtMatmulAlgoCapGetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulAlgoCapGetAttribute')
        except:
            pass

        global __cublasLtMatmulAlgoConfigSetAttribute
        try:
            __cublasLtMatmulAlgoConfigSetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulAlgoConfigSetAttribute')
        except:
            pass

        global __cublasLtMatmulAlgoConfigGetAttribute
        try:
            __cublasLtMatmulAlgoConfigGetAttribute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtMatmulAlgoConfigGetAttribute')
        except:
            pass

        global __cublasLtLoggerSetCallback
        try:
            __cublasLtLoggerSetCallback = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtLoggerSetCallback')
        except:
            pass

        global __cublasLtLoggerSetFile
        try:
            __cublasLtLoggerSetFile = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtLoggerSetFile')
        except:
            pass

        global __cublasLtLoggerOpenFile
        try:
            __cublasLtLoggerOpenFile = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtLoggerOpenFile')
        except:
            pass

        global __cublasLtLoggerSetLevel
        try:
            __cublasLtLoggerSetLevel = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtLoggerSetLevel')
        except:
            pass

        global __cublasLtLoggerSetMask
        try:
            __cublasLtLoggerSetMask = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtLoggerSetMask')
        except:
            pass

        global __cublasLtLoggerForceDisable
        try:
            __cublasLtLoggerForceDisable = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtLoggerForceDisable')
        except:
            pass

        global __cublasLtGetStatusName
        try:
            __cublasLtGetStatusName = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtGetStatusName')
        except:
            pass

        global __cublasLtGetStatusString
        try:
            __cublasLtGetStatusString = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtGetStatusString')
        except:
            pass

        global __cublasLtHeuristicsCacheGetCapacity
        try:
            __cublasLtHeuristicsCacheGetCapacity = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtHeuristicsCacheGetCapacity')
        except:
            pass

        global __cublasLtHeuristicsCacheSetCapacity
        try:
            __cublasLtHeuristicsCacheSetCapacity = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtHeuristicsCacheSetCapacity')
        except:
            pass

        global __cublasLtDisableCpuInstructionsSetMask
        try:
            __cublasLtDisableCpuInstructionsSetMask = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasLtDisableCpuInstructionsSetMask')
        except:
            pass

    __py_cublasLt_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cublasLt()
    cdef dict data = {}

    global __cublasLtCreate
    data["__cublasLtCreate"] = <intptr_t>__cublasLtCreate

    global __cublasLtDestroy
    data["__cublasLtDestroy"] = <intptr_t>__cublasLtDestroy

    global __cublasLtGetVersion
    data["__cublasLtGetVersion"] = <intptr_t>__cublasLtGetVersion

    global __cublasLtGetCudartVersion
    data["__cublasLtGetCudartVersion"] = <intptr_t>__cublasLtGetCudartVersion

    global __cublasLtGetProperty
    data["__cublasLtGetProperty"] = <intptr_t>__cublasLtGetProperty

    global __cublasLtMatmul
    data["__cublasLtMatmul"] = <intptr_t>__cublasLtMatmul

    global __cublasLtMatrixTransform
    data["__cublasLtMatrixTransform"] = <intptr_t>__cublasLtMatrixTransform

    global __cublasLtMatrixLayoutCreate
    data["__cublasLtMatrixLayoutCreate"] = <intptr_t>__cublasLtMatrixLayoutCreate

    global __cublasLtMatrixLayoutDestroy
    data["__cublasLtMatrixLayoutDestroy"] = <intptr_t>__cublasLtMatrixLayoutDestroy

    global __cublasLtMatrixLayoutSetAttribute
    data["__cublasLtMatrixLayoutSetAttribute"] = <intptr_t>__cublasLtMatrixLayoutSetAttribute

    global __cublasLtMatrixLayoutGetAttribute
    data["__cublasLtMatrixLayoutGetAttribute"] = <intptr_t>__cublasLtMatrixLayoutGetAttribute

    global __cublasLtMatmulDescCreate
    data["__cublasLtMatmulDescCreate"] = <intptr_t>__cublasLtMatmulDescCreate

    global __cublasLtMatmulDescDestroy
    data["__cublasLtMatmulDescDestroy"] = <intptr_t>__cublasLtMatmulDescDestroy

    global __cublasLtMatmulDescSetAttribute
    data["__cublasLtMatmulDescSetAttribute"] = <intptr_t>__cublasLtMatmulDescSetAttribute

    global __cublasLtMatmulDescGetAttribute
    data["__cublasLtMatmulDescGetAttribute"] = <intptr_t>__cublasLtMatmulDescGetAttribute

    global __cublasLtMatrixTransformDescCreate
    data["__cublasLtMatrixTransformDescCreate"] = <intptr_t>__cublasLtMatrixTransformDescCreate

    global __cublasLtMatrixTransformDescDestroy
    data["__cublasLtMatrixTransformDescDestroy"] = <intptr_t>__cublasLtMatrixTransformDescDestroy

    global __cublasLtMatrixTransformDescSetAttribute
    data["__cublasLtMatrixTransformDescSetAttribute"] = <intptr_t>__cublasLtMatrixTransformDescSetAttribute

    global __cublasLtMatrixTransformDescGetAttribute
    data["__cublasLtMatrixTransformDescGetAttribute"] = <intptr_t>__cublasLtMatrixTransformDescGetAttribute

    global __cublasLtMatmulPreferenceCreate
    data["__cublasLtMatmulPreferenceCreate"] = <intptr_t>__cublasLtMatmulPreferenceCreate

    global __cublasLtMatmulPreferenceDestroy
    data["__cublasLtMatmulPreferenceDestroy"] = <intptr_t>__cublasLtMatmulPreferenceDestroy

    global __cublasLtMatmulPreferenceSetAttribute
    data["__cublasLtMatmulPreferenceSetAttribute"] = <intptr_t>__cublasLtMatmulPreferenceSetAttribute

    global __cublasLtMatmulPreferenceGetAttribute
    data["__cublasLtMatmulPreferenceGetAttribute"] = <intptr_t>__cublasLtMatmulPreferenceGetAttribute

    global __cublasLtMatmulAlgoGetHeuristic
    data["__cublasLtMatmulAlgoGetHeuristic"] = <intptr_t>__cublasLtMatmulAlgoGetHeuristic

    global __cublasLtMatmulAlgoGetIds
    data["__cublasLtMatmulAlgoGetIds"] = <intptr_t>__cublasLtMatmulAlgoGetIds

    global __cublasLtMatmulAlgoInit
    data["__cublasLtMatmulAlgoInit"] = <intptr_t>__cublasLtMatmulAlgoInit

    global __cublasLtMatmulAlgoCheck
    data["__cublasLtMatmulAlgoCheck"] = <intptr_t>__cublasLtMatmulAlgoCheck

    global __cublasLtMatmulAlgoCapGetAttribute
    data["__cublasLtMatmulAlgoCapGetAttribute"] = <intptr_t>__cublasLtMatmulAlgoCapGetAttribute

    global __cublasLtMatmulAlgoConfigSetAttribute
    data["__cublasLtMatmulAlgoConfigSetAttribute"] = <intptr_t>__cublasLtMatmulAlgoConfigSetAttribute

    global __cublasLtMatmulAlgoConfigGetAttribute
    data["__cublasLtMatmulAlgoConfigGetAttribute"] = <intptr_t>__cublasLtMatmulAlgoConfigGetAttribute

    global __cublasLtLoggerSetCallback
    data["__cublasLtLoggerSetCallback"] = <intptr_t>__cublasLtLoggerSetCallback

    global __cublasLtLoggerSetFile
    data["__cublasLtLoggerSetFile"] = <intptr_t>__cublasLtLoggerSetFile

    global __cublasLtLoggerOpenFile
    data["__cublasLtLoggerOpenFile"] = <intptr_t>__cublasLtLoggerOpenFile

    global __cublasLtLoggerSetLevel
    data["__cublasLtLoggerSetLevel"] = <intptr_t>__cublasLtLoggerSetLevel

    global __cublasLtLoggerSetMask
    data["__cublasLtLoggerSetMask"] = <intptr_t>__cublasLtLoggerSetMask

    global __cublasLtLoggerForceDisable
    data["__cublasLtLoggerForceDisable"] = <intptr_t>__cublasLtLoggerForceDisable

    global __cublasLtGetStatusName
    data["__cublasLtGetStatusName"] = <intptr_t>__cublasLtGetStatusName

    global __cublasLtGetStatusString
    data["__cublasLtGetStatusString"] = <intptr_t>__cublasLtGetStatusString

    global __cublasLtHeuristicsCacheGetCapacity
    data["__cublasLtHeuristicsCacheGetCapacity"] = <intptr_t>__cublasLtHeuristicsCacheGetCapacity

    global __cublasLtHeuristicsCacheSetCapacity
    data["__cublasLtHeuristicsCacheSetCapacity"] = <intptr_t>__cublasLtHeuristicsCacheSetCapacity

    global __cublasLtDisableCpuInstructionsSetMask
    data["__cublasLtDisableCpuInstructionsSetMask"] = <intptr_t>__cublasLtDisableCpuInstructionsSetMask

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

cdef cublasStatus_t _cublasLtCreate(cublasLtHandle_t* lightHandle) except* nogil:
    global __cublasLtCreate
    _check_or_init_cublasLt()
    if __cublasLtCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtCreate is not found")
    return (<cublasStatus_t (*)(cublasLtHandle_t*) nogil>__cublasLtCreate)(
        lightHandle)


cdef cublasStatus_t _cublasLtDestroy(cublasLtHandle_t lightHandle) except* nogil:
    global __cublasLtDestroy
    _check_or_init_cublasLt()
    if __cublasLtDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtDestroy is not found")
    return (<cublasStatus_t (*)(cublasLtHandle_t) nogil>__cublasLtDestroy)(
        lightHandle)


cdef size_t _cublasLtGetVersion() except* nogil:
    global __cublasLtGetVersion
    _check_or_init_cublasLt()
    if __cublasLtGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtGetVersion is not found")
    return (<size_t (*)() nogil>__cublasLtGetVersion)(
        )


cdef size_t _cublasLtGetCudartVersion() except* nogil:
    global __cublasLtGetCudartVersion
    _check_or_init_cublasLt()
    if __cublasLtGetCudartVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtGetCudartVersion is not found")
    return (<size_t (*)() nogil>__cublasLtGetCudartVersion)(
        )


cdef cublasStatus_t _cublasLtGetProperty(libraryPropertyType type, int* value) except* nogil:
    global __cublasLtGetProperty
    _check_or_init_cublasLt()
    if __cublasLtGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtGetProperty is not found")
    return (<cublasStatus_t (*)(libraryPropertyType, int*) nogil>__cublasLtGetProperty)(
        type, value)


cdef cublasStatus_t _cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream) except* nogil:
    global __cublasLtMatmul
    _check_or_init_cublasLt()
    if __cublasLtMatmul == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmul is not found")
    return (<cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, const void*, const void*, cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t, const void*, const void*, cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t*, void*, size_t, cudaStream_t) nogil>__cublasLtMatmul)(
        lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream)


cdef cublasStatus_t _cublasLtMatrixTransform(cublasLtHandle_t lightHandle, cublasLtMatrixTransformDesc_t transformDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* beta, const void* B, cublasLtMatrixLayout_t Bdesc, void* C, cublasLtMatrixLayout_t Cdesc, cudaStream_t stream) except* nogil:
    global __cublasLtMatrixTransform
    _check_or_init_cublasLt()
    if __cublasLtMatrixTransform == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatrixTransform is not found")
    return (<cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatrixTransformDesc_t, const void*, const void*, cublasLtMatrixLayout_t, const void*, const void*, cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t, cudaStream_t) nogil>__cublasLtMatrixTransform)(
        lightHandle, transformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream)


cdef cublasStatus_t _cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld) except* nogil:
    global __cublasLtMatrixLayoutCreate
    _check_or_init_cublasLt()
    if __cublasLtMatrixLayoutCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatrixLayoutCreate is not found")
    return (<cublasStatus_t (*)(cublasLtMatrixLayout_t*, cudaDataType, uint64_t, uint64_t, int64_t) nogil>__cublasLtMatrixLayoutCreate)(
        matLayout, type, rows, cols, ld)


cdef cublasStatus_t _cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) except* nogil:
    global __cublasLtMatrixLayoutDestroy
    _check_or_init_cublasLt()
    if __cublasLtMatrixLayoutDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatrixLayoutDestroy is not found")
    return (<cublasStatus_t (*)(cublasLtMatrixLayout_t) nogil>__cublasLtMatrixLayoutDestroy)(
        matLayout)


cdef cublasStatus_t _cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    global __cublasLtMatrixLayoutSetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatrixLayoutSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatrixLayoutSetAttribute is not found")
    return (<cublasStatus_t (*)(cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, const void*, size_t) nogil>__cublasLtMatrixLayoutSetAttribute)(
        matLayout, attr, buf, sizeInBytes)


cdef cublasStatus_t _cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    global __cublasLtMatrixLayoutGetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatrixLayoutGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatrixLayoutGetAttribute is not found")
    return (<cublasStatus_t (*)(cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, void*, size_t, size_t*) nogil>__cublasLtMatrixLayoutGetAttribute)(
        matLayout, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t _cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType) except* nogil:
    global __cublasLtMatmulDescCreate
    _check_or_init_cublasLt()
    if __cublasLtMatmulDescCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulDescCreate is not found")
    return (<cublasStatus_t (*)(cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t) nogil>__cublasLtMatmulDescCreate)(
        matmulDesc, computeType, scaleType)


cdef cublasStatus_t _cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) except* nogil:
    global __cublasLtMatmulDescDestroy
    _check_or_init_cublasLt()
    if __cublasLtMatmulDescDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulDescDestroy is not found")
    return (<cublasStatus_t (*)(cublasLtMatmulDesc_t) nogil>__cublasLtMatmulDescDestroy)(
        matmulDesc)


cdef cublasStatus_t _cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    global __cublasLtMatmulDescSetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatmulDescSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulDescSetAttribute is not found")
    return (<cublasStatus_t (*)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void*, size_t) nogil>__cublasLtMatmulDescSetAttribute)(
        matmulDesc, attr, buf, sizeInBytes)


cdef cublasStatus_t _cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    global __cublasLtMatmulDescGetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatmulDescGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulDescGetAttribute is not found")
    return (<cublasStatus_t (*)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, void*, size_t, size_t*) nogil>__cublasLtMatmulDescGetAttribute)(
        matmulDesc, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t _cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t* transformDesc, cudaDataType scaleType) except* nogil:
    global __cublasLtMatrixTransformDescCreate
    _check_or_init_cublasLt()
    if __cublasLtMatrixTransformDescCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatrixTransformDescCreate is not found")
    return (<cublasStatus_t (*)(cublasLtMatrixTransformDesc_t*, cudaDataType) nogil>__cublasLtMatrixTransformDescCreate)(
        transformDesc, scaleType)


cdef cublasStatus_t _cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t transformDesc) except* nogil:
    global __cublasLtMatrixTransformDescDestroy
    _check_or_init_cublasLt()
    if __cublasLtMatrixTransformDescDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatrixTransformDescDestroy is not found")
    return (<cublasStatus_t (*)(cublasLtMatrixTransformDesc_t) nogil>__cublasLtMatrixTransformDescDestroy)(
        transformDesc)


cdef cublasStatus_t _cublasLtMatrixTransformDescSetAttribute(cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    global __cublasLtMatrixTransformDescSetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatrixTransformDescSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatrixTransformDescSetAttribute is not found")
    return (<cublasStatus_t (*)(cublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDescAttributes_t, const void*, size_t) nogil>__cublasLtMatrixTransformDescSetAttribute)(
        transformDesc, attr, buf, sizeInBytes)


cdef cublasStatus_t _cublasLtMatrixTransformDescGetAttribute(cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    global __cublasLtMatrixTransformDescGetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatrixTransformDescGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatrixTransformDescGetAttribute is not found")
    return (<cublasStatus_t (*)(cublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDescAttributes_t, void*, size_t, size_t*) nogil>__cublasLtMatrixTransformDescGetAttribute)(
        transformDesc, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t _cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref) except* nogil:
    global __cublasLtMatmulPreferenceCreate
    _check_or_init_cublasLt()
    if __cublasLtMatmulPreferenceCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulPreferenceCreate is not found")
    return (<cublasStatus_t (*)(cublasLtMatmulPreference_t*) nogil>__cublasLtMatmulPreferenceCreate)(
        pref)


cdef cublasStatus_t _cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) except* nogil:
    global __cublasLtMatmulPreferenceDestroy
    _check_or_init_cublasLt()
    if __cublasLtMatmulPreferenceDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulPreferenceDestroy is not found")
    return (<cublasStatus_t (*)(cublasLtMatmulPreference_t) nogil>__cublasLtMatmulPreferenceDestroy)(
        pref)


cdef cublasStatus_t _cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    global __cublasLtMatmulPreferenceSetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatmulPreferenceSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulPreferenceSetAttribute is not found")
    return (<cublasStatus_t (*)(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void*, size_t) nogil>__cublasLtMatmulPreferenceSetAttribute)(
        pref, attr, buf, sizeInBytes)


cdef cublasStatus_t _cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    global __cublasLtMatmulPreferenceGetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatmulPreferenceGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulPreferenceGetAttribute is not found")
    return (<cublasStatus_t (*)(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, void*, size_t, size_t*) nogil>__cublasLtMatmulPreferenceGetAttribute)(
        pref, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t _cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount) except* nogil:
    global __cublasLtMatmulAlgoGetHeuristic
    _check_or_init_cublasLt()
    if __cublasLtMatmulAlgoGetHeuristic == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulAlgoGetHeuristic is not found")
    return (<cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t*, int*) nogil>__cublasLtMatmulAlgoGetHeuristic)(
        lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount)


cdef cublasStatus_t _cublasLtMatmulAlgoGetIds(cublasLtHandle_t lightHandle, cublasComputeType_t computeType, cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype, int requestedAlgoCount, int algoIdsArray[], int* returnAlgoCount) except* nogil:
    global __cublasLtMatmulAlgoGetIds
    _check_or_init_cublasLt()
    if __cublasLtMatmulAlgoGetIds == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulAlgoGetIds is not found")
    return (<cublasStatus_t (*)(cublasLtHandle_t, cublasComputeType_t, cudaDataType_t, cudaDataType_t, cudaDataType_t, cudaDataType_t, cudaDataType_t, int, int*, int*) nogil>__cublasLtMatmulAlgoGetIds)(
        lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, requestedAlgoCount, algoIdsArray, returnAlgoCount)


cdef cublasStatus_t _cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle, cublasComputeType_t computeType, cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype, int algoId, cublasLtMatmulAlgo_t* algo) except* nogil:
    global __cublasLtMatmulAlgoInit
    _check_or_init_cublasLt()
    if __cublasLtMatmulAlgoInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulAlgoInit is not found")
    return (<cublasStatus_t (*)(cublasLtHandle_t, cublasComputeType_t, cudaDataType_t, cudaDataType_t, cudaDataType_t, cudaDataType_t, cudaDataType_t, int, cublasLtMatmulAlgo_t*) nogil>__cublasLtMatmulAlgoInit)(
        lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, algoId, algo)


cdef cublasStatus_t _cublasLtMatmulAlgoCheck(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo, cublasLtMatmulHeuristicResult_t* result) except* nogil:
    global __cublasLtMatmulAlgoCheck
    _check_or_init_cublasLt()
    if __cublasLtMatmulAlgoCheck == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulAlgoCheck is not found")
    return (<cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t*, cublasLtMatmulHeuristicResult_t*) nogil>__cublasLtMatmulAlgoCheck)(
        lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, algo, result)


cdef cublasStatus_t _cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoCapAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    global __cublasLtMatmulAlgoCapGetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatmulAlgoCapGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulAlgoCapGetAttribute is not found")
    return (<cublasStatus_t (*)(const cublasLtMatmulAlgo_t*, cublasLtMatmulAlgoCapAttributes_t, void*, size_t, size_t*) nogil>__cublasLtMatmulAlgoCapGetAttribute)(
        algo, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t _cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr, const void* buf, size_t sizeInBytes) except* nogil:
    global __cublasLtMatmulAlgoConfigSetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatmulAlgoConfigSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulAlgoConfigSetAttribute is not found")
    return (<cublasStatus_t (*)(cublasLtMatmulAlgo_t*, cublasLtMatmulAlgoConfigAttributes_t, const void*, size_t) nogil>__cublasLtMatmulAlgoConfigSetAttribute)(
        algo, attr, buf, sizeInBytes)


cdef cublasStatus_t _cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoConfigAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except* nogil:
    global __cublasLtMatmulAlgoConfigGetAttribute
    _check_or_init_cublasLt()
    if __cublasLtMatmulAlgoConfigGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtMatmulAlgoConfigGetAttribute is not found")
    return (<cublasStatus_t (*)(const cublasLtMatmulAlgo_t*, cublasLtMatmulAlgoConfigAttributes_t, void*, size_t, size_t*) nogil>__cublasLtMatmulAlgoConfigGetAttribute)(
        algo, attr, buf, sizeInBytes, sizeWritten)


cdef cublasStatus_t _cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback) except* nogil:
    global __cublasLtLoggerSetCallback
    _check_or_init_cublasLt()
    if __cublasLtLoggerSetCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtLoggerSetCallback is not found")
    return (<cublasStatus_t (*)(cublasLtLoggerCallback_t) nogil>__cublasLtLoggerSetCallback)(
        callback)


cdef cublasStatus_t _cublasLtLoggerSetFile(FILE* file) except* nogil:
    global __cublasLtLoggerSetFile
    _check_or_init_cublasLt()
    if __cublasLtLoggerSetFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtLoggerSetFile is not found")
    return (<cublasStatus_t (*)(FILE*) nogil>__cublasLtLoggerSetFile)(
        file)


cdef cublasStatus_t _cublasLtLoggerOpenFile(const char* logFile) except* nogil:
    global __cublasLtLoggerOpenFile
    _check_or_init_cublasLt()
    if __cublasLtLoggerOpenFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtLoggerOpenFile is not found")
    return (<cublasStatus_t (*)(const char*) nogil>__cublasLtLoggerOpenFile)(
        logFile)


cdef cublasStatus_t _cublasLtLoggerSetLevel(int level) except* nogil:
    global __cublasLtLoggerSetLevel
    _check_or_init_cublasLt()
    if __cublasLtLoggerSetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtLoggerSetLevel is not found")
    return (<cublasStatus_t (*)(int) nogil>__cublasLtLoggerSetLevel)(
        level)


cdef cublasStatus_t _cublasLtLoggerSetMask(int mask) except* nogil:
    global __cublasLtLoggerSetMask
    _check_or_init_cublasLt()
    if __cublasLtLoggerSetMask == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtLoggerSetMask is not found")
    return (<cublasStatus_t (*)(int) nogil>__cublasLtLoggerSetMask)(
        mask)


cdef cublasStatus_t _cublasLtLoggerForceDisable() except* nogil:
    global __cublasLtLoggerForceDisable
    _check_or_init_cublasLt()
    if __cublasLtLoggerForceDisable == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtLoggerForceDisable is not found")
    return (<cublasStatus_t (*)() nogil>__cublasLtLoggerForceDisable)(
        )


cdef const char* _cublasLtGetStatusName(cublasStatus_t status) except* nogil:
    global __cublasLtGetStatusName
    _check_or_init_cublasLt()
    if __cublasLtGetStatusName == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtGetStatusName is not found")
    return (<const char* (*)(cublasStatus_t) nogil>__cublasLtGetStatusName)(
        status)


cdef const char* _cublasLtGetStatusString(cublasStatus_t status) except* nogil:
    global __cublasLtGetStatusString
    _check_or_init_cublasLt()
    if __cublasLtGetStatusString == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtGetStatusString is not found")
    return (<const char* (*)(cublasStatus_t) nogil>__cublasLtGetStatusString)(
        status)


cdef cublasStatus_t _cublasLtHeuristicsCacheGetCapacity(size_t* capacity) except* nogil:
    global __cublasLtHeuristicsCacheGetCapacity
    _check_or_init_cublasLt()
    if __cublasLtHeuristicsCacheGetCapacity == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtHeuristicsCacheGetCapacity is not found")
    return (<cublasStatus_t (*)(size_t*) nogil>__cublasLtHeuristicsCacheGetCapacity)(
        capacity)


cdef cublasStatus_t _cublasLtHeuristicsCacheSetCapacity(size_t capacity) except* nogil:
    global __cublasLtHeuristicsCacheSetCapacity
    _check_or_init_cublasLt()
    if __cublasLtHeuristicsCacheSetCapacity == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtHeuristicsCacheSetCapacity is not found")
    return (<cublasStatus_t (*)(size_t) nogil>__cublasLtHeuristicsCacheSetCapacity)(
        capacity)


cdef unsigned _cublasLtDisableCpuInstructionsSetMask(unsigned mask) except* nogil:
    global __cublasLtDisableCpuInstructionsSetMask
    _check_or_init_cublasLt()
    if __cublasLtDisableCpuInstructionsSetMask == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLtDisableCpuInstructionsSetMask is not found")
    return (<unsigned (*)(unsigned) nogil>__cublasLtDisableCpuInstructionsSetMask)(
        mask)
