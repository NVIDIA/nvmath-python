# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_cublas_dso_version_suffix

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


cdef void* load_library(const int driver_ver) except* with gil:
    cdef void* handle
    for suffix in get_cublas_dso_version_suffix(driver_ver):
        so_name = "libcublasLt.so" + (f".{suffix}" if suffix else suffix)
        handle = dlopen(so_name.encode(), RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            break
    else:
        err_msg = dlerror()
        raise RuntimeError(f'Failed to dlopen libcublasLt ({err_msg.decode()})')
    return handle


cdef int _check_or_init_cublasLt() except -1 nogil:
    global __py_cublasLt_init
    if __py_cublasLt_init:
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
    err = (<int (*)(int*) nogil>__cuDriverGetVersion)(&driver_ver)
    if err != 0:
        with gil:
            raise RuntimeError('something went wrong')
    #dlclose(handle)
    handle = NULL

    # Load function
    global __cublasLtCreate
    __cublasLtCreate = dlsym(RTLD_DEFAULT, 'cublasLtCreate')
    if __cublasLtCreate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtCreate = dlsym(handle, 'cublasLtCreate')

    global __cublasLtDestroy
    __cublasLtDestroy = dlsym(RTLD_DEFAULT, 'cublasLtDestroy')
    if __cublasLtDestroy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtDestroy = dlsym(handle, 'cublasLtDestroy')

    global __cublasLtGetVersion
    __cublasLtGetVersion = dlsym(RTLD_DEFAULT, 'cublasLtGetVersion')
    if __cublasLtGetVersion == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtGetVersion = dlsym(handle, 'cublasLtGetVersion')

    global __cublasLtGetCudartVersion
    __cublasLtGetCudartVersion = dlsym(RTLD_DEFAULT, 'cublasLtGetCudartVersion')
    if __cublasLtGetCudartVersion == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtGetCudartVersion = dlsym(handle, 'cublasLtGetCudartVersion')

    global __cublasLtGetProperty
    __cublasLtGetProperty = dlsym(RTLD_DEFAULT, 'cublasLtGetProperty')
    if __cublasLtGetProperty == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtGetProperty = dlsym(handle, 'cublasLtGetProperty')

    global __cublasLtMatmul
    __cublasLtMatmul = dlsym(RTLD_DEFAULT, 'cublasLtMatmul')
    if __cublasLtMatmul == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmul = dlsym(handle, 'cublasLtMatmul')

    global __cublasLtMatrixTransform
    __cublasLtMatrixTransform = dlsym(RTLD_DEFAULT, 'cublasLtMatrixTransform')
    if __cublasLtMatrixTransform == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatrixTransform = dlsym(handle, 'cublasLtMatrixTransform')

    global __cublasLtMatrixLayoutCreate
    __cublasLtMatrixLayoutCreate = dlsym(RTLD_DEFAULT, 'cublasLtMatrixLayoutCreate')
    if __cublasLtMatrixLayoutCreate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatrixLayoutCreate = dlsym(handle, 'cublasLtMatrixLayoutCreate')

    global __cublasLtMatrixLayoutDestroy
    __cublasLtMatrixLayoutDestroy = dlsym(RTLD_DEFAULT, 'cublasLtMatrixLayoutDestroy')
    if __cublasLtMatrixLayoutDestroy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatrixLayoutDestroy = dlsym(handle, 'cublasLtMatrixLayoutDestroy')

    global __cublasLtMatrixLayoutSetAttribute
    __cublasLtMatrixLayoutSetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatrixLayoutSetAttribute')
    if __cublasLtMatrixLayoutSetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatrixLayoutSetAttribute = dlsym(handle, 'cublasLtMatrixLayoutSetAttribute')

    global __cublasLtMatrixLayoutGetAttribute
    __cublasLtMatrixLayoutGetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatrixLayoutGetAttribute')
    if __cublasLtMatrixLayoutGetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatrixLayoutGetAttribute = dlsym(handle, 'cublasLtMatrixLayoutGetAttribute')

    global __cublasLtMatmulDescCreate
    __cublasLtMatmulDescCreate = dlsym(RTLD_DEFAULT, 'cublasLtMatmulDescCreate')
    if __cublasLtMatmulDescCreate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulDescCreate = dlsym(handle, 'cublasLtMatmulDescCreate')

    global __cublasLtMatmulDescDestroy
    __cublasLtMatmulDescDestroy = dlsym(RTLD_DEFAULT, 'cublasLtMatmulDescDestroy')
    if __cublasLtMatmulDescDestroy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulDescDestroy = dlsym(handle, 'cublasLtMatmulDescDestroy')

    global __cublasLtMatmulDescSetAttribute
    __cublasLtMatmulDescSetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatmulDescSetAttribute')
    if __cublasLtMatmulDescSetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulDescSetAttribute = dlsym(handle, 'cublasLtMatmulDescSetAttribute')

    global __cublasLtMatmulDescGetAttribute
    __cublasLtMatmulDescGetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatmulDescGetAttribute')
    if __cublasLtMatmulDescGetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulDescGetAttribute = dlsym(handle, 'cublasLtMatmulDescGetAttribute')

    global __cublasLtMatrixTransformDescCreate
    __cublasLtMatrixTransformDescCreate = dlsym(RTLD_DEFAULT, 'cublasLtMatrixTransformDescCreate')
    if __cublasLtMatrixTransformDescCreate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatrixTransformDescCreate = dlsym(handle, 'cublasLtMatrixTransformDescCreate')

    global __cublasLtMatrixTransformDescDestroy
    __cublasLtMatrixTransformDescDestroy = dlsym(RTLD_DEFAULT, 'cublasLtMatrixTransformDescDestroy')
    if __cublasLtMatrixTransformDescDestroy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatrixTransformDescDestroy = dlsym(handle, 'cublasLtMatrixTransformDescDestroy')

    global __cublasLtMatrixTransformDescSetAttribute
    __cublasLtMatrixTransformDescSetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatrixTransformDescSetAttribute')
    if __cublasLtMatrixTransformDescSetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatrixTransformDescSetAttribute = dlsym(handle, 'cublasLtMatrixTransformDescSetAttribute')

    global __cublasLtMatrixTransformDescGetAttribute
    __cublasLtMatrixTransformDescGetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatrixTransformDescGetAttribute')
    if __cublasLtMatrixTransformDescGetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatrixTransformDescGetAttribute = dlsym(handle, 'cublasLtMatrixTransformDescGetAttribute')

    global __cublasLtMatmulPreferenceCreate
    __cublasLtMatmulPreferenceCreate = dlsym(RTLD_DEFAULT, 'cublasLtMatmulPreferenceCreate')
    if __cublasLtMatmulPreferenceCreate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulPreferenceCreate = dlsym(handle, 'cublasLtMatmulPreferenceCreate')

    global __cublasLtMatmulPreferenceDestroy
    __cublasLtMatmulPreferenceDestroy = dlsym(RTLD_DEFAULT, 'cublasLtMatmulPreferenceDestroy')
    if __cublasLtMatmulPreferenceDestroy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulPreferenceDestroy = dlsym(handle, 'cublasLtMatmulPreferenceDestroy')

    global __cublasLtMatmulPreferenceSetAttribute
    __cublasLtMatmulPreferenceSetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatmulPreferenceSetAttribute')
    if __cublasLtMatmulPreferenceSetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulPreferenceSetAttribute = dlsym(handle, 'cublasLtMatmulPreferenceSetAttribute')

    global __cublasLtMatmulPreferenceGetAttribute
    __cublasLtMatmulPreferenceGetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatmulPreferenceGetAttribute')
    if __cublasLtMatmulPreferenceGetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulPreferenceGetAttribute = dlsym(handle, 'cublasLtMatmulPreferenceGetAttribute')

    global __cublasLtMatmulAlgoGetHeuristic
    __cublasLtMatmulAlgoGetHeuristic = dlsym(RTLD_DEFAULT, 'cublasLtMatmulAlgoGetHeuristic')
    if __cublasLtMatmulAlgoGetHeuristic == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulAlgoGetHeuristic = dlsym(handle, 'cublasLtMatmulAlgoGetHeuristic')

    global __cublasLtMatmulAlgoGetIds
    __cublasLtMatmulAlgoGetIds = dlsym(RTLD_DEFAULT, 'cublasLtMatmulAlgoGetIds')
    if __cublasLtMatmulAlgoGetIds == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulAlgoGetIds = dlsym(handle, 'cublasLtMatmulAlgoGetIds')

    global __cublasLtMatmulAlgoInit
    __cublasLtMatmulAlgoInit = dlsym(RTLD_DEFAULT, 'cublasLtMatmulAlgoInit')
    if __cublasLtMatmulAlgoInit == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulAlgoInit = dlsym(handle, 'cublasLtMatmulAlgoInit')

    global __cublasLtMatmulAlgoCheck
    __cublasLtMatmulAlgoCheck = dlsym(RTLD_DEFAULT, 'cublasLtMatmulAlgoCheck')
    if __cublasLtMatmulAlgoCheck == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulAlgoCheck = dlsym(handle, 'cublasLtMatmulAlgoCheck')

    global __cublasLtMatmulAlgoCapGetAttribute
    __cublasLtMatmulAlgoCapGetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatmulAlgoCapGetAttribute')
    if __cublasLtMatmulAlgoCapGetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulAlgoCapGetAttribute = dlsym(handle, 'cublasLtMatmulAlgoCapGetAttribute')

    global __cublasLtMatmulAlgoConfigSetAttribute
    __cublasLtMatmulAlgoConfigSetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatmulAlgoConfigSetAttribute')
    if __cublasLtMatmulAlgoConfigSetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulAlgoConfigSetAttribute = dlsym(handle, 'cublasLtMatmulAlgoConfigSetAttribute')

    global __cublasLtMatmulAlgoConfigGetAttribute
    __cublasLtMatmulAlgoConfigGetAttribute = dlsym(RTLD_DEFAULT, 'cublasLtMatmulAlgoConfigGetAttribute')
    if __cublasLtMatmulAlgoConfigGetAttribute == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtMatmulAlgoConfigGetAttribute = dlsym(handle, 'cublasLtMatmulAlgoConfigGetAttribute')

    global __cublasLtLoggerSetCallback
    __cublasLtLoggerSetCallback = dlsym(RTLD_DEFAULT, 'cublasLtLoggerSetCallback')
    if __cublasLtLoggerSetCallback == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtLoggerSetCallback = dlsym(handle, 'cublasLtLoggerSetCallback')

    global __cublasLtLoggerSetFile
    __cublasLtLoggerSetFile = dlsym(RTLD_DEFAULT, 'cublasLtLoggerSetFile')
    if __cublasLtLoggerSetFile == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtLoggerSetFile = dlsym(handle, 'cublasLtLoggerSetFile')

    global __cublasLtLoggerOpenFile
    __cublasLtLoggerOpenFile = dlsym(RTLD_DEFAULT, 'cublasLtLoggerOpenFile')
    if __cublasLtLoggerOpenFile == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtLoggerOpenFile = dlsym(handle, 'cublasLtLoggerOpenFile')

    global __cublasLtLoggerSetLevel
    __cublasLtLoggerSetLevel = dlsym(RTLD_DEFAULT, 'cublasLtLoggerSetLevel')
    if __cublasLtLoggerSetLevel == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtLoggerSetLevel = dlsym(handle, 'cublasLtLoggerSetLevel')

    global __cublasLtLoggerSetMask
    __cublasLtLoggerSetMask = dlsym(RTLD_DEFAULT, 'cublasLtLoggerSetMask')
    if __cublasLtLoggerSetMask == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtLoggerSetMask = dlsym(handle, 'cublasLtLoggerSetMask')

    global __cublasLtLoggerForceDisable
    __cublasLtLoggerForceDisable = dlsym(RTLD_DEFAULT, 'cublasLtLoggerForceDisable')
    if __cublasLtLoggerForceDisable == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtLoggerForceDisable = dlsym(handle, 'cublasLtLoggerForceDisable')

    global __cublasLtGetStatusName
    __cublasLtGetStatusName = dlsym(RTLD_DEFAULT, 'cublasLtGetStatusName')
    if __cublasLtGetStatusName == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtGetStatusName = dlsym(handle, 'cublasLtGetStatusName')

    global __cublasLtGetStatusString
    __cublasLtGetStatusString = dlsym(RTLD_DEFAULT, 'cublasLtGetStatusString')
    if __cublasLtGetStatusString == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtGetStatusString = dlsym(handle, 'cublasLtGetStatusString')

    global __cublasLtHeuristicsCacheGetCapacity
    __cublasLtHeuristicsCacheGetCapacity = dlsym(RTLD_DEFAULT, 'cublasLtHeuristicsCacheGetCapacity')
    if __cublasLtHeuristicsCacheGetCapacity == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtHeuristicsCacheGetCapacity = dlsym(handle, 'cublasLtHeuristicsCacheGetCapacity')

    global __cublasLtHeuristicsCacheSetCapacity
    __cublasLtHeuristicsCacheSetCapacity = dlsym(RTLD_DEFAULT, 'cublasLtHeuristicsCacheSetCapacity')
    if __cublasLtHeuristicsCacheSetCapacity == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtHeuristicsCacheSetCapacity = dlsym(handle, 'cublasLtHeuristicsCacheSetCapacity')

    global __cublasLtDisableCpuInstructionsSetMask
    __cublasLtDisableCpuInstructionsSetMask = dlsym(RTLD_DEFAULT, 'cublasLtDisableCpuInstructionsSetMask')
    if __cublasLtDisableCpuInstructionsSetMask == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLtDisableCpuInstructionsSetMask = dlsym(handle, 'cublasLtDisableCpuInstructionsSetMask')

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
