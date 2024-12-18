# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_cufft_dso_version_suffix

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

cdef bint __py_cufft_init = False
cdef void* __cuDriverGetVersion = NULL

cdef void* __cufftPlan1d = NULL
cdef void* __cufftPlan2d = NULL
cdef void* __cufftPlan3d = NULL
cdef void* __cufftPlanMany = NULL
cdef void* __cufftMakePlan1d = NULL
cdef void* __cufftMakePlan2d = NULL
cdef void* __cufftMakePlan3d = NULL
cdef void* __cufftMakePlanMany = NULL
cdef void* __cufftMakePlanMany64 = NULL
cdef void* __cufftGetSizeMany64 = NULL
cdef void* __cufftEstimate1d = NULL
cdef void* __cufftEstimate2d = NULL
cdef void* __cufftEstimate3d = NULL
cdef void* __cufftEstimateMany = NULL
cdef void* __cufftCreate = NULL
cdef void* __cufftGetSize1d = NULL
cdef void* __cufftGetSize2d = NULL
cdef void* __cufftGetSize3d = NULL
cdef void* __cufftGetSizeMany = NULL
cdef void* __cufftGetSize = NULL
cdef void* __cufftSetWorkArea = NULL
cdef void* __cufftSetAutoAllocation = NULL
cdef void* __cufftExecC2C = NULL
cdef void* __cufftExecR2C = NULL
cdef void* __cufftExecC2R = NULL
cdef void* __cufftExecZ2Z = NULL
cdef void* __cufftExecD2Z = NULL
cdef void* __cufftExecZ2D = NULL
cdef void* __cufftSetStream = NULL
cdef void* __cufftDestroy = NULL
cdef void* __cufftGetVersion = NULL
cdef void* __cufftGetProperty = NULL
cdef void* __cufftXtSetGPUs = NULL
cdef void* __cufftXtMalloc = NULL
cdef void* __cufftXtMemcpy = NULL
cdef void* __cufftXtFree = NULL
cdef void* __cufftXtSetWorkArea = NULL
cdef void* __cufftXtExecDescriptorC2C = NULL
cdef void* __cufftXtExecDescriptorR2C = NULL
cdef void* __cufftXtExecDescriptorC2R = NULL
cdef void* __cufftXtExecDescriptorZ2Z = NULL
cdef void* __cufftXtExecDescriptorD2Z = NULL
cdef void* __cufftXtExecDescriptorZ2D = NULL
cdef void* __cufftXtQueryPlan = NULL
cdef void* __cufftXtClearCallback = NULL
cdef void* __cufftXtSetCallbackSharedSize = NULL
cdef void* __cufftXtMakePlanMany = NULL
cdef void* __cufftXtGetSizeMany = NULL
cdef void* __cufftXtExec = NULL
cdef void* __cufftXtExecDescriptor = NULL
cdef void* __cufftXtSetWorkAreaPolicy = NULL
cdef void* __cufftXtSetJITCallback = NULL
cdef void* __cufftXtSetSubformatDefault = NULL
cdef void* __cufftSetPlanPropertyInt64 = NULL
cdef void* __cufftGetPlanPropertyInt64 = NULL
cdef void* __cufftResetPlanProperty = NULL


cdef void* load_library(const int driver_ver) except* with gil:
    cdef void* handle
    for suffix in get_cufft_dso_version_suffix(driver_ver):
        so_name = "libcufft.so" + (f".{suffix}" if suffix else suffix)
        handle = dlopen(so_name.encode(), RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            break
    else:
        err_msg = dlerror()
        raise RuntimeError(f'Failed to dlopen libcufft ({err_msg.decode()})')
    return handle


cdef int _check_or_init_cufft() except -1 nogil:
    global __py_cufft_init
    if __py_cufft_init:
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
    global __cufftPlan1d
    __cufftPlan1d = dlsym(RTLD_DEFAULT, 'cufftPlan1d')
    if __cufftPlan1d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftPlan1d = dlsym(handle, 'cufftPlan1d')

    global __cufftPlan2d
    __cufftPlan2d = dlsym(RTLD_DEFAULT, 'cufftPlan2d')
    if __cufftPlan2d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftPlan2d = dlsym(handle, 'cufftPlan2d')

    global __cufftPlan3d
    __cufftPlan3d = dlsym(RTLD_DEFAULT, 'cufftPlan3d')
    if __cufftPlan3d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftPlan3d = dlsym(handle, 'cufftPlan3d')

    global __cufftPlanMany
    __cufftPlanMany = dlsym(RTLD_DEFAULT, 'cufftPlanMany')
    if __cufftPlanMany == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftPlanMany = dlsym(handle, 'cufftPlanMany')

    global __cufftMakePlan1d
    __cufftMakePlan1d = dlsym(RTLD_DEFAULT, 'cufftMakePlan1d')
    if __cufftMakePlan1d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftMakePlan1d = dlsym(handle, 'cufftMakePlan1d')

    global __cufftMakePlan2d
    __cufftMakePlan2d = dlsym(RTLD_DEFAULT, 'cufftMakePlan2d')
    if __cufftMakePlan2d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftMakePlan2d = dlsym(handle, 'cufftMakePlan2d')

    global __cufftMakePlan3d
    __cufftMakePlan3d = dlsym(RTLD_DEFAULT, 'cufftMakePlan3d')
    if __cufftMakePlan3d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftMakePlan3d = dlsym(handle, 'cufftMakePlan3d')

    global __cufftMakePlanMany
    __cufftMakePlanMany = dlsym(RTLD_DEFAULT, 'cufftMakePlanMany')
    if __cufftMakePlanMany == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftMakePlanMany = dlsym(handle, 'cufftMakePlanMany')

    global __cufftMakePlanMany64
    __cufftMakePlanMany64 = dlsym(RTLD_DEFAULT, 'cufftMakePlanMany64')
    if __cufftMakePlanMany64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftMakePlanMany64 = dlsym(handle, 'cufftMakePlanMany64')

    global __cufftGetSizeMany64
    __cufftGetSizeMany64 = dlsym(RTLD_DEFAULT, 'cufftGetSizeMany64')
    if __cufftGetSizeMany64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftGetSizeMany64 = dlsym(handle, 'cufftGetSizeMany64')

    global __cufftEstimate1d
    __cufftEstimate1d = dlsym(RTLD_DEFAULT, 'cufftEstimate1d')
    if __cufftEstimate1d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftEstimate1d = dlsym(handle, 'cufftEstimate1d')

    global __cufftEstimate2d
    __cufftEstimate2d = dlsym(RTLD_DEFAULT, 'cufftEstimate2d')
    if __cufftEstimate2d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftEstimate2d = dlsym(handle, 'cufftEstimate2d')

    global __cufftEstimate3d
    __cufftEstimate3d = dlsym(RTLD_DEFAULT, 'cufftEstimate3d')
    if __cufftEstimate3d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftEstimate3d = dlsym(handle, 'cufftEstimate3d')

    global __cufftEstimateMany
    __cufftEstimateMany = dlsym(RTLD_DEFAULT, 'cufftEstimateMany')
    if __cufftEstimateMany == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftEstimateMany = dlsym(handle, 'cufftEstimateMany')

    global __cufftCreate
    __cufftCreate = dlsym(RTLD_DEFAULT, 'cufftCreate')
    if __cufftCreate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftCreate = dlsym(handle, 'cufftCreate')

    global __cufftGetSize1d
    __cufftGetSize1d = dlsym(RTLD_DEFAULT, 'cufftGetSize1d')
    if __cufftGetSize1d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftGetSize1d = dlsym(handle, 'cufftGetSize1d')

    global __cufftGetSize2d
    __cufftGetSize2d = dlsym(RTLD_DEFAULT, 'cufftGetSize2d')
    if __cufftGetSize2d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftGetSize2d = dlsym(handle, 'cufftGetSize2d')

    global __cufftGetSize3d
    __cufftGetSize3d = dlsym(RTLD_DEFAULT, 'cufftGetSize3d')
    if __cufftGetSize3d == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftGetSize3d = dlsym(handle, 'cufftGetSize3d')

    global __cufftGetSizeMany
    __cufftGetSizeMany = dlsym(RTLD_DEFAULT, 'cufftGetSizeMany')
    if __cufftGetSizeMany == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftGetSizeMany = dlsym(handle, 'cufftGetSizeMany')

    global __cufftGetSize
    __cufftGetSize = dlsym(RTLD_DEFAULT, 'cufftGetSize')
    if __cufftGetSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftGetSize = dlsym(handle, 'cufftGetSize')

    global __cufftSetWorkArea
    __cufftSetWorkArea = dlsym(RTLD_DEFAULT, 'cufftSetWorkArea')
    if __cufftSetWorkArea == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftSetWorkArea = dlsym(handle, 'cufftSetWorkArea')

    global __cufftSetAutoAllocation
    __cufftSetAutoAllocation = dlsym(RTLD_DEFAULT, 'cufftSetAutoAllocation')
    if __cufftSetAutoAllocation == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftSetAutoAllocation = dlsym(handle, 'cufftSetAutoAllocation')

    global __cufftExecC2C
    __cufftExecC2C = dlsym(RTLD_DEFAULT, 'cufftExecC2C')
    if __cufftExecC2C == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftExecC2C = dlsym(handle, 'cufftExecC2C')

    global __cufftExecR2C
    __cufftExecR2C = dlsym(RTLD_DEFAULT, 'cufftExecR2C')
    if __cufftExecR2C == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftExecR2C = dlsym(handle, 'cufftExecR2C')

    global __cufftExecC2R
    __cufftExecC2R = dlsym(RTLD_DEFAULT, 'cufftExecC2R')
    if __cufftExecC2R == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftExecC2R = dlsym(handle, 'cufftExecC2R')

    global __cufftExecZ2Z
    __cufftExecZ2Z = dlsym(RTLD_DEFAULT, 'cufftExecZ2Z')
    if __cufftExecZ2Z == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftExecZ2Z = dlsym(handle, 'cufftExecZ2Z')

    global __cufftExecD2Z
    __cufftExecD2Z = dlsym(RTLD_DEFAULT, 'cufftExecD2Z')
    if __cufftExecD2Z == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftExecD2Z = dlsym(handle, 'cufftExecD2Z')

    global __cufftExecZ2D
    __cufftExecZ2D = dlsym(RTLD_DEFAULT, 'cufftExecZ2D')
    if __cufftExecZ2D == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftExecZ2D = dlsym(handle, 'cufftExecZ2D')

    global __cufftSetStream
    __cufftSetStream = dlsym(RTLD_DEFAULT, 'cufftSetStream')
    if __cufftSetStream == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftSetStream = dlsym(handle, 'cufftSetStream')

    global __cufftDestroy
    __cufftDestroy = dlsym(RTLD_DEFAULT, 'cufftDestroy')
    if __cufftDestroy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftDestroy = dlsym(handle, 'cufftDestroy')

    global __cufftGetVersion
    __cufftGetVersion = dlsym(RTLD_DEFAULT, 'cufftGetVersion')
    if __cufftGetVersion == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftGetVersion = dlsym(handle, 'cufftGetVersion')

    global __cufftGetProperty
    __cufftGetProperty = dlsym(RTLD_DEFAULT, 'cufftGetProperty')
    if __cufftGetProperty == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftGetProperty = dlsym(handle, 'cufftGetProperty')

    global __cufftXtSetGPUs
    __cufftXtSetGPUs = dlsym(RTLD_DEFAULT, 'cufftXtSetGPUs')
    if __cufftXtSetGPUs == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtSetGPUs = dlsym(handle, 'cufftXtSetGPUs')

    global __cufftXtMalloc
    __cufftXtMalloc = dlsym(RTLD_DEFAULT, 'cufftXtMalloc')
    if __cufftXtMalloc == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtMalloc = dlsym(handle, 'cufftXtMalloc')

    global __cufftXtMemcpy
    __cufftXtMemcpy = dlsym(RTLD_DEFAULT, 'cufftXtMemcpy')
    if __cufftXtMemcpy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtMemcpy = dlsym(handle, 'cufftXtMemcpy')

    global __cufftXtFree
    __cufftXtFree = dlsym(RTLD_DEFAULT, 'cufftXtFree')
    if __cufftXtFree == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtFree = dlsym(handle, 'cufftXtFree')

    global __cufftXtSetWorkArea
    __cufftXtSetWorkArea = dlsym(RTLD_DEFAULT, 'cufftXtSetWorkArea')
    if __cufftXtSetWorkArea == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtSetWorkArea = dlsym(handle, 'cufftXtSetWorkArea')

    global __cufftXtExecDescriptorC2C
    __cufftXtExecDescriptorC2C = dlsym(RTLD_DEFAULT, 'cufftXtExecDescriptorC2C')
    if __cufftXtExecDescriptorC2C == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtExecDescriptorC2C = dlsym(handle, 'cufftXtExecDescriptorC2C')

    global __cufftXtExecDescriptorR2C
    __cufftXtExecDescriptorR2C = dlsym(RTLD_DEFAULT, 'cufftXtExecDescriptorR2C')
    if __cufftXtExecDescriptorR2C == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtExecDescriptorR2C = dlsym(handle, 'cufftXtExecDescriptorR2C')

    global __cufftXtExecDescriptorC2R
    __cufftXtExecDescriptorC2R = dlsym(RTLD_DEFAULT, 'cufftXtExecDescriptorC2R')
    if __cufftXtExecDescriptorC2R == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtExecDescriptorC2R = dlsym(handle, 'cufftXtExecDescriptorC2R')

    global __cufftXtExecDescriptorZ2Z
    __cufftXtExecDescriptorZ2Z = dlsym(RTLD_DEFAULT, 'cufftXtExecDescriptorZ2Z')
    if __cufftXtExecDescriptorZ2Z == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtExecDescriptorZ2Z = dlsym(handle, 'cufftXtExecDescriptorZ2Z')

    global __cufftXtExecDescriptorD2Z
    __cufftXtExecDescriptorD2Z = dlsym(RTLD_DEFAULT, 'cufftXtExecDescriptorD2Z')
    if __cufftXtExecDescriptorD2Z == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtExecDescriptorD2Z = dlsym(handle, 'cufftXtExecDescriptorD2Z')

    global __cufftXtExecDescriptorZ2D
    __cufftXtExecDescriptorZ2D = dlsym(RTLD_DEFAULT, 'cufftXtExecDescriptorZ2D')
    if __cufftXtExecDescriptorZ2D == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtExecDescriptorZ2D = dlsym(handle, 'cufftXtExecDescriptorZ2D')

    global __cufftXtQueryPlan
    __cufftXtQueryPlan = dlsym(RTLD_DEFAULT, 'cufftXtQueryPlan')
    if __cufftXtQueryPlan == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtQueryPlan = dlsym(handle, 'cufftXtQueryPlan')

    global __cufftXtClearCallback
    __cufftXtClearCallback = dlsym(RTLD_DEFAULT, 'cufftXtClearCallback')
    if __cufftXtClearCallback == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtClearCallback = dlsym(handle, 'cufftXtClearCallback')

    global __cufftXtSetCallbackSharedSize
    __cufftXtSetCallbackSharedSize = dlsym(RTLD_DEFAULT, 'cufftXtSetCallbackSharedSize')
    if __cufftXtSetCallbackSharedSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtSetCallbackSharedSize = dlsym(handle, 'cufftXtSetCallbackSharedSize')

    global __cufftXtMakePlanMany
    __cufftXtMakePlanMany = dlsym(RTLD_DEFAULT, 'cufftXtMakePlanMany')
    if __cufftXtMakePlanMany == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtMakePlanMany = dlsym(handle, 'cufftXtMakePlanMany')

    global __cufftXtGetSizeMany
    __cufftXtGetSizeMany = dlsym(RTLD_DEFAULT, 'cufftXtGetSizeMany')
    if __cufftXtGetSizeMany == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtGetSizeMany = dlsym(handle, 'cufftXtGetSizeMany')

    global __cufftXtExec
    __cufftXtExec = dlsym(RTLD_DEFAULT, 'cufftXtExec')
    if __cufftXtExec == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtExec = dlsym(handle, 'cufftXtExec')

    global __cufftXtExecDescriptor
    __cufftXtExecDescriptor = dlsym(RTLD_DEFAULT, 'cufftXtExecDescriptor')
    if __cufftXtExecDescriptor == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtExecDescriptor = dlsym(handle, 'cufftXtExecDescriptor')

    global __cufftXtSetWorkAreaPolicy
    __cufftXtSetWorkAreaPolicy = dlsym(RTLD_DEFAULT, 'cufftXtSetWorkAreaPolicy')
    if __cufftXtSetWorkAreaPolicy == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtSetWorkAreaPolicy = dlsym(handle, 'cufftXtSetWorkAreaPolicy')

    global __cufftXtSetJITCallback
    __cufftXtSetJITCallback = dlsym(RTLD_DEFAULT, 'cufftXtSetJITCallback')
    if __cufftXtSetJITCallback == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtSetJITCallback = dlsym(handle, 'cufftXtSetJITCallback')

    global __cufftXtSetSubformatDefault
    __cufftXtSetSubformatDefault = dlsym(RTLD_DEFAULT, 'cufftXtSetSubformatDefault')
    if __cufftXtSetSubformatDefault == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftXtSetSubformatDefault = dlsym(handle, 'cufftXtSetSubformatDefault')

    global __cufftSetPlanPropertyInt64
    __cufftSetPlanPropertyInt64 = dlsym(RTLD_DEFAULT, 'cufftSetPlanPropertyInt64')
    if __cufftSetPlanPropertyInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftSetPlanPropertyInt64 = dlsym(handle, 'cufftSetPlanPropertyInt64')

    global __cufftGetPlanPropertyInt64
    __cufftGetPlanPropertyInt64 = dlsym(RTLD_DEFAULT, 'cufftGetPlanPropertyInt64')
    if __cufftGetPlanPropertyInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftGetPlanPropertyInt64 = dlsym(handle, 'cufftGetPlanPropertyInt64')

    global __cufftResetPlanProperty
    __cufftResetPlanProperty = dlsym(RTLD_DEFAULT, 'cufftResetPlanProperty')
    if __cufftResetPlanProperty == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftResetPlanProperty = dlsym(handle, 'cufftResetPlanProperty')

    __py_cufft_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cufft()
    cdef dict data = {}

    global __cufftPlan1d
    data["__cufftPlan1d"] = <intptr_t>__cufftPlan1d

    global __cufftPlan2d
    data["__cufftPlan2d"] = <intptr_t>__cufftPlan2d

    global __cufftPlan3d
    data["__cufftPlan3d"] = <intptr_t>__cufftPlan3d

    global __cufftPlanMany
    data["__cufftPlanMany"] = <intptr_t>__cufftPlanMany

    global __cufftMakePlan1d
    data["__cufftMakePlan1d"] = <intptr_t>__cufftMakePlan1d

    global __cufftMakePlan2d
    data["__cufftMakePlan2d"] = <intptr_t>__cufftMakePlan2d

    global __cufftMakePlan3d
    data["__cufftMakePlan3d"] = <intptr_t>__cufftMakePlan3d

    global __cufftMakePlanMany
    data["__cufftMakePlanMany"] = <intptr_t>__cufftMakePlanMany

    global __cufftMakePlanMany64
    data["__cufftMakePlanMany64"] = <intptr_t>__cufftMakePlanMany64

    global __cufftGetSizeMany64
    data["__cufftGetSizeMany64"] = <intptr_t>__cufftGetSizeMany64

    global __cufftEstimate1d
    data["__cufftEstimate1d"] = <intptr_t>__cufftEstimate1d

    global __cufftEstimate2d
    data["__cufftEstimate2d"] = <intptr_t>__cufftEstimate2d

    global __cufftEstimate3d
    data["__cufftEstimate3d"] = <intptr_t>__cufftEstimate3d

    global __cufftEstimateMany
    data["__cufftEstimateMany"] = <intptr_t>__cufftEstimateMany

    global __cufftCreate
    data["__cufftCreate"] = <intptr_t>__cufftCreate

    global __cufftGetSize1d
    data["__cufftGetSize1d"] = <intptr_t>__cufftGetSize1d

    global __cufftGetSize2d
    data["__cufftGetSize2d"] = <intptr_t>__cufftGetSize2d

    global __cufftGetSize3d
    data["__cufftGetSize3d"] = <intptr_t>__cufftGetSize3d

    global __cufftGetSizeMany
    data["__cufftGetSizeMany"] = <intptr_t>__cufftGetSizeMany

    global __cufftGetSize
    data["__cufftGetSize"] = <intptr_t>__cufftGetSize

    global __cufftSetWorkArea
    data["__cufftSetWorkArea"] = <intptr_t>__cufftSetWorkArea

    global __cufftSetAutoAllocation
    data["__cufftSetAutoAllocation"] = <intptr_t>__cufftSetAutoAllocation

    global __cufftExecC2C
    data["__cufftExecC2C"] = <intptr_t>__cufftExecC2C

    global __cufftExecR2C
    data["__cufftExecR2C"] = <intptr_t>__cufftExecR2C

    global __cufftExecC2R
    data["__cufftExecC2R"] = <intptr_t>__cufftExecC2R

    global __cufftExecZ2Z
    data["__cufftExecZ2Z"] = <intptr_t>__cufftExecZ2Z

    global __cufftExecD2Z
    data["__cufftExecD2Z"] = <intptr_t>__cufftExecD2Z

    global __cufftExecZ2D
    data["__cufftExecZ2D"] = <intptr_t>__cufftExecZ2D

    global __cufftSetStream
    data["__cufftSetStream"] = <intptr_t>__cufftSetStream

    global __cufftDestroy
    data["__cufftDestroy"] = <intptr_t>__cufftDestroy

    global __cufftGetVersion
    data["__cufftGetVersion"] = <intptr_t>__cufftGetVersion

    global __cufftGetProperty
    data["__cufftGetProperty"] = <intptr_t>__cufftGetProperty

    global __cufftXtSetGPUs
    data["__cufftXtSetGPUs"] = <intptr_t>__cufftXtSetGPUs

    global __cufftXtMalloc
    data["__cufftXtMalloc"] = <intptr_t>__cufftXtMalloc

    global __cufftXtMemcpy
    data["__cufftXtMemcpy"] = <intptr_t>__cufftXtMemcpy

    global __cufftXtFree
    data["__cufftXtFree"] = <intptr_t>__cufftXtFree

    global __cufftXtSetWorkArea
    data["__cufftXtSetWorkArea"] = <intptr_t>__cufftXtSetWorkArea

    global __cufftXtExecDescriptorC2C
    data["__cufftXtExecDescriptorC2C"] = <intptr_t>__cufftXtExecDescriptorC2C

    global __cufftXtExecDescriptorR2C
    data["__cufftXtExecDescriptorR2C"] = <intptr_t>__cufftXtExecDescriptorR2C

    global __cufftXtExecDescriptorC2R
    data["__cufftXtExecDescriptorC2R"] = <intptr_t>__cufftXtExecDescriptorC2R

    global __cufftXtExecDescriptorZ2Z
    data["__cufftXtExecDescriptorZ2Z"] = <intptr_t>__cufftXtExecDescriptorZ2Z

    global __cufftXtExecDescriptorD2Z
    data["__cufftXtExecDescriptorD2Z"] = <intptr_t>__cufftXtExecDescriptorD2Z

    global __cufftXtExecDescriptorZ2D
    data["__cufftXtExecDescriptorZ2D"] = <intptr_t>__cufftXtExecDescriptorZ2D

    global __cufftXtQueryPlan
    data["__cufftXtQueryPlan"] = <intptr_t>__cufftXtQueryPlan

    global __cufftXtClearCallback
    data["__cufftXtClearCallback"] = <intptr_t>__cufftXtClearCallback

    global __cufftXtSetCallbackSharedSize
    data["__cufftXtSetCallbackSharedSize"] = <intptr_t>__cufftXtSetCallbackSharedSize

    global __cufftXtMakePlanMany
    data["__cufftXtMakePlanMany"] = <intptr_t>__cufftXtMakePlanMany

    global __cufftXtGetSizeMany
    data["__cufftXtGetSizeMany"] = <intptr_t>__cufftXtGetSizeMany

    global __cufftXtExec
    data["__cufftXtExec"] = <intptr_t>__cufftXtExec

    global __cufftXtExecDescriptor
    data["__cufftXtExecDescriptor"] = <intptr_t>__cufftXtExecDescriptor

    global __cufftXtSetWorkAreaPolicy
    data["__cufftXtSetWorkAreaPolicy"] = <intptr_t>__cufftXtSetWorkAreaPolicy

    global __cufftXtSetJITCallback
    data["__cufftXtSetJITCallback"] = <intptr_t>__cufftXtSetJITCallback

    global __cufftXtSetSubformatDefault
    data["__cufftXtSetSubformatDefault"] = <intptr_t>__cufftXtSetSubformatDefault

    global __cufftSetPlanPropertyInt64
    data["__cufftSetPlanPropertyInt64"] = <intptr_t>__cufftSetPlanPropertyInt64

    global __cufftGetPlanPropertyInt64
    data["__cufftGetPlanPropertyInt64"] = <intptr_t>__cufftGetPlanPropertyInt64

    global __cufftResetPlanProperty
    data["__cufftResetPlanProperty"] = <intptr_t>__cufftResetPlanProperty

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

cdef cufftResult _cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) except* nogil:
    global __cufftPlan1d
    _check_or_init_cufft()
    if __cufftPlan1d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftPlan1d is not found")
    return (<cufftResult (*)(cufftHandle*, int, cufftType, int) nogil>__cufftPlan1d)(
        plan, nx, type, batch)


cdef cufftResult _cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) except* nogil:
    global __cufftPlan2d
    _check_or_init_cufft()
    if __cufftPlan2d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftPlan2d is not found")
    return (<cufftResult (*)(cufftHandle*, int, int, cufftType) nogil>__cufftPlan2d)(
        plan, nx, ny, type)


cdef cufftResult _cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) except* nogil:
    global __cufftPlan3d
    _check_or_init_cufft()
    if __cufftPlan3d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftPlan3d is not found")
    return (<cufftResult (*)(cufftHandle*, int, int, int, cufftType) nogil>__cufftPlan3d)(
        plan, nx, ny, nz, type)


cdef cufftResult _cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch) except* nogil:
    global __cufftPlanMany
    _check_or_init_cufft()
    if __cufftPlanMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftPlanMany is not found")
    return (<cufftResult (*)(cufftHandle*, int, int*, int*, int, int, int*, int, int, cufftType, int) nogil>__cufftPlanMany)(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)


cdef cufftResult _cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t* workSize) except* nogil:
    global __cufftMakePlan1d
    _check_or_init_cufft()
    if __cufftMakePlan1d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMakePlan1d is not found")
    return (<cufftResult (*)(cufftHandle, int, cufftType, int, size_t*) nogil>__cufftMakePlan1d)(
        plan, nx, type, batch, workSize)


cdef cufftResult _cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize) except* nogil:
    global __cufftMakePlan2d
    _check_or_init_cufft()
    if __cufftMakePlan2d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMakePlan2d is not found")
    return (<cufftResult (*)(cufftHandle, int, int, cufftType, size_t*) nogil>__cufftMakePlan2d)(
        plan, nx, ny, type, workSize)


cdef cufftResult _cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t* workSize) except* nogil:
    global __cufftMakePlan3d
    _check_or_init_cufft()
    if __cufftMakePlan3d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMakePlan3d is not found")
    return (<cufftResult (*)(cufftHandle, int, int, int, cufftType, size_t*) nogil>__cufftMakePlan3d)(
        plan, nx, ny, nz, type, workSize)


cdef cufftResult _cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except* nogil:
    global __cufftMakePlanMany
    _check_or_init_cufft()
    if __cufftMakePlanMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMakePlanMany is not found")
    return (<cufftResult (*)(cufftHandle, int, int*, int*, int, int, int*, int, int, cufftType, int, size_t*) nogil>__cufftMakePlanMany)(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult _cufftMakePlanMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except* nogil:
    global __cufftMakePlanMany64
    _check_or_init_cufft()
    if __cufftMakePlanMany64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMakePlanMany64 is not found")
    return (<cufftResult (*)(cufftHandle, int, long long int*, long long int*, long long int, long long int, long long int*, long long int, long long int, cufftType, long long int, size_t*) nogil>__cufftMakePlanMany64)(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult _cufftGetSizeMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except* nogil:
    global __cufftGetSizeMany64
    _check_or_init_cufft()
    if __cufftGetSizeMany64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSizeMany64 is not found")
    return (<cufftResult (*)(cufftHandle, int, long long int*, long long int*, long long int, long long int, long long int*, long long int, long long int, cufftType, long long int, size_t*) nogil>__cufftGetSizeMany64)(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult _cufftEstimate1d(int nx, cufftType type, int batch, size_t* workSize) except* nogil:
    global __cufftEstimate1d
    _check_or_init_cufft()
    if __cufftEstimate1d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftEstimate1d is not found")
    return (<cufftResult (*)(int, cufftType, int, size_t*) nogil>__cufftEstimate1d)(
        nx, type, batch, workSize)


cdef cufftResult _cufftEstimate2d(int nx, int ny, cufftType type, size_t* workSize) except* nogil:
    global __cufftEstimate2d
    _check_or_init_cufft()
    if __cufftEstimate2d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftEstimate2d is not found")
    return (<cufftResult (*)(int, int, cufftType, size_t*) nogil>__cufftEstimate2d)(
        nx, ny, type, workSize)


cdef cufftResult _cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t* workSize) except* nogil:
    global __cufftEstimate3d
    _check_or_init_cufft()
    if __cufftEstimate3d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftEstimate3d is not found")
    return (<cufftResult (*)(int, int, int, cufftType, size_t*) nogil>__cufftEstimate3d)(
        nx, ny, nz, type, workSize)


cdef cufftResult _cufftEstimateMany(int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except* nogil:
    global __cufftEstimateMany
    _check_or_init_cufft()
    if __cufftEstimateMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftEstimateMany is not found")
    return (<cufftResult (*)(int, int*, int*, int, int, int*, int, int, cufftType, int, size_t*) nogil>__cufftEstimateMany)(
        rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult _cufftCreate(cufftHandle* handle) except* nogil:
    global __cufftCreate
    _check_or_init_cufft()
    if __cufftCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftCreate is not found")
    return (<cufftResult (*)(cufftHandle*) nogil>__cufftCreate)(
        handle)


cdef cufftResult _cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t* workSize) except* nogil:
    global __cufftGetSize1d
    _check_or_init_cufft()
    if __cufftGetSize1d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSize1d is not found")
    return (<cufftResult (*)(cufftHandle, int, cufftType, int, size_t*) nogil>__cufftGetSize1d)(
        handle, nx, type, batch, workSize)


cdef cufftResult _cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t* workSize) except* nogil:
    global __cufftGetSize2d
    _check_or_init_cufft()
    if __cufftGetSize2d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSize2d is not found")
    return (<cufftResult (*)(cufftHandle, int, int, cufftType, size_t*) nogil>__cufftGetSize2d)(
        handle, nx, ny, type, workSize)


cdef cufftResult _cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t* workSize) except* nogil:
    global __cufftGetSize3d
    _check_or_init_cufft()
    if __cufftGetSize3d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSize3d is not found")
    return (<cufftResult (*)(cufftHandle, int, int, int, cufftType, size_t*) nogil>__cufftGetSize3d)(
        handle, nx, ny, nz, type, workSize)


cdef cufftResult _cufftGetSizeMany(cufftHandle handle, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workArea) except* nogil:
    global __cufftGetSizeMany
    _check_or_init_cufft()
    if __cufftGetSizeMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSizeMany is not found")
    return (<cufftResult (*)(cufftHandle, int, int*, int*, int, int, int*, int, int, cufftType, int, size_t*) nogil>__cufftGetSizeMany)(
        handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workArea)


cdef cufftResult _cufftGetSize(cufftHandle handle, size_t* workSize) except* nogil:
    global __cufftGetSize
    _check_or_init_cufft()
    if __cufftGetSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSize is not found")
    return (<cufftResult (*)(cufftHandle, size_t*) nogil>__cufftGetSize)(
        handle, workSize)


cdef cufftResult _cufftSetWorkArea(cufftHandle plan, void* workArea) except* nogil:
    global __cufftSetWorkArea
    _check_or_init_cufft()
    if __cufftSetWorkArea == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftSetWorkArea is not found")
    return (<cufftResult (*)(cufftHandle, void*) nogil>__cufftSetWorkArea)(
        plan, workArea)


cdef cufftResult _cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) except* nogil:
    global __cufftSetAutoAllocation
    _check_or_init_cufft()
    if __cufftSetAutoAllocation == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftSetAutoAllocation is not found")
    return (<cufftResult (*)(cufftHandle, int) nogil>__cufftSetAutoAllocation)(
        plan, autoAllocate)


cdef cufftResult _cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) except* nogil:
    global __cufftExecC2C
    _check_or_init_cufft()
    if __cufftExecC2C == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecC2C is not found")
    return (<cufftResult (*)(cufftHandle, cufftComplex*, cufftComplex*, int) nogil>__cufftExecC2C)(
        plan, idata, odata, direction)


cdef cufftResult _cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata) except* nogil:
    global __cufftExecR2C
    _check_or_init_cufft()
    if __cufftExecR2C == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecR2C is not found")
    return (<cufftResult (*)(cufftHandle, cufftReal*, cufftComplex*) nogil>__cufftExecR2C)(
        plan, idata, odata)


cdef cufftResult _cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) except* nogil:
    global __cufftExecC2R
    _check_or_init_cufft()
    if __cufftExecC2R == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecC2R is not found")
    return (<cufftResult (*)(cufftHandle, cufftComplex*, cufftReal*) nogil>__cufftExecC2R)(
        plan, idata, odata)


cdef cufftResult _cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction) except* nogil:
    global __cufftExecZ2Z
    _check_or_init_cufft()
    if __cufftExecZ2Z == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecZ2Z is not found")
    return (<cufftResult (*)(cufftHandle, cufftDoubleComplex*, cufftDoubleComplex*, int) nogil>__cufftExecZ2Z)(
        plan, idata, odata, direction)


cdef cufftResult _cufftExecD2Z(cufftHandle plan, cufftDoubleReal* idata, cufftDoubleComplex* odata) except* nogil:
    global __cufftExecD2Z
    _check_or_init_cufft()
    if __cufftExecD2Z == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecD2Z is not found")
    return (<cufftResult (*)(cufftHandle, cufftDoubleReal*, cufftDoubleComplex*) nogil>__cufftExecD2Z)(
        plan, idata, odata)


cdef cufftResult _cufftExecZ2D(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleReal* odata) except* nogil:
    global __cufftExecZ2D
    _check_or_init_cufft()
    if __cufftExecZ2D == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecZ2D is not found")
    return (<cufftResult (*)(cufftHandle, cufftDoubleComplex*, cufftDoubleReal*) nogil>__cufftExecZ2D)(
        plan, idata, odata)


cdef cufftResult _cufftSetStream(cufftHandle plan, cudaStream_t stream) except* nogil:
    global __cufftSetStream
    _check_or_init_cufft()
    if __cufftSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftSetStream is not found")
    return (<cufftResult (*)(cufftHandle, cudaStream_t) nogil>__cufftSetStream)(
        plan, stream)


cdef cufftResult _cufftDestroy(cufftHandle plan) except* nogil:
    global __cufftDestroy
    _check_or_init_cufft()
    if __cufftDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftDestroy is not found")
    return (<cufftResult (*)(cufftHandle) nogil>__cufftDestroy)(
        plan)


cdef cufftResult _cufftGetVersion(int* version) except* nogil:
    global __cufftGetVersion
    _check_or_init_cufft()
    if __cufftGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetVersion is not found")
    return (<cufftResult (*)(int*) nogil>__cufftGetVersion)(
        version)


cdef cufftResult _cufftGetProperty(libraryPropertyType type, int* value) except* nogil:
    global __cufftGetProperty
    _check_or_init_cufft()
    if __cufftGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetProperty is not found")
    return (<cufftResult (*)(libraryPropertyType, int*) nogil>__cufftGetProperty)(
        type, value)


cdef cufftResult _cufftXtSetGPUs(cufftHandle handle, int nGPUs, int* whichGPUs) except* nogil:
    global __cufftXtSetGPUs
    _check_or_init_cufft()
    if __cufftXtSetGPUs == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetGPUs is not found")
    return (<cufftResult (*)(cufftHandle, int, int*) nogil>__cufftXtSetGPUs)(
        handle, nGPUs, whichGPUs)


cdef cufftResult _cufftXtMalloc(cufftHandle plan, cudaLibXtDesc** descriptor, cufftXtSubFormat format) except* nogil:
    global __cufftXtMalloc
    _check_or_init_cufft()
    if __cufftXtMalloc == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtMalloc is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc**, cufftXtSubFormat) nogil>__cufftXtMalloc)(
        plan, descriptor, format)


cdef cufftResult _cufftXtMemcpy(cufftHandle plan, void* dstPointer, void* srcPointer, cufftXtCopyType type) except* nogil:
    global __cufftXtMemcpy
    _check_or_init_cufft()
    if __cufftXtMemcpy == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtMemcpy is not found")
    return (<cufftResult (*)(cufftHandle, void*, void*, cufftXtCopyType) nogil>__cufftXtMemcpy)(
        plan, dstPointer, srcPointer, type)


cdef cufftResult _cufftXtFree(cudaLibXtDesc* descriptor) except* nogil:
    global __cufftXtFree
    _check_or_init_cufft()
    if __cufftXtFree == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtFree is not found")
    return (<cufftResult (*)(cudaLibXtDesc*) nogil>__cufftXtFree)(
        descriptor)


cdef cufftResult _cufftXtSetWorkArea(cufftHandle plan, void** workArea) except* nogil:
    global __cufftXtSetWorkArea
    _check_or_init_cufft()
    if __cufftXtSetWorkArea == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetWorkArea is not found")
    return (<cufftResult (*)(cufftHandle, void**) nogil>__cufftXtSetWorkArea)(
        plan, workArea)


cdef cufftResult _cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except* nogil:
    global __cufftXtExecDescriptorC2C
    _check_or_init_cufft()
    if __cufftXtExecDescriptorC2C == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorC2C is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*, int) nogil>__cufftXtExecDescriptorC2C)(
        plan, input, output, direction)


cdef cufftResult _cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil:
    global __cufftXtExecDescriptorR2C
    _check_or_init_cufft()
    if __cufftXtExecDescriptorR2C == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorR2C is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*) nogil>__cufftXtExecDescriptorR2C)(
        plan, input, output)


cdef cufftResult _cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil:
    global __cufftXtExecDescriptorC2R
    _check_or_init_cufft()
    if __cufftXtExecDescriptorC2R == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorC2R is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*) nogil>__cufftXtExecDescriptorC2R)(
        plan, input, output)


cdef cufftResult _cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except* nogil:
    global __cufftXtExecDescriptorZ2Z
    _check_or_init_cufft()
    if __cufftXtExecDescriptorZ2Z == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorZ2Z is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*, int) nogil>__cufftXtExecDescriptorZ2Z)(
        plan, input, output, direction)


cdef cufftResult _cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil:
    global __cufftXtExecDescriptorD2Z
    _check_or_init_cufft()
    if __cufftXtExecDescriptorD2Z == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorD2Z is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*) nogil>__cufftXtExecDescriptorD2Z)(
        plan, input, output)


cdef cufftResult _cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil:
    global __cufftXtExecDescriptorZ2D
    _check_or_init_cufft()
    if __cufftXtExecDescriptorZ2D == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorZ2D is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*) nogil>__cufftXtExecDescriptorZ2D)(
        plan, input, output)


cdef cufftResult _cufftXtQueryPlan(cufftHandle plan, void* queryStruct, cufftXtQueryType queryType) except* nogil:
    global __cufftXtQueryPlan
    _check_or_init_cufft()
    if __cufftXtQueryPlan == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtQueryPlan is not found")
    return (<cufftResult (*)(cufftHandle, void*, cufftXtQueryType) nogil>__cufftXtQueryPlan)(
        plan, queryStruct, queryType)


cdef cufftResult _cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType) except* nogil:
    global __cufftXtClearCallback
    _check_or_init_cufft()
    if __cufftXtClearCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtClearCallback is not found")
    return (<cufftResult (*)(cufftHandle, cufftXtCallbackType) nogil>__cufftXtClearCallback)(
        plan, cbType)


cdef cufftResult _cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType, size_t sharedSize) except* nogil:
    global __cufftXtSetCallbackSharedSize
    _check_or_init_cufft()
    if __cufftXtSetCallbackSharedSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetCallbackSharedSize is not found")
    return (<cufftResult (*)(cufftHandle, cufftXtCallbackType, size_t) nogil>__cufftXtSetCallbackSharedSize)(
        plan, cbType, sharedSize)


cdef cufftResult _cufftXtMakePlanMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except* nogil:
    global __cufftXtMakePlanMany
    _check_or_init_cufft()
    if __cufftXtMakePlanMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtMakePlanMany is not found")
    return (<cufftResult (*)(cufftHandle, int, long long int*, long long int*, long long int, long long int, cudaDataType, long long int*, long long int, long long int, cudaDataType, long long int, size_t*, cudaDataType) nogil>__cufftXtMakePlanMany)(
        plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch, workSize, executiontype)


cdef cufftResult _cufftXtGetSizeMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except* nogil:
    global __cufftXtGetSizeMany
    _check_or_init_cufft()
    if __cufftXtGetSizeMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtGetSizeMany is not found")
    return (<cufftResult (*)(cufftHandle, int, long long int*, long long int*, long long int, long long int, cudaDataType, long long int*, long long int, long long int, cudaDataType, long long int, size_t*, cudaDataType) nogil>__cufftXtGetSizeMany)(
        plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch, workSize, executiontype)


cdef cufftResult _cufftXtExec(cufftHandle plan, void* input, void* output, int direction) except* nogil:
    global __cufftXtExec
    _check_or_init_cufft()
    if __cufftXtExec == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExec is not found")
    return (<cufftResult (*)(cufftHandle, void*, void*, int) nogil>__cufftXtExec)(
        plan, input, output, direction)


cdef cufftResult _cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except* nogil:
    global __cufftXtExecDescriptor
    _check_or_init_cufft()
    if __cufftXtExecDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptor is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*, int) nogil>__cufftXtExecDescriptor)(
        plan, input, output, direction)


cdef cufftResult _cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy, size_t* workSize) except* nogil:
    global __cufftXtSetWorkAreaPolicy
    _check_or_init_cufft()
    if __cufftXtSetWorkAreaPolicy == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetWorkAreaPolicy is not found")
    return (<cufftResult (*)(cufftHandle, cufftXtWorkAreaPolicy, size_t*) nogil>__cufftXtSetWorkAreaPolicy)(
        plan, policy, workSize)


cdef cufftResult _cufftXtSetJITCallback(cufftHandle plan, const void* lto_callback_fatbin, size_t lto_callback_fatbin_size, cufftXtCallbackType type, void** caller_info) except* nogil:
    global __cufftXtSetJITCallback
    _check_or_init_cufft()
    if __cufftXtSetJITCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetJITCallback is not found")
    return (<cufftResult (*)(cufftHandle, const void*, size_t, cufftXtCallbackType, void**) nogil>__cufftXtSetJITCallback)(
        plan, lto_callback_fatbin, lto_callback_fatbin_size, type, caller_info)


cdef cufftResult _cufftXtSetSubformatDefault(cufftHandle plan, cufftXtSubFormat subformat_forward, cufftXtSubFormat subformat_inverse) except* nogil:
    global __cufftXtSetSubformatDefault
    _check_or_init_cufft()
    if __cufftXtSetSubformatDefault == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetSubformatDefault is not found")
    return (<cufftResult (*)(cufftHandle, cufftXtSubFormat, cufftXtSubFormat) nogil>__cufftXtSetSubformatDefault)(
        plan, subformat_forward, subformat_inverse)


cdef cufftResult _cufftSetPlanPropertyInt64(cufftHandle plan, cufftProperty property, const long long int inputValueInt) except* nogil:
    global __cufftSetPlanPropertyInt64
    _check_or_init_cufft()
    if __cufftSetPlanPropertyInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftSetPlanPropertyInt64 is not found")
    return (<cufftResult (*)(cufftHandle, cufftProperty, const long long int) nogil>__cufftSetPlanPropertyInt64)(
        plan, property, inputValueInt)


cdef cufftResult _cufftGetPlanPropertyInt64(cufftHandle plan, cufftProperty property, long long int* returnPtrValue) except* nogil:
    global __cufftGetPlanPropertyInt64
    _check_or_init_cufft()
    if __cufftGetPlanPropertyInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetPlanPropertyInt64 is not found")
    return (<cufftResult (*)(cufftHandle, cufftProperty, long long int*) nogil>__cufftGetPlanPropertyInt64)(
        plan, property, returnPtrValue)


cdef cufftResult _cufftResetPlanProperty(cufftHandle plan, cufftProperty property) except* nogil:
    global __cufftResetPlanProperty
    _check_or_init_cufft()
    if __cufftResetPlanProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftResetPlanProperty is not found")
    return (<cufftResult (*)(cufftHandle, cufftProperty) nogil>__cufftResetPlanProperty)(
        plan, property)
