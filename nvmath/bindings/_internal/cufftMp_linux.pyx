# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.2.6 to 11.4.0. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib

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
        RTLD_DEEPBIND

    const void* RTLD_DEFAULT 'RTLD_DEFAULT'


###############################################################################
# Wrapper init
###############################################################################

cdef bint __py_cufftMp_init = False

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
cdef void* __cufftSetPlanPropertyInt64 = NULL
cdef void* __cufftGetPlanPropertyInt64 = NULL
cdef void* __cufftResetPlanProperty = NULL
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
cdef void* __cufftMpAttachComm = NULL
cdef void* __cufftXtSetDistribution = NULL
cdef void* __cufftXtSetSubformatDefault = NULL
cdef void* __cufftMpCreateReshape = NULL
cdef void* __cufftMpAttachReshapeComm = NULL
cdef void* __cufftMpGetReshapeSize = NULL
cdef void* __cufftMpMakeReshape = NULL
cdef void* __cufftMpExecReshapeAsync = NULL
cdef void* __cufftMpDestroyReshape = NULL
cdef void* ____cufftMpMakeReshape_11_4 = NULL


cdef void* load_library() except* with gil:
    # NOTE: libcufftMp.so shares most of the symbol names with libcufft.so. When extracting
    # the symbols below with dlsym, we need to extract from the library handle instead of
    # RTLD_DEFAULT to avoid picking up the wrong function pointers.
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cufftMp")._handle_uint
    return <void*>handle


cdef int _check_or_init_cufftMp() except -1 nogil:
    global __py_cufftMp_init
    if __py_cufftMp_init:
        return 0

    # Load function
    cdef void* handle = NULL
    global __cufftPlan1d
    if __cufftPlan1d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftPlan1d = dlsym(handle, 'cufftPlan1d')

    global __cufftPlan2d
    if __cufftPlan2d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftPlan2d = dlsym(handle, 'cufftPlan2d')

    global __cufftPlan3d
    if __cufftPlan3d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftPlan3d = dlsym(handle, 'cufftPlan3d')

    global __cufftPlanMany
    if __cufftPlanMany == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftPlanMany = dlsym(handle, 'cufftPlanMany')

    global __cufftMakePlan1d
    if __cufftMakePlan1d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMakePlan1d = dlsym(handle, 'cufftMakePlan1d')

    global __cufftMakePlan2d
    if __cufftMakePlan2d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMakePlan2d = dlsym(handle, 'cufftMakePlan2d')

    global __cufftMakePlan3d
    if __cufftMakePlan3d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMakePlan3d = dlsym(handle, 'cufftMakePlan3d')

    global __cufftMakePlanMany
    if __cufftMakePlanMany == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMakePlanMany = dlsym(handle, 'cufftMakePlanMany')

    global __cufftMakePlanMany64
    if __cufftMakePlanMany64 == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMakePlanMany64 = dlsym(handle, 'cufftMakePlanMany64')

    global __cufftGetSizeMany64
    if __cufftGetSizeMany64 == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftGetSizeMany64 = dlsym(handle, 'cufftGetSizeMany64')

    global __cufftEstimate1d
    if __cufftEstimate1d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftEstimate1d = dlsym(handle, 'cufftEstimate1d')

    global __cufftEstimate2d
    if __cufftEstimate2d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftEstimate2d = dlsym(handle, 'cufftEstimate2d')

    global __cufftEstimate3d
    if __cufftEstimate3d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftEstimate3d = dlsym(handle, 'cufftEstimate3d')

    global __cufftEstimateMany
    if __cufftEstimateMany == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftEstimateMany = dlsym(handle, 'cufftEstimateMany')

    global __cufftCreate
    if __cufftCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftCreate = dlsym(handle, 'cufftCreate')

    global __cufftGetSize1d
    if __cufftGetSize1d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftGetSize1d = dlsym(handle, 'cufftGetSize1d')

    global __cufftGetSize2d
    if __cufftGetSize2d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftGetSize2d = dlsym(handle, 'cufftGetSize2d')

    global __cufftGetSize3d
    if __cufftGetSize3d == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftGetSize3d = dlsym(handle, 'cufftGetSize3d')

    global __cufftGetSizeMany
    if __cufftGetSizeMany == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftGetSizeMany = dlsym(handle, 'cufftGetSizeMany')

    global __cufftGetSize
    if __cufftGetSize == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftGetSize = dlsym(handle, 'cufftGetSize')

    global __cufftSetWorkArea
    if __cufftSetWorkArea == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftSetWorkArea = dlsym(handle, 'cufftSetWorkArea')

    global __cufftSetAutoAllocation
    if __cufftSetAutoAllocation == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftSetAutoAllocation = dlsym(handle, 'cufftSetAutoAllocation')

    global __cufftExecC2C
    if __cufftExecC2C == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftExecC2C = dlsym(handle, 'cufftExecC2C')

    global __cufftExecR2C
    if __cufftExecR2C == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftExecR2C = dlsym(handle, 'cufftExecR2C')

    global __cufftExecC2R
    if __cufftExecC2R == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftExecC2R = dlsym(handle, 'cufftExecC2R')

    global __cufftExecZ2Z
    if __cufftExecZ2Z == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftExecZ2Z = dlsym(handle, 'cufftExecZ2Z')

    global __cufftExecD2Z
    if __cufftExecD2Z == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftExecD2Z = dlsym(handle, 'cufftExecD2Z')

    global __cufftExecZ2D
    if __cufftExecZ2D == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftExecZ2D = dlsym(handle, 'cufftExecZ2D')

    global __cufftSetStream
    if __cufftSetStream == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftSetStream = dlsym(handle, 'cufftSetStream')

    global __cufftDestroy
    if __cufftDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftDestroy = dlsym(handle, 'cufftDestroy')

    global __cufftGetVersion
    if __cufftGetVersion == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftGetVersion = dlsym(handle, 'cufftGetVersion')

    global __cufftGetProperty
    if __cufftGetProperty == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftGetProperty = dlsym(handle, 'cufftGetProperty')

    global __cufftSetPlanPropertyInt64
    if __cufftSetPlanPropertyInt64 == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftSetPlanPropertyInt64 = dlsym(handle, 'cufftSetPlanPropertyInt64')

    global __cufftGetPlanPropertyInt64
    if __cufftGetPlanPropertyInt64 == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftGetPlanPropertyInt64 = dlsym(handle, 'cufftGetPlanPropertyInt64')

    global __cufftResetPlanProperty
    if __cufftResetPlanProperty == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftResetPlanProperty = dlsym(handle, 'cufftResetPlanProperty')

    global __cufftXtSetGPUs
    if __cufftXtSetGPUs == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtSetGPUs = dlsym(handle, 'cufftXtSetGPUs')

    global __cufftXtMalloc
    if __cufftXtMalloc == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtMalloc = dlsym(handle, 'cufftXtMalloc')

    global __cufftXtMemcpy
    if __cufftXtMemcpy == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtMemcpy = dlsym(handle, 'cufftXtMemcpy')

    global __cufftXtFree
    if __cufftXtFree == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtFree = dlsym(handle, 'cufftXtFree')

    global __cufftXtSetWorkArea
    if __cufftXtSetWorkArea == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtSetWorkArea = dlsym(handle, 'cufftXtSetWorkArea')

    global __cufftXtExecDescriptorC2C
    if __cufftXtExecDescriptorC2C == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtExecDescriptorC2C = dlsym(handle, 'cufftXtExecDescriptorC2C')

    global __cufftXtExecDescriptorR2C
    if __cufftXtExecDescriptorR2C == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtExecDescriptorR2C = dlsym(handle, 'cufftXtExecDescriptorR2C')

    global __cufftXtExecDescriptorC2R
    if __cufftXtExecDescriptorC2R == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtExecDescriptorC2R = dlsym(handle, 'cufftXtExecDescriptorC2R')

    global __cufftXtExecDescriptorZ2Z
    if __cufftXtExecDescriptorZ2Z == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtExecDescriptorZ2Z = dlsym(handle, 'cufftXtExecDescriptorZ2Z')

    global __cufftXtExecDescriptorD2Z
    if __cufftXtExecDescriptorD2Z == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtExecDescriptorD2Z = dlsym(handle, 'cufftXtExecDescriptorD2Z')

    global __cufftXtExecDescriptorZ2D
    if __cufftXtExecDescriptorZ2D == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtExecDescriptorZ2D = dlsym(handle, 'cufftXtExecDescriptorZ2D')

    global __cufftXtQueryPlan
    if __cufftXtQueryPlan == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtQueryPlan = dlsym(handle, 'cufftXtQueryPlan')

    global __cufftXtClearCallback
    if __cufftXtClearCallback == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtClearCallback = dlsym(handle, 'cufftXtClearCallback')

    global __cufftXtSetCallbackSharedSize
    if __cufftXtSetCallbackSharedSize == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtSetCallbackSharedSize = dlsym(handle, 'cufftXtSetCallbackSharedSize')

    global __cufftXtMakePlanMany
    if __cufftXtMakePlanMany == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtMakePlanMany = dlsym(handle, 'cufftXtMakePlanMany')

    global __cufftXtGetSizeMany
    if __cufftXtGetSizeMany == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtGetSizeMany = dlsym(handle, 'cufftXtGetSizeMany')

    global __cufftXtExec
    if __cufftXtExec == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtExec = dlsym(handle, 'cufftXtExec')

    global __cufftXtExecDescriptor
    if __cufftXtExecDescriptor == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtExecDescriptor = dlsym(handle, 'cufftXtExecDescriptor')

    global __cufftXtSetWorkAreaPolicy
    if __cufftXtSetWorkAreaPolicy == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtSetWorkAreaPolicy = dlsym(handle, 'cufftXtSetWorkAreaPolicy')

    global __cufftMpAttachComm
    if __cufftMpAttachComm == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMpAttachComm = dlsym(handle, 'cufftMpAttachComm')

    global __cufftXtSetDistribution
    if __cufftXtSetDistribution == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtSetDistribution = dlsym(handle, 'cufftXtSetDistribution')

    global __cufftXtSetSubformatDefault
    if __cufftXtSetSubformatDefault == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftXtSetSubformatDefault = dlsym(handle, 'cufftXtSetSubformatDefault')

    global __cufftMpCreateReshape
    if __cufftMpCreateReshape == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMpCreateReshape = dlsym(handle, 'cufftMpCreateReshape')

    global __cufftMpAttachReshapeComm
    if __cufftMpAttachReshapeComm == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMpAttachReshapeComm = dlsym(handle, 'cufftMpAttachReshapeComm')

    global __cufftMpGetReshapeSize
    if __cufftMpGetReshapeSize == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMpGetReshapeSize = dlsym(handle, 'cufftMpGetReshapeSize')

    global __cufftMpMakeReshape
    if __cufftMpMakeReshape == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMpMakeReshape = dlsym(handle, 'cufftMpMakeReshape')

    global __cufftMpExecReshapeAsync
    if __cufftMpExecReshapeAsync == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMpExecReshapeAsync = dlsym(handle, 'cufftMpExecReshapeAsync')

    global __cufftMpDestroyReshape
    if __cufftMpDestroyReshape == NULL:
        if handle == NULL:
            handle = load_library()
        __cufftMpDestroyReshape = dlsym(handle, 'cufftMpDestroyReshape')

    global ____cufftMpMakeReshape_11_4
    if ____cufftMpMakeReshape_11_4 == NULL:
        if handle == NULL:
            handle = load_library()
        ____cufftMpMakeReshape_11_4 = dlsym(handle, '__cufftMpMakeReshape_11_4')

    __py_cufftMp_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cufftMp()
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

    global __cufftSetPlanPropertyInt64
    data["__cufftSetPlanPropertyInt64"] = <intptr_t>__cufftSetPlanPropertyInt64

    global __cufftGetPlanPropertyInt64
    data["__cufftGetPlanPropertyInt64"] = <intptr_t>__cufftGetPlanPropertyInt64

    global __cufftResetPlanProperty
    data["__cufftResetPlanProperty"] = <intptr_t>__cufftResetPlanProperty

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

    global __cufftMpAttachComm
    data["__cufftMpAttachComm"] = <intptr_t>__cufftMpAttachComm

    global __cufftXtSetDistribution
    data["__cufftXtSetDistribution"] = <intptr_t>__cufftXtSetDistribution

    global __cufftXtSetSubformatDefault
    data["__cufftXtSetSubformatDefault"] = <intptr_t>__cufftXtSetSubformatDefault

    global __cufftMpCreateReshape
    data["__cufftMpCreateReshape"] = <intptr_t>__cufftMpCreateReshape

    global __cufftMpAttachReshapeComm
    data["__cufftMpAttachReshapeComm"] = <intptr_t>__cufftMpAttachReshapeComm

    global __cufftMpGetReshapeSize
    data["__cufftMpGetReshapeSize"] = <intptr_t>__cufftMpGetReshapeSize

    global __cufftMpMakeReshape
    data["__cufftMpMakeReshape"] = <intptr_t>__cufftMpMakeReshape

    global __cufftMpExecReshapeAsync
    data["__cufftMpExecReshapeAsync"] = <intptr_t>__cufftMpExecReshapeAsync

    global __cufftMpDestroyReshape
    data["__cufftMpDestroyReshape"] = <intptr_t>__cufftMpDestroyReshape

    global ____cufftMpMakeReshape_11_4
    data["____cufftMpMakeReshape_11_4"] = <intptr_t>____cufftMpMakeReshape_11_4

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

cdef cufftResult _cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftPlan1d
    _check_or_init_cufftMp()
    if __cufftPlan1d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftPlan1d is not found")
    return (<cufftResult (*)(cufftHandle*, int, cufftType, int) noexcept nogil>__cufftPlan1d)(
        plan, nx, type, batch)


cdef cufftResult _cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftPlan2d
    _check_or_init_cufftMp()
    if __cufftPlan2d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftPlan2d is not found")
    return (<cufftResult (*)(cufftHandle*, int, int, cufftType) noexcept nogil>__cufftPlan2d)(
        plan, nx, ny, type)


cdef cufftResult _cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftPlan3d
    _check_or_init_cufftMp()
    if __cufftPlan3d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftPlan3d is not found")
    return (<cufftResult (*)(cufftHandle*, int, int, int, cufftType) noexcept nogil>__cufftPlan3d)(
        plan, nx, ny, nz, type)


cdef cufftResult _cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftPlanMany
    _check_or_init_cufftMp()
    if __cufftPlanMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftPlanMany is not found")
    return (<cufftResult (*)(cufftHandle*, int, int*, int*, int, int, int*, int, int, cufftType, int) noexcept nogil>__cufftPlanMany)(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)


cdef cufftResult _cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMakePlan1d
    _check_or_init_cufftMp()
    if __cufftMakePlan1d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMakePlan1d is not found")
    return (<cufftResult (*)(cufftHandle, int, cufftType, int, size_t*) noexcept nogil>__cufftMakePlan1d)(
        plan, nx, type, batch, workSize)


cdef cufftResult _cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMakePlan2d
    _check_or_init_cufftMp()
    if __cufftMakePlan2d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMakePlan2d is not found")
    return (<cufftResult (*)(cufftHandle, int, int, cufftType, size_t*) noexcept nogil>__cufftMakePlan2d)(
        plan, nx, ny, type, workSize)


cdef cufftResult _cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMakePlan3d
    _check_or_init_cufftMp()
    if __cufftMakePlan3d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMakePlan3d is not found")
    return (<cufftResult (*)(cufftHandle, int, int, int, cufftType, size_t*) noexcept nogil>__cufftMakePlan3d)(
        plan, nx, ny, nz, type, workSize)


cdef cufftResult _cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMakePlanMany
    _check_or_init_cufftMp()
    if __cufftMakePlanMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMakePlanMany is not found")
    return (<cufftResult (*)(cufftHandle, int, int*, int*, int, int, int*, int, int, cufftType, int, size_t*) noexcept nogil>__cufftMakePlanMany)(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult _cufftMakePlanMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMakePlanMany64
    _check_or_init_cufftMp()
    if __cufftMakePlanMany64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMakePlanMany64 is not found")
    return (<cufftResult (*)(cufftHandle, int, long long int*, long long int*, long long int, long long int, long long int*, long long int, long long int, cufftType, long long int, size_t*) noexcept nogil>__cufftMakePlanMany64)(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult _cufftGetSizeMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftGetSizeMany64
    _check_or_init_cufftMp()
    if __cufftGetSizeMany64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSizeMany64 is not found")
    return (<cufftResult (*)(cufftHandle, int, long long int*, long long int*, long long int, long long int, long long int*, long long int, long long int, cufftType, long long int, size_t*) noexcept nogil>__cufftGetSizeMany64)(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult _cufftEstimate1d(int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftEstimate1d
    _check_or_init_cufftMp()
    if __cufftEstimate1d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftEstimate1d is not found")
    return (<cufftResult (*)(int, cufftType, int, size_t*) noexcept nogil>__cufftEstimate1d)(
        nx, type, batch, workSize)


cdef cufftResult _cufftEstimate2d(int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftEstimate2d
    _check_or_init_cufftMp()
    if __cufftEstimate2d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftEstimate2d is not found")
    return (<cufftResult (*)(int, int, cufftType, size_t*) noexcept nogil>__cufftEstimate2d)(
        nx, ny, type, workSize)


cdef cufftResult _cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftEstimate3d
    _check_or_init_cufftMp()
    if __cufftEstimate3d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftEstimate3d is not found")
    return (<cufftResult (*)(int, int, int, cufftType, size_t*) noexcept nogil>__cufftEstimate3d)(
        nx, ny, nz, type, workSize)


cdef cufftResult _cufftEstimateMany(int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftEstimateMany
    _check_or_init_cufftMp()
    if __cufftEstimateMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftEstimateMany is not found")
    return (<cufftResult (*)(int, int*, int*, int, int, int*, int, int, cufftType, int, size_t*) noexcept nogil>__cufftEstimateMany)(
        rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult _cufftCreate(cufftHandle* handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftCreate
    _check_or_init_cufftMp()
    if __cufftCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftCreate is not found")
    return (<cufftResult (*)(cufftHandle*) noexcept nogil>__cufftCreate)(
        handle)


cdef cufftResult _cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftGetSize1d
    _check_or_init_cufftMp()
    if __cufftGetSize1d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSize1d is not found")
    return (<cufftResult (*)(cufftHandle, int, cufftType, int, size_t*) noexcept nogil>__cufftGetSize1d)(
        handle, nx, type, batch, workSize)


cdef cufftResult _cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftGetSize2d
    _check_or_init_cufftMp()
    if __cufftGetSize2d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSize2d is not found")
    return (<cufftResult (*)(cufftHandle, int, int, cufftType, size_t*) noexcept nogil>__cufftGetSize2d)(
        handle, nx, ny, type, workSize)


cdef cufftResult _cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftGetSize3d
    _check_or_init_cufftMp()
    if __cufftGetSize3d == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSize3d is not found")
    return (<cufftResult (*)(cufftHandle, int, int, int, cufftType, size_t*) noexcept nogil>__cufftGetSize3d)(
        handle, nx, ny, nz, type, workSize)


cdef cufftResult _cufftGetSizeMany(cufftHandle handle, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftGetSizeMany
    _check_or_init_cufftMp()
    if __cufftGetSizeMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSizeMany is not found")
    return (<cufftResult (*)(cufftHandle, int, int*, int*, int, int, int*, int, int, cufftType, int, size_t*) noexcept nogil>__cufftGetSizeMany)(
        handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workArea)


cdef cufftResult _cufftGetSize(cufftHandle handle, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftGetSize
    _check_or_init_cufftMp()
    if __cufftGetSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetSize is not found")
    return (<cufftResult (*)(cufftHandle, size_t*) noexcept nogil>__cufftGetSize)(
        handle, workSize)


cdef cufftResult _cufftSetWorkArea(cufftHandle plan, void* workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftSetWorkArea
    _check_or_init_cufftMp()
    if __cufftSetWorkArea == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftSetWorkArea is not found")
    return (<cufftResult (*)(cufftHandle, void*) noexcept nogil>__cufftSetWorkArea)(
        plan, workArea)


cdef cufftResult _cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftSetAutoAllocation
    _check_or_init_cufftMp()
    if __cufftSetAutoAllocation == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftSetAutoAllocation is not found")
    return (<cufftResult (*)(cufftHandle, int) noexcept nogil>__cufftSetAutoAllocation)(
        plan, autoAllocate)


cdef cufftResult _cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftExecC2C
    _check_or_init_cufftMp()
    if __cufftExecC2C == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecC2C is not found")
    return (<cufftResult (*)(cufftHandle, cufftComplex*, cufftComplex*, int) noexcept nogil>__cufftExecC2C)(
        plan, idata, odata, direction)


cdef cufftResult _cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftExecR2C
    _check_or_init_cufftMp()
    if __cufftExecR2C == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecR2C is not found")
    return (<cufftResult (*)(cufftHandle, cufftReal*, cufftComplex*) noexcept nogil>__cufftExecR2C)(
        plan, idata, odata)


cdef cufftResult _cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftExecC2R
    _check_or_init_cufftMp()
    if __cufftExecC2R == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecC2R is not found")
    return (<cufftResult (*)(cufftHandle, cufftComplex*, cufftReal*) noexcept nogil>__cufftExecC2R)(
        plan, idata, odata)


cdef cufftResult _cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftExecZ2Z
    _check_or_init_cufftMp()
    if __cufftExecZ2Z == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecZ2Z is not found")
    return (<cufftResult (*)(cufftHandle, cufftDoubleComplex*, cufftDoubleComplex*, int) noexcept nogil>__cufftExecZ2Z)(
        plan, idata, odata, direction)


cdef cufftResult _cufftExecD2Z(cufftHandle plan, cufftDoubleReal* idata, cufftDoubleComplex* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftExecD2Z
    _check_or_init_cufftMp()
    if __cufftExecD2Z == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecD2Z is not found")
    return (<cufftResult (*)(cufftHandle, cufftDoubleReal*, cufftDoubleComplex*) noexcept nogil>__cufftExecD2Z)(
        plan, idata, odata)


cdef cufftResult _cufftExecZ2D(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleReal* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftExecZ2D
    _check_or_init_cufftMp()
    if __cufftExecZ2D == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftExecZ2D is not found")
    return (<cufftResult (*)(cufftHandle, cufftDoubleComplex*, cufftDoubleReal*) noexcept nogil>__cufftExecZ2D)(
        plan, idata, odata)


cdef cufftResult _cufftSetStream(cufftHandle plan, cudaStream_t stream) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftSetStream
    _check_or_init_cufftMp()
    if __cufftSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftSetStream is not found")
    return (<cufftResult (*)(cufftHandle, cudaStream_t) noexcept nogil>__cufftSetStream)(
        plan, stream)


cdef cufftResult _cufftDestroy(cufftHandle plan) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftDestroy
    _check_or_init_cufftMp()
    if __cufftDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftDestroy is not found")
    return (<cufftResult (*)(cufftHandle) noexcept nogil>__cufftDestroy)(
        plan)


cdef cufftResult _cufftGetVersion(int* version) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftGetVersion
    _check_or_init_cufftMp()
    if __cufftGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetVersion is not found")
    return (<cufftResult (*)(int*) noexcept nogil>__cufftGetVersion)(
        version)


cdef cufftResult _cufftGetProperty(libraryPropertyType type, int* value) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftGetProperty
    _check_or_init_cufftMp()
    if __cufftGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetProperty is not found")
    return (<cufftResult (*)(libraryPropertyType, int*) noexcept nogil>__cufftGetProperty)(
        type, value)


cdef cufftResult _cufftSetPlanPropertyInt64(cufftHandle plan, cufftProperty property, const long long int inputValueInt) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftSetPlanPropertyInt64
    _check_or_init_cufftMp()
    if __cufftSetPlanPropertyInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftSetPlanPropertyInt64 is not found")
    return (<cufftResult (*)(cufftHandle, cufftProperty, const long long int) noexcept nogil>__cufftSetPlanPropertyInt64)(
        plan, property, inputValueInt)


cdef cufftResult _cufftGetPlanPropertyInt64(cufftHandle plan, cufftProperty property, long long int* returnPtrValue) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftGetPlanPropertyInt64
    _check_or_init_cufftMp()
    if __cufftGetPlanPropertyInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftGetPlanPropertyInt64 is not found")
    return (<cufftResult (*)(cufftHandle, cufftProperty, long long int*) noexcept nogil>__cufftGetPlanPropertyInt64)(
        plan, property, returnPtrValue)


cdef cufftResult _cufftResetPlanProperty(cufftHandle plan, cufftProperty property) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftResetPlanProperty
    _check_or_init_cufftMp()
    if __cufftResetPlanProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftResetPlanProperty is not found")
    return (<cufftResult (*)(cufftHandle, cufftProperty) noexcept nogil>__cufftResetPlanProperty)(
        plan, property)


cdef cufftResult _cufftXtSetGPUs(cufftHandle handle, int nGPUs, int* whichGPUs) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtSetGPUs
    _check_or_init_cufftMp()
    if __cufftXtSetGPUs == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetGPUs is not found")
    return (<cufftResult (*)(cufftHandle, int, int*) noexcept nogil>__cufftXtSetGPUs)(
        handle, nGPUs, whichGPUs)


cdef cufftResult _cufftXtMalloc(cufftHandle plan, cudaLibXtDesc** descriptor, cufftXtSubFormat format) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtMalloc
    _check_or_init_cufftMp()
    if __cufftXtMalloc == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtMalloc is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc**, cufftXtSubFormat) noexcept nogil>__cufftXtMalloc)(
        plan, descriptor, format)


cdef cufftResult _cufftXtMemcpy(cufftHandle plan, void* dstPointer, void* srcPointer, cufftXtCopyType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtMemcpy
    _check_or_init_cufftMp()
    if __cufftXtMemcpy == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtMemcpy is not found")
    return (<cufftResult (*)(cufftHandle, void*, void*, cufftXtCopyType) noexcept nogil>__cufftXtMemcpy)(
        plan, dstPointer, srcPointer, type)


cdef cufftResult _cufftXtFree(cudaLibXtDesc* descriptor) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtFree
    _check_or_init_cufftMp()
    if __cufftXtFree == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtFree is not found")
    return (<cufftResult (*)(cudaLibXtDesc*) noexcept nogil>__cufftXtFree)(
        descriptor)


cdef cufftResult _cufftXtSetWorkArea(cufftHandle plan, void** workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtSetWorkArea
    _check_or_init_cufftMp()
    if __cufftXtSetWorkArea == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetWorkArea is not found")
    return (<cufftResult (*)(cufftHandle, void**) noexcept nogil>__cufftXtSetWorkArea)(
        plan, workArea)


cdef cufftResult _cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtExecDescriptorC2C
    _check_or_init_cufftMp()
    if __cufftXtExecDescriptorC2C == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorC2C is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*, int) noexcept nogil>__cufftXtExecDescriptorC2C)(
        plan, input, output, direction)


cdef cufftResult _cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtExecDescriptorR2C
    _check_or_init_cufftMp()
    if __cufftXtExecDescriptorR2C == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorR2C is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*) noexcept nogil>__cufftXtExecDescriptorR2C)(
        plan, input, output)


cdef cufftResult _cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtExecDescriptorC2R
    _check_or_init_cufftMp()
    if __cufftXtExecDescriptorC2R == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorC2R is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*) noexcept nogil>__cufftXtExecDescriptorC2R)(
        plan, input, output)


cdef cufftResult _cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtExecDescriptorZ2Z
    _check_or_init_cufftMp()
    if __cufftXtExecDescriptorZ2Z == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorZ2Z is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*, int) noexcept nogil>__cufftXtExecDescriptorZ2Z)(
        plan, input, output, direction)


cdef cufftResult _cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtExecDescriptorD2Z
    _check_or_init_cufftMp()
    if __cufftXtExecDescriptorD2Z == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorD2Z is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*) noexcept nogil>__cufftXtExecDescriptorD2Z)(
        plan, input, output)


cdef cufftResult _cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtExecDescriptorZ2D
    _check_or_init_cufftMp()
    if __cufftXtExecDescriptorZ2D == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptorZ2D is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*) noexcept nogil>__cufftXtExecDescriptorZ2D)(
        plan, input, output)


cdef cufftResult _cufftXtQueryPlan(cufftHandle plan, void* queryStruct, cufftXtQueryType queryType) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtQueryPlan
    _check_or_init_cufftMp()
    if __cufftXtQueryPlan == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtQueryPlan is not found")
    return (<cufftResult (*)(cufftHandle, void*, cufftXtQueryType) noexcept nogil>__cufftXtQueryPlan)(
        plan, queryStruct, queryType)


cdef cufftResult _cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtClearCallback
    _check_or_init_cufftMp()
    if __cufftXtClearCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtClearCallback is not found")
    return (<cufftResult (*)(cufftHandle, cufftXtCallbackType) noexcept nogil>__cufftXtClearCallback)(
        plan, cbType)


cdef cufftResult _cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType, size_t sharedSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtSetCallbackSharedSize
    _check_or_init_cufftMp()
    if __cufftXtSetCallbackSharedSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetCallbackSharedSize is not found")
    return (<cufftResult (*)(cufftHandle, cufftXtCallbackType, size_t) noexcept nogil>__cufftXtSetCallbackSharedSize)(
        plan, cbType, sharedSize)


cdef cufftResult _cufftXtMakePlanMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtMakePlanMany
    _check_or_init_cufftMp()
    if __cufftXtMakePlanMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtMakePlanMany is not found")
    return (<cufftResult (*)(cufftHandle, int, long long int*, long long int*, long long int, long long int, cudaDataType, long long int*, long long int, long long int, cudaDataType, long long int, size_t*, cudaDataType) noexcept nogil>__cufftXtMakePlanMany)(
        plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch, workSize, executiontype)


cdef cufftResult _cufftXtGetSizeMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtGetSizeMany
    _check_or_init_cufftMp()
    if __cufftXtGetSizeMany == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtGetSizeMany is not found")
    return (<cufftResult (*)(cufftHandle, int, long long int*, long long int*, long long int, long long int, cudaDataType, long long int*, long long int, long long int, cudaDataType, long long int, size_t*, cudaDataType) noexcept nogil>__cufftXtGetSizeMany)(
        plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch, workSize, executiontype)


cdef cufftResult _cufftXtExec(cufftHandle plan, void* input, void* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtExec
    _check_or_init_cufftMp()
    if __cufftXtExec == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExec is not found")
    return (<cufftResult (*)(cufftHandle, void*, void*, int) noexcept nogil>__cufftXtExec)(
        plan, input, output, direction)


cdef cufftResult _cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtExecDescriptor
    _check_or_init_cufftMp()
    if __cufftXtExecDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtExecDescriptor is not found")
    return (<cufftResult (*)(cufftHandle, cudaLibXtDesc*, cudaLibXtDesc*, int) noexcept nogil>__cufftXtExecDescriptor)(
        plan, input, output, direction)


cdef cufftResult _cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtSetWorkAreaPolicy
    _check_or_init_cufftMp()
    if __cufftXtSetWorkAreaPolicy == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetWorkAreaPolicy is not found")
    return (<cufftResult (*)(cufftHandle, cufftXtWorkAreaPolicy, size_t*) noexcept nogil>__cufftXtSetWorkAreaPolicy)(
        plan, policy, workSize)


cdef cufftResult _cufftMpAttachComm(cufftHandle plan, cufftMpCommType comm_type, void* comm_handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMpAttachComm
    _check_or_init_cufftMp()
    if __cufftMpAttachComm == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMpAttachComm is not found")
    return (<cufftResult (*)(cufftHandle, cufftMpCommType, void*) noexcept nogil>__cufftMpAttachComm)(
        plan, comm_type, comm_handle)


cdef cufftResult _cufftXtSetDistribution(cufftHandle plan, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_input, const long long int* strides_output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtSetDistribution
    _check_or_init_cufftMp()
    if __cufftXtSetDistribution == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetDistribution is not found")
    return (<cufftResult (*)(cufftHandle, int, const long long int*, const long long int*, const long long int*, const long long int*, const long long int*, const long long int*) noexcept nogil>__cufftXtSetDistribution)(
        plan, rank, lower_input, upper_input, lower_output, upper_output, strides_input, strides_output)


cdef cufftResult _cufftXtSetSubformatDefault(cufftHandle plan, cufftXtSubFormat subformat_forward, cufftXtSubFormat subformat_inverse) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftXtSetSubformatDefault
    _check_or_init_cufftMp()
    if __cufftXtSetSubformatDefault == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftXtSetSubformatDefault is not found")
    return (<cufftResult (*)(cufftHandle, cufftXtSubFormat, cufftXtSubFormat) noexcept nogil>__cufftXtSetSubformatDefault)(
        plan, subformat_forward, subformat_inverse)


cdef cufftResult _cufftMpCreateReshape(cufftReshapeHandle* handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMpCreateReshape
    _check_or_init_cufftMp()
    if __cufftMpCreateReshape == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMpCreateReshape is not found")
    return (<cufftResult (*)(cufftReshapeHandle*) noexcept nogil>__cufftMpCreateReshape)(
        handle)


cdef cufftResult _cufftMpAttachReshapeComm(cufftReshapeHandle handle, cufftMpCommType comm_type, void* comm_handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMpAttachReshapeComm
    _check_or_init_cufftMp()
    if __cufftMpAttachReshapeComm == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMpAttachReshapeComm is not found")
    return (<cufftResult (*)(cufftReshapeHandle, cufftMpCommType, void*) noexcept nogil>__cufftMpAttachReshapeComm)(
        handle, comm_type, comm_handle)


cdef cufftResult _cufftMpGetReshapeSize(cufftReshapeHandle handle, size_t* workspace_size) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMpGetReshapeSize
    _check_or_init_cufftMp()
    if __cufftMpGetReshapeSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMpGetReshapeSize is not found")
    return (<cufftResult (*)(cufftReshapeHandle, size_t*) noexcept nogil>__cufftMpGetReshapeSize)(
        handle, workspace_size)


cdef cufftResult ___cufftMpMakeReshape_11_2(cufftReshapeHandle handle, size_t element_size, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_input, const long long int* strides_output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMpMakeReshape
    _check_or_init_cufftMp()
    if __cufftMpMakeReshape == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMpMakeReshape is not found")
    return (<cufftResult (*)(cufftReshapeHandle, size_t, int, const long long int*, const long long int*, const long long int*, const long long int*, const long long int*, const long long int*) noexcept nogil>__cufftMpMakeReshape)(
        handle, element_size, rank, lower_input, upper_input, lower_output, upper_output, strides_input, strides_output)


cdef cufftResult _cufftMpExecReshapeAsync(cufftReshapeHandle handle, void* data_out, const void* data_in, void* workspace, cudaStream_t stream) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMpExecReshapeAsync
    _check_or_init_cufftMp()
    if __cufftMpExecReshapeAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMpExecReshapeAsync is not found")
    return (<cufftResult (*)(cufftReshapeHandle, void*, const void*, void*, cudaStream_t) noexcept nogil>__cufftMpExecReshapeAsync)(
        handle, data_out, data_in, workspace, stream)


cdef cufftResult _cufftMpDestroyReshape(cufftReshapeHandle handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global __cufftMpDestroyReshape
    _check_or_init_cufftMp()
    if __cufftMpDestroyReshape == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftMpDestroyReshape is not found")
    return (<cufftResult (*)(cufftReshapeHandle) noexcept nogil>__cufftMpDestroyReshape)(
        handle)


cdef cufftResult ___cufftMpMakeReshape_11_4(cufftReshapeHandle handle, size_t element_size, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* strides_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_output, void* comm_handle, cufftMpCommType comm_type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    global ____cufftMpMakeReshape_11_4
    _check_or_init_cufftMp()
    if ____cufftMpMakeReshape_11_4 == NULL:
        with gil:
            raise FunctionNotFoundError("function __cufftMpMakeReshape_11_4 is not found")
    return (<cufftResult (*)(cufftReshapeHandle, size_t, int, const long long int*, const long long int*, const long long int*, const long long int*, const long long int*, const long long int*, void*, cufftMpCommType) noexcept nogil>____cufftMpMakeReshape_11_4)(
        handle, element_size, rank, lower_input, upper_input, strides_input, lower_output, upper_output, strides_output, comm_handle, comm_type)
