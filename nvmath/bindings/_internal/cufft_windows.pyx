# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_cufft_dso_version_suffix

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


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef load_library(const int driver_ver):
    handle = 0

    for suffix in get_cufft_dso_version_suffix(driver_ver):
        if len(suffix) == 0:
            continue
        dll_name = f"cufft64_{suffix}.dll"

        # First check if the DLL has been loaded by 3rd parties
        try:
            handle = win32api.GetModuleHandle(dll_name)
        except:
            pass
        else:
            break

        # Next, check if DLLs are installed via pip
        for sp in get_site_packages():
            mod_path = os.path.join(sp, "nvidia", "cufft", "bin")
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
        raise RuntimeError('Failed to load cufft')

    assert handle != 0
    return handle


cdef int _check_or_init_cufft() except -1 nogil:
    global __py_cufft_init
    if __py_cufft_init:
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
        global __cufftPlan1d
        try:
            __cufftPlan1d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftPlan1d')
        except:
            pass

        global __cufftPlan2d
        try:
            __cufftPlan2d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftPlan2d')
        except:
            pass

        global __cufftPlan3d
        try:
            __cufftPlan3d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftPlan3d')
        except:
            pass

        global __cufftPlanMany
        try:
            __cufftPlanMany = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftPlanMany')
        except:
            pass

        global __cufftMakePlan1d
        try:
            __cufftMakePlan1d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftMakePlan1d')
        except:
            pass

        global __cufftMakePlan2d
        try:
            __cufftMakePlan2d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftMakePlan2d')
        except:
            pass

        global __cufftMakePlan3d
        try:
            __cufftMakePlan3d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftMakePlan3d')
        except:
            pass

        global __cufftMakePlanMany
        try:
            __cufftMakePlanMany = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftMakePlanMany')
        except:
            pass

        global __cufftMakePlanMany64
        try:
            __cufftMakePlanMany64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftMakePlanMany64')
        except:
            pass

        global __cufftGetSizeMany64
        try:
            __cufftGetSizeMany64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftGetSizeMany64')
        except:
            pass

        global __cufftEstimate1d
        try:
            __cufftEstimate1d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftEstimate1d')
        except:
            pass

        global __cufftEstimate2d
        try:
            __cufftEstimate2d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftEstimate2d')
        except:
            pass

        global __cufftEstimate3d
        try:
            __cufftEstimate3d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftEstimate3d')
        except:
            pass

        global __cufftEstimateMany
        try:
            __cufftEstimateMany = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftEstimateMany')
        except:
            pass

        global __cufftCreate
        try:
            __cufftCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftCreate')
        except:
            pass

        global __cufftGetSize1d
        try:
            __cufftGetSize1d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftGetSize1d')
        except:
            pass

        global __cufftGetSize2d
        try:
            __cufftGetSize2d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftGetSize2d')
        except:
            pass

        global __cufftGetSize3d
        try:
            __cufftGetSize3d = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftGetSize3d')
        except:
            pass

        global __cufftGetSizeMany
        try:
            __cufftGetSizeMany = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftGetSizeMany')
        except:
            pass

        global __cufftGetSize
        try:
            __cufftGetSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftGetSize')
        except:
            pass

        global __cufftSetWorkArea
        try:
            __cufftSetWorkArea = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftSetWorkArea')
        except:
            pass

        global __cufftSetAutoAllocation
        try:
            __cufftSetAutoAllocation = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftSetAutoAllocation')
        except:
            pass

        global __cufftExecC2C
        try:
            __cufftExecC2C = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftExecC2C')
        except:
            pass

        global __cufftExecR2C
        try:
            __cufftExecR2C = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftExecR2C')
        except:
            pass

        global __cufftExecC2R
        try:
            __cufftExecC2R = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftExecC2R')
        except:
            pass

        global __cufftExecZ2Z
        try:
            __cufftExecZ2Z = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftExecZ2Z')
        except:
            pass

        global __cufftExecD2Z
        try:
            __cufftExecD2Z = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftExecD2Z')
        except:
            pass

        global __cufftExecZ2D
        try:
            __cufftExecZ2D = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftExecZ2D')
        except:
            pass

        global __cufftSetStream
        try:
            __cufftSetStream = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftSetStream')
        except:
            pass

        global __cufftDestroy
        try:
            __cufftDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftDestroy')
        except:
            pass

        global __cufftGetVersion
        try:
            __cufftGetVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftGetVersion')
        except:
            pass

        global __cufftGetProperty
        try:
            __cufftGetProperty = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftGetProperty')
        except:
            pass

        global __cufftXtSetGPUs
        try:
            __cufftXtSetGPUs = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtSetGPUs')
        except:
            pass

        global __cufftXtMalloc
        try:
            __cufftXtMalloc = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtMalloc')
        except:
            pass

        global __cufftXtMemcpy
        try:
            __cufftXtMemcpy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtMemcpy')
        except:
            pass

        global __cufftXtFree
        try:
            __cufftXtFree = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtFree')
        except:
            pass

        global __cufftXtSetWorkArea
        try:
            __cufftXtSetWorkArea = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtSetWorkArea')
        except:
            pass

        global __cufftXtExecDescriptorC2C
        try:
            __cufftXtExecDescriptorC2C = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtExecDescriptorC2C')
        except:
            pass

        global __cufftXtExecDescriptorR2C
        try:
            __cufftXtExecDescriptorR2C = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtExecDescriptorR2C')
        except:
            pass

        global __cufftXtExecDescriptorC2R
        try:
            __cufftXtExecDescriptorC2R = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtExecDescriptorC2R')
        except:
            pass

        global __cufftXtExecDescriptorZ2Z
        try:
            __cufftXtExecDescriptorZ2Z = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtExecDescriptorZ2Z')
        except:
            pass

        global __cufftXtExecDescriptorD2Z
        try:
            __cufftXtExecDescriptorD2Z = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtExecDescriptorD2Z')
        except:
            pass

        global __cufftXtExecDescriptorZ2D
        try:
            __cufftXtExecDescriptorZ2D = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtExecDescriptorZ2D')
        except:
            pass

        global __cufftXtQueryPlan
        try:
            __cufftXtQueryPlan = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtQueryPlan')
        except:
            pass

        global __cufftXtClearCallback
        try:
            __cufftXtClearCallback = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtClearCallback')
        except:
            pass

        global __cufftXtSetCallbackSharedSize
        try:
            __cufftXtSetCallbackSharedSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtSetCallbackSharedSize')
        except:
            pass

        global __cufftXtMakePlanMany
        try:
            __cufftXtMakePlanMany = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtMakePlanMany')
        except:
            pass

        global __cufftXtGetSizeMany
        try:
            __cufftXtGetSizeMany = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtGetSizeMany')
        except:
            pass

        global __cufftXtExec
        try:
            __cufftXtExec = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtExec')
        except:
            pass

        global __cufftXtExecDescriptor
        try:
            __cufftXtExecDescriptor = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtExecDescriptor')
        except:
            pass

        global __cufftXtSetWorkAreaPolicy
        try:
            __cufftXtSetWorkAreaPolicy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtSetWorkAreaPolicy')
        except:
            pass

        global __cufftXtSetJITCallback
        try:
            __cufftXtSetJITCallback = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtSetJITCallback')
        except:
            pass

        global __cufftXtSetSubformatDefault
        try:
            __cufftXtSetSubformatDefault = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftXtSetSubformatDefault')
        except:
            pass

        global __cufftSetPlanPropertyInt64
        try:
            __cufftSetPlanPropertyInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftSetPlanPropertyInt64')
        except:
            pass

        global __cufftGetPlanPropertyInt64
        try:
            __cufftGetPlanPropertyInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftGetPlanPropertyInt64')
        except:
            pass

        global __cufftResetPlanProperty
        try:
            __cufftResetPlanProperty = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftResetPlanProperty')
        except:
            pass

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
