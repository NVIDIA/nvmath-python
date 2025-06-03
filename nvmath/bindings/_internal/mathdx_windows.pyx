# This code was automatically generated with version 0.2.1. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_mathdx_dso_version_suffix

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
cdef bint __py_mathdx_init = False
cdef void* __cuDriverGetVersion = NULL

cdef void* __commondxCreateCode = NULL
cdef void* __commondxSetCodeOptionInt64 = NULL
cdef void* __commondxGetCodeOptionInt64 = NULL
cdef void* __commondxGetCodeOptionsInt64s = NULL
cdef void* __commondxGetCodeLTOIRSize = NULL
cdef void* __commondxGetCodeLTOIR = NULL
cdef void* __commondxDestroyCode = NULL
cdef void* __commondxStatusToStr = NULL
cdef void* __cublasdxCreateDescriptor = NULL
cdef void* __cublasdxSetOptionStr = NULL
cdef void* __cublasdxSetOperatorInt64 = NULL
cdef void* __cublasdxSetOperatorInt64s = NULL
cdef void* __cublasdxBindTensor = NULL
cdef void* __cublasdxSetTensorOptionInt64 = NULL
cdef void* __cublasdxFinalizeTensors = NULL
cdef void* __cublasdxGetTensorTraitInt64 = NULL
cdef void* __cublasdxGetTensorTraitStrSize = NULL
cdef void* __cublasdxGetTensorTraitStr = NULL
cdef void* __cublasdxBindDeviceFunction = NULL
cdef void* __cublasdxFinalizeDeviceFunctions = NULL
cdef void* __cublasdxGetDeviceFunctionTraitStrSize = NULL
cdef void* __cublasdxGetDeviceFunctionTraitStr = NULL
cdef void* __cublasdxGetLTOIRSize = NULL
cdef void* __cublasdxGetLTOIR = NULL
cdef void* __cublasdxGetTraitStrSize = NULL
cdef void* __cublasdxGetTraitStr = NULL
cdef void* __cublasdxGetTraitInt64 = NULL
cdef void* __cublasdxGetTraitInt64s = NULL
cdef void* __cublasdxOperatorTypeToStr = NULL
cdef void* __cublasdxTraitTypeToStr = NULL
cdef void* __cublasdxFinalizeCode = NULL
cdef void* __cublasdxDestroyDescriptor = NULL
cdef void* __cufftdxCreateDescriptor = NULL
cdef void* __cufftdxSetOptionStr = NULL
cdef void* __cufftdxGetKnobInt64Size = NULL
cdef void* __cufftdxGetKnobInt64s = NULL
cdef void* __cufftdxSetOperatorInt64 = NULL
cdef void* __cufftdxSetOperatorInt64s = NULL
cdef void* __cufftdxGetLTOIRSize = NULL
cdef void* __cufftdxGetLTOIR = NULL
cdef void* __cufftdxGetTraitStrSize = NULL
cdef void* __cufftdxGetTraitStr = NULL
cdef void* __cufftdxGetTraitInt64 = NULL
cdef void* __cufftdxGetTraitInt64s = NULL
cdef void* __cufftdxGetTraitCommondxDataType = NULL
cdef void* __cufftdxFinalizeCode = NULL
cdef void* __cufftdxDestroyDescriptor = NULL
cdef void* __cufftdxOperatorTypeToStr = NULL
cdef void* __cufftdxTraitTypeToStr = NULL
cdef void* __cusolverdxCreateDescriptor = NULL
cdef void* __cusolverdxSetOptionStr = NULL
cdef void* __cusolverdxSetOperatorInt64 = NULL
cdef void* __cusolverdxSetOperatorInt64s = NULL
cdef void* __cusolverdxGetLTOIRSize = NULL
cdef void* __cusolverdxGetLTOIR = NULL
cdef void* __cusolverdxGetUniversalFATBINSize = NULL
cdef void* __cusolverdxGetUniversalFATBIN = NULL
cdef void* __cusolverdxGetTraitStrSize = NULL
cdef void* __cusolverdxGetTraitStr = NULL
cdef void* __cusolverdxGetTraitInt64 = NULL
cdef void* __cusolverdxFinalizeCode = NULL
cdef void* __cusolverdxDestroyDescriptor = NULL
cdef void* __cusolverdxOperatorTypeToStr = NULL
cdef void* __cusolverdxTraitTypeToStr = NULL
cdef void* __mathdxGetVersion = NULL
cdef void* __mathdxGetVersionEx = NULL


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef load_library(const int driver_ver):
    handle = 0

    for suffix in get_mathdx_dso_version_suffix(driver_ver):
        if len(suffix) == 0:
            continue
        dll_name = f"mathdx64_{suffix}.dll"

        # First check if the DLL has been loaded by 3rd parties
        try:
            handle = win32api.GetModuleHandle(dll_name)
        except:
            pass
        else:
            break

        # Next, check if DLLs are installed via pip
        for sp in get_site_packages():
            mod_path = os.path.join(sp, "nvidia", "mathdx", "bin")
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
        raise RuntimeError('Failed to load mathdx')

    assert handle != 0
    return handle


cdef int _check_or_init_mathdx() except -1 nogil:
    global __py_mathdx_init
    if __py_mathdx_init:
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
        global __commondxCreateCode
        try:
            __commondxCreateCode = <void*><intptr_t>win32api.GetProcAddress(handle, 'commondxCreateCode')
        except:
            pass

        global __commondxSetCodeOptionInt64
        try:
            __commondxSetCodeOptionInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'commondxSetCodeOptionInt64')
        except:
            pass

        global __commondxGetCodeOptionInt64
        try:
            __commondxGetCodeOptionInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'commondxGetCodeOptionInt64')
        except:
            pass

        global __commondxGetCodeOptionsInt64s
        try:
            __commondxGetCodeOptionsInt64s = <void*><intptr_t>win32api.GetProcAddress(handle, 'commondxGetCodeOptionsInt64s')
        except:
            pass

        global __commondxGetCodeLTOIRSize
        try:
            __commondxGetCodeLTOIRSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'commondxGetCodeLTOIRSize')
        except:
            pass

        global __commondxGetCodeLTOIR
        try:
            __commondxGetCodeLTOIR = <void*><intptr_t>win32api.GetProcAddress(handle, 'commondxGetCodeLTOIR')
        except:
            pass

        global __commondxDestroyCode
        try:
            __commondxDestroyCode = <void*><intptr_t>win32api.GetProcAddress(handle, 'commondxDestroyCode')
        except:
            pass

        global __commondxStatusToStr
        try:
            __commondxStatusToStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'commondxStatusToStr')
        except:
            pass

        global __cublasdxCreateDescriptor
        try:
            __cublasdxCreateDescriptor = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxCreateDescriptor')
        except:
            pass

        global __cublasdxSetOptionStr
        try:
            __cublasdxSetOptionStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxSetOptionStr')
        except:
            pass

        global __cublasdxSetOperatorInt64
        try:
            __cublasdxSetOperatorInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxSetOperatorInt64')
        except:
            pass

        global __cublasdxSetOperatorInt64s
        try:
            __cublasdxSetOperatorInt64s = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxSetOperatorInt64s')
        except:
            pass

        global __cublasdxBindTensor
        try:
            __cublasdxBindTensor = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxBindTensor')
        except:
            pass

        global __cublasdxSetTensorOptionInt64
        try:
            __cublasdxSetTensorOptionInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxSetTensorOptionInt64')
        except:
            pass

        global __cublasdxFinalizeTensors
        try:
            __cublasdxFinalizeTensors = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxFinalizeTensors')
        except:
            pass

        global __cublasdxGetTensorTraitInt64
        try:
            __cublasdxGetTensorTraitInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetTensorTraitInt64')
        except:
            pass

        global __cublasdxGetTensorTraitStrSize
        try:
            __cublasdxGetTensorTraitStrSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetTensorTraitStrSize')
        except:
            pass

        global __cublasdxGetTensorTraitStr
        try:
            __cublasdxGetTensorTraitStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetTensorTraitStr')
        except:
            pass

        global __cublasdxBindDeviceFunction
        try:
            __cublasdxBindDeviceFunction = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxBindDeviceFunction')
        except:
            pass

        global __cublasdxFinalizeDeviceFunctions
        try:
            __cublasdxFinalizeDeviceFunctions = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxFinalizeDeviceFunctions')
        except:
            pass

        global __cublasdxGetDeviceFunctionTraitStrSize
        try:
            __cublasdxGetDeviceFunctionTraitStrSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetDeviceFunctionTraitStrSize')
        except:
            pass

        global __cublasdxGetDeviceFunctionTraitStr
        try:
            __cublasdxGetDeviceFunctionTraitStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetDeviceFunctionTraitStr')
        except:
            pass

        global __cublasdxGetLTOIRSize
        try:
            __cublasdxGetLTOIRSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetLTOIRSize')
        except:
            pass

        global __cublasdxGetLTOIR
        try:
            __cublasdxGetLTOIR = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetLTOIR')
        except:
            pass

        global __cublasdxGetTraitStrSize
        try:
            __cublasdxGetTraitStrSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetTraitStrSize')
        except:
            pass

        global __cublasdxGetTraitStr
        try:
            __cublasdxGetTraitStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetTraitStr')
        except:
            pass

        global __cublasdxGetTraitInt64
        try:
            __cublasdxGetTraitInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetTraitInt64')
        except:
            pass

        global __cublasdxGetTraitInt64s
        try:
            __cublasdxGetTraitInt64s = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxGetTraitInt64s')
        except:
            pass

        global __cublasdxOperatorTypeToStr
        try:
            __cublasdxOperatorTypeToStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxOperatorTypeToStr')
        except:
            pass

        global __cublasdxTraitTypeToStr
        try:
            __cublasdxTraitTypeToStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxTraitTypeToStr')
        except:
            pass

        global __cublasdxFinalizeCode
        try:
            __cublasdxFinalizeCode = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxFinalizeCode')
        except:
            pass

        global __cublasdxDestroyDescriptor
        try:
            __cublasdxDestroyDescriptor = <void*><intptr_t>win32api.GetProcAddress(handle, 'cublasdxDestroyDescriptor')
        except:
            pass

        global __cufftdxCreateDescriptor
        try:
            __cufftdxCreateDescriptor = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxCreateDescriptor')
        except:
            pass

        global __cufftdxSetOptionStr
        try:
            __cufftdxSetOptionStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxSetOptionStr')
        except:
            pass

        global __cufftdxGetKnobInt64Size
        try:
            __cufftdxGetKnobInt64Size = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxGetKnobInt64Size')
        except:
            pass

        global __cufftdxGetKnobInt64s
        try:
            __cufftdxGetKnobInt64s = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxGetKnobInt64s')
        except:
            pass

        global __cufftdxSetOperatorInt64
        try:
            __cufftdxSetOperatorInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxSetOperatorInt64')
        except:
            pass

        global __cufftdxSetOperatorInt64s
        try:
            __cufftdxSetOperatorInt64s = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxSetOperatorInt64s')
        except:
            pass

        global __cufftdxGetLTOIRSize
        try:
            __cufftdxGetLTOIRSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxGetLTOIRSize')
        except:
            pass

        global __cufftdxGetLTOIR
        try:
            __cufftdxGetLTOIR = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxGetLTOIR')
        except:
            pass

        global __cufftdxGetTraitStrSize
        try:
            __cufftdxGetTraitStrSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxGetTraitStrSize')
        except:
            pass

        global __cufftdxGetTraitStr
        try:
            __cufftdxGetTraitStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxGetTraitStr')
        except:
            pass

        global __cufftdxGetTraitInt64
        try:
            __cufftdxGetTraitInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxGetTraitInt64')
        except:
            pass

        global __cufftdxGetTraitInt64s
        try:
            __cufftdxGetTraitInt64s = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxGetTraitInt64s')
        except:
            pass

        global __cufftdxGetTraitCommondxDataType
        try:
            __cufftdxGetTraitCommondxDataType = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxGetTraitCommondxDataType')
        except:
            pass

        global __cufftdxFinalizeCode
        try:
            __cufftdxFinalizeCode = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxFinalizeCode')
        except:
            pass

        global __cufftdxDestroyDescriptor
        try:
            __cufftdxDestroyDescriptor = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxDestroyDescriptor')
        except:
            pass

        global __cufftdxOperatorTypeToStr
        try:
            __cufftdxOperatorTypeToStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxOperatorTypeToStr')
        except:
            pass

        global __cufftdxTraitTypeToStr
        try:
            __cufftdxTraitTypeToStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cufftdxTraitTypeToStr')
        except:
            pass

        global __cusolverdxCreateDescriptor
        try:
            __cusolverdxCreateDescriptor = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxCreateDescriptor')
        except:
            pass

        global __cusolverdxSetOptionStr
        try:
            __cusolverdxSetOptionStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxSetOptionStr')
        except:
            pass

        global __cusolverdxSetOperatorInt64
        try:
            __cusolverdxSetOperatorInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxSetOperatorInt64')
        except:
            pass

        global __cusolverdxSetOperatorInt64s
        try:
            __cusolverdxSetOperatorInt64s = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxSetOperatorInt64s')
        except:
            pass

        global __cusolverdxGetLTOIRSize
        try:
            __cusolverdxGetLTOIRSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxGetLTOIRSize')
        except:
            pass

        global __cusolverdxGetLTOIR
        try:
            __cusolverdxGetLTOIR = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxGetLTOIR')
        except:
            pass

        global __cusolverdxGetUniversalFATBINSize
        try:
            __cusolverdxGetUniversalFATBINSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxGetUniversalFATBINSize')
        except:
            pass

        global __cusolverdxGetUniversalFATBIN
        try:
            __cusolverdxGetUniversalFATBIN = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxGetUniversalFATBIN')
        except:
            pass

        global __cusolverdxGetTraitStrSize
        try:
            __cusolverdxGetTraitStrSize = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxGetTraitStrSize')
        except:
            pass

        global __cusolverdxGetTraitStr
        try:
            __cusolverdxGetTraitStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxGetTraitStr')
        except:
            pass

        global __cusolverdxGetTraitInt64
        try:
            __cusolverdxGetTraitInt64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxGetTraitInt64')
        except:
            pass

        global __cusolverdxFinalizeCode
        try:
            __cusolverdxFinalizeCode = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxFinalizeCode')
        except:
            pass

        global __cusolverdxDestroyDescriptor
        try:
            __cusolverdxDestroyDescriptor = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxDestroyDescriptor')
        except:
            pass

        global __cusolverdxOperatorTypeToStr
        try:
            __cusolverdxOperatorTypeToStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxOperatorTypeToStr')
        except:
            pass

        global __cusolverdxTraitTypeToStr
        try:
            __cusolverdxTraitTypeToStr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverdxTraitTypeToStr')
        except:
            pass

        global __mathdxGetVersion
        try:
            __mathdxGetVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'mathdxGetVersion')
        except:
            pass

        global __mathdxGetVersionEx
        try:
            __mathdxGetVersionEx = <void*><intptr_t>win32api.GetProcAddress(handle, 'mathdxGetVersionEx')
        except:
            pass

    __py_mathdx_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_mathdx()
    cdef dict data = {}

    global __commondxCreateCode
    data["__commondxCreateCode"] = <intptr_t>__commondxCreateCode

    global __commondxSetCodeOptionInt64
    data["__commondxSetCodeOptionInt64"] = <intptr_t>__commondxSetCodeOptionInt64

    global __commondxGetCodeOptionInt64
    data["__commondxGetCodeOptionInt64"] = <intptr_t>__commondxGetCodeOptionInt64

    global __commondxGetCodeOptionsInt64s
    data["__commondxGetCodeOptionsInt64s"] = <intptr_t>__commondxGetCodeOptionsInt64s

    global __commondxGetCodeLTOIRSize
    data["__commondxGetCodeLTOIRSize"] = <intptr_t>__commondxGetCodeLTOIRSize

    global __commondxGetCodeLTOIR
    data["__commondxGetCodeLTOIR"] = <intptr_t>__commondxGetCodeLTOIR

    global __commondxDestroyCode
    data["__commondxDestroyCode"] = <intptr_t>__commondxDestroyCode

    global __commondxStatusToStr
    data["__commondxStatusToStr"] = <intptr_t>__commondxStatusToStr

    global __cublasdxCreateDescriptor
    data["__cublasdxCreateDescriptor"] = <intptr_t>__cublasdxCreateDescriptor

    global __cublasdxSetOptionStr
    data["__cublasdxSetOptionStr"] = <intptr_t>__cublasdxSetOptionStr

    global __cublasdxSetOperatorInt64
    data["__cublasdxSetOperatorInt64"] = <intptr_t>__cublasdxSetOperatorInt64

    global __cublasdxSetOperatorInt64s
    data["__cublasdxSetOperatorInt64s"] = <intptr_t>__cublasdxSetOperatorInt64s

    global __cublasdxBindTensor
    data["__cublasdxBindTensor"] = <intptr_t>__cublasdxBindTensor

    global __cublasdxSetTensorOptionInt64
    data["__cublasdxSetTensorOptionInt64"] = <intptr_t>__cublasdxSetTensorOptionInt64

    global __cublasdxFinalizeTensors
    data["__cublasdxFinalizeTensors"] = <intptr_t>__cublasdxFinalizeTensors

    global __cublasdxGetTensorTraitInt64
    data["__cublasdxGetTensorTraitInt64"] = <intptr_t>__cublasdxGetTensorTraitInt64

    global __cublasdxGetTensorTraitStrSize
    data["__cublasdxGetTensorTraitStrSize"] = <intptr_t>__cublasdxGetTensorTraitStrSize

    global __cublasdxGetTensorTraitStr
    data["__cublasdxGetTensorTraitStr"] = <intptr_t>__cublasdxGetTensorTraitStr

    global __cublasdxBindDeviceFunction
    data["__cublasdxBindDeviceFunction"] = <intptr_t>__cublasdxBindDeviceFunction

    global __cublasdxFinalizeDeviceFunctions
    data["__cublasdxFinalizeDeviceFunctions"] = <intptr_t>__cublasdxFinalizeDeviceFunctions

    global __cublasdxGetDeviceFunctionTraitStrSize
    data["__cublasdxGetDeviceFunctionTraitStrSize"] = <intptr_t>__cublasdxGetDeviceFunctionTraitStrSize

    global __cublasdxGetDeviceFunctionTraitStr
    data["__cublasdxGetDeviceFunctionTraitStr"] = <intptr_t>__cublasdxGetDeviceFunctionTraitStr

    global __cublasdxGetLTOIRSize
    data["__cublasdxGetLTOIRSize"] = <intptr_t>__cublasdxGetLTOIRSize

    global __cublasdxGetLTOIR
    data["__cublasdxGetLTOIR"] = <intptr_t>__cublasdxGetLTOIR

    global __cublasdxGetTraitStrSize
    data["__cublasdxGetTraitStrSize"] = <intptr_t>__cublasdxGetTraitStrSize

    global __cublasdxGetTraitStr
    data["__cublasdxGetTraitStr"] = <intptr_t>__cublasdxGetTraitStr

    global __cublasdxGetTraitInt64
    data["__cublasdxGetTraitInt64"] = <intptr_t>__cublasdxGetTraitInt64

    global __cublasdxGetTraitInt64s
    data["__cublasdxGetTraitInt64s"] = <intptr_t>__cublasdxGetTraitInt64s

    global __cublasdxOperatorTypeToStr
    data["__cublasdxOperatorTypeToStr"] = <intptr_t>__cublasdxOperatorTypeToStr

    global __cublasdxTraitTypeToStr
    data["__cublasdxTraitTypeToStr"] = <intptr_t>__cublasdxTraitTypeToStr

    global __cublasdxFinalizeCode
    data["__cublasdxFinalizeCode"] = <intptr_t>__cublasdxFinalizeCode

    global __cublasdxDestroyDescriptor
    data["__cublasdxDestroyDescriptor"] = <intptr_t>__cublasdxDestroyDescriptor

    global __cufftdxCreateDescriptor
    data["__cufftdxCreateDescriptor"] = <intptr_t>__cufftdxCreateDescriptor

    global __cufftdxSetOptionStr
    data["__cufftdxSetOptionStr"] = <intptr_t>__cufftdxSetOptionStr

    global __cufftdxGetKnobInt64Size
    data["__cufftdxGetKnobInt64Size"] = <intptr_t>__cufftdxGetKnobInt64Size

    global __cufftdxGetKnobInt64s
    data["__cufftdxGetKnobInt64s"] = <intptr_t>__cufftdxGetKnobInt64s

    global __cufftdxSetOperatorInt64
    data["__cufftdxSetOperatorInt64"] = <intptr_t>__cufftdxSetOperatorInt64

    global __cufftdxSetOperatorInt64s
    data["__cufftdxSetOperatorInt64s"] = <intptr_t>__cufftdxSetOperatorInt64s

    global __cufftdxGetLTOIRSize
    data["__cufftdxGetLTOIRSize"] = <intptr_t>__cufftdxGetLTOIRSize

    global __cufftdxGetLTOIR
    data["__cufftdxGetLTOIR"] = <intptr_t>__cufftdxGetLTOIR

    global __cufftdxGetTraitStrSize
    data["__cufftdxGetTraitStrSize"] = <intptr_t>__cufftdxGetTraitStrSize

    global __cufftdxGetTraitStr
    data["__cufftdxGetTraitStr"] = <intptr_t>__cufftdxGetTraitStr

    global __cufftdxGetTraitInt64
    data["__cufftdxGetTraitInt64"] = <intptr_t>__cufftdxGetTraitInt64

    global __cufftdxGetTraitInt64s
    data["__cufftdxGetTraitInt64s"] = <intptr_t>__cufftdxGetTraitInt64s

    global __cufftdxGetTraitCommondxDataType
    data["__cufftdxGetTraitCommondxDataType"] = <intptr_t>__cufftdxGetTraitCommondxDataType

    global __cufftdxFinalizeCode
    data["__cufftdxFinalizeCode"] = <intptr_t>__cufftdxFinalizeCode

    global __cufftdxDestroyDescriptor
    data["__cufftdxDestroyDescriptor"] = <intptr_t>__cufftdxDestroyDescriptor

    global __cufftdxOperatorTypeToStr
    data["__cufftdxOperatorTypeToStr"] = <intptr_t>__cufftdxOperatorTypeToStr

    global __cufftdxTraitTypeToStr
    data["__cufftdxTraitTypeToStr"] = <intptr_t>__cufftdxTraitTypeToStr

    global __cusolverdxCreateDescriptor
    data["__cusolverdxCreateDescriptor"] = <intptr_t>__cusolverdxCreateDescriptor

    global __cusolverdxSetOptionStr
    data["__cusolverdxSetOptionStr"] = <intptr_t>__cusolverdxSetOptionStr

    global __cusolverdxSetOperatorInt64
    data["__cusolverdxSetOperatorInt64"] = <intptr_t>__cusolverdxSetOperatorInt64

    global __cusolverdxSetOperatorInt64s
    data["__cusolverdxSetOperatorInt64s"] = <intptr_t>__cusolverdxSetOperatorInt64s

    global __cusolverdxGetLTOIRSize
    data["__cusolverdxGetLTOIRSize"] = <intptr_t>__cusolverdxGetLTOIRSize

    global __cusolverdxGetLTOIR
    data["__cusolverdxGetLTOIR"] = <intptr_t>__cusolverdxGetLTOIR

    global __cusolverdxGetUniversalFATBINSize
    data["__cusolverdxGetUniversalFATBINSize"] = <intptr_t>__cusolverdxGetUniversalFATBINSize

    global __cusolverdxGetUniversalFATBIN
    data["__cusolverdxGetUniversalFATBIN"] = <intptr_t>__cusolverdxGetUniversalFATBIN

    global __cusolverdxGetTraitStrSize
    data["__cusolverdxGetTraitStrSize"] = <intptr_t>__cusolverdxGetTraitStrSize

    global __cusolverdxGetTraitStr
    data["__cusolverdxGetTraitStr"] = <intptr_t>__cusolverdxGetTraitStr

    global __cusolverdxGetTraitInt64
    data["__cusolverdxGetTraitInt64"] = <intptr_t>__cusolverdxGetTraitInt64

    global __cusolverdxFinalizeCode
    data["__cusolverdxFinalizeCode"] = <intptr_t>__cusolverdxFinalizeCode

    global __cusolverdxDestroyDescriptor
    data["__cusolverdxDestroyDescriptor"] = <intptr_t>__cusolverdxDestroyDescriptor

    global __cusolverdxOperatorTypeToStr
    data["__cusolverdxOperatorTypeToStr"] = <intptr_t>__cusolverdxOperatorTypeToStr

    global __cusolverdxTraitTypeToStr
    data["__cusolverdxTraitTypeToStr"] = <intptr_t>__cusolverdxTraitTypeToStr

    global __mathdxGetVersion
    data["__mathdxGetVersion"] = <intptr_t>__mathdxGetVersion

    global __mathdxGetVersionEx
    data["__mathdxGetVersionEx"] = <intptr_t>__mathdxGetVersionEx

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

cdef commondxStatusType _commondxCreateCode(commondxCode* code) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxCreateCode
    _check_or_init_mathdx()
    if __commondxCreateCode == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxCreateCode is not found")
    return (<commondxStatusType (*)(commondxCode*) noexcept nogil>__commondxCreateCode)(
        code)


cdef commondxStatusType _commondxSetCodeOptionInt64(commondxCode code, commondxOption option, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxSetCodeOptionInt64
    _check_or_init_mathdx()
    if __commondxSetCodeOptionInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxSetCodeOptionInt64 is not found")
    return (<commondxStatusType (*)(commondxCode, commondxOption, long long int) noexcept nogil>__commondxSetCodeOptionInt64)(
        code, option, value)


cdef commondxStatusType _commondxGetCodeOptionInt64(commondxCode code, commondxOption option, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetCodeOptionInt64
    _check_or_init_mathdx()
    if __commondxGetCodeOptionInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetCodeOptionInt64 is not found")
    return (<commondxStatusType (*)(commondxCode, commondxOption, long long int*) noexcept nogil>__commondxGetCodeOptionInt64)(
        code, option, value)


cdef commondxStatusType _commondxGetCodeOptionsInt64s(commondxCode code, commondxOption option, size_t size, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetCodeOptionsInt64s
    _check_or_init_mathdx()
    if __commondxGetCodeOptionsInt64s == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetCodeOptionsInt64s is not found")
    return (<commondxStatusType (*)(commondxCode, commondxOption, size_t, long long int*) noexcept nogil>__commondxGetCodeOptionsInt64s)(
        code, option, size, array)


cdef commondxStatusType _commondxGetCodeLTOIRSize(commondxCode code, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetCodeLTOIRSize
    _check_or_init_mathdx()
    if __commondxGetCodeLTOIRSize == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetCodeLTOIRSize is not found")
    return (<commondxStatusType (*)(commondxCode, size_t*) noexcept nogil>__commondxGetCodeLTOIRSize)(
        code, size)


cdef commondxStatusType _commondxGetCodeLTOIR(commondxCode code, size_t size, void* out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetCodeLTOIR
    _check_or_init_mathdx()
    if __commondxGetCodeLTOIR == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetCodeLTOIR is not found")
    return (<commondxStatusType (*)(commondxCode, size_t, void*) noexcept nogil>__commondxGetCodeLTOIR)(
        code, size, out)


cdef commondxStatusType _commondxDestroyCode(commondxCode code) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxDestroyCode
    _check_or_init_mathdx()
    if __commondxDestroyCode == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxDestroyCode is not found")
    return (<commondxStatusType (*)(commondxCode) noexcept nogil>__commondxDestroyCode)(
        code)


cdef const char* _commondxStatusToStr(commondxStatusType status) except?NULL nogil:
    global __commondxStatusToStr
    _check_or_init_mathdx()
    if __commondxStatusToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxStatusToStr is not found")
    return (<const char* (*)(commondxStatusType) noexcept nogil>__commondxStatusToStr)(
        status)


cdef commondxStatusType _cublasdxCreateDescriptor(cublasdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxCreateDescriptor
    _check_or_init_mathdx()
    if __cublasdxCreateDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxCreateDescriptor is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor*) noexcept nogil>__cublasdxCreateDescriptor)(
        handle)


cdef commondxStatusType _cublasdxSetOptionStr(cublasdxDescriptor handle, commondxOption option, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxSetOptionStr
    _check_or_init_mathdx()
    if __cublasdxSetOptionStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxSetOptionStr is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, commondxOption, const char*) noexcept nogil>__cublasdxSetOptionStr)(
        handle, option, value)


cdef commondxStatusType _cublasdxSetOperatorInt64(cublasdxDescriptor handle, cublasdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxSetOperatorInt64
    _check_or_init_mathdx()
    if __cublasdxSetOperatorInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxSetOperatorInt64 is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxOperatorType, long long int) noexcept nogil>__cublasdxSetOperatorInt64)(
        handle, op, value)


cdef commondxStatusType _cublasdxSetOperatorInt64s(cublasdxDescriptor handle, cublasdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxSetOperatorInt64s
    _check_or_init_mathdx()
    if __cublasdxSetOperatorInt64s == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxSetOperatorInt64s is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxOperatorType, size_t, const long long int*) noexcept nogil>__cublasdxSetOperatorInt64s)(
        handle, op, count, array)


cdef commondxStatusType _cublasdxBindTensor(cublasdxDescriptor handle, cublasdxTensorType tensor_type, cublasdxTensor* tensor) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxBindTensor
    _check_or_init_mathdx()
    if __cublasdxBindTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxBindTensor is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxTensorType, cublasdxTensor*) noexcept nogil>__cublasdxBindTensor)(
        handle, tensor_type, tensor)


cdef commondxStatusType _cublasdxSetTensorOptionInt64(cublasdxTensor tensor, cublasdxTensorOption option, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxSetTensorOptionInt64
    _check_or_init_mathdx()
    if __cublasdxSetTensorOptionInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxSetTensorOptionInt64 is not found")
    return (<commondxStatusType (*)(cublasdxTensor, cublasdxTensorOption, long long int) noexcept nogil>__cublasdxSetTensorOptionInt64)(
        tensor, option, value)


cdef commondxStatusType _cublasdxFinalizeTensors(cublasdxDescriptor handle, size_t count, const cublasdxTensor* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxFinalizeTensors
    _check_or_init_mathdx()
    if __cublasdxFinalizeTensors == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxFinalizeTensors is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, size_t, const cublasdxTensor*) noexcept nogil>__cublasdxFinalizeTensors)(
        handle, count, array)


cdef commondxStatusType _cublasdxGetTensorTraitInt64(cublasdxTensor tensor, cublasdxTensorTrait trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetTensorTraitInt64
    _check_or_init_mathdx()
    if __cublasdxGetTensorTraitInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetTensorTraitInt64 is not found")
    return (<commondxStatusType (*)(cublasdxTensor, cublasdxTensorTrait, long long int*) noexcept nogil>__cublasdxGetTensorTraitInt64)(
        tensor, trait, value)


cdef commondxStatusType _cublasdxGetTensorTraitStrSize(cublasdxTensor tensor, cublasdxTensorTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetTensorTraitStrSize
    _check_or_init_mathdx()
    if __cublasdxGetTensorTraitStrSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetTensorTraitStrSize is not found")
    return (<commondxStatusType (*)(cublasdxTensor, cublasdxTensorTrait, size_t*) noexcept nogil>__cublasdxGetTensorTraitStrSize)(
        tensor, trait, size)


cdef commondxStatusType _cublasdxGetTensorTraitStr(cublasdxTensor tensor, cublasdxTensorTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetTensorTraitStr
    _check_or_init_mathdx()
    if __cublasdxGetTensorTraitStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetTensorTraitStr is not found")
    return (<commondxStatusType (*)(cublasdxTensor, cublasdxTensorTrait, size_t, char*) noexcept nogil>__cublasdxGetTensorTraitStr)(
        tensor, trait, size, value)


cdef commondxStatusType _cublasdxBindDeviceFunction(cublasdxDescriptor handle, cublasdxDeviceFunctionType device_function_type, size_t count, const cublasdxTensor* array, cublasdxDeviceFunction* device_function) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxBindDeviceFunction
    _check_or_init_mathdx()
    if __cublasdxBindDeviceFunction == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxBindDeviceFunction is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxDeviceFunctionType, size_t, const cublasdxTensor*, cublasdxDeviceFunction*) noexcept nogil>__cublasdxBindDeviceFunction)(
        handle, device_function_type, count, array, device_function)


cdef commondxStatusType _cublasdxFinalizeDeviceFunctions(commondxCode code, size_t count, const cublasdxDeviceFunction* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxFinalizeDeviceFunctions
    _check_or_init_mathdx()
    if __cublasdxFinalizeDeviceFunctions == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxFinalizeDeviceFunctions is not found")
    return (<commondxStatusType (*)(commondxCode, size_t, const cublasdxDeviceFunction*) noexcept nogil>__cublasdxFinalizeDeviceFunctions)(
        code, count, array)


cdef commondxStatusType _cublasdxGetDeviceFunctionTraitStrSize(cublasdxDeviceFunction device_function, cublasdxDeviceFunctionTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetDeviceFunctionTraitStrSize
    _check_or_init_mathdx()
    if __cublasdxGetDeviceFunctionTraitStrSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetDeviceFunctionTraitStrSize is not found")
    return (<commondxStatusType (*)(cublasdxDeviceFunction, cublasdxDeviceFunctionTrait, size_t*) noexcept nogil>__cublasdxGetDeviceFunctionTraitStrSize)(
        device_function, trait, size)


cdef commondxStatusType _cublasdxGetDeviceFunctionTraitStr(cublasdxDeviceFunction device_function, cublasdxDeviceFunctionTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetDeviceFunctionTraitStr
    _check_or_init_mathdx()
    if __cublasdxGetDeviceFunctionTraitStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetDeviceFunctionTraitStr is not found")
    return (<commondxStatusType (*)(cublasdxDeviceFunction, cublasdxDeviceFunctionTrait, size_t, char*) noexcept nogil>__cublasdxGetDeviceFunctionTraitStr)(
        device_function, trait, size, value)


cdef commondxStatusType _cublasdxGetLTOIRSize(cublasdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetLTOIRSize
    _check_or_init_mathdx()
    if __cublasdxGetLTOIRSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetLTOIRSize is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, size_t*) noexcept nogil>__cublasdxGetLTOIRSize)(
        handle, lto_size)


cdef commondxStatusType _cublasdxGetLTOIR(cublasdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetLTOIR
    _check_or_init_mathdx()
    if __cublasdxGetLTOIR == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetLTOIR is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, size_t, void*) noexcept nogil>__cublasdxGetLTOIR)(
        handle, size, lto)


cdef commondxStatusType _cublasdxGetTraitStrSize(cublasdxDescriptor handle, cublasdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetTraitStrSize
    _check_or_init_mathdx()
    if __cublasdxGetTraitStrSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetTraitStrSize is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxTraitType, size_t*) noexcept nogil>__cublasdxGetTraitStrSize)(
        handle, trait, size)


cdef commondxStatusType _cublasdxGetTraitStr(cublasdxDescriptor handle, cublasdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetTraitStr
    _check_or_init_mathdx()
    if __cublasdxGetTraitStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetTraitStr is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxTraitType, size_t, char*) noexcept nogil>__cublasdxGetTraitStr)(
        handle, trait, size, value)


cdef commondxStatusType _cublasdxGetTraitInt64(cublasdxDescriptor handle, cublasdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetTraitInt64
    _check_or_init_mathdx()
    if __cublasdxGetTraitInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetTraitInt64 is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxTraitType, long long int*) noexcept nogil>__cublasdxGetTraitInt64)(
        handle, trait, value)


cdef commondxStatusType _cublasdxGetTraitInt64s(cublasdxDescriptor handle, cublasdxTraitType trait, size_t count, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetTraitInt64s
    _check_or_init_mathdx()
    if __cublasdxGetTraitInt64s == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetTraitInt64s is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxTraitType, size_t, long long int*) noexcept nogil>__cublasdxGetTraitInt64s)(
        handle, trait, count, array)


cdef const char* _cublasdxOperatorTypeToStr(cublasdxOperatorType op) except?NULL nogil:
    global __cublasdxOperatorTypeToStr
    _check_or_init_mathdx()
    if __cublasdxOperatorTypeToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxOperatorTypeToStr is not found")
    return (<const char* (*)(cublasdxOperatorType) noexcept nogil>__cublasdxOperatorTypeToStr)(
        op)


cdef const char* _cublasdxTraitTypeToStr(cublasdxTraitType trait) except?NULL nogil:
    global __cublasdxTraitTypeToStr
    _check_or_init_mathdx()
    if __cublasdxTraitTypeToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxTraitTypeToStr is not found")
    return (<const char* (*)(cublasdxTraitType) noexcept nogil>__cublasdxTraitTypeToStr)(
        trait)


cdef commondxStatusType _cublasdxFinalizeCode(commondxCode code, cublasdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxFinalizeCode
    _check_or_init_mathdx()
    if __cublasdxFinalizeCode == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxFinalizeCode is not found")
    return (<commondxStatusType (*)(commondxCode, cublasdxDescriptor) noexcept nogil>__cublasdxFinalizeCode)(
        code, handle)


cdef commondxStatusType _cublasdxDestroyDescriptor(cublasdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxDestroyDescriptor
    _check_or_init_mathdx()
    if __cublasdxDestroyDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxDestroyDescriptor is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor) noexcept nogil>__cublasdxDestroyDescriptor)(
        handle)


cdef commondxStatusType _cufftdxCreateDescriptor(cufftdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxCreateDescriptor
    _check_or_init_mathdx()
    if __cufftdxCreateDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxCreateDescriptor is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor*) noexcept nogil>__cufftdxCreateDescriptor)(
        handle)


cdef commondxStatusType _cufftdxSetOptionStr(cufftdxDescriptor handle, commondxOption opt, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxSetOptionStr
    _check_or_init_mathdx()
    if __cufftdxSetOptionStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxSetOptionStr is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, commondxOption, const char*) noexcept nogil>__cufftdxSetOptionStr)(
        handle, opt, value)


cdef commondxStatusType _cufftdxGetKnobInt64Size(cufftdxDescriptor handle, size_t num_knobs, cufftdxKnobType* knobs_ptr, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxGetKnobInt64Size
    _check_or_init_mathdx()
    if __cufftdxGetKnobInt64Size == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxGetKnobInt64Size is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, size_t, cufftdxKnobType*, size_t*) noexcept nogil>__cufftdxGetKnobInt64Size)(
        handle, num_knobs, knobs_ptr, size)


cdef commondxStatusType _cufftdxGetKnobInt64s(cufftdxDescriptor handle, size_t num_knobs, cufftdxKnobType* knobs_ptr, size_t size, long long int* values) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxGetKnobInt64s
    _check_or_init_mathdx()
    if __cufftdxGetKnobInt64s == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxGetKnobInt64s is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, size_t, cufftdxKnobType*, size_t, long long int*) noexcept nogil>__cufftdxGetKnobInt64s)(
        handle, num_knobs, knobs_ptr, size, values)


cdef commondxStatusType _cufftdxSetOperatorInt64(cufftdxDescriptor handle, cufftdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxSetOperatorInt64
    _check_or_init_mathdx()
    if __cufftdxSetOperatorInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxSetOperatorInt64 is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, cufftdxOperatorType, long long int) noexcept nogil>__cufftdxSetOperatorInt64)(
        handle, op, value)


cdef commondxStatusType _cufftdxSetOperatorInt64s(cufftdxDescriptor handle, cufftdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxSetOperatorInt64s
    _check_or_init_mathdx()
    if __cufftdxSetOperatorInt64s == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxSetOperatorInt64s is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, cufftdxOperatorType, size_t, const long long int*) noexcept nogil>__cufftdxSetOperatorInt64s)(
        handle, op, count, array)


cdef commondxStatusType _cufftdxGetLTOIRSize(cufftdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxGetLTOIRSize
    _check_or_init_mathdx()
    if __cufftdxGetLTOIRSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxGetLTOIRSize is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, size_t*) noexcept nogil>__cufftdxGetLTOIRSize)(
        handle, lto_size)


cdef commondxStatusType _cufftdxGetLTOIR(cufftdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxGetLTOIR
    _check_or_init_mathdx()
    if __cufftdxGetLTOIR == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxGetLTOIR is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, size_t, void*) noexcept nogil>__cufftdxGetLTOIR)(
        handle, size, lto)


cdef commondxStatusType _cufftdxGetTraitStrSize(cufftdxDescriptor handle, cufftdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxGetTraitStrSize
    _check_or_init_mathdx()
    if __cufftdxGetTraitStrSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxGetTraitStrSize is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, cufftdxTraitType, size_t*) noexcept nogil>__cufftdxGetTraitStrSize)(
        handle, trait, size)


cdef commondxStatusType _cufftdxGetTraitStr(cufftdxDescriptor handle, cufftdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxGetTraitStr
    _check_or_init_mathdx()
    if __cufftdxGetTraitStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxGetTraitStr is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, cufftdxTraitType, size_t, char*) noexcept nogil>__cufftdxGetTraitStr)(
        handle, trait, size, value)


cdef commondxStatusType _cufftdxGetTraitInt64(cufftdxDescriptor handle, cufftdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxGetTraitInt64
    _check_or_init_mathdx()
    if __cufftdxGetTraitInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxGetTraitInt64 is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, cufftdxTraitType, long long int*) noexcept nogil>__cufftdxGetTraitInt64)(
        handle, trait, value)


cdef commondxStatusType _cufftdxGetTraitInt64s(cufftdxDescriptor handle, cufftdxTraitType trait, size_t count, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxGetTraitInt64s
    _check_or_init_mathdx()
    if __cufftdxGetTraitInt64s == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxGetTraitInt64s is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, cufftdxTraitType, size_t, long long int*) noexcept nogil>__cufftdxGetTraitInt64s)(
        handle, trait, count, array)


cdef commondxStatusType _cufftdxGetTraitCommondxDataType(cufftdxDescriptor handle, cufftdxTraitType trait, commondxValueType* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxGetTraitCommondxDataType
    _check_or_init_mathdx()
    if __cufftdxGetTraitCommondxDataType == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxGetTraitCommondxDataType is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor, cufftdxTraitType, commondxValueType*) noexcept nogil>__cufftdxGetTraitCommondxDataType)(
        handle, trait, value)


cdef commondxStatusType _cufftdxFinalizeCode(commondxCode code, cufftdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxFinalizeCode
    _check_or_init_mathdx()
    if __cufftdxFinalizeCode == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxFinalizeCode is not found")
    return (<commondxStatusType (*)(commondxCode, cufftdxDescriptor) noexcept nogil>__cufftdxFinalizeCode)(
        code, handle)


cdef commondxStatusType _cufftdxDestroyDescriptor(cufftdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cufftdxDestroyDescriptor
    _check_or_init_mathdx()
    if __cufftdxDestroyDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxDestroyDescriptor is not found")
    return (<commondxStatusType (*)(cufftdxDescriptor) noexcept nogil>__cufftdxDestroyDescriptor)(
        handle)


cdef const char* _cufftdxOperatorTypeToStr(cufftdxOperatorType op) except?NULL nogil:
    global __cufftdxOperatorTypeToStr
    _check_or_init_mathdx()
    if __cufftdxOperatorTypeToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxOperatorTypeToStr is not found")
    return (<const char* (*)(cufftdxOperatorType) noexcept nogil>__cufftdxOperatorTypeToStr)(
        op)


cdef const char* _cufftdxTraitTypeToStr(cufftdxTraitType op) except?NULL nogil:
    global __cufftdxTraitTypeToStr
    _check_or_init_mathdx()
    if __cufftdxTraitTypeToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cufftdxTraitTypeToStr is not found")
    return (<const char* (*)(cufftdxTraitType) noexcept nogil>__cufftdxTraitTypeToStr)(
        op)


cdef commondxStatusType _cusolverdxCreateDescriptor(cusolverdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxCreateDescriptor
    _check_or_init_mathdx()
    if __cusolverdxCreateDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxCreateDescriptor is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor*) noexcept nogil>__cusolverdxCreateDescriptor)(
        handle)


cdef commondxStatusType _cusolverdxSetOptionStr(cusolverdxDescriptor handle, commondxOption opt, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxSetOptionStr
    _check_or_init_mathdx()
    if __cusolverdxSetOptionStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxSetOptionStr is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, commondxOption, const char*) noexcept nogil>__cusolverdxSetOptionStr)(
        handle, opt, value)


cdef commondxStatusType _cusolverdxSetOperatorInt64(cusolverdxDescriptor handle, cusolverdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxSetOperatorInt64
    _check_or_init_mathdx()
    if __cusolverdxSetOperatorInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxSetOperatorInt64 is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, cusolverdxOperatorType, long long int) noexcept nogil>__cusolverdxSetOperatorInt64)(
        handle, op, value)


cdef commondxStatusType _cusolverdxSetOperatorInt64s(cusolverdxDescriptor handle, cusolverdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxSetOperatorInt64s
    _check_or_init_mathdx()
    if __cusolverdxSetOperatorInt64s == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxSetOperatorInt64s is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, cusolverdxOperatorType, size_t, const long long int*) noexcept nogil>__cusolverdxSetOperatorInt64s)(
        handle, op, count, array)


cdef commondxStatusType _cusolverdxGetLTOIRSize(cusolverdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxGetLTOIRSize
    _check_or_init_mathdx()
    if __cusolverdxGetLTOIRSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxGetLTOIRSize is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, size_t*) noexcept nogil>__cusolverdxGetLTOIRSize)(
        handle, lto_size)


cdef commondxStatusType _cusolverdxGetLTOIR(cusolverdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxGetLTOIR
    _check_or_init_mathdx()
    if __cusolverdxGetLTOIR == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxGetLTOIR is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, size_t, void*) noexcept nogil>__cusolverdxGetLTOIR)(
        handle, size, lto)


cdef commondxStatusType _cusolverdxGetUniversalFATBINSize(cusolverdxDescriptor handle, size_t* fatbin_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxGetUniversalFATBINSize
    _check_or_init_mathdx()
    if __cusolverdxGetUniversalFATBINSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxGetUniversalFATBINSize is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, size_t*) noexcept nogil>__cusolverdxGetUniversalFATBINSize)(
        handle, fatbin_size)


cdef commondxStatusType _cusolverdxGetUniversalFATBIN(cusolverdxDescriptor handle, size_t fatbin_size, void* fatbin) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxGetUniversalFATBIN
    _check_or_init_mathdx()
    if __cusolverdxGetUniversalFATBIN == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxGetUniversalFATBIN is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, size_t, void*) noexcept nogil>__cusolverdxGetUniversalFATBIN)(
        handle, fatbin_size, fatbin)


cdef commondxStatusType _cusolverdxGetTraitStrSize(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxGetTraitStrSize
    _check_or_init_mathdx()
    if __cusolverdxGetTraitStrSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxGetTraitStrSize is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, cusolverdxTraitType, size_t*) noexcept nogil>__cusolverdxGetTraitStrSize)(
        handle, trait, size)


cdef commondxStatusType _cusolverdxGetTraitStr(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxGetTraitStr
    _check_or_init_mathdx()
    if __cusolverdxGetTraitStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxGetTraitStr is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, cusolverdxTraitType, size_t, char*) noexcept nogil>__cusolverdxGetTraitStr)(
        handle, trait, size, value)


cdef commondxStatusType _cusolverdxGetTraitInt64(cusolverdxDescriptor handle, cusolverdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxGetTraitInt64
    _check_or_init_mathdx()
    if __cusolverdxGetTraitInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxGetTraitInt64 is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, cusolverdxTraitType, long long int*) noexcept nogil>__cusolverdxGetTraitInt64)(
        handle, trait, value)


cdef commondxStatusType _cusolverdxFinalizeCode(commondxCode code, cusolverdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxFinalizeCode
    _check_or_init_mathdx()
    if __cusolverdxFinalizeCode == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxFinalizeCode is not found")
    return (<commondxStatusType (*)(commondxCode, cusolverdxDescriptor) noexcept nogil>__cusolverdxFinalizeCode)(
        code, handle)


cdef commondxStatusType _cusolverdxDestroyDescriptor(cusolverdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxDestroyDescriptor
    _check_or_init_mathdx()
    if __cusolverdxDestroyDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxDestroyDescriptor is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor) noexcept nogil>__cusolverdxDestroyDescriptor)(
        handle)


cdef const char* _cusolverdxOperatorTypeToStr(cusolverdxOperatorType op) except?NULL nogil:
    global __cusolverdxOperatorTypeToStr
    _check_or_init_mathdx()
    if __cusolverdxOperatorTypeToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxOperatorTypeToStr is not found")
    return (<const char* (*)(cusolverdxOperatorType) noexcept nogil>__cusolverdxOperatorTypeToStr)(
        op)


cdef const char* _cusolverdxTraitTypeToStr(cusolverdxTraitType trait) except?NULL nogil:
    global __cusolverdxTraitTypeToStr
    _check_or_init_mathdx()
    if __cusolverdxTraitTypeToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxTraitTypeToStr is not found")
    return (<const char* (*)(cusolverdxTraitType) noexcept nogil>__cusolverdxTraitTypeToStr)(
        trait)


cdef commondxStatusType _mathdxGetVersion(int* version) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __mathdxGetVersion
    _check_or_init_mathdx()
    if __mathdxGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function mathdxGetVersion is not found")
    return (<commondxStatusType (*)(int*) noexcept nogil>__mathdxGetVersion)(
        version)


cdef commondxStatusType _mathdxGetVersionEx(int* major, int* minor, int* patch) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __mathdxGetVersionEx
    _check_or_init_mathdx()
    if __mathdxGetVersionEx == NULL:
        with gil:
            raise FunctionNotFoundError("function mathdxGetVersionEx is not found")
    return (<commondxStatusType (*)(int*, int*, int*) noexcept nogil>__mathdxGetVersionEx)(
        major, minor, patch)
