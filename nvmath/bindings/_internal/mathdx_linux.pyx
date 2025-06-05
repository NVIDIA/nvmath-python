# This code was automatically generated with version 0.2.1. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_mathdx_dso_version_suffix

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


cdef void* load_library(const int driver_ver) except* with gil:
    cdef void* handle
    for suffix in get_mathdx_dso_version_suffix(driver_ver):
        so_name = "libmathdx.so" + (f".{suffix}" if suffix else suffix)
        handle = dlopen(so_name.encode(), RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            break
    else:
        err_msg = dlerror()
        raise RuntimeError(f'Failed to dlopen libmathdx ({err_msg.decode()})')
    return handle


cdef int _check_or_init_mathdx() except -1 nogil:
    global __py_mathdx_init
    if __py_mathdx_init:
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
    global __commondxCreateCode
    __commondxCreateCode = dlsym(RTLD_DEFAULT, 'commondxCreateCode')
    if __commondxCreateCode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __commondxCreateCode = dlsym(handle, 'commondxCreateCode')

    global __commondxSetCodeOptionInt64
    __commondxSetCodeOptionInt64 = dlsym(RTLD_DEFAULT, 'commondxSetCodeOptionInt64')
    if __commondxSetCodeOptionInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __commondxSetCodeOptionInt64 = dlsym(handle, 'commondxSetCodeOptionInt64')

    global __commondxGetCodeOptionInt64
    __commondxGetCodeOptionInt64 = dlsym(RTLD_DEFAULT, 'commondxGetCodeOptionInt64')
    if __commondxGetCodeOptionInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __commondxGetCodeOptionInt64 = dlsym(handle, 'commondxGetCodeOptionInt64')

    global __commondxGetCodeOptionsInt64s
    __commondxGetCodeOptionsInt64s = dlsym(RTLD_DEFAULT, 'commondxGetCodeOptionsInt64s')
    if __commondxGetCodeOptionsInt64s == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __commondxGetCodeOptionsInt64s = dlsym(handle, 'commondxGetCodeOptionsInt64s')

    global __commondxGetCodeLTOIRSize
    __commondxGetCodeLTOIRSize = dlsym(RTLD_DEFAULT, 'commondxGetCodeLTOIRSize')
    if __commondxGetCodeLTOIRSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __commondxGetCodeLTOIRSize = dlsym(handle, 'commondxGetCodeLTOIRSize')

    global __commondxGetCodeLTOIR
    __commondxGetCodeLTOIR = dlsym(RTLD_DEFAULT, 'commondxGetCodeLTOIR')
    if __commondxGetCodeLTOIR == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __commondxGetCodeLTOIR = dlsym(handle, 'commondxGetCodeLTOIR')

    global __commondxDestroyCode
    __commondxDestroyCode = dlsym(RTLD_DEFAULT, 'commondxDestroyCode')
    if __commondxDestroyCode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __commondxDestroyCode = dlsym(handle, 'commondxDestroyCode')

    global __commondxStatusToStr
    __commondxStatusToStr = dlsym(RTLD_DEFAULT, 'commondxStatusToStr')
    if __commondxStatusToStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __commondxStatusToStr = dlsym(handle, 'commondxStatusToStr')

    global __cublasdxCreateDescriptor
    __cublasdxCreateDescriptor = dlsym(RTLD_DEFAULT, 'cublasdxCreateDescriptor')
    if __cublasdxCreateDescriptor == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxCreateDescriptor = dlsym(handle, 'cublasdxCreateDescriptor')

    global __cublasdxSetOptionStr
    __cublasdxSetOptionStr = dlsym(RTLD_DEFAULT, 'cublasdxSetOptionStr')
    if __cublasdxSetOptionStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxSetOptionStr = dlsym(handle, 'cublasdxSetOptionStr')

    global __cublasdxSetOperatorInt64
    __cublasdxSetOperatorInt64 = dlsym(RTLD_DEFAULT, 'cublasdxSetOperatorInt64')
    if __cublasdxSetOperatorInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxSetOperatorInt64 = dlsym(handle, 'cublasdxSetOperatorInt64')

    global __cublasdxSetOperatorInt64s
    __cublasdxSetOperatorInt64s = dlsym(RTLD_DEFAULT, 'cublasdxSetOperatorInt64s')
    if __cublasdxSetOperatorInt64s == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxSetOperatorInt64s = dlsym(handle, 'cublasdxSetOperatorInt64s')

    global __cublasdxBindTensor
    __cublasdxBindTensor = dlsym(RTLD_DEFAULT, 'cublasdxBindTensor')
    if __cublasdxBindTensor == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxBindTensor = dlsym(handle, 'cublasdxBindTensor')

    global __cublasdxSetTensorOptionInt64
    __cublasdxSetTensorOptionInt64 = dlsym(RTLD_DEFAULT, 'cublasdxSetTensorOptionInt64')
    if __cublasdxSetTensorOptionInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxSetTensorOptionInt64 = dlsym(handle, 'cublasdxSetTensorOptionInt64')

    global __cublasdxFinalizeTensors
    __cublasdxFinalizeTensors = dlsym(RTLD_DEFAULT, 'cublasdxFinalizeTensors')
    if __cublasdxFinalizeTensors == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxFinalizeTensors = dlsym(handle, 'cublasdxFinalizeTensors')

    global __cublasdxGetTensorTraitInt64
    __cublasdxGetTensorTraitInt64 = dlsym(RTLD_DEFAULT, 'cublasdxGetTensorTraitInt64')
    if __cublasdxGetTensorTraitInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetTensorTraitInt64 = dlsym(handle, 'cublasdxGetTensorTraitInt64')

    global __cublasdxGetTensorTraitStrSize
    __cublasdxGetTensorTraitStrSize = dlsym(RTLD_DEFAULT, 'cublasdxGetTensorTraitStrSize')
    if __cublasdxGetTensorTraitStrSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetTensorTraitStrSize = dlsym(handle, 'cublasdxGetTensorTraitStrSize')

    global __cublasdxGetTensorTraitStr
    __cublasdxGetTensorTraitStr = dlsym(RTLD_DEFAULT, 'cublasdxGetTensorTraitStr')
    if __cublasdxGetTensorTraitStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetTensorTraitStr = dlsym(handle, 'cublasdxGetTensorTraitStr')

    global __cublasdxBindDeviceFunction
    __cublasdxBindDeviceFunction = dlsym(RTLD_DEFAULT, 'cublasdxBindDeviceFunction')
    if __cublasdxBindDeviceFunction == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxBindDeviceFunction = dlsym(handle, 'cublasdxBindDeviceFunction')

    global __cublasdxFinalizeDeviceFunctions
    __cublasdxFinalizeDeviceFunctions = dlsym(RTLD_DEFAULT, 'cublasdxFinalizeDeviceFunctions')
    if __cublasdxFinalizeDeviceFunctions == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxFinalizeDeviceFunctions = dlsym(handle, 'cublasdxFinalizeDeviceFunctions')

    global __cublasdxGetDeviceFunctionTraitStrSize
    __cublasdxGetDeviceFunctionTraitStrSize = dlsym(RTLD_DEFAULT, 'cublasdxGetDeviceFunctionTraitStrSize')
    if __cublasdxGetDeviceFunctionTraitStrSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetDeviceFunctionTraitStrSize = dlsym(handle, 'cublasdxGetDeviceFunctionTraitStrSize')

    global __cublasdxGetDeviceFunctionTraitStr
    __cublasdxGetDeviceFunctionTraitStr = dlsym(RTLD_DEFAULT, 'cublasdxGetDeviceFunctionTraitStr')
    if __cublasdxGetDeviceFunctionTraitStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetDeviceFunctionTraitStr = dlsym(handle, 'cublasdxGetDeviceFunctionTraitStr')

    global __cublasdxGetLTOIRSize
    __cublasdxGetLTOIRSize = dlsym(RTLD_DEFAULT, 'cublasdxGetLTOIRSize')
    if __cublasdxGetLTOIRSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetLTOIRSize = dlsym(handle, 'cublasdxGetLTOIRSize')

    global __cublasdxGetLTOIR
    __cublasdxGetLTOIR = dlsym(RTLD_DEFAULT, 'cublasdxGetLTOIR')
    if __cublasdxGetLTOIR == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetLTOIR = dlsym(handle, 'cublasdxGetLTOIR')

    global __cublasdxGetTraitStrSize
    __cublasdxGetTraitStrSize = dlsym(RTLD_DEFAULT, 'cublasdxGetTraitStrSize')
    if __cublasdxGetTraitStrSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetTraitStrSize = dlsym(handle, 'cublasdxGetTraitStrSize')

    global __cublasdxGetTraitStr
    __cublasdxGetTraitStr = dlsym(RTLD_DEFAULT, 'cublasdxGetTraitStr')
    if __cublasdxGetTraitStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetTraitStr = dlsym(handle, 'cublasdxGetTraitStr')

    global __cublasdxGetTraitInt64
    __cublasdxGetTraitInt64 = dlsym(RTLD_DEFAULT, 'cublasdxGetTraitInt64')
    if __cublasdxGetTraitInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetTraitInt64 = dlsym(handle, 'cublasdxGetTraitInt64')

    global __cublasdxGetTraitInt64s
    __cublasdxGetTraitInt64s = dlsym(RTLD_DEFAULT, 'cublasdxGetTraitInt64s')
    if __cublasdxGetTraitInt64s == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxGetTraitInt64s = dlsym(handle, 'cublasdxGetTraitInt64s')

    global __cublasdxOperatorTypeToStr
    __cublasdxOperatorTypeToStr = dlsym(RTLD_DEFAULT, 'cublasdxOperatorTypeToStr')
    if __cublasdxOperatorTypeToStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxOperatorTypeToStr = dlsym(handle, 'cublasdxOperatorTypeToStr')

    global __cublasdxTraitTypeToStr
    __cublasdxTraitTypeToStr = dlsym(RTLD_DEFAULT, 'cublasdxTraitTypeToStr')
    if __cublasdxTraitTypeToStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxTraitTypeToStr = dlsym(handle, 'cublasdxTraitTypeToStr')

    global __cublasdxFinalizeCode
    __cublasdxFinalizeCode = dlsym(RTLD_DEFAULT, 'cublasdxFinalizeCode')
    if __cublasdxFinalizeCode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxFinalizeCode = dlsym(handle, 'cublasdxFinalizeCode')

    global __cublasdxDestroyDescriptor
    __cublasdxDestroyDescriptor = dlsym(RTLD_DEFAULT, 'cublasdxDestroyDescriptor')
    if __cublasdxDestroyDescriptor == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasdxDestroyDescriptor = dlsym(handle, 'cublasdxDestroyDescriptor')

    global __cufftdxCreateDescriptor
    __cufftdxCreateDescriptor = dlsym(RTLD_DEFAULT, 'cufftdxCreateDescriptor')
    if __cufftdxCreateDescriptor == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxCreateDescriptor = dlsym(handle, 'cufftdxCreateDescriptor')

    global __cufftdxSetOptionStr
    __cufftdxSetOptionStr = dlsym(RTLD_DEFAULT, 'cufftdxSetOptionStr')
    if __cufftdxSetOptionStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxSetOptionStr = dlsym(handle, 'cufftdxSetOptionStr')

    global __cufftdxGetKnobInt64Size
    __cufftdxGetKnobInt64Size = dlsym(RTLD_DEFAULT, 'cufftdxGetKnobInt64Size')
    if __cufftdxGetKnobInt64Size == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxGetKnobInt64Size = dlsym(handle, 'cufftdxGetKnobInt64Size')

    global __cufftdxGetKnobInt64s
    __cufftdxGetKnobInt64s = dlsym(RTLD_DEFAULT, 'cufftdxGetKnobInt64s')
    if __cufftdxGetKnobInt64s == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxGetKnobInt64s = dlsym(handle, 'cufftdxGetKnobInt64s')

    global __cufftdxSetOperatorInt64
    __cufftdxSetOperatorInt64 = dlsym(RTLD_DEFAULT, 'cufftdxSetOperatorInt64')
    if __cufftdxSetOperatorInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxSetOperatorInt64 = dlsym(handle, 'cufftdxSetOperatorInt64')

    global __cufftdxSetOperatorInt64s
    __cufftdxSetOperatorInt64s = dlsym(RTLD_DEFAULT, 'cufftdxSetOperatorInt64s')
    if __cufftdxSetOperatorInt64s == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxSetOperatorInt64s = dlsym(handle, 'cufftdxSetOperatorInt64s')

    global __cufftdxGetLTOIRSize
    __cufftdxGetLTOIRSize = dlsym(RTLD_DEFAULT, 'cufftdxGetLTOIRSize')
    if __cufftdxGetLTOIRSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxGetLTOIRSize = dlsym(handle, 'cufftdxGetLTOIRSize')

    global __cufftdxGetLTOIR
    __cufftdxGetLTOIR = dlsym(RTLD_DEFAULT, 'cufftdxGetLTOIR')
    if __cufftdxGetLTOIR == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxGetLTOIR = dlsym(handle, 'cufftdxGetLTOIR')

    global __cufftdxGetTraitStrSize
    __cufftdxGetTraitStrSize = dlsym(RTLD_DEFAULT, 'cufftdxGetTraitStrSize')
    if __cufftdxGetTraitStrSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxGetTraitStrSize = dlsym(handle, 'cufftdxGetTraitStrSize')

    global __cufftdxGetTraitStr
    __cufftdxGetTraitStr = dlsym(RTLD_DEFAULT, 'cufftdxGetTraitStr')
    if __cufftdxGetTraitStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxGetTraitStr = dlsym(handle, 'cufftdxGetTraitStr')

    global __cufftdxGetTraitInt64
    __cufftdxGetTraitInt64 = dlsym(RTLD_DEFAULT, 'cufftdxGetTraitInt64')
    if __cufftdxGetTraitInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxGetTraitInt64 = dlsym(handle, 'cufftdxGetTraitInt64')

    global __cufftdxGetTraitInt64s
    __cufftdxGetTraitInt64s = dlsym(RTLD_DEFAULT, 'cufftdxGetTraitInt64s')
    if __cufftdxGetTraitInt64s == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxGetTraitInt64s = dlsym(handle, 'cufftdxGetTraitInt64s')

    global __cufftdxGetTraitCommondxDataType
    __cufftdxGetTraitCommondxDataType = dlsym(RTLD_DEFAULT, 'cufftdxGetTraitCommondxDataType')
    if __cufftdxGetTraitCommondxDataType == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxGetTraitCommondxDataType = dlsym(handle, 'cufftdxGetTraitCommondxDataType')

    global __cufftdxFinalizeCode
    __cufftdxFinalizeCode = dlsym(RTLD_DEFAULT, 'cufftdxFinalizeCode')
    if __cufftdxFinalizeCode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxFinalizeCode = dlsym(handle, 'cufftdxFinalizeCode')

    global __cufftdxDestroyDescriptor
    __cufftdxDestroyDescriptor = dlsym(RTLD_DEFAULT, 'cufftdxDestroyDescriptor')
    if __cufftdxDestroyDescriptor == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxDestroyDescriptor = dlsym(handle, 'cufftdxDestroyDescriptor')

    global __cufftdxOperatorTypeToStr
    __cufftdxOperatorTypeToStr = dlsym(RTLD_DEFAULT, 'cufftdxOperatorTypeToStr')
    if __cufftdxOperatorTypeToStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxOperatorTypeToStr = dlsym(handle, 'cufftdxOperatorTypeToStr')

    global __cufftdxTraitTypeToStr
    __cufftdxTraitTypeToStr = dlsym(RTLD_DEFAULT, 'cufftdxTraitTypeToStr')
    if __cufftdxTraitTypeToStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cufftdxTraitTypeToStr = dlsym(handle, 'cufftdxTraitTypeToStr')

    global __cusolverdxCreateDescriptor
    __cusolverdxCreateDescriptor = dlsym(RTLD_DEFAULT, 'cusolverdxCreateDescriptor')
    if __cusolverdxCreateDescriptor == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxCreateDescriptor = dlsym(handle, 'cusolverdxCreateDescriptor')

    global __cusolverdxSetOptionStr
    __cusolverdxSetOptionStr = dlsym(RTLD_DEFAULT, 'cusolverdxSetOptionStr')
    if __cusolverdxSetOptionStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxSetOptionStr = dlsym(handle, 'cusolverdxSetOptionStr')

    global __cusolverdxSetOperatorInt64
    __cusolverdxSetOperatorInt64 = dlsym(RTLD_DEFAULT, 'cusolverdxSetOperatorInt64')
    if __cusolverdxSetOperatorInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxSetOperatorInt64 = dlsym(handle, 'cusolverdxSetOperatorInt64')

    global __cusolverdxSetOperatorInt64s
    __cusolverdxSetOperatorInt64s = dlsym(RTLD_DEFAULT, 'cusolverdxSetOperatorInt64s')
    if __cusolverdxSetOperatorInt64s == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxSetOperatorInt64s = dlsym(handle, 'cusolverdxSetOperatorInt64s')

    global __cusolverdxGetLTOIRSize
    __cusolverdxGetLTOIRSize = dlsym(RTLD_DEFAULT, 'cusolverdxGetLTOIRSize')
    if __cusolverdxGetLTOIRSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxGetLTOIRSize = dlsym(handle, 'cusolverdxGetLTOIRSize')

    global __cusolverdxGetLTOIR
    __cusolverdxGetLTOIR = dlsym(RTLD_DEFAULT, 'cusolverdxGetLTOIR')
    if __cusolverdxGetLTOIR == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxGetLTOIR = dlsym(handle, 'cusolverdxGetLTOIR')

    global __cusolverdxGetUniversalFATBINSize
    __cusolverdxGetUniversalFATBINSize = dlsym(RTLD_DEFAULT, 'cusolverdxGetUniversalFATBINSize')
    if __cusolverdxGetUniversalFATBINSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxGetUniversalFATBINSize = dlsym(handle, 'cusolverdxGetUniversalFATBINSize')

    global __cusolverdxGetUniversalFATBIN
    __cusolverdxGetUniversalFATBIN = dlsym(RTLD_DEFAULT, 'cusolverdxGetUniversalFATBIN')
    if __cusolverdxGetUniversalFATBIN == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxGetUniversalFATBIN = dlsym(handle, 'cusolverdxGetUniversalFATBIN')

    global __cusolverdxGetTraitStrSize
    __cusolverdxGetTraitStrSize = dlsym(RTLD_DEFAULT, 'cusolverdxGetTraitStrSize')
    if __cusolverdxGetTraitStrSize == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxGetTraitStrSize = dlsym(handle, 'cusolverdxGetTraitStrSize')

    global __cusolverdxGetTraitStr
    __cusolverdxGetTraitStr = dlsym(RTLD_DEFAULT, 'cusolverdxGetTraitStr')
    if __cusolverdxGetTraitStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxGetTraitStr = dlsym(handle, 'cusolverdxGetTraitStr')

    global __cusolverdxGetTraitInt64
    __cusolverdxGetTraitInt64 = dlsym(RTLD_DEFAULT, 'cusolverdxGetTraitInt64')
    if __cusolverdxGetTraitInt64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxGetTraitInt64 = dlsym(handle, 'cusolverdxGetTraitInt64')

    global __cusolverdxFinalizeCode
    __cusolverdxFinalizeCode = dlsym(RTLD_DEFAULT, 'cusolverdxFinalizeCode')
    if __cusolverdxFinalizeCode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxFinalizeCode = dlsym(handle, 'cusolverdxFinalizeCode')

    global __cusolverdxDestroyDescriptor
    __cusolverdxDestroyDescriptor = dlsym(RTLD_DEFAULT, 'cusolverdxDestroyDescriptor')
    if __cusolverdxDestroyDescriptor == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxDestroyDescriptor = dlsym(handle, 'cusolverdxDestroyDescriptor')

    global __cusolverdxOperatorTypeToStr
    __cusolverdxOperatorTypeToStr = dlsym(RTLD_DEFAULT, 'cusolverdxOperatorTypeToStr')
    if __cusolverdxOperatorTypeToStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxOperatorTypeToStr = dlsym(handle, 'cusolverdxOperatorTypeToStr')

    global __cusolverdxTraitTypeToStr
    __cusolverdxTraitTypeToStr = dlsym(RTLD_DEFAULT, 'cusolverdxTraitTypeToStr')
    if __cusolverdxTraitTypeToStr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cusolverdxTraitTypeToStr = dlsym(handle, 'cusolverdxTraitTypeToStr')

    global __mathdxGetVersion
    __mathdxGetVersion = dlsym(RTLD_DEFAULT, 'mathdxGetVersion')
    if __mathdxGetVersion == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __mathdxGetVersion = dlsym(handle, 'mathdxGetVersion')

    global __mathdxGetVersionEx
    __mathdxGetVersionEx = dlsym(RTLD_DEFAULT, 'mathdxGetVersionEx')
    if __mathdxGetVersionEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __mathdxGetVersionEx = dlsym(handle, 'mathdxGetVersionEx')

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
