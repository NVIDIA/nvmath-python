# This code was automatically generated across versions from 0.3.1 to 0.3.2, generator version 0.3.1.dev1418+g63712a33a.d20260318. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

import threading

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Extern
###############################################################################

# You must 'from .utils import NotSupportedError' before using this template

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

cdef int get_cuda_version():
    cdef void* handle = NULL
    cdef int err, driver_ver = 0

    # Load driver to check version
    handle = dlopen('libcuda.so.1', RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        err_msg = dlerror()
        raise NotSupportedError(f'CUDA driver is not found ({err_msg.decode()})')
    cuDriverGetVersion = dlsym(handle, "cuDriverGetVersion")
    if cuDriverGetVersion == NULL:
        raise RuntimeError('Did not find cuDriverGetVersion symbol in libcuda.so.1')
    err = (<int (*)(int*) noexcept nogil>cuDriverGetVersion)(&driver_ver)
    if err != 0:
        raise RuntimeError(f'cuDriverGetVersion returned error code {err}')

    return driver_ver



###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_mathdx_init = False

cdef void* __commondxCreateCode = NULL
cdef void* __commondxSetCodeOptionInt64 = NULL
cdef void* __commondxSetCodeOptionInt64s = NULL
cdef void* __commondxSetCodeOptionStr = NULL
cdef void* __commondxGetCodeOptionInt64 = NULL
cdef void* __commondxGetCodeOptionsInt64s = NULL
cdef void* __commondxGetCodeLTOIRSize = NULL
cdef void* __commondxGetCodeLTOIR = NULL
cdef void* __commondxGetCodeNumLTOIRs = NULL
cdef void* __commondxGetCodeLTOIRSizes = NULL
cdef void* __commondxGetCodeLTOIRs = NULL
cdef void* __commondxDestroyCode = NULL
cdef void* __commondxStatusToStr = NULL
cdef void* __commondxGetLastErrorStrSize = NULL
cdef void* __commondxGetLastErrorStr = NULL
cdef void* __mathdxGetVersion = NULL
cdef void* __mathdxGetVersionEx = NULL
cdef void* __cublasdxCreateDescriptor = NULL
cdef void* __cublasdxSetOptionStr = NULL
cdef void* __cublasdxSetOperatorInt64 = NULL
cdef void* __cublasdxSetOperatorInt64s = NULL
cdef void* __cublasdxCreateTensor = NULL
cdef void* __cublasdxCreateTensorStrided = NULL
cdef void* __cublasdxMakeTensorLike = NULL
cdef void* __cublasdxDestroyTensor = NULL
cdef void* __cublasdxDestroyPipeline = NULL
cdef void* __cublasdxCreateDevicePipeline = NULL
cdef void* __cublasdxCreateTilePipeline = NULL
cdef void* __cublasdxSetTensorOptionInt64 = NULL
cdef void* __cublasdxFinalizeTensors = NULL
cdef void* __cublasdxFinalizePipelines = NULL
cdef void* __cublasdxFinalize = NULL
cdef void* __cublasdxGetTensorTraitInt64 = NULL
cdef void* __cublasdxGetPipelineTraitInt64 = NULL
cdef void* __cublasdxGetPipelineTraitInt64s = NULL
cdef void* __cublasdxGetTensorTraitStrSize = NULL
cdef void* __cublasdxGetPipelineTraitStrSize = NULL
cdef void* __cublasdxGetTensorTraitStr = NULL
cdef void* __cublasdxGetPipelineTraitStr = NULL
cdef void* __cublasdxCreateDeviceFunction = NULL
cdef void* __cublasdxDestroyDeviceFunction = NULL
cdef void* __cublasdxCreateDeviceFunctionWithPipelines = NULL
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
cdef void* __cusolverdxGetTraitInt64s = NULL
cdef void* __cusolverdxFinalizeCode = NULL
cdef void* __cusolverdxDestroyDescriptor = NULL
cdef void* __cusolverdxOperatorTypeToStr = NULL
cdef void* __cusolverdxTraitTypeToStr = NULL
cdef void* __curanddxGetVersion = NULL
cdef void* __curanddxCreateDescriptor = NULL
cdef void* __curanddxSetOptionStr = NULL
cdef void* __curanddxSetOperatorInt64 = NULL
cdef void* __curanddxSetOperatorDoubles = NULL
cdef void* __curanddxGetTraitStrSize = NULL
cdef void* __curanddxGetTraitStr = NULL
cdef void* __curanddxGetTraitInt64 = NULL
cdef void* __curanddxFinalizeCode = NULL
cdef void* __curanddxDestroyDescriptor = NULL
cdef void* __curanddxOperatorTypeToStr = NULL
cdef void* __curanddxDistributionToStr = NULL
cdef void* __curanddxGeneratorToStr = NULL
cdef void* __curanddxGenerateMethodToStr = NULL
cdef void* __curanddxNormalMethodToStr = NULL
cdef void* __curanddxTraitTypeToStr = NULL
cdef void* __commondxGetCodePTXSize = NULL
cdef void* __commondxGetCodePTX = NULL
cdef void* __cusolverdxGetTraitCommondxDataTypes = NULL


cdef void* load_library(const int driver_ver) except* with gil:
    load_nvidia_dynamic_lib("nvrtc")
    cdef uintptr_t handle = load_nvidia_dynamic_lib("mathdx")._handle_uint
    return <void*>handle


cdef int _check_or_init_mathdx() except -1 nogil:
    global __py_mathdx_init
    if __py_mathdx_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_mathdx_init:
            return 0

        driver_ver = get_cuda_version()

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

        global __commondxSetCodeOptionInt64s
        __commondxSetCodeOptionInt64s = dlsym(RTLD_DEFAULT, 'commondxSetCodeOptionInt64s')
        if __commondxSetCodeOptionInt64s == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __commondxSetCodeOptionInt64s = dlsym(handle, 'commondxSetCodeOptionInt64s')

        global __commondxSetCodeOptionStr
        __commondxSetCodeOptionStr = dlsym(RTLD_DEFAULT, 'commondxSetCodeOptionStr')
        if __commondxSetCodeOptionStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __commondxSetCodeOptionStr = dlsym(handle, 'commondxSetCodeOptionStr')

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

        global __commondxGetCodeNumLTOIRs
        __commondxGetCodeNumLTOIRs = dlsym(RTLD_DEFAULT, 'commondxGetCodeNumLTOIRs')
        if __commondxGetCodeNumLTOIRs == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __commondxGetCodeNumLTOIRs = dlsym(handle, 'commondxGetCodeNumLTOIRs')

        global __commondxGetCodeLTOIRSizes
        __commondxGetCodeLTOIRSizes = dlsym(RTLD_DEFAULT, 'commondxGetCodeLTOIRSizes')
        if __commondxGetCodeLTOIRSizes == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __commondxGetCodeLTOIRSizes = dlsym(handle, 'commondxGetCodeLTOIRSizes')

        global __commondxGetCodeLTOIRs
        __commondxGetCodeLTOIRs = dlsym(RTLD_DEFAULT, 'commondxGetCodeLTOIRs')
        if __commondxGetCodeLTOIRs == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __commondxGetCodeLTOIRs = dlsym(handle, 'commondxGetCodeLTOIRs')

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

        global __commondxGetLastErrorStrSize
        __commondxGetLastErrorStrSize = dlsym(RTLD_DEFAULT, 'commondxGetLastErrorStrSize')
        if __commondxGetLastErrorStrSize == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __commondxGetLastErrorStrSize = dlsym(handle, 'commondxGetLastErrorStrSize')

        global __commondxGetLastErrorStr
        __commondxGetLastErrorStr = dlsym(RTLD_DEFAULT, 'commondxGetLastErrorStr')
        if __commondxGetLastErrorStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __commondxGetLastErrorStr = dlsym(handle, 'commondxGetLastErrorStr')

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

        global __cublasdxCreateTensor
        __cublasdxCreateTensor = dlsym(RTLD_DEFAULT, 'cublasdxCreateTensor')
        if __cublasdxCreateTensor == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxCreateTensor = dlsym(handle, 'cublasdxCreateTensor')

        global __cublasdxCreateTensorStrided
        __cublasdxCreateTensorStrided = dlsym(RTLD_DEFAULT, 'cublasdxCreateTensorStrided')
        if __cublasdxCreateTensorStrided == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxCreateTensorStrided = dlsym(handle, 'cublasdxCreateTensorStrided')

        global __cublasdxMakeTensorLike
        __cublasdxMakeTensorLike = dlsym(RTLD_DEFAULT, 'cublasdxMakeTensorLike')
        if __cublasdxMakeTensorLike == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxMakeTensorLike = dlsym(handle, 'cublasdxMakeTensorLike')

        global __cublasdxDestroyTensor
        __cublasdxDestroyTensor = dlsym(RTLD_DEFAULT, 'cublasdxDestroyTensor')
        if __cublasdxDestroyTensor == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxDestroyTensor = dlsym(handle, 'cublasdxDestroyTensor')

        global __cublasdxDestroyPipeline
        __cublasdxDestroyPipeline = dlsym(RTLD_DEFAULT, 'cublasdxDestroyPipeline')
        if __cublasdxDestroyPipeline == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxDestroyPipeline = dlsym(handle, 'cublasdxDestroyPipeline')

        global __cublasdxCreateDevicePipeline
        __cublasdxCreateDevicePipeline = dlsym(RTLD_DEFAULT, 'cublasdxCreateDevicePipeline')
        if __cublasdxCreateDevicePipeline == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxCreateDevicePipeline = dlsym(handle, 'cublasdxCreateDevicePipeline')

        global __cublasdxCreateTilePipeline
        __cublasdxCreateTilePipeline = dlsym(RTLD_DEFAULT, 'cublasdxCreateTilePipeline')
        if __cublasdxCreateTilePipeline == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxCreateTilePipeline = dlsym(handle, 'cublasdxCreateTilePipeline')

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

        global __cublasdxFinalizePipelines
        __cublasdxFinalizePipelines = dlsym(RTLD_DEFAULT, 'cublasdxFinalizePipelines')
        if __cublasdxFinalizePipelines == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxFinalizePipelines = dlsym(handle, 'cublasdxFinalizePipelines')

        global __cublasdxFinalize
        __cublasdxFinalize = dlsym(RTLD_DEFAULT, 'cublasdxFinalize')
        if __cublasdxFinalize == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxFinalize = dlsym(handle, 'cublasdxFinalize')

        global __cublasdxGetTensorTraitInt64
        __cublasdxGetTensorTraitInt64 = dlsym(RTLD_DEFAULT, 'cublasdxGetTensorTraitInt64')
        if __cublasdxGetTensorTraitInt64 == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxGetTensorTraitInt64 = dlsym(handle, 'cublasdxGetTensorTraitInt64')

        global __cublasdxGetPipelineTraitInt64
        __cublasdxGetPipelineTraitInt64 = dlsym(RTLD_DEFAULT, 'cublasdxGetPipelineTraitInt64')
        if __cublasdxGetPipelineTraitInt64 == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxGetPipelineTraitInt64 = dlsym(handle, 'cublasdxGetPipelineTraitInt64')

        global __cublasdxGetPipelineTraitInt64s
        __cublasdxGetPipelineTraitInt64s = dlsym(RTLD_DEFAULT, 'cublasdxGetPipelineTraitInt64s')
        if __cublasdxGetPipelineTraitInt64s == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxGetPipelineTraitInt64s = dlsym(handle, 'cublasdxGetPipelineTraitInt64s')

        global __cublasdxGetTensorTraitStrSize
        __cublasdxGetTensorTraitStrSize = dlsym(RTLD_DEFAULT, 'cublasdxGetTensorTraitStrSize')
        if __cublasdxGetTensorTraitStrSize == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxGetTensorTraitStrSize = dlsym(handle, 'cublasdxGetTensorTraitStrSize')

        global __cublasdxGetPipelineTraitStrSize
        __cublasdxGetPipelineTraitStrSize = dlsym(RTLD_DEFAULT, 'cublasdxGetPipelineTraitStrSize')
        if __cublasdxGetPipelineTraitStrSize == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxGetPipelineTraitStrSize = dlsym(handle, 'cublasdxGetPipelineTraitStrSize')

        global __cublasdxGetTensorTraitStr
        __cublasdxGetTensorTraitStr = dlsym(RTLD_DEFAULT, 'cublasdxGetTensorTraitStr')
        if __cublasdxGetTensorTraitStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxGetTensorTraitStr = dlsym(handle, 'cublasdxGetTensorTraitStr')

        global __cublasdxGetPipelineTraitStr
        __cublasdxGetPipelineTraitStr = dlsym(RTLD_DEFAULT, 'cublasdxGetPipelineTraitStr')
        if __cublasdxGetPipelineTraitStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxGetPipelineTraitStr = dlsym(handle, 'cublasdxGetPipelineTraitStr')

        global __cublasdxCreateDeviceFunction
        __cublasdxCreateDeviceFunction = dlsym(RTLD_DEFAULT, 'cublasdxCreateDeviceFunction')
        if __cublasdxCreateDeviceFunction == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxCreateDeviceFunction = dlsym(handle, 'cublasdxCreateDeviceFunction')

        global __cublasdxDestroyDeviceFunction
        __cublasdxDestroyDeviceFunction = dlsym(RTLD_DEFAULT, 'cublasdxDestroyDeviceFunction')
        if __cublasdxDestroyDeviceFunction == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxDestroyDeviceFunction = dlsym(handle, 'cublasdxDestroyDeviceFunction')

        global __cublasdxCreateDeviceFunctionWithPipelines
        __cublasdxCreateDeviceFunctionWithPipelines = dlsym(RTLD_DEFAULT, 'cublasdxCreateDeviceFunctionWithPipelines')
        if __cublasdxCreateDeviceFunctionWithPipelines == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cublasdxCreateDeviceFunctionWithPipelines = dlsym(handle, 'cublasdxCreateDeviceFunctionWithPipelines')

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

        global __cusolverdxGetTraitInt64s
        __cusolverdxGetTraitInt64s = dlsym(RTLD_DEFAULT, 'cusolverdxGetTraitInt64s')
        if __cusolverdxGetTraitInt64s == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cusolverdxGetTraitInt64s = dlsym(handle, 'cusolverdxGetTraitInt64s')

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

        global __curanddxGetVersion
        __curanddxGetVersion = dlsym(RTLD_DEFAULT, 'curanddxGetVersion')
        if __curanddxGetVersion == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxGetVersion = dlsym(handle, 'curanddxGetVersion')

        global __curanddxCreateDescriptor
        __curanddxCreateDescriptor = dlsym(RTLD_DEFAULT, 'curanddxCreateDescriptor')
        if __curanddxCreateDescriptor == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxCreateDescriptor = dlsym(handle, 'curanddxCreateDescriptor')

        global __curanddxSetOptionStr
        __curanddxSetOptionStr = dlsym(RTLD_DEFAULT, 'curanddxSetOptionStr')
        if __curanddxSetOptionStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxSetOptionStr = dlsym(handle, 'curanddxSetOptionStr')

        global __curanddxSetOperatorInt64
        __curanddxSetOperatorInt64 = dlsym(RTLD_DEFAULT, 'curanddxSetOperatorInt64')
        if __curanddxSetOperatorInt64 == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxSetOperatorInt64 = dlsym(handle, 'curanddxSetOperatorInt64')

        global __curanddxSetOperatorDoubles
        __curanddxSetOperatorDoubles = dlsym(RTLD_DEFAULT, 'curanddxSetOperatorDoubles')
        if __curanddxSetOperatorDoubles == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxSetOperatorDoubles = dlsym(handle, 'curanddxSetOperatorDoubles')

        global __curanddxGetTraitStrSize
        __curanddxGetTraitStrSize = dlsym(RTLD_DEFAULT, 'curanddxGetTraitStrSize')
        if __curanddxGetTraitStrSize == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxGetTraitStrSize = dlsym(handle, 'curanddxGetTraitStrSize')

        global __curanddxGetTraitStr
        __curanddxGetTraitStr = dlsym(RTLD_DEFAULT, 'curanddxGetTraitStr')
        if __curanddxGetTraitStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxGetTraitStr = dlsym(handle, 'curanddxGetTraitStr')

        global __curanddxGetTraitInt64
        __curanddxGetTraitInt64 = dlsym(RTLD_DEFAULT, 'curanddxGetTraitInt64')
        if __curanddxGetTraitInt64 == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxGetTraitInt64 = dlsym(handle, 'curanddxGetTraitInt64')

        global __curanddxFinalizeCode
        __curanddxFinalizeCode = dlsym(RTLD_DEFAULT, 'curanddxFinalizeCode')
        if __curanddxFinalizeCode == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxFinalizeCode = dlsym(handle, 'curanddxFinalizeCode')

        global __curanddxDestroyDescriptor
        __curanddxDestroyDescriptor = dlsym(RTLD_DEFAULT, 'curanddxDestroyDescriptor')
        if __curanddxDestroyDescriptor == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxDestroyDescriptor = dlsym(handle, 'curanddxDestroyDescriptor')

        global __curanddxOperatorTypeToStr
        __curanddxOperatorTypeToStr = dlsym(RTLD_DEFAULT, 'curanddxOperatorTypeToStr')
        if __curanddxOperatorTypeToStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxOperatorTypeToStr = dlsym(handle, 'curanddxOperatorTypeToStr')

        global __curanddxDistributionToStr
        __curanddxDistributionToStr = dlsym(RTLD_DEFAULT, 'curanddxDistributionToStr')
        if __curanddxDistributionToStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxDistributionToStr = dlsym(handle, 'curanddxDistributionToStr')

        global __curanddxGeneratorToStr
        __curanddxGeneratorToStr = dlsym(RTLD_DEFAULT, 'curanddxGeneratorToStr')
        if __curanddxGeneratorToStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxGeneratorToStr = dlsym(handle, 'curanddxGeneratorToStr')

        global __curanddxGenerateMethodToStr
        __curanddxGenerateMethodToStr = dlsym(RTLD_DEFAULT, 'curanddxGenerateMethodToStr')
        if __curanddxGenerateMethodToStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxGenerateMethodToStr = dlsym(handle, 'curanddxGenerateMethodToStr')

        global __curanddxNormalMethodToStr
        __curanddxNormalMethodToStr = dlsym(RTLD_DEFAULT, 'curanddxNormalMethodToStr')
        if __curanddxNormalMethodToStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxNormalMethodToStr = dlsym(handle, 'curanddxNormalMethodToStr')

        global __curanddxTraitTypeToStr
        __curanddxTraitTypeToStr = dlsym(RTLD_DEFAULT, 'curanddxTraitTypeToStr')
        if __curanddxTraitTypeToStr == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __curanddxTraitTypeToStr = dlsym(handle, 'curanddxTraitTypeToStr')

        global __commondxGetCodePTXSize
        __commondxGetCodePTXSize = dlsym(RTLD_DEFAULT, 'commondxGetCodePTXSize')
        if __commondxGetCodePTXSize == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __commondxGetCodePTXSize = dlsym(handle, 'commondxGetCodePTXSize')

        global __commondxGetCodePTX
        __commondxGetCodePTX = dlsym(RTLD_DEFAULT, 'commondxGetCodePTX')
        if __commondxGetCodePTX == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __commondxGetCodePTX = dlsym(handle, 'commondxGetCodePTX')

        global __cusolverdxGetTraitCommondxDataTypes
        __cusolverdxGetTraitCommondxDataTypes = dlsym(RTLD_DEFAULT, 'cusolverdxGetTraitCommondxDataTypes')
        if __cusolverdxGetTraitCommondxDataTypes == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cusolverdxGetTraitCommondxDataTypes = dlsym(handle, 'cusolverdxGetTraitCommondxDataTypes')

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

    global __commondxSetCodeOptionInt64s
    data["__commondxSetCodeOptionInt64s"] = <intptr_t>__commondxSetCodeOptionInt64s

    global __commondxSetCodeOptionStr
    data["__commondxSetCodeOptionStr"] = <intptr_t>__commondxSetCodeOptionStr

    global __commondxGetCodeOptionInt64
    data["__commondxGetCodeOptionInt64"] = <intptr_t>__commondxGetCodeOptionInt64

    global __commondxGetCodeOptionsInt64s
    data["__commondxGetCodeOptionsInt64s"] = <intptr_t>__commondxGetCodeOptionsInt64s

    global __commondxGetCodeLTOIRSize
    data["__commondxGetCodeLTOIRSize"] = <intptr_t>__commondxGetCodeLTOIRSize

    global __commondxGetCodeLTOIR
    data["__commondxGetCodeLTOIR"] = <intptr_t>__commondxGetCodeLTOIR

    global __commondxGetCodeNumLTOIRs
    data["__commondxGetCodeNumLTOIRs"] = <intptr_t>__commondxGetCodeNumLTOIRs

    global __commondxGetCodeLTOIRSizes
    data["__commondxGetCodeLTOIRSizes"] = <intptr_t>__commondxGetCodeLTOIRSizes

    global __commondxGetCodeLTOIRs
    data["__commondxGetCodeLTOIRs"] = <intptr_t>__commondxGetCodeLTOIRs

    global __commondxDestroyCode
    data["__commondxDestroyCode"] = <intptr_t>__commondxDestroyCode

    global __commondxStatusToStr
    data["__commondxStatusToStr"] = <intptr_t>__commondxStatusToStr

    global __commondxGetLastErrorStrSize
    data["__commondxGetLastErrorStrSize"] = <intptr_t>__commondxGetLastErrorStrSize

    global __commondxGetLastErrorStr
    data["__commondxGetLastErrorStr"] = <intptr_t>__commondxGetLastErrorStr

    global __mathdxGetVersion
    data["__mathdxGetVersion"] = <intptr_t>__mathdxGetVersion

    global __mathdxGetVersionEx
    data["__mathdxGetVersionEx"] = <intptr_t>__mathdxGetVersionEx

    global __cublasdxCreateDescriptor
    data["__cublasdxCreateDescriptor"] = <intptr_t>__cublasdxCreateDescriptor

    global __cublasdxSetOptionStr
    data["__cublasdxSetOptionStr"] = <intptr_t>__cublasdxSetOptionStr

    global __cublasdxSetOperatorInt64
    data["__cublasdxSetOperatorInt64"] = <intptr_t>__cublasdxSetOperatorInt64

    global __cublasdxSetOperatorInt64s
    data["__cublasdxSetOperatorInt64s"] = <intptr_t>__cublasdxSetOperatorInt64s

    global __cublasdxCreateTensor
    data["__cublasdxCreateTensor"] = <intptr_t>__cublasdxCreateTensor

    global __cublasdxCreateTensorStrided
    data["__cublasdxCreateTensorStrided"] = <intptr_t>__cublasdxCreateTensorStrided

    global __cublasdxMakeTensorLike
    data["__cublasdxMakeTensorLike"] = <intptr_t>__cublasdxMakeTensorLike

    global __cublasdxDestroyTensor
    data["__cublasdxDestroyTensor"] = <intptr_t>__cublasdxDestroyTensor

    global __cublasdxDestroyPipeline
    data["__cublasdxDestroyPipeline"] = <intptr_t>__cublasdxDestroyPipeline

    global __cublasdxCreateDevicePipeline
    data["__cublasdxCreateDevicePipeline"] = <intptr_t>__cublasdxCreateDevicePipeline

    global __cublasdxCreateTilePipeline
    data["__cublasdxCreateTilePipeline"] = <intptr_t>__cublasdxCreateTilePipeline

    global __cublasdxSetTensorOptionInt64
    data["__cublasdxSetTensorOptionInt64"] = <intptr_t>__cublasdxSetTensorOptionInt64

    global __cublasdxFinalizeTensors
    data["__cublasdxFinalizeTensors"] = <intptr_t>__cublasdxFinalizeTensors

    global __cublasdxFinalizePipelines
    data["__cublasdxFinalizePipelines"] = <intptr_t>__cublasdxFinalizePipelines

    global __cublasdxFinalize
    data["__cublasdxFinalize"] = <intptr_t>__cublasdxFinalize

    global __cublasdxGetTensorTraitInt64
    data["__cublasdxGetTensorTraitInt64"] = <intptr_t>__cublasdxGetTensorTraitInt64

    global __cublasdxGetPipelineTraitInt64
    data["__cublasdxGetPipelineTraitInt64"] = <intptr_t>__cublasdxGetPipelineTraitInt64

    global __cublasdxGetPipelineTraitInt64s
    data["__cublasdxGetPipelineTraitInt64s"] = <intptr_t>__cublasdxGetPipelineTraitInt64s

    global __cublasdxGetTensorTraitStrSize
    data["__cublasdxGetTensorTraitStrSize"] = <intptr_t>__cublasdxGetTensorTraitStrSize

    global __cublasdxGetPipelineTraitStrSize
    data["__cublasdxGetPipelineTraitStrSize"] = <intptr_t>__cublasdxGetPipelineTraitStrSize

    global __cublasdxGetTensorTraitStr
    data["__cublasdxGetTensorTraitStr"] = <intptr_t>__cublasdxGetTensorTraitStr

    global __cublasdxGetPipelineTraitStr
    data["__cublasdxGetPipelineTraitStr"] = <intptr_t>__cublasdxGetPipelineTraitStr

    global __cublasdxCreateDeviceFunction
    data["__cublasdxCreateDeviceFunction"] = <intptr_t>__cublasdxCreateDeviceFunction

    global __cublasdxDestroyDeviceFunction
    data["__cublasdxDestroyDeviceFunction"] = <intptr_t>__cublasdxDestroyDeviceFunction

    global __cublasdxCreateDeviceFunctionWithPipelines
    data["__cublasdxCreateDeviceFunctionWithPipelines"] = <intptr_t>__cublasdxCreateDeviceFunctionWithPipelines

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

    global __cusolverdxGetTraitInt64s
    data["__cusolverdxGetTraitInt64s"] = <intptr_t>__cusolverdxGetTraitInt64s

    global __cusolverdxFinalizeCode
    data["__cusolverdxFinalizeCode"] = <intptr_t>__cusolverdxFinalizeCode

    global __cusolverdxDestroyDescriptor
    data["__cusolverdxDestroyDescriptor"] = <intptr_t>__cusolverdxDestroyDescriptor

    global __cusolverdxOperatorTypeToStr
    data["__cusolverdxOperatorTypeToStr"] = <intptr_t>__cusolverdxOperatorTypeToStr

    global __cusolverdxTraitTypeToStr
    data["__cusolverdxTraitTypeToStr"] = <intptr_t>__cusolverdxTraitTypeToStr

    global __curanddxGetVersion
    data["__curanddxGetVersion"] = <intptr_t>__curanddxGetVersion

    global __curanddxCreateDescriptor
    data["__curanddxCreateDescriptor"] = <intptr_t>__curanddxCreateDescriptor

    global __curanddxSetOptionStr
    data["__curanddxSetOptionStr"] = <intptr_t>__curanddxSetOptionStr

    global __curanddxSetOperatorInt64
    data["__curanddxSetOperatorInt64"] = <intptr_t>__curanddxSetOperatorInt64

    global __curanddxSetOperatorDoubles
    data["__curanddxSetOperatorDoubles"] = <intptr_t>__curanddxSetOperatorDoubles

    global __curanddxGetTraitStrSize
    data["__curanddxGetTraitStrSize"] = <intptr_t>__curanddxGetTraitStrSize

    global __curanddxGetTraitStr
    data["__curanddxGetTraitStr"] = <intptr_t>__curanddxGetTraitStr

    global __curanddxGetTraitInt64
    data["__curanddxGetTraitInt64"] = <intptr_t>__curanddxGetTraitInt64

    global __curanddxFinalizeCode
    data["__curanddxFinalizeCode"] = <intptr_t>__curanddxFinalizeCode

    global __curanddxDestroyDescriptor
    data["__curanddxDestroyDescriptor"] = <intptr_t>__curanddxDestroyDescriptor

    global __curanddxOperatorTypeToStr
    data["__curanddxOperatorTypeToStr"] = <intptr_t>__curanddxOperatorTypeToStr

    global __curanddxDistributionToStr
    data["__curanddxDistributionToStr"] = <intptr_t>__curanddxDistributionToStr

    global __curanddxGeneratorToStr
    data["__curanddxGeneratorToStr"] = <intptr_t>__curanddxGeneratorToStr

    global __curanddxGenerateMethodToStr
    data["__curanddxGenerateMethodToStr"] = <intptr_t>__curanddxGenerateMethodToStr

    global __curanddxNormalMethodToStr
    data["__curanddxNormalMethodToStr"] = <intptr_t>__curanddxNormalMethodToStr

    global __curanddxTraitTypeToStr
    data["__curanddxTraitTypeToStr"] = <intptr_t>__curanddxTraitTypeToStr

    global __commondxGetCodePTXSize
    data["__commondxGetCodePTXSize"] = <intptr_t>__commondxGetCodePTXSize

    global __commondxGetCodePTX
    data["__commondxGetCodePTX"] = <intptr_t>__commondxGetCodePTX

    global __cusolverdxGetTraitCommondxDataTypes
    data["__cusolverdxGetTraitCommondxDataTypes"] = <intptr_t>__cusolverdxGetTraitCommondxDataTypes

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


cdef commondxStatusType _commondxSetCodeOptionInt64s(commondxCode code, commondxOption option, size_t count, long long int* values) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxSetCodeOptionInt64s
    _check_or_init_mathdx()
    if __commondxSetCodeOptionInt64s == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxSetCodeOptionInt64s is not found")
    return (<commondxStatusType (*)(commondxCode, commondxOption, size_t, long long int*) noexcept nogil>__commondxSetCodeOptionInt64s)(
        code, option, count, values)


cdef commondxStatusType _commondxSetCodeOptionStr(commondxCode code, commondxOption option, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxSetCodeOptionStr
    _check_or_init_mathdx()
    if __commondxSetCodeOptionStr == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxSetCodeOptionStr is not found")
    return (<commondxStatusType (*)(commondxCode, commondxOption, const char*) noexcept nogil>__commondxSetCodeOptionStr)(
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


cdef commondxStatusType _commondxGetCodeNumLTOIRs(commondxCode code, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetCodeNumLTOIRs
    _check_or_init_mathdx()
    if __commondxGetCodeNumLTOIRs == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetCodeNumLTOIRs is not found")
    return (<commondxStatusType (*)(commondxCode, size_t*) noexcept nogil>__commondxGetCodeNumLTOIRs)(
        code, size)


cdef commondxStatusType _commondxGetCodeLTOIRSizes(commondxCode code, size_t size, size_t* out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetCodeLTOIRSizes
    _check_or_init_mathdx()
    if __commondxGetCodeLTOIRSizes == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetCodeLTOIRSizes is not found")
    return (<commondxStatusType (*)(commondxCode, size_t, size_t*) noexcept nogil>__commondxGetCodeLTOIRSizes)(
        code, size, out)


cdef commondxStatusType _commondxGetCodeLTOIRs(commondxCode code, size_t size, void** out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetCodeLTOIRs
    _check_or_init_mathdx()
    if __commondxGetCodeLTOIRs == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetCodeLTOIRs is not found")
    return (<commondxStatusType (*)(commondxCode, size_t, void**) noexcept nogil>__commondxGetCodeLTOIRs)(
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


cdef commondxStatusType _commondxGetLastErrorStrSize(size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetLastErrorStrSize
    _check_or_init_mathdx()
    if __commondxGetLastErrorStrSize == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetLastErrorStrSize is not found")
    return (<commondxStatusType (*)(size_t*) noexcept nogil>__commondxGetLastErrorStrSize)(
        size)


cdef commondxStatusType _commondxGetLastErrorStr(commondxStatusType* code, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetLastErrorStr
    _check_or_init_mathdx()
    if __commondxGetLastErrorStr == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetLastErrorStr is not found")
    return (<commondxStatusType (*)(commondxStatusType*, size_t, char*) noexcept nogil>__commondxGetLastErrorStr)(
        code, size, value)


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


cdef commondxStatusType _cublasdxCreateTensor(cublasdxDescriptor handle, cublasdxTensorType tensor_type, cublasdxTensor* tensor) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxCreateTensor
    _check_or_init_mathdx()
    if __cublasdxCreateTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxCreateTensor is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxTensorType, cublasdxTensor*) noexcept nogil>__cublasdxCreateTensor)(
        handle, tensor_type, tensor)


cdef commondxStatusType _cublasdxCreateTensorStrided(cublasdxMemorySpace memory_space, commondxValueType value_type, void* ptr, long long int rank, long long int* shape, long long int* stride, cublasdxTensor* tensor) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxCreateTensorStrided
    _check_or_init_mathdx()
    if __cublasdxCreateTensorStrided == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxCreateTensorStrided is not found")
    return (<commondxStatusType (*)(cublasdxMemorySpace, commondxValueType, void*, long long int, long long int*, long long int*, cublasdxTensor*) noexcept nogil>__cublasdxCreateTensorStrided)(
        memory_space, value_type, ptr, rank, shape, stride, tensor)


cdef commondxStatusType _cublasdxMakeTensorLike(cublasdxTensor input, commondxValueType value_type, cublasdxTensor* output) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxMakeTensorLike
    _check_or_init_mathdx()
    if __cublasdxMakeTensorLike == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxMakeTensorLike is not found")
    return (<commondxStatusType (*)(cublasdxTensor, commondxValueType, cublasdxTensor*) noexcept nogil>__cublasdxMakeTensorLike)(
        input, value_type, output)


cdef commondxStatusType _cublasdxDestroyTensor(cublasdxTensor tensor) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxDestroyTensor
    _check_or_init_mathdx()
    if __cublasdxDestroyTensor == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxDestroyTensor is not found")
    return (<commondxStatusType (*)(cublasdxTensor) noexcept nogil>__cublasdxDestroyTensor)(
        tensor)


cdef commondxStatusType _cublasdxDestroyPipeline(cublasdxPipeline pipeline) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxDestroyPipeline
    _check_or_init_mathdx()
    if __cublasdxDestroyPipeline == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxDestroyPipeline is not found")
    return (<commondxStatusType (*)(cublasdxPipeline) noexcept nogil>__cublasdxDestroyPipeline)(
        pipeline)


cdef commondxStatusType _cublasdxCreateDevicePipeline(cublasdxDescriptor handle, cublasdxDevicePipelineType device_pipeline_type, long long int pipeline_depth, cublasdxBlockSizeStrategy block_size_strategy, cublasdxTensor tensor_a, cublasdxTensor tensor_b, cublasdxPipeline* device_pipeline) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxCreateDevicePipeline
    _check_or_init_mathdx()
    if __cublasdxCreateDevicePipeline == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxCreateDevicePipeline is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxDevicePipelineType, long long int, cublasdxBlockSizeStrategy, cublasdxTensor, cublasdxTensor, cublasdxPipeline*) noexcept nogil>__cublasdxCreateDevicePipeline)(
        handle, device_pipeline_type, pipeline_depth, block_size_strategy, tensor_a, tensor_b, device_pipeline)


cdef commondxStatusType _cublasdxCreateTilePipeline(cublasdxDescriptor handle, cublasdxTilePipelineType tile_pipeline_type, cublasdxPipeline device_pipeline, cublasdxPipeline* tile_pipeline) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxCreateTilePipeline
    _check_or_init_mathdx()
    if __cublasdxCreateTilePipeline == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxCreateTilePipeline is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxTilePipelineType, cublasdxPipeline, cublasdxPipeline*) noexcept nogil>__cublasdxCreateTilePipeline)(
        handle, tile_pipeline_type, device_pipeline, tile_pipeline)


cdef commondxStatusType _cublasdxSetTensorOptionInt64(cublasdxTensor tensor, cublasdxTensorOption option, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxSetTensorOptionInt64
    _check_or_init_mathdx()
    if __cublasdxSetTensorOptionInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxSetTensorOptionInt64 is not found")
    return (<commondxStatusType (*)(cublasdxTensor, cublasdxTensorOption, long long int) noexcept nogil>__cublasdxSetTensorOptionInt64)(
        tensor, option, value)


cdef commondxStatusType _cublasdxFinalizeTensors(size_t count, const cublasdxTensor* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxFinalizeTensors
    _check_or_init_mathdx()
    if __cublasdxFinalizeTensors == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxFinalizeTensors is not found")
    return (<commondxStatusType (*)(size_t, const cublasdxTensor*) noexcept nogil>__cublasdxFinalizeTensors)(
        count, array)


cdef commondxStatusType _cublasdxFinalizePipelines(size_t count, const cublasdxPipeline* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxFinalizePipelines
    _check_or_init_mathdx()
    if __cublasdxFinalizePipelines == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxFinalizePipelines is not found")
    return (<commondxStatusType (*)(size_t, const cublasdxPipeline*) noexcept nogil>__cublasdxFinalizePipelines)(
        count, array)


cdef commondxStatusType _cublasdxFinalize(size_t countTensors, const cublasdxTensor* tensors, size_t countPipelines, const cublasdxPipeline* pipelines) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxFinalize
    _check_or_init_mathdx()
    if __cublasdxFinalize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxFinalize is not found")
    return (<commondxStatusType (*)(size_t, const cublasdxTensor*, size_t, const cublasdxPipeline*) noexcept nogil>__cublasdxFinalize)(
        countTensors, tensors, countPipelines, pipelines)


cdef commondxStatusType _cublasdxGetTensorTraitInt64(cublasdxTensor tensor, cublasdxTensorTrait trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetTensorTraitInt64
    _check_or_init_mathdx()
    if __cublasdxGetTensorTraitInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetTensorTraitInt64 is not found")
    return (<commondxStatusType (*)(cublasdxTensor, cublasdxTensorTrait, long long int*) noexcept nogil>__cublasdxGetTensorTraitInt64)(
        tensor, trait, value)


cdef commondxStatusType _cublasdxGetPipelineTraitInt64(cublasdxPipeline pipeline, cublasdxPipelineTrait trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetPipelineTraitInt64
    _check_or_init_mathdx()
    if __cublasdxGetPipelineTraitInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetPipelineTraitInt64 is not found")
    return (<commondxStatusType (*)(cublasdxPipeline, cublasdxPipelineTrait, long long int*) noexcept nogil>__cublasdxGetPipelineTraitInt64)(
        pipeline, trait, value)


cdef commondxStatusType _cublasdxGetPipelineTraitInt64s(cublasdxPipeline pipeline, cublasdxPipelineTrait trait, size_t count, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetPipelineTraitInt64s
    _check_or_init_mathdx()
    if __cublasdxGetPipelineTraitInt64s == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetPipelineTraitInt64s is not found")
    return (<commondxStatusType (*)(cublasdxPipeline, cublasdxPipelineTrait, size_t, long long int*) noexcept nogil>__cublasdxGetPipelineTraitInt64s)(
        pipeline, trait, count, array)


cdef commondxStatusType _cublasdxGetTensorTraitStrSize(cublasdxTensor tensor, cublasdxTensorTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetTensorTraitStrSize
    _check_or_init_mathdx()
    if __cublasdxGetTensorTraitStrSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetTensorTraitStrSize is not found")
    return (<commondxStatusType (*)(cublasdxTensor, cublasdxTensorTrait, size_t*) noexcept nogil>__cublasdxGetTensorTraitStrSize)(
        tensor, trait, size)


cdef commondxStatusType _cublasdxGetPipelineTraitStrSize(cublasdxPipeline pipeline, cublasdxPipelineTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetPipelineTraitStrSize
    _check_or_init_mathdx()
    if __cublasdxGetPipelineTraitStrSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetPipelineTraitStrSize is not found")
    return (<commondxStatusType (*)(cublasdxPipeline, cublasdxPipelineTrait, size_t*) noexcept nogil>__cublasdxGetPipelineTraitStrSize)(
        pipeline, trait, size)


cdef commondxStatusType _cublasdxGetTensorTraitStr(cublasdxTensor tensor, cublasdxTensorTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetTensorTraitStr
    _check_or_init_mathdx()
    if __cublasdxGetTensorTraitStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetTensorTraitStr is not found")
    return (<commondxStatusType (*)(cublasdxTensor, cublasdxTensorTrait, size_t, char*) noexcept nogil>__cublasdxGetTensorTraitStr)(
        tensor, trait, size, value)


cdef commondxStatusType _cublasdxGetPipelineTraitStr(cublasdxPipeline pipeline, cublasdxPipelineTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxGetPipelineTraitStr
    _check_or_init_mathdx()
    if __cublasdxGetPipelineTraitStr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxGetPipelineTraitStr is not found")
    return (<commondxStatusType (*)(cublasdxPipeline, cublasdxPipelineTrait, size_t, char*) noexcept nogil>__cublasdxGetPipelineTraitStr)(
        pipeline, trait, size, value)


cdef commondxStatusType _cublasdxCreateDeviceFunction(cublasdxDescriptor handle, cublasdxDeviceFunctionType device_function_type, size_t count, const cublasdxTensor* array, cublasdxDeviceFunction* device_function) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxCreateDeviceFunction
    _check_or_init_mathdx()
    if __cublasdxCreateDeviceFunction == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxCreateDeviceFunction is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxDeviceFunctionType, size_t, const cublasdxTensor*, cublasdxDeviceFunction*) noexcept nogil>__cublasdxCreateDeviceFunction)(
        handle, device_function_type, count, array, device_function)


cdef commondxStatusType _cublasdxDestroyDeviceFunction(cublasdxDeviceFunction device_function) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxDestroyDeviceFunction
    _check_or_init_mathdx()
    if __cublasdxDestroyDeviceFunction == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxDestroyDeviceFunction is not found")
    return (<commondxStatusType (*)(cublasdxDeviceFunction) noexcept nogil>__cublasdxDestroyDeviceFunction)(
        device_function)


cdef commondxStatusType _cublasdxCreateDeviceFunctionWithPipelines(cublasdxDescriptor handle, cublasdxDeviceFunctionType device_function_type, size_t tensor_count, const cublasdxTensor* tensors, size_t pipeline_count, const cublasdxPipeline* pipelines, cublasdxDeviceFunction* device_function) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cublasdxCreateDeviceFunctionWithPipelines
    _check_or_init_mathdx()
    if __cublasdxCreateDeviceFunctionWithPipelines == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasdxCreateDeviceFunctionWithPipelines is not found")
    return (<commondxStatusType (*)(cublasdxDescriptor, cublasdxDeviceFunctionType, size_t, const cublasdxTensor*, size_t, const cublasdxPipeline*, cublasdxDeviceFunction*) noexcept nogil>__cublasdxCreateDeviceFunctionWithPipelines)(
        handle, device_function_type, tensor_count, tensors, pipeline_count, pipelines, device_function)


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


cdef commondxStatusType _cusolverdxGetTraitInt64s(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t count, long long int* values) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxGetTraitInt64s
    _check_or_init_mathdx()
    if __cusolverdxGetTraitInt64s == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxGetTraitInt64s is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, cusolverdxTraitType, size_t, long long int*) noexcept nogil>__cusolverdxGetTraitInt64s)(
        handle, trait, count, values)


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


cdef commondxStatusType _curanddxGetVersion(int* major, int* minor, int* patch) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __curanddxGetVersion
    _check_or_init_mathdx()
    if __curanddxGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxGetVersion is not found")
    return (<commondxStatusType (*)(int*, int*, int*) noexcept nogil>__curanddxGetVersion)(
        major, minor, patch)


cdef commondxStatusType _curanddxCreateDescriptor(curanddxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __curanddxCreateDescriptor
    _check_or_init_mathdx()
    if __curanddxCreateDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxCreateDescriptor is not found")
    return (<commondxStatusType (*)(curanddxDescriptor*) noexcept nogil>__curanddxCreateDescriptor)(
        handle)


cdef commondxStatusType _curanddxSetOptionStr(curanddxDescriptor handle, commondxOption opt, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __curanddxSetOptionStr
    _check_or_init_mathdx()
    if __curanddxSetOptionStr == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxSetOptionStr is not found")
    return (<commondxStatusType (*)(curanddxDescriptor, commondxOption, const char*) noexcept nogil>__curanddxSetOptionStr)(
        handle, opt, value)


cdef commondxStatusType _curanddxSetOperatorInt64(curanddxDescriptor handle, curanddxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __curanddxSetOperatorInt64
    _check_or_init_mathdx()
    if __curanddxSetOperatorInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxSetOperatorInt64 is not found")
    return (<commondxStatusType (*)(curanddxDescriptor, curanddxOperatorType, long long int) noexcept nogil>__curanddxSetOperatorInt64)(
        handle, op, value)


cdef commondxStatusType _curanddxSetOperatorDoubles(curanddxDescriptor handle, curanddxOperatorType op, size_t count, double* values) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __curanddxSetOperatorDoubles
    _check_or_init_mathdx()
    if __curanddxSetOperatorDoubles == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxSetOperatorDoubles is not found")
    return (<commondxStatusType (*)(curanddxDescriptor, curanddxOperatorType, size_t, double*) noexcept nogil>__curanddxSetOperatorDoubles)(
        handle, op, count, values)


cdef commondxStatusType _curanddxGetTraitStrSize(curanddxDescriptor handle, curanddxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __curanddxGetTraitStrSize
    _check_or_init_mathdx()
    if __curanddxGetTraitStrSize == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxGetTraitStrSize is not found")
    return (<commondxStatusType (*)(curanddxDescriptor, curanddxTraitType, size_t*) noexcept nogil>__curanddxGetTraitStrSize)(
        handle, trait, size)


cdef commondxStatusType _curanddxGetTraitStr(curanddxDescriptor handle, curanddxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __curanddxGetTraitStr
    _check_or_init_mathdx()
    if __curanddxGetTraitStr == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxGetTraitStr is not found")
    return (<commondxStatusType (*)(curanddxDescriptor, curanddxTraitType, size_t, char*) noexcept nogil>__curanddxGetTraitStr)(
        handle, trait, size, value)


cdef commondxStatusType _curanddxGetTraitInt64(curanddxDescriptor handle, curanddxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __curanddxGetTraitInt64
    _check_or_init_mathdx()
    if __curanddxGetTraitInt64 == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxGetTraitInt64 is not found")
    return (<commondxStatusType (*)(curanddxDescriptor, curanddxTraitType, long long int*) noexcept nogil>__curanddxGetTraitInt64)(
        handle, trait, value)


cdef commondxStatusType _curanddxFinalizeCode(commondxCode code, curanddxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __curanddxFinalizeCode
    _check_or_init_mathdx()
    if __curanddxFinalizeCode == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxFinalizeCode is not found")
    return (<commondxStatusType (*)(commondxCode, curanddxDescriptor) noexcept nogil>__curanddxFinalizeCode)(
        code, handle)


cdef commondxStatusType _curanddxDestroyDescriptor(curanddxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __curanddxDestroyDescriptor
    _check_or_init_mathdx()
    if __curanddxDestroyDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxDestroyDescriptor is not found")
    return (<commondxStatusType (*)(curanddxDescriptor) noexcept nogil>__curanddxDestroyDescriptor)(
        handle)


cdef const char* _curanddxOperatorTypeToStr(curanddxOperatorType op) except?NULL nogil:
    global __curanddxOperatorTypeToStr
    _check_or_init_mathdx()
    if __curanddxOperatorTypeToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxOperatorTypeToStr is not found")
    return (<const char* (*)(curanddxOperatorType) noexcept nogil>__curanddxOperatorTypeToStr)(
        op)


cdef const char* _curanddxDistributionToStr(curanddxDistribution dist) except?NULL nogil:
    global __curanddxDistributionToStr
    _check_or_init_mathdx()
    if __curanddxDistributionToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxDistributionToStr is not found")
    return (<const char* (*)(curanddxDistribution) noexcept nogil>__curanddxDistributionToStr)(
        dist)


cdef const char* _curanddxGeneratorToStr(curanddxGenerator generator) except?NULL nogil:
    global __curanddxGeneratorToStr
    _check_or_init_mathdx()
    if __curanddxGeneratorToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxGeneratorToStr is not found")
    return (<const char* (*)(curanddxGenerator) noexcept nogil>__curanddxGeneratorToStr)(
        generator)


cdef const char* _curanddxGenerateMethodToStr(curanddxGenerateMethod generate_method) except?NULL nogil:
    global __curanddxGenerateMethodToStr
    _check_or_init_mathdx()
    if __curanddxGenerateMethodToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxGenerateMethodToStr is not found")
    return (<const char* (*)(curanddxGenerateMethod) noexcept nogil>__curanddxGenerateMethodToStr)(
        generate_method)


cdef const char* _curanddxNormalMethodToStr(curanddxNormalMethod normal_method) except?NULL nogil:
    global __curanddxNormalMethodToStr
    _check_or_init_mathdx()
    if __curanddxNormalMethodToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxNormalMethodToStr is not found")
    return (<const char* (*)(curanddxNormalMethod) noexcept nogil>__curanddxNormalMethodToStr)(
        normal_method)


cdef const char* _curanddxTraitTypeToStr(curanddxTraitType trait) except?NULL nogil:
    global __curanddxTraitTypeToStr
    _check_or_init_mathdx()
    if __curanddxTraitTypeToStr == NULL:
        with gil:
            raise FunctionNotFoundError("function curanddxTraitTypeToStr is not found")
    return (<const char* (*)(curanddxTraitType) noexcept nogil>__curanddxTraitTypeToStr)(
        trait)


cdef commondxStatusType _commondxGetCodePTXSize(commondxCode code, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetCodePTXSize
    _check_or_init_mathdx()
    if __commondxGetCodePTXSize == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetCodePTXSize is not found")
    return (<commondxStatusType (*)(commondxCode, size_t*) noexcept nogil>__commondxGetCodePTXSize)(
        code, size)


cdef commondxStatusType _commondxGetCodePTX(commondxCode code, size_t size, void* out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __commondxGetCodePTX
    _check_or_init_mathdx()
    if __commondxGetCodePTX == NULL:
        with gil:
            raise FunctionNotFoundError("function commondxGetCodePTX is not found")
    return (<commondxStatusType (*)(commondxCode, size_t, void*) noexcept nogil>__commondxGetCodePTX)(
        code, size, out)


cdef commondxStatusType _cusolverdxGetTraitCommondxDataTypes(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t count, commondxValueType* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    global __cusolverdxGetTraitCommondxDataTypes
    _check_or_init_mathdx()
    if __cusolverdxGetTraitCommondxDataTypes == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverdxGetTraitCommondxDataTypes is not found")
    return (<commondxStatusType (*)(cusolverdxDescriptor, cusolverdxTraitType, size_t, commondxValueType*) noexcept nogil>__cusolverdxGetTraitCommondxDataTypes)(
        handle, trait, count, array)
