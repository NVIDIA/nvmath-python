# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.3.1. Do not modify it directly.

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
cdef bint __py_cutensor_init = False

cdef void* __cutensorCreate = NULL
cdef void* __cutensorDestroy = NULL
cdef void* __cutensorHandleResizePlanCache = NULL
cdef void* __cutensorHandleWritePlanCacheToFile = NULL
cdef void* __cutensorHandleReadPlanCacheFromFile = NULL
cdef void* __cutensorWriteKernelCacheToFile = NULL
cdef void* __cutensorReadKernelCacheFromFile = NULL
cdef void* __cutensorCreateTensorDescriptor = NULL
cdef void* __cutensorDestroyTensorDescriptor = NULL
cdef void* __cutensorCreateElementwiseTrinary = NULL
cdef void* __cutensorElementwiseTrinaryExecute = NULL
cdef void* __cutensorCreateElementwiseBinary = NULL
cdef void* __cutensorElementwiseBinaryExecute = NULL
cdef void* __cutensorCreatePermutation = NULL
cdef void* __cutensorPermute = NULL
cdef void* __cutensorCreateContraction = NULL
cdef void* __cutensorDestroyOperationDescriptor = NULL
cdef void* __cutensorOperationDescriptorSetAttribute = NULL
cdef void* __cutensorOperationDescriptorGetAttribute = NULL
cdef void* __cutensorCreatePlanPreference = NULL
cdef void* __cutensorDestroyPlanPreference = NULL
cdef void* __cutensorPlanPreferenceSetAttribute = NULL
cdef void* __cutensorPlanGetAttribute = NULL
cdef void* __cutensorEstimateWorkspaceSize = NULL
cdef void* __cutensorCreatePlan = NULL
cdef void* __cutensorDestroyPlan = NULL
cdef void* __cutensorContract = NULL
cdef void* __cutensorCreateReduction = NULL
cdef void* __cutensorReduce = NULL
cdef void* __cutensorCreateContractionTrinary = NULL
cdef void* __cutensorContractTrinary = NULL
cdef void* __cutensorCreateBlockSparseTensorDescriptor = NULL
cdef void* __cutensorDestroyBlockSparseTensorDescriptor = NULL
cdef void* __cutensorCreateBlockSparseContraction = NULL
cdef void* __cutensorBlockSparseContract = NULL
cdef void* __cutensorGetErrorString = NULL
cdef void* __cutensorGetVersion = NULL
cdef void* __cutensorGetCudartVersion = NULL
cdef void* __cutensorLoggerSetCallback = NULL
cdef void* __cutensorLoggerSetFile = NULL
cdef void* __cutensorLoggerOpenFile = NULL
cdef void* __cutensorLoggerSetLevel = NULL
cdef void* __cutensorLoggerSetMask = NULL
cdef void* __cutensorLoggerForceDisable = NULL


cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cutensor")._handle_uint
    return <void*>handle


cdef int _check_or_init_cutensor() except -1 nogil:
    global __py_cutensor_init
    if __py_cutensor_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_cutensor_init:
            return 0

        # Load function
        global __cutensorCreate
        __cutensorCreate = dlsym(RTLD_DEFAULT, 'cutensorCreate')
        if __cutensorCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreate = dlsym(handle, 'cutensorCreate')

        global __cutensorDestroy
        __cutensorDestroy = dlsym(RTLD_DEFAULT, 'cutensorDestroy')
        if __cutensorDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorDestroy = dlsym(handle, 'cutensorDestroy')

        global __cutensorHandleResizePlanCache
        __cutensorHandleResizePlanCache = dlsym(RTLD_DEFAULT, 'cutensorHandleResizePlanCache')
        if __cutensorHandleResizePlanCache == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorHandleResizePlanCache = dlsym(handle, 'cutensorHandleResizePlanCache')

        global __cutensorHandleWritePlanCacheToFile
        __cutensorHandleWritePlanCacheToFile = dlsym(RTLD_DEFAULT, 'cutensorHandleWritePlanCacheToFile')
        if __cutensorHandleWritePlanCacheToFile == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorHandleWritePlanCacheToFile = dlsym(handle, 'cutensorHandleWritePlanCacheToFile')

        global __cutensorHandleReadPlanCacheFromFile
        __cutensorHandleReadPlanCacheFromFile = dlsym(RTLD_DEFAULT, 'cutensorHandleReadPlanCacheFromFile')
        if __cutensorHandleReadPlanCacheFromFile == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorHandleReadPlanCacheFromFile = dlsym(handle, 'cutensorHandleReadPlanCacheFromFile')

        global __cutensorWriteKernelCacheToFile
        __cutensorWriteKernelCacheToFile = dlsym(RTLD_DEFAULT, 'cutensorWriteKernelCacheToFile')
        if __cutensorWriteKernelCacheToFile == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorWriteKernelCacheToFile = dlsym(handle, 'cutensorWriteKernelCacheToFile')

        global __cutensorReadKernelCacheFromFile
        __cutensorReadKernelCacheFromFile = dlsym(RTLD_DEFAULT, 'cutensorReadKernelCacheFromFile')
        if __cutensorReadKernelCacheFromFile == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorReadKernelCacheFromFile = dlsym(handle, 'cutensorReadKernelCacheFromFile')

        global __cutensorCreateTensorDescriptor
        __cutensorCreateTensorDescriptor = dlsym(RTLD_DEFAULT, 'cutensorCreateTensorDescriptor')
        if __cutensorCreateTensorDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreateTensorDescriptor = dlsym(handle, 'cutensorCreateTensorDescriptor')

        global __cutensorDestroyTensorDescriptor
        __cutensorDestroyTensorDescriptor = dlsym(RTLD_DEFAULT, 'cutensorDestroyTensorDescriptor')
        if __cutensorDestroyTensorDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorDestroyTensorDescriptor = dlsym(handle, 'cutensorDestroyTensorDescriptor')

        global __cutensorCreateElementwiseTrinary
        __cutensorCreateElementwiseTrinary = dlsym(RTLD_DEFAULT, 'cutensorCreateElementwiseTrinary')
        if __cutensorCreateElementwiseTrinary == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreateElementwiseTrinary = dlsym(handle, 'cutensorCreateElementwiseTrinary')

        global __cutensorElementwiseTrinaryExecute
        __cutensorElementwiseTrinaryExecute = dlsym(RTLD_DEFAULT, 'cutensorElementwiseTrinaryExecute')
        if __cutensorElementwiseTrinaryExecute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorElementwiseTrinaryExecute = dlsym(handle, 'cutensorElementwiseTrinaryExecute')

        global __cutensorCreateElementwiseBinary
        __cutensorCreateElementwiseBinary = dlsym(RTLD_DEFAULT, 'cutensorCreateElementwiseBinary')
        if __cutensorCreateElementwiseBinary == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreateElementwiseBinary = dlsym(handle, 'cutensorCreateElementwiseBinary')

        global __cutensorElementwiseBinaryExecute
        __cutensorElementwiseBinaryExecute = dlsym(RTLD_DEFAULT, 'cutensorElementwiseBinaryExecute')
        if __cutensorElementwiseBinaryExecute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorElementwiseBinaryExecute = dlsym(handle, 'cutensorElementwiseBinaryExecute')

        global __cutensorCreatePermutation
        __cutensorCreatePermutation = dlsym(RTLD_DEFAULT, 'cutensorCreatePermutation')
        if __cutensorCreatePermutation == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreatePermutation = dlsym(handle, 'cutensorCreatePermutation')

        global __cutensorPermute
        __cutensorPermute = dlsym(RTLD_DEFAULT, 'cutensorPermute')
        if __cutensorPermute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorPermute = dlsym(handle, 'cutensorPermute')

        global __cutensorCreateContraction
        __cutensorCreateContraction = dlsym(RTLD_DEFAULT, 'cutensorCreateContraction')
        if __cutensorCreateContraction == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreateContraction = dlsym(handle, 'cutensorCreateContraction')

        global __cutensorDestroyOperationDescriptor
        __cutensorDestroyOperationDescriptor = dlsym(RTLD_DEFAULT, 'cutensorDestroyOperationDescriptor')
        if __cutensorDestroyOperationDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorDestroyOperationDescriptor = dlsym(handle, 'cutensorDestroyOperationDescriptor')

        global __cutensorOperationDescriptorSetAttribute
        __cutensorOperationDescriptorSetAttribute = dlsym(RTLD_DEFAULT, 'cutensorOperationDescriptorSetAttribute')
        if __cutensorOperationDescriptorSetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorOperationDescriptorSetAttribute = dlsym(handle, 'cutensorOperationDescriptorSetAttribute')

        global __cutensorOperationDescriptorGetAttribute
        __cutensorOperationDescriptorGetAttribute = dlsym(RTLD_DEFAULT, 'cutensorOperationDescriptorGetAttribute')
        if __cutensorOperationDescriptorGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorOperationDescriptorGetAttribute = dlsym(handle, 'cutensorOperationDescriptorGetAttribute')

        global __cutensorCreatePlanPreference
        __cutensorCreatePlanPreference = dlsym(RTLD_DEFAULT, 'cutensorCreatePlanPreference')
        if __cutensorCreatePlanPreference == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreatePlanPreference = dlsym(handle, 'cutensorCreatePlanPreference')

        global __cutensorDestroyPlanPreference
        __cutensorDestroyPlanPreference = dlsym(RTLD_DEFAULT, 'cutensorDestroyPlanPreference')
        if __cutensorDestroyPlanPreference == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorDestroyPlanPreference = dlsym(handle, 'cutensorDestroyPlanPreference')

        global __cutensorPlanPreferenceSetAttribute
        __cutensorPlanPreferenceSetAttribute = dlsym(RTLD_DEFAULT, 'cutensorPlanPreferenceSetAttribute')
        if __cutensorPlanPreferenceSetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorPlanPreferenceSetAttribute = dlsym(handle, 'cutensorPlanPreferenceSetAttribute')

        global __cutensorPlanGetAttribute
        __cutensorPlanGetAttribute = dlsym(RTLD_DEFAULT, 'cutensorPlanGetAttribute')
        if __cutensorPlanGetAttribute == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorPlanGetAttribute = dlsym(handle, 'cutensorPlanGetAttribute')

        global __cutensorEstimateWorkspaceSize
        __cutensorEstimateWorkspaceSize = dlsym(RTLD_DEFAULT, 'cutensorEstimateWorkspaceSize')
        if __cutensorEstimateWorkspaceSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorEstimateWorkspaceSize = dlsym(handle, 'cutensorEstimateWorkspaceSize')

        global __cutensorCreatePlan
        __cutensorCreatePlan = dlsym(RTLD_DEFAULT, 'cutensorCreatePlan')
        if __cutensorCreatePlan == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreatePlan = dlsym(handle, 'cutensorCreatePlan')

        global __cutensorDestroyPlan
        __cutensorDestroyPlan = dlsym(RTLD_DEFAULT, 'cutensorDestroyPlan')
        if __cutensorDestroyPlan == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorDestroyPlan = dlsym(handle, 'cutensorDestroyPlan')

        global __cutensorContract
        __cutensorContract = dlsym(RTLD_DEFAULT, 'cutensorContract')
        if __cutensorContract == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorContract = dlsym(handle, 'cutensorContract')

        global __cutensorCreateReduction
        __cutensorCreateReduction = dlsym(RTLD_DEFAULT, 'cutensorCreateReduction')
        if __cutensorCreateReduction == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreateReduction = dlsym(handle, 'cutensorCreateReduction')

        global __cutensorReduce
        __cutensorReduce = dlsym(RTLD_DEFAULT, 'cutensorReduce')
        if __cutensorReduce == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorReduce = dlsym(handle, 'cutensorReduce')

        global __cutensorCreateContractionTrinary
        __cutensorCreateContractionTrinary = dlsym(RTLD_DEFAULT, 'cutensorCreateContractionTrinary')
        if __cutensorCreateContractionTrinary == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreateContractionTrinary = dlsym(handle, 'cutensorCreateContractionTrinary')

        global __cutensorContractTrinary
        __cutensorContractTrinary = dlsym(RTLD_DEFAULT, 'cutensorContractTrinary')
        if __cutensorContractTrinary == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorContractTrinary = dlsym(handle, 'cutensorContractTrinary')

        global __cutensorCreateBlockSparseTensorDescriptor
        __cutensorCreateBlockSparseTensorDescriptor = dlsym(RTLD_DEFAULT, 'cutensorCreateBlockSparseTensorDescriptor')
        if __cutensorCreateBlockSparseTensorDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreateBlockSparseTensorDescriptor = dlsym(handle, 'cutensorCreateBlockSparseTensorDescriptor')

        global __cutensorDestroyBlockSparseTensorDescriptor
        __cutensorDestroyBlockSparseTensorDescriptor = dlsym(RTLD_DEFAULT, 'cutensorDestroyBlockSparseTensorDescriptor')
        if __cutensorDestroyBlockSparseTensorDescriptor == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorDestroyBlockSparseTensorDescriptor = dlsym(handle, 'cutensorDestroyBlockSparseTensorDescriptor')

        global __cutensorCreateBlockSparseContraction
        __cutensorCreateBlockSparseContraction = dlsym(RTLD_DEFAULT, 'cutensorCreateBlockSparseContraction')
        if __cutensorCreateBlockSparseContraction == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorCreateBlockSparseContraction = dlsym(handle, 'cutensorCreateBlockSparseContraction')

        global __cutensorBlockSparseContract
        __cutensorBlockSparseContract = dlsym(RTLD_DEFAULT, 'cutensorBlockSparseContract')
        if __cutensorBlockSparseContract == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorBlockSparseContract = dlsym(handle, 'cutensorBlockSparseContract')

        global __cutensorGetErrorString
        __cutensorGetErrorString = dlsym(RTLD_DEFAULT, 'cutensorGetErrorString')
        if __cutensorGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorGetErrorString = dlsym(handle, 'cutensorGetErrorString')

        global __cutensorGetVersion
        __cutensorGetVersion = dlsym(RTLD_DEFAULT, 'cutensorGetVersion')
        if __cutensorGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorGetVersion = dlsym(handle, 'cutensorGetVersion')

        global __cutensorGetCudartVersion
        __cutensorGetCudartVersion = dlsym(RTLD_DEFAULT, 'cutensorGetCudartVersion')
        if __cutensorGetCudartVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorGetCudartVersion = dlsym(handle, 'cutensorGetCudartVersion')

        global __cutensorLoggerSetCallback
        __cutensorLoggerSetCallback = dlsym(RTLD_DEFAULT, 'cutensorLoggerSetCallback')
        if __cutensorLoggerSetCallback == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorLoggerSetCallback = dlsym(handle, 'cutensorLoggerSetCallback')

        global __cutensorLoggerSetFile
        __cutensorLoggerSetFile = dlsym(RTLD_DEFAULT, 'cutensorLoggerSetFile')
        if __cutensorLoggerSetFile == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorLoggerSetFile = dlsym(handle, 'cutensorLoggerSetFile')

        global __cutensorLoggerOpenFile
        __cutensorLoggerOpenFile = dlsym(RTLD_DEFAULT, 'cutensorLoggerOpenFile')
        if __cutensorLoggerOpenFile == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorLoggerOpenFile = dlsym(handle, 'cutensorLoggerOpenFile')

        global __cutensorLoggerSetLevel
        __cutensorLoggerSetLevel = dlsym(RTLD_DEFAULT, 'cutensorLoggerSetLevel')
        if __cutensorLoggerSetLevel == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorLoggerSetLevel = dlsym(handle, 'cutensorLoggerSetLevel')

        global __cutensorLoggerSetMask
        __cutensorLoggerSetMask = dlsym(RTLD_DEFAULT, 'cutensorLoggerSetMask')
        if __cutensorLoggerSetMask == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorLoggerSetMask = dlsym(handle, 'cutensorLoggerSetMask')

        global __cutensorLoggerForceDisable
        __cutensorLoggerForceDisable = dlsym(RTLD_DEFAULT, 'cutensorLoggerForceDisable')
        if __cutensorLoggerForceDisable == NULL:
            if handle == NULL:
                handle = load_library()
            __cutensorLoggerForceDisable = dlsym(handle, 'cutensorLoggerForceDisable')
        __py_cutensor_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cutensor()
    cdef dict data = {}

    global __cutensorCreate
    data["__cutensorCreate"] = <intptr_t>__cutensorCreate

    global __cutensorDestroy
    data["__cutensorDestroy"] = <intptr_t>__cutensorDestroy

    global __cutensorHandleResizePlanCache
    data["__cutensorHandleResizePlanCache"] = <intptr_t>__cutensorHandleResizePlanCache

    global __cutensorHandleWritePlanCacheToFile
    data["__cutensorHandleWritePlanCacheToFile"] = <intptr_t>__cutensorHandleWritePlanCacheToFile

    global __cutensorHandleReadPlanCacheFromFile
    data["__cutensorHandleReadPlanCacheFromFile"] = <intptr_t>__cutensorHandleReadPlanCacheFromFile

    global __cutensorWriteKernelCacheToFile
    data["__cutensorWriteKernelCacheToFile"] = <intptr_t>__cutensorWriteKernelCacheToFile

    global __cutensorReadKernelCacheFromFile
    data["__cutensorReadKernelCacheFromFile"] = <intptr_t>__cutensorReadKernelCacheFromFile

    global __cutensorCreateTensorDescriptor
    data["__cutensorCreateTensorDescriptor"] = <intptr_t>__cutensorCreateTensorDescriptor

    global __cutensorDestroyTensorDescriptor
    data["__cutensorDestroyTensorDescriptor"] = <intptr_t>__cutensorDestroyTensorDescriptor

    global __cutensorCreateElementwiseTrinary
    data["__cutensorCreateElementwiseTrinary"] = <intptr_t>__cutensorCreateElementwiseTrinary

    global __cutensorElementwiseTrinaryExecute
    data["__cutensorElementwiseTrinaryExecute"] = <intptr_t>__cutensorElementwiseTrinaryExecute

    global __cutensorCreateElementwiseBinary
    data["__cutensorCreateElementwiseBinary"] = <intptr_t>__cutensorCreateElementwiseBinary

    global __cutensorElementwiseBinaryExecute
    data["__cutensorElementwiseBinaryExecute"] = <intptr_t>__cutensorElementwiseBinaryExecute

    global __cutensorCreatePermutation
    data["__cutensorCreatePermutation"] = <intptr_t>__cutensorCreatePermutation

    global __cutensorPermute
    data["__cutensorPermute"] = <intptr_t>__cutensorPermute

    global __cutensorCreateContraction
    data["__cutensorCreateContraction"] = <intptr_t>__cutensorCreateContraction

    global __cutensorDestroyOperationDescriptor
    data["__cutensorDestroyOperationDescriptor"] = <intptr_t>__cutensorDestroyOperationDescriptor

    global __cutensorOperationDescriptorSetAttribute
    data["__cutensorOperationDescriptorSetAttribute"] = <intptr_t>__cutensorOperationDescriptorSetAttribute

    global __cutensorOperationDescriptorGetAttribute
    data["__cutensorOperationDescriptorGetAttribute"] = <intptr_t>__cutensorOperationDescriptorGetAttribute

    global __cutensorCreatePlanPreference
    data["__cutensorCreatePlanPreference"] = <intptr_t>__cutensorCreatePlanPreference

    global __cutensorDestroyPlanPreference
    data["__cutensorDestroyPlanPreference"] = <intptr_t>__cutensorDestroyPlanPreference

    global __cutensorPlanPreferenceSetAttribute
    data["__cutensorPlanPreferenceSetAttribute"] = <intptr_t>__cutensorPlanPreferenceSetAttribute

    global __cutensorPlanGetAttribute
    data["__cutensorPlanGetAttribute"] = <intptr_t>__cutensorPlanGetAttribute

    global __cutensorEstimateWorkspaceSize
    data["__cutensorEstimateWorkspaceSize"] = <intptr_t>__cutensorEstimateWorkspaceSize

    global __cutensorCreatePlan
    data["__cutensorCreatePlan"] = <intptr_t>__cutensorCreatePlan

    global __cutensorDestroyPlan
    data["__cutensorDestroyPlan"] = <intptr_t>__cutensorDestroyPlan

    global __cutensorContract
    data["__cutensorContract"] = <intptr_t>__cutensorContract

    global __cutensorCreateReduction
    data["__cutensorCreateReduction"] = <intptr_t>__cutensorCreateReduction

    global __cutensorReduce
    data["__cutensorReduce"] = <intptr_t>__cutensorReduce

    global __cutensorCreateContractionTrinary
    data["__cutensorCreateContractionTrinary"] = <intptr_t>__cutensorCreateContractionTrinary

    global __cutensorContractTrinary
    data["__cutensorContractTrinary"] = <intptr_t>__cutensorContractTrinary

    global __cutensorCreateBlockSparseTensorDescriptor
    data["__cutensorCreateBlockSparseTensorDescriptor"] = <intptr_t>__cutensorCreateBlockSparseTensorDescriptor

    global __cutensorDestroyBlockSparseTensorDescriptor
    data["__cutensorDestroyBlockSparseTensorDescriptor"] = <intptr_t>__cutensorDestroyBlockSparseTensorDescriptor

    global __cutensorCreateBlockSparseContraction
    data["__cutensorCreateBlockSparseContraction"] = <intptr_t>__cutensorCreateBlockSparseContraction

    global __cutensorBlockSparseContract
    data["__cutensorBlockSparseContract"] = <intptr_t>__cutensorBlockSparseContract

    global __cutensorGetErrorString
    data["__cutensorGetErrorString"] = <intptr_t>__cutensorGetErrorString

    global __cutensorGetVersion
    data["__cutensorGetVersion"] = <intptr_t>__cutensorGetVersion

    global __cutensorGetCudartVersion
    data["__cutensorGetCudartVersion"] = <intptr_t>__cutensorGetCudartVersion

    global __cutensorLoggerSetCallback
    data["__cutensorLoggerSetCallback"] = <intptr_t>__cutensorLoggerSetCallback

    global __cutensorLoggerSetFile
    data["__cutensorLoggerSetFile"] = <intptr_t>__cutensorLoggerSetFile

    global __cutensorLoggerOpenFile
    data["__cutensorLoggerOpenFile"] = <intptr_t>__cutensorLoggerOpenFile

    global __cutensorLoggerSetLevel
    data["__cutensorLoggerSetLevel"] = <intptr_t>__cutensorLoggerSetLevel

    global __cutensorLoggerSetMask
    data["__cutensorLoggerSetMask"] = <intptr_t>__cutensorLoggerSetMask

    global __cutensorLoggerForceDisable
    data["__cutensorLoggerForceDisable"] = <intptr_t>__cutensorLoggerForceDisable

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

cdef cutensorStatus_t _cutensorCreate(cutensorHandle_t* handle) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreate
    _check_or_init_cutensor()
    if __cutensorCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreate is not found")
    return (<cutensorStatus_t (*)(cutensorHandle_t*) noexcept nogil>__cutensorCreate)(
        handle)


cdef cutensorStatus_t _cutensorDestroy(cutensorHandle_t handle) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorDestroy
    _check_or_init_cutensor()
    if __cutensorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorDestroy is not found")
    return (<cutensorStatus_t (*)(cutensorHandle_t) noexcept nogil>__cutensorDestroy)(
        handle)


cdef cutensorStatus_t _cutensorHandleResizePlanCache(cutensorHandle_t handle, const uint32_t numEntries) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorHandleResizePlanCache
    _check_or_init_cutensor()
    if __cutensorHandleResizePlanCache == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorHandleResizePlanCache is not found")
    return (<cutensorStatus_t (*)(cutensorHandle_t, const uint32_t) noexcept nogil>__cutensorHandleResizePlanCache)(
        handle, numEntries)


cdef cutensorStatus_t _cutensorHandleWritePlanCacheToFile(const cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorHandleWritePlanCacheToFile
    _check_or_init_cutensor()
    if __cutensorHandleWritePlanCacheToFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorHandleWritePlanCacheToFile is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const char*) noexcept nogil>__cutensorHandleWritePlanCacheToFile)(
        handle, filename)


cdef cutensorStatus_t _cutensorHandleReadPlanCacheFromFile(cutensorHandle_t handle, const char filename[], uint32_t* numCachelinesRead) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorHandleReadPlanCacheFromFile
    _check_or_init_cutensor()
    if __cutensorHandleReadPlanCacheFromFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorHandleReadPlanCacheFromFile is not found")
    return (<cutensorStatus_t (*)(cutensorHandle_t, const char*, uint32_t*) noexcept nogil>__cutensorHandleReadPlanCacheFromFile)(
        handle, filename, numCachelinesRead)


cdef cutensorStatus_t _cutensorWriteKernelCacheToFile(const cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorWriteKernelCacheToFile
    _check_or_init_cutensor()
    if __cutensorWriteKernelCacheToFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorWriteKernelCacheToFile is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const char*) noexcept nogil>__cutensorWriteKernelCacheToFile)(
        handle, filename)


cdef cutensorStatus_t _cutensorReadKernelCacheFromFile(cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorReadKernelCacheFromFile
    _check_or_init_cutensor()
    if __cutensorReadKernelCacheFromFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorReadKernelCacheFromFile is not found")
    return (<cutensorStatus_t (*)(cutensorHandle_t, const char*) noexcept nogil>__cutensorReadKernelCacheFromFile)(
        handle, filename)


cdef cutensorStatus_t _cutensorCreateTensorDescriptor(const cutensorHandle_t handle, cutensorTensorDescriptor_t* desc, const uint32_t numModes, const int64_t extent[], const int64_t stride[], cudaDataType_t dataType, uint32_t alignmentRequirement) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreateTensorDescriptor
    _check_or_init_cutensor()
    if __cutensorCreateTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreateTensorDescriptor is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorTensorDescriptor_t*, const uint32_t, const int64_t*, const int64_t*, cudaDataType_t, uint32_t) noexcept nogil>__cutensorCreateTensorDescriptor)(
        handle, desc, numModes, extent, stride, dataType, alignmentRequirement)


cdef cutensorStatus_t _cutensorDestroyTensorDescriptor(cutensorTensorDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorDestroyTensorDescriptor
    _check_or_init_cutensor()
    if __cutensorDestroyTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorDestroyTensorDescriptor is not found")
    return (<cutensorStatus_t (*)(cutensorTensorDescriptor_t) noexcept nogil>__cutensorDestroyTensorDescriptor)(
        desc)


cdef cutensorStatus_t _cutensorCreateElementwiseTrinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAB, cutensorOperator_t opABC, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreateElementwiseTrinary
    _check_or_init_cutensor()
    if __cutensorCreateElementwiseTrinary == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreateElementwiseTrinary is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorOperationDescriptor_t*, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, cutensorOperator_t, const cutensorComputeDescriptor_t) noexcept nogil>__cutensorCreateElementwiseTrinary)(
        handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, opAB, opABC, descCompute)


cdef cutensorStatus_t _cutensorElementwiseTrinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* B, const void* gamma, const void* C, void* D, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorElementwiseTrinaryExecute
    _check_or_init_cutensor()
    if __cutensorElementwiseTrinaryExecute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorElementwiseTrinaryExecute is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const cutensorPlan_t, const void*, const void*, const void*, const void*, const void*, const void*, void*, cudaStream_t) noexcept nogil>__cutensorElementwiseTrinaryExecute)(
        handle, plan, alpha, A, beta, B, gamma, C, D, stream)


cdef cutensorStatus_t _cutensorCreateElementwiseBinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAC, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreateElementwiseBinary
    _check_or_init_cutensor()
    if __cutensorCreateElementwiseBinary == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreateElementwiseBinary is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorOperationDescriptor_t*, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorComputeDescriptor_t) noexcept nogil>__cutensorCreateElementwiseBinary)(
        handle, desc, descA, modeA, opA, descC, modeC, opC, descD, modeD, opAC, descCompute)


cdef cutensorStatus_t _cutensorElementwiseBinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* gamma, const void* C, void* D, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorElementwiseBinaryExecute
    _check_or_init_cutensor()
    if __cutensorElementwiseBinaryExecute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorElementwiseBinaryExecute is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const cutensorPlan_t, const void*, const void*, const void*, const void*, void*, cudaStream_t) noexcept nogil>__cutensorElementwiseBinaryExecute)(
        handle, plan, alpha, A, gamma, C, D, stream)


cdef cutensorStatus_t _cutensorCreatePermutation(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreatePermutation
    _check_or_init_cutensor()
    if __cutensorCreatePermutation == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreatePermutation is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorOperationDescriptor_t*, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, const cutensorComputeDescriptor_t) noexcept nogil>__cutensorCreatePermutation)(
        handle, desc, descA, modeA, opA, descB, modeB, descCompute)


cdef cutensorStatus_t _cutensorPermute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, void* B, const cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorPermute
    _check_or_init_cutensor()
    if __cutensorPermute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorPermute is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const cutensorPlan_t, const void*, const void*, void*, const cudaStream_t) noexcept nogil>__cutensorPermute)(
        handle, plan, alpha, A, B, stream)


cdef cutensorStatus_t _cutensorCreateContraction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreateContraction
    _check_or_init_cutensor()
    if __cutensorCreateContraction == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreateContraction is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorOperationDescriptor_t*, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, const cutensorComputeDescriptor_t) noexcept nogil>__cutensorCreateContraction)(
        handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, descCompute)


cdef cutensorStatus_t _cutensorDestroyOperationDescriptor(cutensorOperationDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorDestroyOperationDescriptor
    _check_or_init_cutensor()
    if __cutensorDestroyOperationDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorDestroyOperationDescriptor is not found")
    return (<cutensorStatus_t (*)(cutensorOperationDescriptor_t) noexcept nogil>__cutensorDestroyOperationDescriptor)(
        desc)


cdef cutensorStatus_t _cutensorOperationDescriptorSetAttribute(const cutensorHandle_t handle, cutensorOperationDescriptor_t desc, cutensorOperationDescriptorAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorOperationDescriptorSetAttribute
    _check_or_init_cutensor()
    if __cutensorOperationDescriptorSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorOperationDescriptorSetAttribute is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorOperationDescriptor_t, cutensorOperationDescriptorAttribute_t, const void*, size_t) noexcept nogil>__cutensorOperationDescriptorSetAttribute)(
        handle, desc, attr, buf, sizeInBytes)


cdef cutensorStatus_t _cutensorOperationDescriptorGetAttribute(const cutensorHandle_t handle, cutensorOperationDescriptor_t desc, cutensorOperationDescriptorAttribute_t attr, void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorOperationDescriptorGetAttribute
    _check_or_init_cutensor()
    if __cutensorOperationDescriptorGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorOperationDescriptorGetAttribute is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorOperationDescriptor_t, cutensorOperationDescriptorAttribute_t, void*, size_t) noexcept nogil>__cutensorOperationDescriptorGetAttribute)(
        handle, desc, attr, buf, sizeInBytes)


cdef cutensorStatus_t _cutensorCreatePlanPreference(const cutensorHandle_t handle, cutensorPlanPreference_t* pref, cutensorAlgo_t algo, cutensorJitMode_t jitMode) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreatePlanPreference
    _check_or_init_cutensor()
    if __cutensorCreatePlanPreference == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreatePlanPreference is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorPlanPreference_t*, cutensorAlgo_t, cutensorJitMode_t) noexcept nogil>__cutensorCreatePlanPreference)(
        handle, pref, algo, jitMode)


cdef cutensorStatus_t _cutensorDestroyPlanPreference(cutensorPlanPreference_t pref) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorDestroyPlanPreference
    _check_or_init_cutensor()
    if __cutensorDestroyPlanPreference == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorDestroyPlanPreference is not found")
    return (<cutensorStatus_t (*)(cutensorPlanPreference_t) noexcept nogil>__cutensorDestroyPlanPreference)(
        pref)


cdef cutensorStatus_t _cutensorPlanPreferenceSetAttribute(const cutensorHandle_t handle, cutensorPlanPreference_t pref, cutensorPlanPreferenceAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorPlanPreferenceSetAttribute
    _check_or_init_cutensor()
    if __cutensorPlanPreferenceSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorPlanPreferenceSetAttribute is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorPlanPreference_t, cutensorPlanPreferenceAttribute_t, const void*, size_t) noexcept nogil>__cutensorPlanPreferenceSetAttribute)(
        handle, pref, attr, buf, sizeInBytes)


cdef cutensorStatus_t _cutensorPlanGetAttribute(const cutensorHandle_t handle, const cutensorPlan_t plan, cutensorPlanAttribute_t attr, void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorPlanGetAttribute
    _check_or_init_cutensor()
    if __cutensorPlanGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorPlanGetAttribute is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const cutensorPlan_t, cutensorPlanAttribute_t, void*, size_t) noexcept nogil>__cutensorPlanGetAttribute)(
        handle, plan, attr, buf, sizeInBytes)


cdef cutensorStatus_t _cutensorEstimateWorkspaceSize(const cutensorHandle_t handle, const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t planPref, const cutensorWorksizePreference_t workspacePref, uint64_t* workspaceSizeEstimate) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorEstimateWorkspaceSize
    _check_or_init_cutensor()
    if __cutensorEstimateWorkspaceSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorEstimateWorkspaceSize is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const cutensorOperationDescriptor_t, const cutensorPlanPreference_t, const cutensorWorksizePreference_t, uint64_t*) noexcept nogil>__cutensorEstimateWorkspaceSize)(
        handle, desc, planPref, workspacePref, workspaceSizeEstimate)


cdef cutensorStatus_t _cutensorCreatePlan(const cutensorHandle_t handle, cutensorPlan_t* plan, const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t pref, uint64_t workspaceSizeLimit) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreatePlan
    _check_or_init_cutensor()
    if __cutensorCreatePlan == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreatePlan is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorPlan_t*, const cutensorOperationDescriptor_t, const cutensorPlanPreference_t, uint64_t) noexcept nogil>__cutensorCreatePlan)(
        handle, plan, desc, pref, workspaceSizeLimit)


cdef cutensorStatus_t _cutensorDestroyPlan(cutensorPlan_t plan) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorDestroyPlan
    _check_or_init_cutensor()
    if __cutensorDestroyPlan == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorDestroyPlan is not found")
    return (<cutensorStatus_t (*)(cutensorPlan_t) noexcept nogil>__cutensorDestroyPlan)(
        plan)


cdef cutensorStatus_t _cutensorContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* B, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorContract
    _check_or_init_cutensor()
    if __cutensorContract == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorContract is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const cutensorPlan_t, const void*, const void*, const void*, const void*, const void*, void*, void*, uint64_t, cudaStream_t) noexcept nogil>__cutensorContract)(
        handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)


cdef cutensorStatus_t _cutensorCreateReduction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opReduce, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreateReduction
    _check_or_init_cutensor()
    if __cutensorCreateReduction == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreateReduction is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorOperationDescriptor_t*, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorComputeDescriptor_t) noexcept nogil>__cutensorCreateReduction)(
        handle, desc, descA, modeA, opA, descC, modeC, opC, descD, modeD, opReduce, descCompute)


cdef cutensorStatus_t _cutensorReduce(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorReduce
    _check_or_init_cutensor()
    if __cutensorReduce == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorReduce is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const cutensorPlan_t, const void*, const void*, const void*, const void*, void*, void*, uint64_t, cudaStream_t) noexcept nogil>__cutensorReduce)(
        handle, plan, alpha, A, beta, C, D, workspace, workspaceSize, stream)


cdef cutensorStatus_t _cutensorCreateContractionTrinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opD, const cutensorTensorDescriptor_t descE, const int32_t modeE[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreateContractionTrinary
    _check_or_init_cutensor()
    if __cutensorCreateContractionTrinary == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreateContractionTrinary is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorOperationDescriptor_t*, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorTensorDescriptor_t, const int32_t*, const cutensorComputeDescriptor_t) noexcept nogil>__cutensorCreateContractionTrinary)(
        handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, opD, descE, modeE, descCompute)


cdef cutensorStatus_t _cutensorContractTrinary(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* B, const void* C, const void* beta, const void* D, void* E, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorContractTrinary
    _check_or_init_cutensor()
    if __cutensorContractTrinary == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorContractTrinary is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const cutensorPlan_t, const void*, const void*, const void*, const void*, const void*, const void*, void*, void*, uint64_t, cudaStream_t) noexcept nogil>__cutensorContractTrinary)(
        handle, plan, alpha, A, B, C, beta, D, E, workspace, workspaceSize, stream)


cdef cutensorStatus_t _cutensorCreateBlockSparseTensorDescriptor(cutensorHandle_t handle, cutensorBlockSparseTensorDescriptor_t* desc, const uint32_t numModes, const uint64_t numNonZeroBlocks, const uint32_t numSectionsPerMode[], const int64_t extent[], const int32_t nonZeroCoordinates[], const int64_t stride[], cudaDataType_t dataType) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreateBlockSparseTensorDescriptor
    _check_or_init_cutensor()
    if __cutensorCreateBlockSparseTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreateBlockSparseTensorDescriptor is not found")
    return (<cutensorStatus_t (*)(cutensorHandle_t, cutensorBlockSparseTensorDescriptor_t*, const uint32_t, const uint64_t, const uint32_t*, const int64_t*, const int32_t*, const int64_t*, cudaDataType_t) noexcept nogil>__cutensorCreateBlockSparseTensorDescriptor)(
        handle, desc, numModes, numNonZeroBlocks, numSectionsPerMode, extent, nonZeroCoordinates, stride, dataType)


cdef cutensorStatus_t _cutensorDestroyBlockSparseTensorDescriptor(cutensorBlockSparseTensorDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorDestroyBlockSparseTensorDescriptor
    _check_or_init_cutensor()
    if __cutensorDestroyBlockSparseTensorDescriptor == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorDestroyBlockSparseTensorDescriptor is not found")
    return (<cutensorStatus_t (*)(cutensorBlockSparseTensorDescriptor_t) noexcept nogil>__cutensorDestroyBlockSparseTensorDescriptor)(
        desc)


cdef cutensorStatus_t _cutensorCreateBlockSparseContraction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorBlockSparseTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorBlockSparseTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorBlockSparseTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorBlockSparseTensorDescriptor_t descD, const int32_t modeD[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorCreateBlockSparseContraction
    _check_or_init_cutensor()
    if __cutensorCreateBlockSparseContraction == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorCreateBlockSparseContraction is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, cutensorOperationDescriptor_t*, const cutensorBlockSparseTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorBlockSparseTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorBlockSparseTensorDescriptor_t, const int32_t*, cutensorOperator_t, const cutensorBlockSparseTensorDescriptor_t, const int32_t*, const cutensorComputeDescriptor_t) noexcept nogil>__cutensorCreateBlockSparseContraction)(
        handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, descCompute)


cdef cutensorStatus_t _cutensorBlockSparseContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* const A[], const void* const B[], const void* beta, const void* const C[], void* const D[], void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorBlockSparseContract
    _check_or_init_cutensor()
    if __cutensorBlockSparseContract == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorBlockSparseContract is not found")
    return (<cutensorStatus_t (*)(const cutensorHandle_t, const cutensorPlan_t, const void*, const void* const*, const void* const*, const void*, const void* const*, void* const*, void*, uint64_t, cudaStream_t) noexcept nogil>__cutensorBlockSparseContract)(
        handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)


cdef const char* _cutensorGetErrorString(const cutensorStatus_t error) except?NULL nogil:
    global __cutensorGetErrorString
    _check_or_init_cutensor()
    if __cutensorGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorGetErrorString is not found")
    return (<const char* (*)(const cutensorStatus_t) noexcept nogil>__cutensorGetErrorString)(
        error)


cdef size_t _cutensorGetVersion() except?0 nogil:
    global __cutensorGetVersion
    _check_or_init_cutensor()
    if __cutensorGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorGetVersion is not found")
    return (<size_t (*)() noexcept nogil>__cutensorGetVersion)(
        )


cdef size_t _cutensorGetCudartVersion() except?0 nogil:
    global __cutensorGetCudartVersion
    _check_or_init_cutensor()
    if __cutensorGetCudartVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorGetCudartVersion is not found")
    return (<size_t (*)() noexcept nogil>__cutensorGetCudartVersion)(
        )


cdef cutensorStatus_t _cutensorLoggerSetCallback(cutensorLoggerCallback_t callback) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorLoggerSetCallback
    _check_or_init_cutensor()
    if __cutensorLoggerSetCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorLoggerSetCallback is not found")
    return (<cutensorStatus_t (*)(cutensorLoggerCallback_t) noexcept nogil>__cutensorLoggerSetCallback)(
        callback)


cdef cutensorStatus_t _cutensorLoggerSetFile(FILE* file) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorLoggerSetFile
    _check_or_init_cutensor()
    if __cutensorLoggerSetFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorLoggerSetFile is not found")
    return (<cutensorStatus_t (*)(FILE*) noexcept nogil>__cutensorLoggerSetFile)(
        file)


cdef cutensorStatus_t _cutensorLoggerOpenFile(const char* logFile) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorLoggerOpenFile
    _check_or_init_cutensor()
    if __cutensorLoggerOpenFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorLoggerOpenFile is not found")
    return (<cutensorStatus_t (*)(const char*) noexcept nogil>__cutensorLoggerOpenFile)(
        logFile)


cdef cutensorStatus_t _cutensorLoggerSetLevel(int32_t level) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorLoggerSetLevel
    _check_or_init_cutensor()
    if __cutensorLoggerSetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorLoggerSetLevel is not found")
    return (<cutensorStatus_t (*)(int32_t) noexcept nogil>__cutensorLoggerSetLevel)(
        level)


cdef cutensorStatus_t _cutensorLoggerSetMask(int32_t mask) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorLoggerSetMask
    _check_or_init_cutensor()
    if __cutensorLoggerSetMask == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorLoggerSetMask is not found")
    return (<cutensorStatus_t (*)(int32_t) noexcept nogil>__cutensorLoggerSetMask)(
        mask)


cdef cutensorStatus_t _cutensorLoggerForceDisable() except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cutensorLoggerForceDisable
    _check_or_init_cutensor()
    if __cutensorLoggerForceDisable == NULL:
        with gil:
            raise FunctionNotFoundError("function cutensorLoggerForceDisable is not found")
    return (<cutensorStatus_t (*)() noexcept nogil>__cutensorLoggerForceDisable)(
        )
