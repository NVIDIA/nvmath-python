# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.5.0. Do not modify it directly.

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

    const void* RTLD_DEFAULT 'RTLD_DEFAULT'


###############################################################################
# Wrapper init
###############################################################################

cdef bint __py_cudss_init = False

cdef void* __cudssConfigSet = NULL
cdef void* __cudssConfigGet = NULL
cdef void* __cudssDataSet = NULL
cdef void* __cudssDataGet = NULL
cdef void* __cudssExecute = NULL
cdef void* __cudssSetStream = NULL
cdef void* __cudssSetCommLayer = NULL
cdef void* __cudssSetThreadingLayer = NULL
cdef void* __cudssConfigCreate = NULL
cdef void* __cudssConfigDestroy = NULL
cdef void* __cudssDataCreate = NULL
cdef void* __cudssDataDestroy = NULL
cdef void* __cudssCreate = NULL
cdef void* __cudssDestroy = NULL
cdef void* __cudssGetProperty = NULL
cdef void* __cudssMatrixCreateDn = NULL
cdef void* __cudssMatrixCreateCsr = NULL
cdef void* __cudssMatrixCreateBatchDn = NULL
cdef void* __cudssMatrixCreateBatchCsr = NULL
cdef void* __cudssMatrixDestroy = NULL
cdef void* __cudssMatrixGetDn = NULL
cdef void* __cudssMatrixGetCsr = NULL
cdef void* __cudssMatrixSetValues = NULL
cdef void* __cudssMatrixSetCsrPointers = NULL
cdef void* __cudssMatrixGetBatchDn = NULL
cdef void* __cudssMatrixGetBatchCsr = NULL
cdef void* __cudssMatrixSetBatchValues = NULL
cdef void* __cudssMatrixSetBatchCsrPointers = NULL
cdef void* __cudssMatrixGetFormat = NULL
cdef void* __cudssGetDeviceMemHandler = NULL
cdef void* __cudssSetDeviceMemHandler = NULL


cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cudss")._handle_uint
    return <void*>handle


cdef int _check_or_init_cudss() except -1 nogil:
    global __py_cudss_init
    if __py_cudss_init:
        return 0

    cdef void* handle = NULL
    # Load function
    global __cudssConfigSet
    __cudssConfigSet = dlsym(RTLD_DEFAULT, 'cudssConfigSet')
    if __cudssConfigSet == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssConfigSet = dlsym(handle, 'cudssConfigSet')

    global __cudssConfigGet
    __cudssConfigGet = dlsym(RTLD_DEFAULT, 'cudssConfigGet')
    if __cudssConfigGet == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssConfigGet = dlsym(handle, 'cudssConfigGet')

    global __cudssDataSet
    __cudssDataSet = dlsym(RTLD_DEFAULT, 'cudssDataSet')
    if __cudssDataSet == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssDataSet = dlsym(handle, 'cudssDataSet')

    global __cudssDataGet
    __cudssDataGet = dlsym(RTLD_DEFAULT, 'cudssDataGet')
    if __cudssDataGet == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssDataGet = dlsym(handle, 'cudssDataGet')

    global __cudssExecute
    __cudssExecute = dlsym(RTLD_DEFAULT, 'cudssExecute')
    if __cudssExecute == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssExecute = dlsym(handle, 'cudssExecute')

    global __cudssSetStream
    __cudssSetStream = dlsym(RTLD_DEFAULT, 'cudssSetStream')
    if __cudssSetStream == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssSetStream = dlsym(handle, 'cudssSetStream')

    global __cudssSetCommLayer
    __cudssSetCommLayer = dlsym(RTLD_DEFAULT, 'cudssSetCommLayer')
    if __cudssSetCommLayer == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssSetCommLayer = dlsym(handle, 'cudssSetCommLayer')

    global __cudssSetThreadingLayer
    __cudssSetThreadingLayer = dlsym(RTLD_DEFAULT, 'cudssSetThreadingLayer')
    if __cudssSetThreadingLayer == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssSetThreadingLayer = dlsym(handle, 'cudssSetThreadingLayer')

    global __cudssConfigCreate
    __cudssConfigCreate = dlsym(RTLD_DEFAULT, 'cudssConfigCreate')
    if __cudssConfigCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssConfigCreate = dlsym(handle, 'cudssConfigCreate')

    global __cudssConfigDestroy
    __cudssConfigDestroy = dlsym(RTLD_DEFAULT, 'cudssConfigDestroy')
    if __cudssConfigDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssConfigDestroy = dlsym(handle, 'cudssConfigDestroy')

    global __cudssDataCreate
    __cudssDataCreate = dlsym(RTLD_DEFAULT, 'cudssDataCreate')
    if __cudssDataCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssDataCreate = dlsym(handle, 'cudssDataCreate')

    global __cudssDataDestroy
    __cudssDataDestroy = dlsym(RTLD_DEFAULT, 'cudssDataDestroy')
    if __cudssDataDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssDataDestroy = dlsym(handle, 'cudssDataDestroy')

    global __cudssCreate
    __cudssCreate = dlsym(RTLD_DEFAULT, 'cudssCreate')
    if __cudssCreate == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssCreate = dlsym(handle, 'cudssCreate')

    global __cudssDestroy
    __cudssDestroy = dlsym(RTLD_DEFAULT, 'cudssDestroy')
    if __cudssDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssDestroy = dlsym(handle, 'cudssDestroy')

    global __cudssGetProperty
    __cudssGetProperty = dlsym(RTLD_DEFAULT, 'cudssGetProperty')
    if __cudssGetProperty == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssGetProperty = dlsym(handle, 'cudssGetProperty')

    global __cudssMatrixCreateDn
    __cudssMatrixCreateDn = dlsym(RTLD_DEFAULT, 'cudssMatrixCreateDn')
    if __cudssMatrixCreateDn == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixCreateDn = dlsym(handle, 'cudssMatrixCreateDn')

    global __cudssMatrixCreateCsr
    __cudssMatrixCreateCsr = dlsym(RTLD_DEFAULT, 'cudssMatrixCreateCsr')
    if __cudssMatrixCreateCsr == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixCreateCsr = dlsym(handle, 'cudssMatrixCreateCsr')

    global __cudssMatrixCreateBatchDn
    __cudssMatrixCreateBatchDn = dlsym(RTLD_DEFAULT, 'cudssMatrixCreateBatchDn')
    if __cudssMatrixCreateBatchDn == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixCreateBatchDn = dlsym(handle, 'cudssMatrixCreateBatchDn')

    global __cudssMatrixCreateBatchCsr
    __cudssMatrixCreateBatchCsr = dlsym(RTLD_DEFAULT, 'cudssMatrixCreateBatchCsr')
    if __cudssMatrixCreateBatchCsr == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixCreateBatchCsr = dlsym(handle, 'cudssMatrixCreateBatchCsr')

    global __cudssMatrixDestroy
    __cudssMatrixDestroy = dlsym(RTLD_DEFAULT, 'cudssMatrixDestroy')
    if __cudssMatrixDestroy == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixDestroy = dlsym(handle, 'cudssMatrixDestroy')

    global __cudssMatrixGetDn
    __cudssMatrixGetDn = dlsym(RTLD_DEFAULT, 'cudssMatrixGetDn')
    if __cudssMatrixGetDn == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixGetDn = dlsym(handle, 'cudssMatrixGetDn')

    global __cudssMatrixGetCsr
    __cudssMatrixGetCsr = dlsym(RTLD_DEFAULT, 'cudssMatrixGetCsr')
    if __cudssMatrixGetCsr == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixGetCsr = dlsym(handle, 'cudssMatrixGetCsr')

    global __cudssMatrixSetValues
    __cudssMatrixSetValues = dlsym(RTLD_DEFAULT, 'cudssMatrixSetValues')
    if __cudssMatrixSetValues == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixSetValues = dlsym(handle, 'cudssMatrixSetValues')

    global __cudssMatrixSetCsrPointers
    __cudssMatrixSetCsrPointers = dlsym(RTLD_DEFAULT, 'cudssMatrixSetCsrPointers')
    if __cudssMatrixSetCsrPointers == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixSetCsrPointers = dlsym(handle, 'cudssMatrixSetCsrPointers')

    global __cudssMatrixGetBatchDn
    __cudssMatrixGetBatchDn = dlsym(RTLD_DEFAULT, 'cudssMatrixGetBatchDn')
    if __cudssMatrixGetBatchDn == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixGetBatchDn = dlsym(handle, 'cudssMatrixGetBatchDn')

    global __cudssMatrixGetBatchCsr
    __cudssMatrixGetBatchCsr = dlsym(RTLD_DEFAULT, 'cudssMatrixGetBatchCsr')
    if __cudssMatrixGetBatchCsr == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixGetBatchCsr = dlsym(handle, 'cudssMatrixGetBatchCsr')

    global __cudssMatrixSetBatchValues
    __cudssMatrixSetBatchValues = dlsym(RTLD_DEFAULT, 'cudssMatrixSetBatchValues')
    if __cudssMatrixSetBatchValues == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixSetBatchValues = dlsym(handle, 'cudssMatrixSetBatchValues')

    global __cudssMatrixSetBatchCsrPointers
    __cudssMatrixSetBatchCsrPointers = dlsym(RTLD_DEFAULT, 'cudssMatrixSetBatchCsrPointers')
    if __cudssMatrixSetBatchCsrPointers == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixSetBatchCsrPointers = dlsym(handle, 'cudssMatrixSetBatchCsrPointers')

    global __cudssMatrixGetFormat
    __cudssMatrixGetFormat = dlsym(RTLD_DEFAULT, 'cudssMatrixGetFormat')
    if __cudssMatrixGetFormat == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssMatrixGetFormat = dlsym(handle, 'cudssMatrixGetFormat')

    global __cudssGetDeviceMemHandler
    __cudssGetDeviceMemHandler = dlsym(RTLD_DEFAULT, 'cudssGetDeviceMemHandler')
    if __cudssGetDeviceMemHandler == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssGetDeviceMemHandler = dlsym(handle, 'cudssGetDeviceMemHandler')

    global __cudssSetDeviceMemHandler
    __cudssSetDeviceMemHandler = dlsym(RTLD_DEFAULT, 'cudssSetDeviceMemHandler')
    if __cudssSetDeviceMemHandler == NULL:
        if handle == NULL:
            handle = load_library()
        __cudssSetDeviceMemHandler = dlsym(handle, 'cudssSetDeviceMemHandler')

    __py_cudss_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cudss()
    cdef dict data = {}

    global __cudssConfigSet
    data["__cudssConfigSet"] = <intptr_t>__cudssConfigSet

    global __cudssConfigGet
    data["__cudssConfigGet"] = <intptr_t>__cudssConfigGet

    global __cudssDataSet
    data["__cudssDataSet"] = <intptr_t>__cudssDataSet

    global __cudssDataGet
    data["__cudssDataGet"] = <intptr_t>__cudssDataGet

    global __cudssExecute
    data["__cudssExecute"] = <intptr_t>__cudssExecute

    global __cudssSetStream
    data["__cudssSetStream"] = <intptr_t>__cudssSetStream

    global __cudssSetCommLayer
    data["__cudssSetCommLayer"] = <intptr_t>__cudssSetCommLayer

    global __cudssSetThreadingLayer
    data["__cudssSetThreadingLayer"] = <intptr_t>__cudssSetThreadingLayer

    global __cudssConfigCreate
    data["__cudssConfigCreate"] = <intptr_t>__cudssConfigCreate

    global __cudssConfigDestroy
    data["__cudssConfigDestroy"] = <intptr_t>__cudssConfigDestroy

    global __cudssDataCreate
    data["__cudssDataCreate"] = <intptr_t>__cudssDataCreate

    global __cudssDataDestroy
    data["__cudssDataDestroy"] = <intptr_t>__cudssDataDestroy

    global __cudssCreate
    data["__cudssCreate"] = <intptr_t>__cudssCreate

    global __cudssDestroy
    data["__cudssDestroy"] = <intptr_t>__cudssDestroy

    global __cudssGetProperty
    data["__cudssGetProperty"] = <intptr_t>__cudssGetProperty

    global __cudssMatrixCreateDn
    data["__cudssMatrixCreateDn"] = <intptr_t>__cudssMatrixCreateDn

    global __cudssMatrixCreateCsr
    data["__cudssMatrixCreateCsr"] = <intptr_t>__cudssMatrixCreateCsr

    global __cudssMatrixCreateBatchDn
    data["__cudssMatrixCreateBatchDn"] = <intptr_t>__cudssMatrixCreateBatchDn

    global __cudssMatrixCreateBatchCsr
    data["__cudssMatrixCreateBatchCsr"] = <intptr_t>__cudssMatrixCreateBatchCsr

    global __cudssMatrixDestroy
    data["__cudssMatrixDestroy"] = <intptr_t>__cudssMatrixDestroy

    global __cudssMatrixGetDn
    data["__cudssMatrixGetDn"] = <intptr_t>__cudssMatrixGetDn

    global __cudssMatrixGetCsr
    data["__cudssMatrixGetCsr"] = <intptr_t>__cudssMatrixGetCsr

    global __cudssMatrixSetValues
    data["__cudssMatrixSetValues"] = <intptr_t>__cudssMatrixSetValues

    global __cudssMatrixSetCsrPointers
    data["__cudssMatrixSetCsrPointers"] = <intptr_t>__cudssMatrixSetCsrPointers

    global __cudssMatrixGetBatchDn
    data["__cudssMatrixGetBatchDn"] = <intptr_t>__cudssMatrixGetBatchDn

    global __cudssMatrixGetBatchCsr
    data["__cudssMatrixGetBatchCsr"] = <intptr_t>__cudssMatrixGetBatchCsr

    global __cudssMatrixSetBatchValues
    data["__cudssMatrixSetBatchValues"] = <intptr_t>__cudssMatrixSetBatchValues

    global __cudssMatrixSetBatchCsrPointers
    data["__cudssMatrixSetBatchCsrPointers"] = <intptr_t>__cudssMatrixSetBatchCsrPointers

    global __cudssMatrixGetFormat
    data["__cudssMatrixGetFormat"] = <intptr_t>__cudssMatrixGetFormat

    global __cudssGetDeviceMemHandler
    data["__cudssGetDeviceMemHandler"] = <intptr_t>__cudssGetDeviceMemHandler

    global __cudssSetDeviceMemHandler
    data["__cudssSetDeviceMemHandler"] = <intptr_t>__cudssSetDeviceMemHandler

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

cdef cudssStatus_t _cudssConfigSet(cudssConfig_t config, cudssConfigParam_t param, void* value, size_t sizeInBytes) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssConfigSet
    _check_or_init_cudss()
    if __cudssConfigSet == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssConfigSet is not found")
    return (<cudssStatus_t (*)(cudssConfig_t, cudssConfigParam_t, void*, size_t) noexcept nogil>__cudssConfigSet)(
        config, param, value, sizeInBytes)


cdef cudssStatus_t _cudssConfigGet(cudssConfig_t config, cudssConfigParam_t param, void* value, size_t sizeInBytes, size_t* sizeWritten) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssConfigGet
    _check_or_init_cudss()
    if __cudssConfigGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssConfigGet is not found")
    return (<cudssStatus_t (*)(cudssConfig_t, cudssConfigParam_t, void*, size_t, size_t*) noexcept nogil>__cudssConfigGet)(
        config, param, value, sizeInBytes, sizeWritten)


cdef cudssStatus_t _cudssDataSet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param, void* value, size_t sizeInBytes) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssDataSet
    _check_or_init_cudss()
    if __cudssDataSet == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssDataSet is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, cudssData_t, cudssDataParam_t, void*, size_t) noexcept nogil>__cudssDataSet)(
        handle, data, param, value, sizeInBytes)


cdef cudssStatus_t _cudssDataGet(cudssHandle_t handle, cudssData_t data, cudssDataParam_t param, void* value, size_t sizeInBytes, size_t* sizeWritten) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssDataGet
    _check_or_init_cudss()
    if __cudssDataGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssDataGet is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, cudssData_t, cudssDataParam_t, void*, size_t, size_t*) noexcept nogil>__cudssDataGet)(
        handle, data, param, value, sizeInBytes, sizeWritten)


cdef cudssStatus_t _cudssExecute(cudssHandle_t handle, cudssPhase_t phase, cudssConfig_t solverConfig, cudssData_t solverData, cudssMatrix_t inputMatrix, cudssMatrix_t solution, cudssMatrix_t rhs) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssExecute
    _check_or_init_cudss()
    if __cudssExecute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssExecute is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, cudssPhase_t, cudssConfig_t, cudssData_t, cudssMatrix_t, cudssMatrix_t, cudssMatrix_t) noexcept nogil>__cudssExecute)(
        handle, phase, solverConfig, solverData, inputMatrix, solution, rhs)


cdef cudssStatus_t _cudssSetStream(cudssHandle_t handle, cudaStream_t stream) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssSetStream
    _check_or_init_cudss()
    if __cudssSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssSetStream is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, cudaStream_t) noexcept nogil>__cudssSetStream)(
        handle, stream)


cdef cudssStatus_t _cudssSetCommLayer(cudssHandle_t handle, const char* commLibFileName) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssSetCommLayer
    _check_or_init_cudss()
    if __cudssSetCommLayer == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssSetCommLayer is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, const char*) noexcept nogil>__cudssSetCommLayer)(
        handle, commLibFileName)


cdef cudssStatus_t _cudssSetThreadingLayer(cudssHandle_t handle, const char* thrLibFileName) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssSetThreadingLayer
    _check_or_init_cudss()
    if __cudssSetThreadingLayer == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssSetThreadingLayer is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, const char*) noexcept nogil>__cudssSetThreadingLayer)(
        handle, thrLibFileName)


cdef cudssStatus_t _cudssConfigCreate(cudssConfig_t* solverConfig) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssConfigCreate
    _check_or_init_cudss()
    if __cudssConfigCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssConfigCreate is not found")
    return (<cudssStatus_t (*)(cudssConfig_t*) noexcept nogil>__cudssConfigCreate)(
        solverConfig)


cdef cudssStatus_t _cudssConfigDestroy(cudssConfig_t solverConfig) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssConfigDestroy
    _check_or_init_cudss()
    if __cudssConfigDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssConfigDestroy is not found")
    return (<cudssStatus_t (*)(cudssConfig_t) noexcept nogil>__cudssConfigDestroy)(
        solverConfig)


cdef cudssStatus_t _cudssDataCreate(cudssHandle_t handle, cudssData_t* solverData) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssDataCreate
    _check_or_init_cudss()
    if __cudssDataCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssDataCreate is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, cudssData_t*) noexcept nogil>__cudssDataCreate)(
        handle, solverData)


cdef cudssStatus_t _cudssDataDestroy(cudssHandle_t handle, cudssData_t solverData) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssDataDestroy
    _check_or_init_cudss()
    if __cudssDataDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssDataDestroy is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, cudssData_t) noexcept nogil>__cudssDataDestroy)(
        handle, solverData)


cdef cudssStatus_t _cudssCreate(cudssHandle_t* handle) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssCreate
    _check_or_init_cudss()
    if __cudssCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssCreate is not found")
    return (<cudssStatus_t (*)(cudssHandle_t*) noexcept nogil>__cudssCreate)(
        handle)


cdef cudssStatus_t _cudssDestroy(cudssHandle_t handle) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssDestroy
    _check_or_init_cudss()
    if __cudssDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssDestroy is not found")
    return (<cudssStatus_t (*)(cudssHandle_t) noexcept nogil>__cudssDestroy)(
        handle)


cdef cudssStatus_t _cudssGetProperty(libraryPropertyType propertyType, int* value) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssGetProperty
    _check_or_init_cudss()
    if __cudssGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssGetProperty is not found")
    return (<cudssStatus_t (*)(libraryPropertyType, int*) noexcept nogil>__cudssGetProperty)(
        propertyType, value)


cdef cudssStatus_t _cudssMatrixCreateDn(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t ld, void* values, cudaDataType_t valueType, cudssLayout_t layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixCreateDn
    _check_or_init_cudss()
    if __cudssMatrixCreateDn == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixCreateDn is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t*, int64_t, int64_t, int64_t, void*, cudaDataType_t, cudssLayout_t) noexcept nogil>__cudssMatrixCreateDn)(
        matrix, nrows, ncols, ld, values, valueType, layout)


cdef cudssStatus_t _cudssMatrixCreateCsr(cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t nnz, void* rowStart, void* rowEnd, void* colIndices, void* values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixCreateCsr
    _check_or_init_cudss()
    if __cudssMatrixCreateCsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixCreateCsr is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t*, int64_t, int64_t, int64_t, void*, void*, void*, void*, cudaDataType_t, cudaDataType_t, cudssMatrixType_t, cudssMatrixViewType_t, cudssIndexBase_t) noexcept nogil>__cudssMatrixCreateCsr)(
        matrix, nrows, ncols, nnz, rowStart, rowEnd, colIndices, values, indexType, valueType, mtype, mview, indexBase)


cdef cudssStatus_t _cudssMatrixCreateBatchDn(cudssMatrix_t* matrix, int64_t batchCount, void* nrows, void* ncols, void* ld, void** values, cudaDataType_t indexType, cudaDataType_t valueType, cudssLayout_t layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixCreateBatchDn
    _check_or_init_cudss()
    if __cudssMatrixCreateBatchDn == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixCreateBatchDn is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t*, int64_t, void*, void*, void*, void**, cudaDataType_t, cudaDataType_t, cudssLayout_t) noexcept nogil>__cudssMatrixCreateBatchDn)(
        matrix, batchCount, nrows, ncols, ld, values, indexType, valueType, layout)


cdef cudssStatus_t _cudssMatrixCreateBatchCsr(cudssMatrix_t* matrix, int64_t batchCount, void* nrows, void* ncols, void* nnz, void** rowStart, void** rowEnd, void** colIndices, void** values, cudaDataType_t indexType, cudaDataType_t valueType, cudssMatrixType_t mtype, cudssMatrixViewType_t mview, cudssIndexBase_t indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixCreateBatchCsr
    _check_or_init_cudss()
    if __cudssMatrixCreateBatchCsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixCreateBatchCsr is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t*, int64_t, void*, void*, void*, void**, void**, void**, void**, cudaDataType_t, cudaDataType_t, cudssMatrixType_t, cudssMatrixViewType_t, cudssIndexBase_t) noexcept nogil>__cudssMatrixCreateBatchCsr)(
        matrix, batchCount, nrows, ncols, nnz, rowStart, rowEnd, colIndices, values, indexType, valueType, mtype, mview, indexBase)


cdef cudssStatus_t _cudssMatrixDestroy(cudssMatrix_t matrix) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixDestroy
    _check_or_init_cudss()
    if __cudssMatrixDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixDestroy is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t) noexcept nogil>__cudssMatrixDestroy)(
        matrix)


cdef cudssStatus_t _cudssMatrixGetDn(cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* ld, void** values, cudaDataType_t* type, cudssLayout_t* layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixGetDn
    _check_or_init_cudss()
    if __cudssMatrixGetDn == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixGetDn is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, int64_t*, int64_t*, int64_t*, void**, cudaDataType_t*, cudssLayout_t*) noexcept nogil>__cudssMatrixGetDn)(
        matrix, nrows, ncols, ld, values, type, layout)


cdef cudssStatus_t _cudssMatrixGetCsr(cudssMatrix_t matrix, int64_t* nrows, int64_t* ncols, int64_t* nnz, void** rowStart, void** rowEnd, void** colIndices, void** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixGetCsr
    _check_or_init_cudss()
    if __cudssMatrixGetCsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixGetCsr is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, int64_t*, int64_t*, int64_t*, void**, void**, void**, void**, cudaDataType_t*, cudaDataType_t*, cudssMatrixType_t*, cudssMatrixViewType_t*, cudssIndexBase_t*) noexcept nogil>__cudssMatrixGetCsr)(
        matrix, nrows, ncols, nnz, rowStart, rowEnd, colIndices, values, indexType, valueType, mtype, mview, indexBase)


cdef cudssStatus_t _cudssMatrixSetValues(cudssMatrix_t matrix, void* values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixSetValues
    _check_or_init_cudss()
    if __cudssMatrixSetValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixSetValues is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, void*) noexcept nogil>__cudssMatrixSetValues)(
        matrix, values)


cdef cudssStatus_t _cudssMatrixSetCsrPointers(cudssMatrix_t matrix, void* rowOffsets, void* rowEnd, void* colIndices, void* values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixSetCsrPointers
    _check_or_init_cudss()
    if __cudssMatrixSetCsrPointers == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixSetCsrPointers is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, void*, void*, void*, void*) noexcept nogil>__cudssMatrixSetCsrPointers)(
        matrix, rowOffsets, rowEnd, colIndices, values)


cdef cudssStatus_t _cudssMatrixGetBatchDn(cudssMatrix_t matrix, int64_t* batchCount, void** nrows, void** ncols, void** ld, void*** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssLayout_t* layout) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixGetBatchDn
    _check_or_init_cudss()
    if __cudssMatrixGetBatchDn == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixGetBatchDn is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, int64_t*, void**, void**, void**, void***, cudaDataType_t*, cudaDataType_t*, cudssLayout_t*) noexcept nogil>__cudssMatrixGetBatchDn)(
        matrix, batchCount, nrows, ncols, ld, values, indexType, valueType, layout)


cdef cudssStatus_t _cudssMatrixGetBatchCsr(cudssMatrix_t matrix, int64_t* batchCount, void** nrows, void** ncols, void** nnz, void*** rowStart, void*** rowEnd, void*** colIndices, void*** values, cudaDataType_t* indexType, cudaDataType_t* valueType, cudssMatrixType_t* mtype, cudssMatrixViewType_t* mview, cudssIndexBase_t* indexBase) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixGetBatchCsr
    _check_or_init_cudss()
    if __cudssMatrixGetBatchCsr == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixGetBatchCsr is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, int64_t*, void**, void**, void**, void***, void***, void***, void***, cudaDataType_t*, cudaDataType_t*, cudssMatrixType_t*, cudssMatrixViewType_t*, cudssIndexBase_t*) noexcept nogil>__cudssMatrixGetBatchCsr)(
        matrix, batchCount, nrows, ncols, nnz, rowStart, rowEnd, colIndices, values, indexType, valueType, mtype, mview, indexBase)


cdef cudssStatus_t _cudssMatrixSetBatchValues(cudssMatrix_t matrix, void** values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixSetBatchValues
    _check_or_init_cudss()
    if __cudssMatrixSetBatchValues == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixSetBatchValues is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, void**) noexcept nogil>__cudssMatrixSetBatchValues)(
        matrix, values)


cdef cudssStatus_t _cudssMatrixSetBatchCsrPointers(cudssMatrix_t matrix, void** rowOffsets, void** rowEnd, void** colIndices, void** values) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixSetBatchCsrPointers
    _check_or_init_cudss()
    if __cudssMatrixSetBatchCsrPointers == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixSetBatchCsrPointers is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, void**, void**, void**, void**) noexcept nogil>__cudssMatrixSetBatchCsrPointers)(
        matrix, rowOffsets, rowEnd, colIndices, values)


cdef cudssStatus_t _cudssMatrixGetFormat(cudssMatrix_t matrix, int* format) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixGetFormat
    _check_or_init_cudss()
    if __cudssMatrixGetFormat == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixGetFormat is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, int*) noexcept nogil>__cudssMatrixGetFormat)(
        matrix, format)


cdef cudssStatus_t _cudssGetDeviceMemHandler(cudssHandle_t handle, cudssDeviceMemHandler_t* handler) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssGetDeviceMemHandler
    _check_or_init_cudss()
    if __cudssGetDeviceMemHandler == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssGetDeviceMemHandler is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, cudssDeviceMemHandler_t*) noexcept nogil>__cudssGetDeviceMemHandler)(
        handle, handler)


cdef cudssStatus_t _cudssSetDeviceMemHandler(cudssHandle_t handle, const cudssDeviceMemHandler_t* handler) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssSetDeviceMemHandler
    _check_or_init_cudss()
    if __cudssSetDeviceMemHandler == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssSetDeviceMemHandler is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, const cudssDeviceMemHandler_t*) noexcept nogil>__cudssSetDeviceMemHandler)(
        handle, handler)
