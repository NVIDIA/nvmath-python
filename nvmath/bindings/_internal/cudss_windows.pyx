# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.5.0. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

import os
import site

import win32api

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################

LOAD_LIBRARY_SEARCH_SYSTEM32     = 0x00000800
cdef bint __py_cudss_init = False
cdef void* __cuDriverGetVersion = NULL

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


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cudss")._handle_uint
    return <void*>handle


cdef int _check_or_init_cudss() except -1 nogil:
    global __py_cudss_init
    if __py_cudss_init:
        return 0

    with gil:
        # Load library
        handle = <intptr_t>load_library()

        # Load function
        global __cudssConfigSet
        try:
            __cudssConfigSet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssConfigSet')
        except:
            pass

        global __cudssConfigGet
        try:
            __cudssConfigGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssConfigGet')
        except:
            pass

        global __cudssDataSet
        try:
            __cudssDataSet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssDataSet')
        except:
            pass

        global __cudssDataGet
        try:
            __cudssDataGet = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssDataGet')
        except:
            pass

        global __cudssExecute
        try:
            __cudssExecute = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssExecute')
        except:
            pass

        global __cudssSetStream
        try:
            __cudssSetStream = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssSetStream')
        except:
            pass

        global __cudssSetCommLayer
        try:
            __cudssSetCommLayer = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssSetCommLayer')
        except:
            pass

        global __cudssSetThreadingLayer
        try:
            __cudssSetThreadingLayer = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssSetThreadingLayer')
        except:
            pass

        global __cudssConfigCreate
        try:
            __cudssConfigCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssConfigCreate')
        except:
            pass

        global __cudssConfigDestroy
        try:
            __cudssConfigDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssConfigDestroy')
        except:
            pass

        global __cudssDataCreate
        try:
            __cudssDataCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssDataCreate')
        except:
            pass

        global __cudssDataDestroy
        try:
            __cudssDataDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssDataDestroy')
        except:
            pass

        global __cudssCreate
        try:
            __cudssCreate = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssCreate')
        except:
            pass

        global __cudssDestroy
        try:
            __cudssDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssDestroy')
        except:
            pass

        global __cudssGetProperty
        try:
            __cudssGetProperty = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssGetProperty')
        except:
            pass

        global __cudssMatrixCreateDn
        try:
            __cudssMatrixCreateDn = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixCreateDn')
        except:
            pass

        global __cudssMatrixCreateCsr
        try:
            __cudssMatrixCreateCsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixCreateCsr')
        except:
            pass

        global __cudssMatrixCreateBatchDn
        try:
            __cudssMatrixCreateBatchDn = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixCreateBatchDn')
        except:
            pass

        global __cudssMatrixCreateBatchCsr
        try:
            __cudssMatrixCreateBatchCsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixCreateBatchCsr')
        except:
            pass

        global __cudssMatrixDestroy
        try:
            __cudssMatrixDestroy = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixDestroy')
        except:
            pass

        global __cudssMatrixGetDn
        try:
            __cudssMatrixGetDn = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixGetDn')
        except:
            pass

        global __cudssMatrixGetCsr
        try:
            __cudssMatrixGetCsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixGetCsr')
        except:
            pass

        global __cudssMatrixSetValues
        try:
            __cudssMatrixSetValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixSetValues')
        except:
            pass

        global __cudssMatrixSetCsrPointers
        try:
            __cudssMatrixSetCsrPointers = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixSetCsrPointers')
        except:
            pass

        global __cudssMatrixGetBatchDn
        try:
            __cudssMatrixGetBatchDn = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixGetBatchDn')
        except:
            pass

        global __cudssMatrixGetBatchCsr
        try:
            __cudssMatrixGetBatchCsr = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixGetBatchCsr')
        except:
            pass

        global __cudssMatrixSetBatchValues
        try:
            __cudssMatrixSetBatchValues = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixSetBatchValues')
        except:
            pass

        global __cudssMatrixSetBatchCsrPointers
        try:
            __cudssMatrixSetBatchCsrPointers = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixSetBatchCsrPointers')
        except:
            pass

        global __cudssMatrixGetFormat
        try:
            __cudssMatrixGetFormat = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssMatrixGetFormat')
        except:
            pass

        global __cudssGetDeviceMemHandler
        try:
            __cudssGetDeviceMemHandler = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssGetDeviceMemHandler')
        except:
            pass

        global __cudssSetDeviceMemHandler
        try:
            __cudssSetDeviceMemHandler = <void*><intptr_t>win32api.GetProcAddress(handle, 'cudssSetDeviceMemHandler')
        except:
            pass

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
