# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.7.0, generator version 0.3.1.dev1303+g031f1197f. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

import os
import site
import threading

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib

from libc.stddef cimport wchar_t
from libc.stdint cimport uintptr_t
from cpython cimport PyUnicode_AsWideCharString, PyMem_Free

# You must 'from .utils import NotSupportedError' before using this template

cdef extern from "windows.h" nogil:
    ctypedef void* HMODULE
    ctypedef void* HANDLE
    ctypedef void* FARPROC
    ctypedef unsigned long DWORD
    ctypedef const wchar_t *LPCWSTR
    ctypedef const char *LPCSTR

    cdef DWORD LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800
    cdef DWORD LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
    cdef DWORD LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100

    HMODULE _LoadLibraryExW "LoadLibraryExW"(
        LPCWSTR lpLibFileName,
        HANDLE hFile,
        DWORD dwFlags
    )

    FARPROC _GetProcAddress "GetProcAddress"(HMODULE hModule, LPCSTR lpProcName)

cdef inline uintptr_t LoadLibraryExW(str path, HANDLE hFile, DWORD dwFlags):
    cdef uintptr_t result
    cdef wchar_t* wpath = PyUnicode_AsWideCharString(path, NULL)
    with nogil:
        result = <uintptr_t>_LoadLibraryExW(
            wpath,
            hFile,
            dwFlags
        )
    PyMem_Free(wpath)
    return result

cdef inline void *GetProcAddress(uintptr_t hModule, const char* lpProcName) nogil:
    return _GetProcAddress(<HMODULE>hModule, lpProcName)

cdef int get_cuda_version():
    cdef int err, driver_ver = 0

    # Load driver to check version
    handle = LoadLibraryExW("nvcuda.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32)
    if handle == 0:
        raise NotSupportedError('CUDA driver is not found')
    cuDriverGetVersion = GetProcAddress(handle, 'cuDriverGetVersion')
    if cuDriverGetVersion == NULL:
        raise RuntimeError('Did not find cuDriverGetVersion symbol in nvcuda.dll')
    err = (<int (*)(int*) noexcept nogil>cuDriverGetVersion)(&driver_ver)
    if err != 0:
        raise RuntimeError(f'cuDriverGetVersion returned error code {err}')

    return driver_ver



###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
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
cdef void* __cudssCreateMg = NULL
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
cdef void* __cudssMatrixSetDistributionRow1d = NULL
cdef void* __cudssMatrixGetDistributionRow1d = NULL
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

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_cudss_init:
            return 0

        # Load library
        handle = <intptr_t>load_library()

        # Load function
        global __cudssConfigSet
        __cudssConfigSet = GetProcAddress(handle, 'cudssConfigSet')

        global __cudssConfigGet
        __cudssConfigGet = GetProcAddress(handle, 'cudssConfigGet')

        global __cudssDataSet
        __cudssDataSet = GetProcAddress(handle, 'cudssDataSet')

        global __cudssDataGet
        __cudssDataGet = GetProcAddress(handle, 'cudssDataGet')

        global __cudssExecute
        __cudssExecute = GetProcAddress(handle, 'cudssExecute')

        global __cudssSetStream
        __cudssSetStream = GetProcAddress(handle, 'cudssSetStream')

        global __cudssSetCommLayer
        __cudssSetCommLayer = GetProcAddress(handle, 'cudssSetCommLayer')

        global __cudssSetThreadingLayer
        __cudssSetThreadingLayer = GetProcAddress(handle, 'cudssSetThreadingLayer')

        global __cudssConfigCreate
        __cudssConfigCreate = GetProcAddress(handle, 'cudssConfigCreate')

        global __cudssConfigDestroy
        __cudssConfigDestroy = GetProcAddress(handle, 'cudssConfigDestroy')

        global __cudssDataCreate
        __cudssDataCreate = GetProcAddress(handle, 'cudssDataCreate')

        global __cudssDataDestroy
        __cudssDataDestroy = GetProcAddress(handle, 'cudssDataDestroy')

        global __cudssCreate
        __cudssCreate = GetProcAddress(handle, 'cudssCreate')

        global __cudssCreateMg
        __cudssCreateMg = GetProcAddress(handle, 'cudssCreateMg')

        global __cudssDestroy
        __cudssDestroy = GetProcAddress(handle, 'cudssDestroy')

        global __cudssGetProperty
        __cudssGetProperty = GetProcAddress(handle, 'cudssGetProperty')

        global __cudssMatrixCreateDn
        __cudssMatrixCreateDn = GetProcAddress(handle, 'cudssMatrixCreateDn')

        global __cudssMatrixCreateCsr
        __cudssMatrixCreateCsr = GetProcAddress(handle, 'cudssMatrixCreateCsr')

        global __cudssMatrixCreateBatchDn
        __cudssMatrixCreateBatchDn = GetProcAddress(handle, 'cudssMatrixCreateBatchDn')

        global __cudssMatrixCreateBatchCsr
        __cudssMatrixCreateBatchCsr = GetProcAddress(handle, 'cudssMatrixCreateBatchCsr')

        global __cudssMatrixDestroy
        __cudssMatrixDestroy = GetProcAddress(handle, 'cudssMatrixDestroy')

        global __cudssMatrixGetDn
        __cudssMatrixGetDn = GetProcAddress(handle, 'cudssMatrixGetDn')

        global __cudssMatrixGetCsr
        __cudssMatrixGetCsr = GetProcAddress(handle, 'cudssMatrixGetCsr')

        global __cudssMatrixSetValues
        __cudssMatrixSetValues = GetProcAddress(handle, 'cudssMatrixSetValues')

        global __cudssMatrixSetCsrPointers
        __cudssMatrixSetCsrPointers = GetProcAddress(handle, 'cudssMatrixSetCsrPointers')

        global __cudssMatrixGetBatchDn
        __cudssMatrixGetBatchDn = GetProcAddress(handle, 'cudssMatrixGetBatchDn')

        global __cudssMatrixGetBatchCsr
        __cudssMatrixGetBatchCsr = GetProcAddress(handle, 'cudssMatrixGetBatchCsr')

        global __cudssMatrixSetBatchValues
        __cudssMatrixSetBatchValues = GetProcAddress(handle, 'cudssMatrixSetBatchValues')

        global __cudssMatrixSetBatchCsrPointers
        __cudssMatrixSetBatchCsrPointers = GetProcAddress(handle, 'cudssMatrixSetBatchCsrPointers')

        global __cudssMatrixGetFormat
        __cudssMatrixGetFormat = GetProcAddress(handle, 'cudssMatrixGetFormat')

        global __cudssMatrixSetDistributionRow1d
        __cudssMatrixSetDistributionRow1d = GetProcAddress(handle, 'cudssMatrixSetDistributionRow1d')

        global __cudssMatrixGetDistributionRow1d
        __cudssMatrixGetDistributionRow1d = GetProcAddress(handle, 'cudssMatrixGetDistributionRow1d')

        global __cudssGetDeviceMemHandler
        __cudssGetDeviceMemHandler = GetProcAddress(handle, 'cudssGetDeviceMemHandler')

        global __cudssSetDeviceMemHandler
        __cudssSetDeviceMemHandler = GetProcAddress(handle, 'cudssSetDeviceMemHandler')

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

    global __cudssCreateMg
    data["__cudssCreateMg"] = <intptr_t>__cudssCreateMg

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

    global __cudssMatrixSetDistributionRow1d
    data["__cudssMatrixSetDistributionRow1d"] = <intptr_t>__cudssMatrixSetDistributionRow1d

    global __cudssMatrixGetDistributionRow1d
    data["__cudssMatrixGetDistributionRow1d"] = <intptr_t>__cudssMatrixGetDistributionRow1d

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


cdef cudssStatus_t _cudssExecute(cudssHandle_t handle, int phase, cudssConfig_t solverConfig, cudssData_t solverData, cudssMatrix_t inputMatrix, cudssMatrix_t solution, cudssMatrix_t rhs) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssExecute
    _check_or_init_cudss()
    if __cudssExecute == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssExecute is not found")
    return (<cudssStatus_t (*)(cudssHandle_t, int, cudssConfig_t, cudssData_t, cudssMatrix_t, cudssMatrix_t, cudssMatrix_t) noexcept nogil>__cudssExecute)(
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


cdef cudssStatus_t _cudssCreateMg(cudssHandle_t* handle_pt, int device_count, int* device_indices) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssCreateMg
    _check_or_init_cudss()
    if __cudssCreateMg == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssCreateMg is not found")
    return (<cudssStatus_t (*)(cudssHandle_t*, int, int*) noexcept nogil>__cudssCreateMg)(
        handle_pt, device_count, device_indices)


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


cdef cudssStatus_t _cudssMatrixSetDistributionRow1d(cudssMatrix_t matrix, int64_t first_row, int64_t last_row) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixSetDistributionRow1d
    _check_or_init_cudss()
    if __cudssMatrixSetDistributionRow1d == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixSetDistributionRow1d is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, int64_t, int64_t) noexcept nogil>__cudssMatrixSetDistributionRow1d)(
        matrix, first_row, last_row)


cdef cudssStatus_t _cudssMatrixGetDistributionRow1d(cudssMatrix_t matrix, int64_t* first_row, int64_t* last_row) except?_CUDSSSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cudssMatrixGetDistributionRow1d
    _check_or_init_cudss()
    if __cudssMatrixGetDistributionRow1d == NULL:
        with gil:
            raise FunctionNotFoundError("function cudssMatrixGetDistributionRow1d is not found")
    return (<cudssStatus_t (*)(cudssMatrix_t, int64_t*, int64_t*) noexcept nogil>__cudssMatrixGetDistributionRow1d)(
        matrix, first_row, last_row)


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
