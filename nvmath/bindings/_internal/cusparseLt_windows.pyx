# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.7.1 to 0.8.1, generator version 0.3.1.dev1565+g7fa82f8eb. Do not modify it directly.

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
cdef bint __py_cusparseLt_init = False

cdef void* __cusparseLtGetErrorName = NULL
cdef void* __cusparseLtGetErrorString = NULL
cdef void* __cusparseLtInit = NULL
cdef void* __cusparseLtDestroy = NULL
cdef void* __cusparseLtGetVersion = NULL
cdef void* __cusparseLtGetProperty = NULL
cdef void* __cusparseLtDenseDescriptorInit = NULL
cdef void* __cusparseLtStructuredDescriptorInit = NULL
cdef void* __cusparseLtMatDescriptorDestroy = NULL
cdef void* __cusparseLtMatDescSetAttribute = NULL
cdef void* __cusparseLtMatDescGetAttribute = NULL
cdef void* __cusparseLtMatmulDescriptorInit = NULL
cdef void* __cusparseLtMatmulDescSetAttribute = NULL
cdef void* __cusparseLtMatmulDescGetAttribute = NULL
cdef void* __cusparseLtMatmulAlgSelectionInit = NULL
cdef void* __cusparseLtMatmulAlgSetAttribute = NULL
cdef void* __cusparseLtMatmulAlgGetAttribute = NULL
cdef void* __cusparseLtMatmulGetWorkspace = NULL
cdef void* __cusparseLtMatmulPlanInit = NULL
cdef void* __cusparseLtMatmulPlanDestroy = NULL
cdef void* __cusparseLtMatmul = NULL
cdef void* __cusparseLtMatmulSearch = NULL
cdef void* __cusparseLtSpMMAPrune = NULL
cdef void* __cusparseLtSpMMAPruneCheck = NULL
cdef void* __cusparseLtSpMMAPrune2 = NULL
cdef void* __cusparseLtSpMMAPruneCheck2 = NULL
cdef void* __cusparseLtSpMMACompressedSize = NULL
cdef void* __cusparseLtSpMMACompress = NULL
cdef void* __cusparseLtSpMMACompressedSize2 = NULL
cdef void* __cusparseLtSpMMACompress2 = NULL
cdef void* __cusparseLtMatmulAlgSelectionDestroy = NULL


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef void* load_library(const int driver_ver) except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cusparseLt")._handle_uint
    return <void*>handle


cdef int _check_or_init_cusparseLt() except -1 nogil:
    global __py_cusparseLt_init
    if __py_cusparseLt_init:
        return 0

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_cusparseLt_init:
            return 0

        driver_ver = get_cuda_version()

        # Load library
        handle = <intptr_t>load_library(driver_ver)

        # Load function
        global __cusparseLtGetErrorName
        __cusparseLtGetErrorName = GetProcAddress(handle, 'cusparseLtGetErrorName')

        global __cusparseLtGetErrorString
        __cusparseLtGetErrorString = GetProcAddress(handle, 'cusparseLtGetErrorString')

        global __cusparseLtInit
        __cusparseLtInit = GetProcAddress(handle, 'cusparseLtInit')

        global __cusparseLtDestroy
        __cusparseLtDestroy = GetProcAddress(handle, 'cusparseLtDestroy')

        global __cusparseLtGetVersion
        __cusparseLtGetVersion = GetProcAddress(handle, 'cusparseLtGetVersion')

        global __cusparseLtGetProperty
        __cusparseLtGetProperty = GetProcAddress(handle, 'cusparseLtGetProperty')

        global __cusparseLtDenseDescriptorInit
        __cusparseLtDenseDescriptorInit = GetProcAddress(handle, 'cusparseLtDenseDescriptorInit')

        global __cusparseLtStructuredDescriptorInit
        __cusparseLtStructuredDescriptorInit = GetProcAddress(handle, 'cusparseLtStructuredDescriptorInit')

        global __cusparseLtMatDescriptorDestroy
        __cusparseLtMatDescriptorDestroy = GetProcAddress(handle, 'cusparseLtMatDescriptorDestroy')

        global __cusparseLtMatDescSetAttribute
        __cusparseLtMatDescSetAttribute = GetProcAddress(handle, 'cusparseLtMatDescSetAttribute')

        global __cusparseLtMatDescGetAttribute
        __cusparseLtMatDescGetAttribute = GetProcAddress(handle, 'cusparseLtMatDescGetAttribute')

        global __cusparseLtMatmulDescriptorInit
        __cusparseLtMatmulDescriptorInit = GetProcAddress(handle, 'cusparseLtMatmulDescriptorInit')

        global __cusparseLtMatmulDescSetAttribute
        __cusparseLtMatmulDescSetAttribute = GetProcAddress(handle, 'cusparseLtMatmulDescSetAttribute')

        global __cusparseLtMatmulDescGetAttribute
        __cusparseLtMatmulDescGetAttribute = GetProcAddress(handle, 'cusparseLtMatmulDescGetAttribute')

        global __cusparseLtMatmulAlgSelectionInit
        __cusparseLtMatmulAlgSelectionInit = GetProcAddress(handle, 'cusparseLtMatmulAlgSelectionInit')

        global __cusparseLtMatmulAlgSetAttribute
        __cusparseLtMatmulAlgSetAttribute = GetProcAddress(handle, 'cusparseLtMatmulAlgSetAttribute')

        global __cusparseLtMatmulAlgGetAttribute
        __cusparseLtMatmulAlgGetAttribute = GetProcAddress(handle, 'cusparseLtMatmulAlgGetAttribute')

        global __cusparseLtMatmulGetWorkspace
        __cusparseLtMatmulGetWorkspace = GetProcAddress(handle, 'cusparseLtMatmulGetWorkspace')

        global __cusparseLtMatmulPlanInit
        __cusparseLtMatmulPlanInit = GetProcAddress(handle, 'cusparseLtMatmulPlanInit')

        global __cusparseLtMatmulPlanDestroy
        __cusparseLtMatmulPlanDestroy = GetProcAddress(handle, 'cusparseLtMatmulPlanDestroy')

        global __cusparseLtMatmul
        __cusparseLtMatmul = GetProcAddress(handle, 'cusparseLtMatmul')

        global __cusparseLtMatmulSearch
        __cusparseLtMatmulSearch = GetProcAddress(handle, 'cusparseLtMatmulSearch')

        global __cusparseLtSpMMAPrune
        __cusparseLtSpMMAPrune = GetProcAddress(handle, 'cusparseLtSpMMAPrune')

        global __cusparseLtSpMMAPruneCheck
        __cusparseLtSpMMAPruneCheck = GetProcAddress(handle, 'cusparseLtSpMMAPruneCheck')

        global __cusparseLtSpMMAPrune2
        __cusparseLtSpMMAPrune2 = GetProcAddress(handle, 'cusparseLtSpMMAPrune2')

        global __cusparseLtSpMMAPruneCheck2
        __cusparseLtSpMMAPruneCheck2 = GetProcAddress(handle, 'cusparseLtSpMMAPruneCheck2')

        global __cusparseLtSpMMACompressedSize
        __cusparseLtSpMMACompressedSize = GetProcAddress(handle, 'cusparseLtSpMMACompressedSize')

        global __cusparseLtSpMMACompress
        __cusparseLtSpMMACompress = GetProcAddress(handle, 'cusparseLtSpMMACompress')

        global __cusparseLtSpMMACompressedSize2
        __cusparseLtSpMMACompressedSize2 = GetProcAddress(handle, 'cusparseLtSpMMACompressedSize2')

        global __cusparseLtSpMMACompress2
        __cusparseLtSpMMACompress2 = GetProcAddress(handle, 'cusparseLtSpMMACompress2')

        global __cusparseLtMatmulAlgSelectionDestroy
        __cusparseLtMatmulAlgSelectionDestroy = GetProcAddress(handle, 'cusparseLtMatmulAlgSelectionDestroy')

        __py_cusparseLt_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cusparseLt()
    cdef dict data = {}

    global __cusparseLtGetErrorName
    data["__cusparseLtGetErrorName"] = <intptr_t>__cusparseLtGetErrorName

    global __cusparseLtGetErrorString
    data["__cusparseLtGetErrorString"] = <intptr_t>__cusparseLtGetErrorString

    global __cusparseLtInit
    data["__cusparseLtInit"] = <intptr_t>__cusparseLtInit

    global __cusparseLtDestroy
    data["__cusparseLtDestroy"] = <intptr_t>__cusparseLtDestroy

    global __cusparseLtGetVersion
    data["__cusparseLtGetVersion"] = <intptr_t>__cusparseLtGetVersion

    global __cusparseLtGetProperty
    data["__cusparseLtGetProperty"] = <intptr_t>__cusparseLtGetProperty

    global __cusparseLtDenseDescriptorInit
    data["__cusparseLtDenseDescriptorInit"] = <intptr_t>__cusparseLtDenseDescriptorInit

    global __cusparseLtStructuredDescriptorInit
    data["__cusparseLtStructuredDescriptorInit"] = <intptr_t>__cusparseLtStructuredDescriptorInit

    global __cusparseLtMatDescriptorDestroy
    data["__cusparseLtMatDescriptorDestroy"] = <intptr_t>__cusparseLtMatDescriptorDestroy

    global __cusparseLtMatDescSetAttribute
    data["__cusparseLtMatDescSetAttribute"] = <intptr_t>__cusparseLtMatDescSetAttribute

    global __cusparseLtMatDescGetAttribute
    data["__cusparseLtMatDescGetAttribute"] = <intptr_t>__cusparseLtMatDescGetAttribute

    global __cusparseLtMatmulDescriptorInit
    data["__cusparseLtMatmulDescriptorInit"] = <intptr_t>__cusparseLtMatmulDescriptorInit

    global __cusparseLtMatmulDescSetAttribute
    data["__cusparseLtMatmulDescSetAttribute"] = <intptr_t>__cusparseLtMatmulDescSetAttribute

    global __cusparseLtMatmulDescGetAttribute
    data["__cusparseLtMatmulDescGetAttribute"] = <intptr_t>__cusparseLtMatmulDescGetAttribute

    global __cusparseLtMatmulAlgSelectionInit
    data["__cusparseLtMatmulAlgSelectionInit"] = <intptr_t>__cusparseLtMatmulAlgSelectionInit

    global __cusparseLtMatmulAlgSetAttribute
    data["__cusparseLtMatmulAlgSetAttribute"] = <intptr_t>__cusparseLtMatmulAlgSetAttribute

    global __cusparseLtMatmulAlgGetAttribute
    data["__cusparseLtMatmulAlgGetAttribute"] = <intptr_t>__cusparseLtMatmulAlgGetAttribute

    global __cusparseLtMatmulGetWorkspace
    data["__cusparseLtMatmulGetWorkspace"] = <intptr_t>__cusparseLtMatmulGetWorkspace

    global __cusparseLtMatmulPlanInit
    data["__cusparseLtMatmulPlanInit"] = <intptr_t>__cusparseLtMatmulPlanInit

    global __cusparseLtMatmulPlanDestroy
    data["__cusparseLtMatmulPlanDestroy"] = <intptr_t>__cusparseLtMatmulPlanDestroy

    global __cusparseLtMatmul
    data["__cusparseLtMatmul"] = <intptr_t>__cusparseLtMatmul

    global __cusparseLtMatmulSearch
    data["__cusparseLtMatmulSearch"] = <intptr_t>__cusparseLtMatmulSearch

    global __cusparseLtSpMMAPrune
    data["__cusparseLtSpMMAPrune"] = <intptr_t>__cusparseLtSpMMAPrune

    global __cusparseLtSpMMAPruneCheck
    data["__cusparseLtSpMMAPruneCheck"] = <intptr_t>__cusparseLtSpMMAPruneCheck

    global __cusparseLtSpMMAPrune2
    data["__cusparseLtSpMMAPrune2"] = <intptr_t>__cusparseLtSpMMAPrune2

    global __cusparseLtSpMMAPruneCheck2
    data["__cusparseLtSpMMAPruneCheck2"] = <intptr_t>__cusparseLtSpMMAPruneCheck2

    global __cusparseLtSpMMACompressedSize
    data["__cusparseLtSpMMACompressedSize"] = <intptr_t>__cusparseLtSpMMACompressedSize

    global __cusparseLtSpMMACompress
    data["__cusparseLtSpMMACompress"] = <intptr_t>__cusparseLtSpMMACompress

    global __cusparseLtSpMMACompressedSize2
    data["__cusparseLtSpMMACompressedSize2"] = <intptr_t>__cusparseLtSpMMACompressedSize2

    global __cusparseLtSpMMACompress2
    data["__cusparseLtSpMMACompress2"] = <intptr_t>__cusparseLtSpMMACompress2

    global __cusparseLtMatmulAlgSelectionDestroy
    data["__cusparseLtMatmulAlgSelectionDestroy"] = <intptr_t>__cusparseLtMatmulAlgSelectionDestroy

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

cdef const char* _cusparseLtGetErrorName(cusparseStatus_t status) except?NULL nogil:
    global __cusparseLtGetErrorName
    _check_or_init_cusparseLt()
    if __cusparseLtGetErrorName == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtGetErrorName is not found")
    return (<const char* (*)(cusparseStatus_t) noexcept nogil>__cusparseLtGetErrorName)(
        status)


cdef const char* _cusparseLtGetErrorString(cusparseStatus_t status) except?NULL nogil:
    global __cusparseLtGetErrorString
    _check_or_init_cusparseLt()
    if __cusparseLtGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtGetErrorString is not found")
    return (<const char* (*)(cusparseStatus_t) noexcept nogil>__cusparseLtGetErrorString)(
        status)


cdef cusparseStatus_t _cusparseLtInit(cusparseLtHandle_t* handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtInit
    _check_or_init_cusparseLt()
    if __cusparseLtInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtInit is not found")
    return (<cusparseStatus_t (*)(cusparseLtHandle_t*) noexcept nogil>__cusparseLtInit)(
        handle)


cdef cusparseStatus_t _cusparseLtDestroy(const cusparseLtHandle_t* handle) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtDestroy
    _check_or_init_cusparseLt()
    if __cusparseLtDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtDestroy is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*) noexcept nogil>__cusparseLtDestroy)(
        handle)


cdef cusparseStatus_t _cusparseLtGetVersion(const cusparseLtHandle_t* handle, int* version) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtGetVersion
    _check_or_init_cusparseLt()
    if __cusparseLtGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtGetVersion is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, int*) noexcept nogil>__cusparseLtGetVersion)(
        handle, version)


cdef cusparseStatus_t _cusparseLtGetProperty(libraryPropertyType propertyType, int* value) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtGetProperty
    _check_or_init_cusparseLt()
    if __cusparseLtGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtGetProperty is not found")
    return (<cusparseStatus_t (*)(libraryPropertyType, int*) noexcept nogil>__cusparseLtGetProperty)(
        propertyType, value)


cdef cusparseStatus_t _cusparseLtDenseDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtDenseDescriptorInit
    _check_or_init_cusparseLt()
    if __cusparseLtDenseDescriptorInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtDenseDescriptorInit is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, cusparseLtMatDescriptor_t*, int64_t, int64_t, int64_t, uint32_t, cudaDataType, cusparseOrder_t) noexcept nogil>__cusparseLtDenseDescriptorInit)(
        handle, matDescr, rows, cols, ld, alignment, valueType, order)


cdef cusparseStatus_t _cusparseLtStructuredDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order, cusparseLtSparsity_t sparsity) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtStructuredDescriptorInit
    _check_or_init_cusparseLt()
    if __cusparseLtStructuredDescriptorInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtStructuredDescriptorInit is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, cusparseLtMatDescriptor_t*, int64_t, int64_t, int64_t, uint32_t, cudaDataType, cusparseOrder_t, cusparseLtSparsity_t) noexcept nogil>__cusparseLtStructuredDescriptorInit)(
        handle, matDescr, rows, cols, ld, alignment, valueType, order, sparsity)


cdef cusparseStatus_t _cusparseLtMatDescriptorDestroy(const cusparseLtMatDescriptor_t* matDescr) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatDescriptorDestroy
    _check_or_init_cusparseLt()
    if __cusparseLtMatDescriptorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatDescriptorDestroy is not found")
    return (<cusparseStatus_t (*)(const cusparseLtMatDescriptor_t*) noexcept nogil>__cusparseLtMatDescriptorDestroy)(
        matDescr)


cdef cusparseStatus_t _cusparseLtMatDescSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matmulDescr, cusparseLtMatDescAttribute_t matAttribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatDescSetAttribute
    _check_or_init_cusparseLt()
    if __cusparseLtMatDescSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatDescSetAttribute is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, cusparseLtMatDescriptor_t*, cusparseLtMatDescAttribute_t, const void*, size_t) noexcept nogil>__cusparseLtMatDescSetAttribute)(
        handle, matmulDescr, matAttribute, data, dataSize)


cdef cusparseStatus_t _cusparseLtMatDescGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* matmulDescr, cusparseLtMatDescAttribute_t matAttribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatDescGetAttribute
    _check_or_init_cusparseLt()
    if __cusparseLtMatDescGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatDescGetAttribute is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatDescriptor_t*, cusparseLtMatDescAttribute_t, void*, size_t) noexcept nogil>__cusparseLtMatDescGetAttribute)(
        handle, matmulDescr, matAttribute, data, dataSize)


cdef cusparseStatus_t _cusparseLtMatmulDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr, cusparseOperation_t opA, cusparseOperation_t opB, const cusparseLtMatDescriptor_t* matA, const cusparseLtMatDescriptor_t* matB, const cusparseLtMatDescriptor_t* matC, const cusparseLtMatDescriptor_t* matD, cusparseComputeType computeType) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulDescriptorInit
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulDescriptorInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulDescriptorInit is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, cusparseLtMatmulDescriptor_t*, cusparseOperation_t, cusparseOperation_t, const cusparseLtMatDescriptor_t*, const cusparseLtMatDescriptor_t*, const cusparseLtMatDescriptor_t*, const cusparseLtMatDescriptor_t*, cusparseComputeType) noexcept nogil>__cusparseLtMatmulDescriptorInit)(
        handle, matmulDescr, opA, opB, matA, matB, matC, matD, computeType)


cdef cusparseStatus_t _cusparseLtMatmulDescSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulDescAttribute_t matmulAttribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulDescSetAttribute
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulDescSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulDescSetAttribute is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, cusparseLtMatmulDescriptor_t*, cusparseLtMatmulDescAttribute_t, const void*, size_t) noexcept nogil>__cusparseLtMatmulDescSetAttribute)(
        handle, matmulDescr, matmulAttribute, data, dataSize)


cdef cusparseStatus_t _cusparseLtMatmulDescGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulDescAttribute_t matmulAttribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulDescGetAttribute
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulDescGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulDescGetAttribute is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatmulDescriptor_t*, cusparseLtMatmulDescAttribute_t, void*, size_t) noexcept nogil>__cusparseLtMatmulDescGetAttribute)(
        handle, matmulDescr, matmulAttribute, data, dataSize)


cdef cusparseStatus_t _cusparseLtMatmulAlgSelectionInit(const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection, const cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulAlg_t alg) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulAlgSelectionInit
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulAlgSelectionInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulAlgSelectionInit is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, cusparseLtMatmulAlgSelection_t*, const cusparseLtMatmulDescriptor_t*, cusparseLtMatmulAlg_t) noexcept nogil>__cusparseLtMatmulAlgSelectionInit)(
        handle, algSelection, matmulDescr, alg)


cdef cusparseStatus_t _cusparseLtMatmulAlgSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection, cusparseLtMatmulAlgAttribute_t attribute, const void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulAlgSetAttribute
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulAlgSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulAlgSetAttribute is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, cusparseLtMatmulAlgSelection_t*, cusparseLtMatmulAlgAttribute_t, const void*, size_t) noexcept nogil>__cusparseLtMatmulAlgSetAttribute)(
        handle, algSelection, attribute, data, dataSize)


cdef cusparseStatus_t _cusparseLtMatmulAlgGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatmulAlgSelection_t* algSelection, cusparseLtMatmulAlgAttribute_t attribute, void* data, size_t dataSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulAlgGetAttribute
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulAlgGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulAlgGetAttribute is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatmulAlgSelection_t*, cusparseLtMatmulAlgAttribute_t, void*, size_t) noexcept nogil>__cusparseLtMatmulAlgGetAttribute)(
        handle, algSelection, attribute, data, dataSize)


cdef cusparseStatus_t _cusparseLtMatmulGetWorkspace(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* workspaceSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulGetWorkspace
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulGetWorkspace == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulGetWorkspace is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatmulPlan_t*, size_t*) noexcept nogil>__cusparseLtMatmulGetWorkspace)(
        handle, plan, workspaceSize)


cdef cusparseStatus_t _cusparseLtMatmulPlanInit(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const cusparseLtMatmulDescriptor_t* matmulDescr, const cusparseLtMatmulAlgSelection_t* algSelection) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulPlanInit
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulPlanInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulPlanInit is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, cusparseLtMatmulPlan_t*, const cusparseLtMatmulDescriptor_t*, const cusparseLtMatmulAlgSelection_t*) noexcept nogil>__cusparseLtMatmulPlanInit)(
        handle, plan, matmulDescr, algSelection)


cdef cusparseStatus_t _cusparseLtMatmulPlanDestroy(const cusparseLtMatmulPlan_t* plan) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulPlanDestroy
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulPlanDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulPlanDestroy is not found")
    return (<cusparseStatus_t (*)(const cusparseLtMatmulPlan_t*) noexcept nogil>__cusparseLtMatmulPlanDestroy)(
        plan)


cdef cusparseStatus_t _cusparseLtMatmul(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, cudaStream_t* streams, int32_t numStreams) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmul
    _check_or_init_cusparseLt()
    if __cusparseLtMatmul == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmul is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatmulPlan_t*, const void*, const void*, const void*, const void*, const void*, void*, void*, cudaStream_t*, int32_t) noexcept nogil>__cusparseLtMatmul)(
        handle, plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, streams, numStreams)


cdef cusparseStatus_t _cusparseLtMatmulSearch(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, cudaStream_t* streams, int32_t numStreams) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulSearch
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulSearch == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulSearch is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, cusparseLtMatmulPlan_t*, const void*, const void*, const void*, const void*, const void*, void*, void*, cudaStream_t*, int32_t) noexcept nogil>__cusparseLtMatmulSearch)(
        handle, plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, streams, numStreams)


cdef cusparseStatus_t _cusparseLtSpMMAPrune(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtSpMMAPrune
    _check_or_init_cusparseLt()
    if __cusparseLtSpMMAPrune == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtSpMMAPrune is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatmulDescriptor_t*, const void*, void*, cusparseLtPruneAlg_t, cudaStream_t) noexcept nogil>__cusparseLtSpMMAPrune)(
        handle, matmulDescr, d_in, d_out, pruneAlg, stream)


cdef cusparseStatus_t _cusparseLtSpMMAPruneCheck(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, int* valid, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtSpMMAPruneCheck
    _check_or_init_cusparseLt()
    if __cusparseLtSpMMAPruneCheck == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtSpMMAPruneCheck is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatmulDescriptor_t*, const void*, int*, cudaStream_t) noexcept nogil>__cusparseLtSpMMAPruneCheck)(
        handle, matmulDescr, d_in, valid, stream)


cdef cusparseStatus_t _cusparseLtSpMMAPrune2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtSpMMAPrune2
    _check_or_init_cusparseLt()
    if __cusparseLtSpMMAPrune2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtSpMMAPrune2 is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatDescriptor_t*, int, cusparseOperation_t, const void*, void*, cusparseLtPruneAlg_t, cudaStream_t) noexcept nogil>__cusparseLtSpMMAPrune2)(
        handle, sparseMatDescr, isSparseA, op, d_in, d_out, pruneAlg, stream)


cdef cusparseStatus_t _cusparseLtSpMMAPruneCheck2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_in, int* d_valid, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtSpMMAPruneCheck2
    _check_or_init_cusparseLt()
    if __cusparseLtSpMMAPruneCheck2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtSpMMAPruneCheck2 is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatDescriptor_t*, int, cusparseOperation_t, const void*, int*, cudaStream_t) noexcept nogil>__cusparseLtSpMMAPruneCheck2)(
        handle, sparseMatDescr, isSparseA, op, d_in, d_valid, stream)


cdef cusparseStatus_t _cusparseLtSpMMACompressedSize(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* compressedSize, size_t* compressedBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtSpMMACompressedSize
    _check_or_init_cusparseLt()
    if __cusparseLtSpMMACompressedSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtSpMMACompressedSize is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatmulPlan_t*, size_t*, size_t*) noexcept nogil>__cusparseLtSpMMACompressedSize)(
        handle, plan, compressedSize, compressedBufferSize)


cdef cusparseStatus_t _cusparseLtSpMMACompress(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* d_dense, void* d_compressed, void* d_compressed_buffer, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtSpMMACompress
    _check_or_init_cusparseLt()
    if __cusparseLtSpMMACompress == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtSpMMACompress is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatmulPlan_t*, const void*, void*, void*, cudaStream_t) noexcept nogil>__cusparseLtSpMMACompress)(
        handle, plan, d_dense, d_compressed, d_compressed_buffer, stream)


cdef cusparseStatus_t _cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, size_t* compressedSize, size_t* compressedBufferSize) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtSpMMACompressedSize2
    _check_or_init_cusparseLt()
    if __cusparseLtSpMMACompressedSize2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtSpMMACompressedSize2 is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatDescriptor_t*, size_t*, size_t*) noexcept nogil>__cusparseLtSpMMACompressedSize2)(
        handle, sparseMatDescr, compressedSize, compressedBufferSize)


cdef cusparseStatus_t _cusparseLtSpMMACompress2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_dense, void* d_compressed, void* d_compressed_buffer, cudaStream_t stream) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtSpMMACompress2
    _check_or_init_cusparseLt()
    if __cusparseLtSpMMACompress2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtSpMMACompress2 is not found")
    return (<cusparseStatus_t (*)(const cusparseLtHandle_t*, const cusparseLtMatDescriptor_t*, int, cusparseOperation_t, const void*, void*, void*, cudaStream_t) noexcept nogil>__cusparseLtSpMMACompress2)(
        handle, sparseMatDescr, isSparseA, op, d_dense, d_compressed, d_compressed_buffer, stream)


cdef cusparseStatus_t _cusparseLtMatmulAlgSelectionDestroy(const cusparseLtMatmulAlgSelection_t* algSelection) except?_CUSPARSESTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusparseLtMatmulAlgSelectionDestroy
    _check_or_init_cusparseLt()
    if __cusparseLtMatmulAlgSelectionDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cusparseLtMatmulAlgSelectionDestroy is not found")
    return (<cusparseStatus_t (*)(const cusparseLtMatmulAlgSelection_t*) noexcept nogil>__cusparseLtMatmulAlgSelectionDestroy)(
        algSelection)
