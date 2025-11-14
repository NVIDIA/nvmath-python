# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 2.11.4 to 2.28.3. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

import threading

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
cdef bint __py_nccl_init = False

cdef void* __ncclGetVersion = NULL
cdef void* __ncclGetUniqueId = NULL
cdef void* __ncclCommInitRank = NULL
cdef void* __ncclCommDestroy = NULL
cdef void* __ncclCommAbort = NULL
cdef void* __ncclGetErrorString = NULL
cdef void* __ncclCommCount = NULL
cdef void* __ncclCommCuDevice = NULL
cdef void* __ncclCommUserRank = NULL
cdef void* __ncclGetLastError = NULL
cdef void* __ncclCommFinalize = NULL


cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("nccl")._handle_uint
    return <void*>handle


cdef int _check_or_init_nccl() except -1 nogil:
    global __py_nccl_init
    if __py_nccl_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Load function
        global __ncclGetVersion
        __ncclGetVersion = dlsym(RTLD_DEFAULT, 'ncclGetVersion')
        if __ncclGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetVersion = dlsym(handle, 'ncclGetVersion')

        global __ncclGetUniqueId
        __ncclGetUniqueId = dlsym(RTLD_DEFAULT, 'ncclGetUniqueId')
        if __ncclGetUniqueId == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetUniqueId = dlsym(handle, 'ncclGetUniqueId')

        global __ncclCommInitRank
        __ncclCommInitRank = dlsym(RTLD_DEFAULT, 'ncclCommInitRank')
        if __ncclCommInitRank == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommInitRank = dlsym(handle, 'ncclCommInitRank')

        global __ncclCommDestroy
        __ncclCommDestroy = dlsym(RTLD_DEFAULT, 'ncclCommDestroy')
        if __ncclCommDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommDestroy = dlsym(handle, 'ncclCommDestroy')

        global __ncclCommAbort
        __ncclCommAbort = dlsym(RTLD_DEFAULT, 'ncclCommAbort')
        if __ncclCommAbort == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommAbort = dlsym(handle, 'ncclCommAbort')

        global __ncclGetErrorString
        __ncclGetErrorString = dlsym(RTLD_DEFAULT, 'ncclGetErrorString')
        if __ncclGetErrorString == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetErrorString = dlsym(handle, 'ncclGetErrorString')

        global __ncclCommCount
        __ncclCommCount = dlsym(RTLD_DEFAULT, 'ncclCommCount')
        if __ncclCommCount == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommCount = dlsym(handle, 'ncclCommCount')

        global __ncclCommCuDevice
        __ncclCommCuDevice = dlsym(RTLD_DEFAULT, 'ncclCommCuDevice')
        if __ncclCommCuDevice == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommCuDevice = dlsym(handle, 'ncclCommCuDevice')

        global __ncclCommUserRank
        __ncclCommUserRank = dlsym(RTLD_DEFAULT, 'ncclCommUserRank')
        if __ncclCommUserRank == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommUserRank = dlsym(handle, 'ncclCommUserRank')

        global __ncclGetLastError
        __ncclGetLastError = dlsym(RTLD_DEFAULT, 'ncclGetLastError')
        if __ncclGetLastError == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclGetLastError = dlsym(handle, 'ncclGetLastError')

        global __ncclCommFinalize
        __ncclCommFinalize = dlsym(RTLD_DEFAULT, 'ncclCommFinalize')
        if __ncclCommFinalize == NULL:
            if handle == NULL:
                handle = load_library()
            __ncclCommFinalize = dlsym(handle, 'ncclCommFinalize')
        __py_nccl_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nccl()
    cdef dict data = {}

    global __ncclGetVersion
    data["__ncclGetVersion"] = <intptr_t>__ncclGetVersion

    global __ncclGetUniqueId
    data["__ncclGetUniqueId"] = <intptr_t>__ncclGetUniqueId

    global __ncclCommInitRank
    data["__ncclCommInitRank"] = <intptr_t>__ncclCommInitRank

    global __ncclCommDestroy
    data["__ncclCommDestroy"] = <intptr_t>__ncclCommDestroy

    global __ncclCommAbort
    data["__ncclCommAbort"] = <intptr_t>__ncclCommAbort

    global __ncclGetErrorString
    data["__ncclGetErrorString"] = <intptr_t>__ncclGetErrorString

    global __ncclCommCount
    data["__ncclCommCount"] = <intptr_t>__ncclCommCount

    global __ncclCommCuDevice
    data["__ncclCommCuDevice"] = <intptr_t>__ncclCommCuDevice

    global __ncclCommUserRank
    data["__ncclCommUserRank"] = <intptr_t>__ncclCommUserRank

    global __ncclGetLastError
    data["__ncclGetLastError"] = <intptr_t>__ncclGetLastError

    global __ncclCommFinalize
    data["__ncclCommFinalize"] = <intptr_t>__ncclCommFinalize

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

cdef ncclResult_t _ncclGetVersion(int* version) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetVersion
    _check_or_init_nccl()
    if __ncclGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetVersion is not found")
    return (<ncclResult_t (*)(int*) noexcept nogil>__ncclGetVersion)(
        version)


cdef ncclResult_t _ncclGetUniqueId(ncclUniqueId* uniqueId) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclGetUniqueId
    _check_or_init_nccl()
    if __ncclGetUniqueId == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetUniqueId is not found")
    return (<ncclResult_t (*)(ncclUniqueId*) noexcept nogil>__ncclGetUniqueId)(
        uniqueId)


cdef ncclResult_t _ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommInitRank
    _check_or_init_nccl()
    if __ncclCommInitRank == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommInitRank is not found")
    return (<ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int) noexcept nogil>__ncclCommInitRank)(
        comm, nranks, commId, rank)


cdef ncclResult_t _ncclCommDestroy(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommDestroy
    _check_or_init_nccl()
    if __ncclCommDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommDestroy is not found")
    return (<ncclResult_t (*)(ncclComm_t) noexcept nogil>__ncclCommDestroy)(
        comm)


cdef ncclResult_t _ncclCommAbort(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommAbort
    _check_or_init_nccl()
    if __ncclCommAbort == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommAbort is not found")
    return (<ncclResult_t (*)(ncclComm_t) noexcept nogil>__ncclCommAbort)(
        comm)


cdef const char* _ncclGetErrorString(ncclResult_t result) except?NULL nogil:
    global __ncclGetErrorString
    _check_or_init_nccl()
    if __ncclGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetErrorString is not found")
    return (<const char* (*)(ncclResult_t) noexcept nogil>__ncclGetErrorString)(
        result)


cdef ncclResult_t _ncclCommCount(const ncclComm_t comm, int* count) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommCount
    _check_or_init_nccl()
    if __ncclCommCount == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommCount is not found")
    return (<ncclResult_t (*)(const ncclComm_t, int*) noexcept nogil>__ncclCommCount)(
        comm, count)


cdef ncclResult_t _ncclCommCuDevice(const ncclComm_t comm, int* device) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommCuDevice
    _check_or_init_nccl()
    if __ncclCommCuDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommCuDevice is not found")
    return (<ncclResult_t (*)(const ncclComm_t, int*) noexcept nogil>__ncclCommCuDevice)(
        comm, device)


cdef ncclResult_t _ncclCommUserRank(const ncclComm_t comm, int* rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommUserRank
    _check_or_init_nccl()
    if __ncclCommUserRank == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommUserRank is not found")
    return (<ncclResult_t (*)(const ncclComm_t, int*) noexcept nogil>__ncclCommUserRank)(
        comm, rank)


cdef const char* _ncclGetLastError(ncclComm_t comm) except?NULL nogil:
    global __ncclGetLastError
    _check_or_init_nccl()
    if __ncclGetLastError == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclGetLastError is not found")
    return (<const char* (*)(ncclComm_t) noexcept nogil>__ncclGetLastError)(
        comm)


cdef ncclResult_t _ncclCommFinalize(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclCommFinalize
    _check_or_init_nccl()
    if __ncclCommFinalize == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclCommFinalize is not found")
    return (<ncclResult_t (*)(ncclComm_t) noexcept nogil>__ncclCommFinalize)(
        comm)
