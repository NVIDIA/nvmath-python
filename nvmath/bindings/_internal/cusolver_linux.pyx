# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.1 to 13.2.1, generator version 0.3.1.dev1380+g2c74a7741. Do not modify it directly.

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
cdef bint __py_cusolver_init = False

cdef void* __cusolverGetProperty = NULL
cdef void* __cusolverGetVersion = NULL


cdef void* load_library(const int driver_ver) except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cusolver")._handle_uint
    return <void*>handle

cdef int _check_or_init_cusolver() except -1 nogil:
    global __py_cusolver_init
    if __py_cusolver_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_cusolver_init:
            return 0

        driver_ver = get_cuda_version()

        # Load function
        global __cusolverGetProperty
        __cusolverGetProperty = dlsym(RTLD_DEFAULT, 'cusolverGetProperty')
        if __cusolverGetProperty == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cusolverGetProperty = dlsym(handle, 'cusolverGetProperty')

        global __cusolverGetVersion
        __cusolverGetVersion = dlsym(RTLD_DEFAULT, 'cusolverGetVersion')
        if __cusolverGetVersion == NULL:
            if handle == NULL:
                handle = load_library(driver_ver)
            __cusolverGetVersion = dlsym(handle, 'cusolverGetVersion')

        __py_cusolver_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cusolver()
    cdef dict data = {}

    global __cusolverGetProperty
    data["__cusolverGetProperty"] = <intptr_t>__cusolverGetProperty

    global __cusolverGetVersion
    data["__cusolverGetVersion"] = <intptr_t>__cusolverGetVersion

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

cdef cusolverStatus_t _cusolverGetProperty(libraryPropertyType type, int* value) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverGetProperty
    _check_or_init_cusolver()
    if __cusolverGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverGetProperty is not found")
    return (<cusolverStatus_t (*)(libraryPropertyType, int*) noexcept nogil>__cusolverGetProperty)(
        type, value)


cdef cusolverStatus_t _cusolverGetVersion(int* version) except?_CUSOLVERSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cusolverGetVersion
    _check_or_init_cusolver()
    if __cusolverGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverGetVersion is not found")
    return (<cusolverStatus_t (*)(int*) noexcept nogil>__cusolverGetVersion)(
        version)
