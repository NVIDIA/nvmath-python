# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_cusolver_dso_version_suffix

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

cdef bint __py_cusolver_init = False
cdef void* __cuDriverGetVersion = NULL

cdef void* __cusolverGetProperty = NULL
cdef void* __cusolverGetVersion = NULL


cdef void* load_library(const int driver_ver) except* with gil:
    cdef void* handle
    for suffix in get_cusolver_dso_version_suffix(driver_ver):
        so_name = "libcusolver.so" + (f".{suffix}" if suffix else suffix)
        handle = dlopen(so_name.encode(), RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            break
    else:
        err_msg = dlerror()
        raise RuntimeError(f'Failed to dlopen libcusolver ({err_msg.decode()})')
    return handle


cdef int _check_or_init_cusolver() except -1 nogil:
    global __py_cusolver_init
    if __py_cusolver_init:
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

cdef cusolverStatus_t _cusolverGetProperty(libraryPropertyType type, int* value) except* nogil:
    global __cusolverGetProperty
    _check_or_init_cusolver()
    if __cusolverGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverGetProperty is not found")
    return (<cusolverStatus_t (*)(libraryPropertyType, int*) nogil>__cusolverGetProperty)(
        type, value)


cdef cusolverStatus_t _cusolverGetVersion(int* version) except* nogil:
    global __cusolverGetVersion
    _check_or_init_cusolver()
    if __cusolverGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cusolverGetVersion is not found")
    return (<cusolverStatus_t (*)(int*) nogil>__cusolverGetVersion)(
        version)
