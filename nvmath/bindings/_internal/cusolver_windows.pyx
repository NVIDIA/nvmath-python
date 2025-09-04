# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

from .cublas cimport load_library as load_cublas
from .cusparse cimport load_library as load_cusparse

import os
import site

import win32api

from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib

###############################################################################
# Wrapper init
###############################################################################

LOAD_LIBRARY_SEARCH_SYSTEM32     = 0x00000800
cdef bint __py_cusolver_init = False
cdef void* __cuDriverGetVersion = NULL

cdef void* __cusolverGetProperty = NULL
cdef void* __cusolverGetVersion = NULL


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef load_library(const int driver_ver):
    return load_nvidia_dynamic_lib("cusolver")._handle_uint


cdef int _check_or_init_cusolver() except -1 nogil:
    global __py_cusolver_init
    if __py_cusolver_init:
        return 0

    cdef int err, driver_ver
    with gil:
        # Load driver to check version
        try:
            handle = win32api.LoadLibraryEx("nvcuda.dll", 0, LOAD_LIBRARY_SEARCH_SYSTEM32)
        except Exception as e:
            raise NotSupportedError(f'CUDA driver is not found ({e})')
        global __cuDriverGetVersion
        if __cuDriverGetVersion == NULL:
            __cuDriverGetVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'cuDriverGetVersion')
            if __cuDriverGetVersion == NULL:
                raise RuntimeError('something went wrong')
        err = (<int (*)(int*) noexcept nogil>__cuDriverGetVersion)(&driver_ver)
        if err != 0:
            raise RuntimeError('something went wrong')

        # Load library
        handle = load_library(driver_ver)

        # Load function
        global __cusolverGetProperty
        try:
            __cusolverGetProperty = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverGetProperty')
        except:
            pass

        global __cusolverGetVersion
        try:
            __cusolverGetVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'cusolverGetVersion')
        except:
            pass

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
