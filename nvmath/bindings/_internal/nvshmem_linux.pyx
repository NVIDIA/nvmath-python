# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 3.1.7. Do not modify it directly.

cimport cython
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

cdef bint __py_nvshmem_init = False

cdef void* __nvshmemx_init_status = NULL
cdef void* __nvshmem_my_pe = NULL
cdef void* __nvshmem_n_pes = NULL
cdef void* __nvshmem_malloc = NULL
cdef void* __nvshmem_calloc = NULL
cdef void* __nvshmem_align = NULL
cdef void* __nvshmem_free = NULL
cdef void* __nvshmem_ptr = NULL
cdef void* __nvshmem_int_p = NULL
cdef void* __nvshmem_team_my_pe = NULL
cdef void* __nvshmemx_barrier_all_on_stream = NULL
cdef void* __nvshmemx_sync_all_on_stream = NULL
cdef void* __nvshmemx_hostlib_init_attr = NULL
cdef void* __nvshmemx_hostlib_finalize = NULL
cdef void* __nvshmemx_set_attr_uniqueid_args = NULL
cdef void* __nvshmemx_get_uniqueid = NULL


cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("nvshmem_host")._handle_uint
    return <void*>handle


cdef int _check_or_init_nvshmem() except -1 nogil:
    global __py_nvshmem_init
    if __py_nvshmem_init:
        return 0

    # Load function
    cdef void* handle = NULL
    global __nvshmemx_init_status
    __nvshmemx_init_status = dlsym(RTLD_DEFAULT, 'nvshmemx_init_status')
    if __nvshmemx_init_status == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_init_status = dlsym(handle, 'nvshmemx_init_status')

    global __nvshmem_my_pe
    __nvshmem_my_pe = dlsym(RTLD_DEFAULT, 'nvshmem_my_pe')
    if __nvshmem_my_pe == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_my_pe = dlsym(handle, 'nvshmem_my_pe')

    global __nvshmem_n_pes
    __nvshmem_n_pes = dlsym(RTLD_DEFAULT, 'nvshmem_n_pes')
    if __nvshmem_n_pes == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_n_pes = dlsym(handle, 'nvshmem_n_pes')

    global __nvshmem_malloc
    __nvshmem_malloc = dlsym(RTLD_DEFAULT, 'nvshmem_malloc')
    if __nvshmem_malloc == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_malloc = dlsym(handle, 'nvshmem_malloc')

    global __nvshmem_calloc
    __nvshmem_calloc = dlsym(RTLD_DEFAULT, 'nvshmem_calloc')
    if __nvshmem_calloc == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_calloc = dlsym(handle, 'nvshmem_calloc')

    global __nvshmem_align
    __nvshmem_align = dlsym(RTLD_DEFAULT, 'nvshmem_align')
    if __nvshmem_align == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_align = dlsym(handle, 'nvshmem_align')

    global __nvshmem_free
    __nvshmem_free = dlsym(RTLD_DEFAULT, 'nvshmem_free')
    if __nvshmem_free == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_free = dlsym(handle, 'nvshmem_free')

    global __nvshmem_ptr
    __nvshmem_ptr = dlsym(RTLD_DEFAULT, 'nvshmem_ptr')
    if __nvshmem_ptr == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_ptr = dlsym(handle, 'nvshmem_ptr')

    global __nvshmem_int_p
    __nvshmem_int_p = dlsym(RTLD_DEFAULT, 'nvshmem_int_p')
    if __nvshmem_int_p == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_int_p = dlsym(handle, 'nvshmem_int_p')

    global __nvshmem_team_my_pe
    __nvshmem_team_my_pe = dlsym(RTLD_DEFAULT, 'nvshmem_team_my_pe')
    if __nvshmem_team_my_pe == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_team_my_pe = dlsym(handle, 'nvshmem_team_my_pe')

    global __nvshmemx_barrier_all_on_stream
    __nvshmemx_barrier_all_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_barrier_all_on_stream')
    if __nvshmemx_barrier_all_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_barrier_all_on_stream = dlsym(handle, 'nvshmemx_barrier_all_on_stream')

    global __nvshmemx_sync_all_on_stream
    __nvshmemx_sync_all_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_sync_all_on_stream')
    if __nvshmemx_sync_all_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_sync_all_on_stream = dlsym(handle, 'nvshmemx_sync_all_on_stream')

    global __nvshmemx_hostlib_init_attr
    __nvshmemx_hostlib_init_attr = dlsym(RTLD_DEFAULT, 'nvshmemx_hostlib_init_attr')
    if __nvshmemx_hostlib_init_attr == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_hostlib_init_attr = dlsym(handle, 'nvshmemx_hostlib_init_attr')

    global __nvshmemx_hostlib_finalize
    __nvshmemx_hostlib_finalize = dlsym(RTLD_DEFAULT, 'nvshmemx_hostlib_finalize')
    if __nvshmemx_hostlib_finalize == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_hostlib_finalize = dlsym(handle, 'nvshmemx_hostlib_finalize')

    global __nvshmemx_set_attr_uniqueid_args
    __nvshmemx_set_attr_uniqueid_args = dlsym(RTLD_DEFAULT, 'nvshmemx_set_attr_uniqueid_args')
    if __nvshmemx_set_attr_uniqueid_args == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_set_attr_uniqueid_args = dlsym(handle, 'nvshmemx_set_attr_uniqueid_args')

    global __nvshmemx_get_uniqueid
    __nvshmemx_get_uniqueid = dlsym(RTLD_DEFAULT, 'nvshmemx_get_uniqueid')
    if __nvshmemx_get_uniqueid == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_get_uniqueid = dlsym(handle, 'nvshmemx_get_uniqueid')

    __py_nvshmem_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nvshmem()
    cdef dict data = {}

    global __nvshmemx_init_status
    data["__nvshmemx_init_status"] = <intptr_t>__nvshmemx_init_status

    global __nvshmem_my_pe
    data["__nvshmem_my_pe"] = <intptr_t>__nvshmem_my_pe

    global __nvshmem_n_pes
    data["__nvshmem_n_pes"] = <intptr_t>__nvshmem_n_pes

    global __nvshmem_malloc
    data["__nvshmem_malloc"] = <intptr_t>__nvshmem_malloc

    global __nvshmem_calloc
    data["__nvshmem_calloc"] = <intptr_t>__nvshmem_calloc

    global __nvshmem_align
    data["__nvshmem_align"] = <intptr_t>__nvshmem_align

    global __nvshmem_free
    data["__nvshmem_free"] = <intptr_t>__nvshmem_free

    global __nvshmem_ptr
    data["__nvshmem_ptr"] = <intptr_t>__nvshmem_ptr

    global __nvshmem_int_p
    data["__nvshmem_int_p"] = <intptr_t>__nvshmem_int_p

    global __nvshmem_team_my_pe
    data["__nvshmem_team_my_pe"] = <intptr_t>__nvshmem_team_my_pe

    global __nvshmemx_barrier_all_on_stream
    data["__nvshmemx_barrier_all_on_stream"] = <intptr_t>__nvshmemx_barrier_all_on_stream

    global __nvshmemx_sync_all_on_stream
    data["__nvshmemx_sync_all_on_stream"] = <intptr_t>__nvshmemx_sync_all_on_stream

    global __nvshmemx_hostlib_init_attr
    data["__nvshmemx_hostlib_init_attr"] = <intptr_t>__nvshmemx_hostlib_init_attr

    global __nvshmemx_hostlib_finalize
    data["__nvshmemx_hostlib_finalize"] = <intptr_t>__nvshmemx_hostlib_finalize

    global __nvshmemx_set_attr_uniqueid_args
    data["__nvshmemx_set_attr_uniqueid_args"] = <intptr_t>__nvshmemx_set_attr_uniqueid_args

    global __nvshmemx_get_uniqueid
    data["__nvshmemx_get_uniqueid"] = <intptr_t>__nvshmemx_get_uniqueid

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

cdef int _nvshmemx_init_status() except?-42 nogil:
    global __nvshmemx_init_status
    _check_or_init_nvshmem()
    if __nvshmemx_init_status == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_init_status is not found")
    return (<int (*)() noexcept nogil>__nvshmemx_init_status)(
        )


cdef int _nvshmem_my_pe() except?-42 nogil:
    global __nvshmem_my_pe
    _check_or_init_nvshmem()
    if __nvshmem_my_pe == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_my_pe is not found")
    return (<int (*)() noexcept nogil>__nvshmem_my_pe)(
        )


cdef int _nvshmem_n_pes() except?-42 nogil:
    global __nvshmem_n_pes
    _check_or_init_nvshmem()
    if __nvshmem_n_pes == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_n_pes is not found")
    return (<int (*)() noexcept nogil>__nvshmem_n_pes)(
        )


cdef void* _nvshmem_malloc(size_t size) except?NULL nogil:
    global __nvshmem_malloc
    _check_or_init_nvshmem()
    if __nvshmem_malloc == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_malloc is not found")
    return (<void* (*)(size_t) noexcept nogil>__nvshmem_malloc)(
        size)


cdef void* _nvshmem_calloc(size_t count, size_t size) except?NULL nogil:
    global __nvshmem_calloc
    _check_or_init_nvshmem()
    if __nvshmem_calloc == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_calloc is not found")
    return (<void* (*)(size_t, size_t) noexcept nogil>__nvshmem_calloc)(
        count, size)


cdef void* _nvshmem_align(size_t alignment, size_t size) except?NULL nogil:
    global __nvshmem_align
    _check_or_init_nvshmem()
    if __nvshmem_align == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_align is not found")
    return (<void* (*)(size_t, size_t) noexcept nogil>__nvshmem_align)(
        alignment, size)


@cython.show_performance_hints(False)
cdef void _nvshmem_free(void* ptr) except* nogil:
    global __nvshmem_free
    _check_or_init_nvshmem()
    if __nvshmem_free == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_free is not found")
    (<void (*)(void*) noexcept nogil>__nvshmem_free)(
        ptr)


cdef void* _nvshmem_ptr(const void* dest, int pe) except?NULL nogil:
    global __nvshmem_ptr
    _check_or_init_nvshmem()
    if __nvshmem_ptr == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_ptr is not found")
    return (<void* (*)(const void*, int) noexcept nogil>__nvshmem_ptr)(
        dest, pe)


@cython.show_performance_hints(False)
cdef void _nvshmem_int_p(int* dest, const int value, int pe) except* nogil:
    global __nvshmem_int_p
    _check_or_init_nvshmem()
    if __nvshmem_int_p == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_int_p is not found")
    (<void (*)(int*, const int, int) noexcept nogil>__nvshmem_int_p)(
        dest, value, pe)


cdef int _nvshmem_team_my_pe(nvshmem_team_t team) except?-42 nogil:
    global __nvshmem_team_my_pe
    _check_or_init_nvshmem()
    if __nvshmem_team_my_pe == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_team_my_pe is not found")
    return (<int (*)(nvshmem_team_t) noexcept nogil>__nvshmem_team_my_pe)(
        team)


@cython.show_performance_hints(False)
cdef void _nvshmemx_barrier_all_on_stream(cudaStream_t stream) except* nogil:
    global __nvshmemx_barrier_all_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_barrier_all_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_barrier_all_on_stream is not found")
    (<void (*)(cudaStream_t) noexcept nogil>__nvshmemx_barrier_all_on_stream)(
        stream)


@cython.show_performance_hints(False)
cdef void _nvshmemx_sync_all_on_stream(cudaStream_t stream) except* nogil:
    global __nvshmemx_sync_all_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_sync_all_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_sync_all_on_stream is not found")
    (<void (*)(cudaStream_t) noexcept nogil>__nvshmemx_sync_all_on_stream)(
        stream)


cdef int _nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t* attr) except?-42 nogil:
    global __nvshmemx_hostlib_init_attr
    _check_or_init_nvshmem()
    if __nvshmemx_hostlib_init_attr == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_hostlib_init_attr is not found")
    return (<int (*)(unsigned int, nvshmemx_init_attr_t*) noexcept nogil>__nvshmemx_hostlib_init_attr)(
        flags, attr)


@cython.show_performance_hints(False)
cdef void _nvshmemx_hostlib_finalize() except* nogil:
    global __nvshmemx_hostlib_finalize
    _check_or_init_nvshmem()
    if __nvshmemx_hostlib_finalize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_hostlib_finalize is not found")
    (<void (*)() noexcept nogil>__nvshmemx_hostlib_finalize)(
        )


cdef int _nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks, const nvshmemx_uniqueid_t* uniqueid, nvshmemx_init_attr_t* attr) except?-42 nogil:
    global __nvshmemx_set_attr_uniqueid_args
    _check_or_init_nvshmem()
    if __nvshmemx_set_attr_uniqueid_args == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_set_attr_uniqueid_args is not found")
    return (<int (*)(const int, const int, const nvshmemx_uniqueid_t*, nvshmemx_init_attr_t*) noexcept nogil>__nvshmemx_set_attr_uniqueid_args)(
        myrank, nranks, uniqueid, attr)


cdef int _nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* uniqueid) except?-42 nogil:
    global __nvshmemx_get_uniqueid
    _check_or_init_nvshmem()
    if __nvshmemx_get_uniqueid == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_get_uniqueid is not found")
    return (<int (*)(nvshmemx_uniqueid_t*) noexcept nogil>__nvshmemx_get_uniqueid)(
        uniqueid)
