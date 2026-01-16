# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.6.0 to 0.7.0. Do not modify it directly.

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
cdef bint __py_cublasMp_init = False

cdef void* __cublasMpCreate = NULL
cdef void* __cublasMpDestroy = NULL
cdef void* __cublasMpStreamSet = NULL
cdef void* __cublasMpStreamGet = NULL
cdef void* __cublasMpGetVersion = NULL
cdef void* __cublasMpGridCreate = NULL
cdef void* __cublasMpGridDestroy = NULL
cdef void* __cublasMpMatrixDescriptorCreate = NULL
cdef void* __cublasMpMatrixDescriptorDestroy = NULL
cdef void* __cublasMpMatrixDescriptorInit = NULL
cdef void* __cublasMpMatmulDescriptorCreate = NULL
cdef void* __cublasMpMatmulDescriptorDestroy = NULL
cdef void* __cublasMpMatmulDescriptorInit = NULL
cdef void* __cublasMpMatmulDescriptorAttributeSet = NULL
cdef void* __cublasMpMatmulDescriptorAttributeGet = NULL
cdef void* __cublasMpTrsm_bufferSize = NULL
cdef void* __cublasMpTrsm = NULL
cdef void* __cublasMpGemm_bufferSize = NULL
cdef void* __cublasMpGemm = NULL
cdef void* __cublasMpMatmul_bufferSize = NULL
cdef void* __cublasMpMatmul = NULL
cdef void* __cublasMpSyrk_bufferSize = NULL
cdef void* __cublasMpSyrk = NULL
cdef void* __cublasMpNumroc = NULL
cdef void* __cublasMpGemr2D_bufferSize = NULL
cdef void* __cublasMpGemr2D = NULL
cdef void* __cublasMpTrmr2D_bufferSize = NULL
cdef void* __cublasMpTrmr2D = NULL
cdef void* __cublasMpGeadd_bufferSize = NULL
cdef void* __cublasMpGeadd = NULL
cdef void* __cublasMpTradd_bufferSize = NULL
cdef void* __cublasMpTradd = NULL
cdef void* __cublasMpSetEmulationStrategy = NULL
cdef void* __cublasMpGetEmulationStrategy = NULL


cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cublasmp")._handle_uint
    return <void*>handle


cdef int _check_or_init_cublasMp() except -1 nogil:
    global __py_cublasMp_init
    if __py_cublasMp_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_cublasMp_init:
            return 0

        # Load function
        global __cublasMpCreate
        __cublasMpCreate = dlsym(RTLD_DEFAULT, 'cublasMpCreate')
        if __cublasMpCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpCreate = dlsym(handle, 'cublasMpCreate')

        global __cublasMpDestroy
        __cublasMpDestroy = dlsym(RTLD_DEFAULT, 'cublasMpDestroy')
        if __cublasMpDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpDestroy = dlsym(handle, 'cublasMpDestroy')

        global __cublasMpStreamSet
        __cublasMpStreamSet = dlsym(RTLD_DEFAULT, 'cublasMpStreamSet')
        if __cublasMpStreamSet == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpStreamSet = dlsym(handle, 'cublasMpStreamSet')

        global __cublasMpStreamGet
        __cublasMpStreamGet = dlsym(RTLD_DEFAULT, 'cublasMpStreamGet')
        if __cublasMpStreamGet == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpStreamGet = dlsym(handle, 'cublasMpStreamGet')

        global __cublasMpGetVersion
        __cublasMpGetVersion = dlsym(RTLD_DEFAULT, 'cublasMpGetVersion')
        if __cublasMpGetVersion == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpGetVersion = dlsym(handle, 'cublasMpGetVersion')

        global __cublasMpGridCreate
        __cublasMpGridCreate = dlsym(RTLD_DEFAULT, 'cublasMpGridCreate')
        if __cublasMpGridCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpGridCreate = dlsym(handle, 'cublasMpGridCreate')

        global __cublasMpGridDestroy
        __cublasMpGridDestroy = dlsym(RTLD_DEFAULT, 'cublasMpGridDestroy')
        if __cublasMpGridDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpGridDestroy = dlsym(handle, 'cublasMpGridDestroy')

        global __cublasMpMatrixDescriptorCreate
        __cublasMpMatrixDescriptorCreate = dlsym(RTLD_DEFAULT, 'cublasMpMatrixDescriptorCreate')
        if __cublasMpMatrixDescriptorCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpMatrixDescriptorCreate = dlsym(handle, 'cublasMpMatrixDescriptorCreate')

        global __cublasMpMatrixDescriptorDestroy
        __cublasMpMatrixDescriptorDestroy = dlsym(RTLD_DEFAULT, 'cublasMpMatrixDescriptorDestroy')
        if __cublasMpMatrixDescriptorDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpMatrixDescriptorDestroy = dlsym(handle, 'cublasMpMatrixDescriptorDestroy')

        global __cublasMpMatrixDescriptorInit
        __cublasMpMatrixDescriptorInit = dlsym(RTLD_DEFAULT, 'cublasMpMatrixDescriptorInit')
        if __cublasMpMatrixDescriptorInit == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpMatrixDescriptorInit = dlsym(handle, 'cublasMpMatrixDescriptorInit')

        global __cublasMpMatmulDescriptorCreate
        __cublasMpMatmulDescriptorCreate = dlsym(RTLD_DEFAULT, 'cublasMpMatmulDescriptorCreate')
        if __cublasMpMatmulDescriptorCreate == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpMatmulDescriptorCreate = dlsym(handle, 'cublasMpMatmulDescriptorCreate')

        global __cublasMpMatmulDescriptorDestroy
        __cublasMpMatmulDescriptorDestroy = dlsym(RTLD_DEFAULT, 'cublasMpMatmulDescriptorDestroy')
        if __cublasMpMatmulDescriptorDestroy == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpMatmulDescriptorDestroy = dlsym(handle, 'cublasMpMatmulDescriptorDestroy')

        global __cublasMpMatmulDescriptorInit
        __cublasMpMatmulDescriptorInit = dlsym(RTLD_DEFAULT, 'cublasMpMatmulDescriptorInit')
        if __cublasMpMatmulDescriptorInit == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpMatmulDescriptorInit = dlsym(handle, 'cublasMpMatmulDescriptorInit')

        global __cublasMpMatmulDescriptorAttributeSet
        __cublasMpMatmulDescriptorAttributeSet = dlsym(RTLD_DEFAULT, 'cublasMpMatmulDescriptorAttributeSet')
        if __cublasMpMatmulDescriptorAttributeSet == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpMatmulDescriptorAttributeSet = dlsym(handle, 'cublasMpMatmulDescriptorAttributeSet')

        global __cublasMpMatmulDescriptorAttributeGet
        __cublasMpMatmulDescriptorAttributeGet = dlsym(RTLD_DEFAULT, 'cublasMpMatmulDescriptorAttributeGet')
        if __cublasMpMatmulDescriptorAttributeGet == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpMatmulDescriptorAttributeGet = dlsym(handle, 'cublasMpMatmulDescriptorAttributeGet')

        global __cublasMpTrsm_bufferSize
        __cublasMpTrsm_bufferSize = dlsym(RTLD_DEFAULT, 'cublasMpTrsm_bufferSize')
        if __cublasMpTrsm_bufferSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpTrsm_bufferSize = dlsym(handle, 'cublasMpTrsm_bufferSize')

        global __cublasMpTrsm
        __cublasMpTrsm = dlsym(RTLD_DEFAULT, 'cublasMpTrsm')
        if __cublasMpTrsm == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpTrsm = dlsym(handle, 'cublasMpTrsm')

        global __cublasMpGemm_bufferSize
        __cublasMpGemm_bufferSize = dlsym(RTLD_DEFAULT, 'cublasMpGemm_bufferSize')
        if __cublasMpGemm_bufferSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpGemm_bufferSize = dlsym(handle, 'cublasMpGemm_bufferSize')

        global __cublasMpGemm
        __cublasMpGemm = dlsym(RTLD_DEFAULT, 'cublasMpGemm')
        if __cublasMpGemm == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpGemm = dlsym(handle, 'cublasMpGemm')

        global __cublasMpMatmul_bufferSize
        __cublasMpMatmul_bufferSize = dlsym(RTLD_DEFAULT, 'cublasMpMatmul_bufferSize')
        if __cublasMpMatmul_bufferSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpMatmul_bufferSize = dlsym(handle, 'cublasMpMatmul_bufferSize')

        global __cublasMpMatmul
        __cublasMpMatmul = dlsym(RTLD_DEFAULT, 'cublasMpMatmul')
        if __cublasMpMatmul == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpMatmul = dlsym(handle, 'cublasMpMatmul')

        global __cublasMpSyrk_bufferSize
        __cublasMpSyrk_bufferSize = dlsym(RTLD_DEFAULT, 'cublasMpSyrk_bufferSize')
        if __cublasMpSyrk_bufferSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpSyrk_bufferSize = dlsym(handle, 'cublasMpSyrk_bufferSize')

        global __cublasMpSyrk
        __cublasMpSyrk = dlsym(RTLD_DEFAULT, 'cublasMpSyrk')
        if __cublasMpSyrk == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpSyrk = dlsym(handle, 'cublasMpSyrk')

        global __cublasMpNumroc
        __cublasMpNumroc = dlsym(RTLD_DEFAULT, 'cublasMpNumroc')
        if __cublasMpNumroc == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpNumroc = dlsym(handle, 'cublasMpNumroc')

        global __cublasMpGemr2D_bufferSize
        __cublasMpGemr2D_bufferSize = dlsym(RTLD_DEFAULT, 'cublasMpGemr2D_bufferSize')
        if __cublasMpGemr2D_bufferSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpGemr2D_bufferSize = dlsym(handle, 'cublasMpGemr2D_bufferSize')

        global __cublasMpGemr2D
        __cublasMpGemr2D = dlsym(RTLD_DEFAULT, 'cublasMpGemr2D')
        if __cublasMpGemr2D == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpGemr2D = dlsym(handle, 'cublasMpGemr2D')

        global __cublasMpTrmr2D_bufferSize
        __cublasMpTrmr2D_bufferSize = dlsym(RTLD_DEFAULT, 'cublasMpTrmr2D_bufferSize')
        if __cublasMpTrmr2D_bufferSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpTrmr2D_bufferSize = dlsym(handle, 'cublasMpTrmr2D_bufferSize')

        global __cublasMpTrmr2D
        __cublasMpTrmr2D = dlsym(RTLD_DEFAULT, 'cublasMpTrmr2D')
        if __cublasMpTrmr2D == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpTrmr2D = dlsym(handle, 'cublasMpTrmr2D')

        global __cublasMpGeadd_bufferSize
        __cublasMpGeadd_bufferSize = dlsym(RTLD_DEFAULT, 'cublasMpGeadd_bufferSize')
        if __cublasMpGeadd_bufferSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpGeadd_bufferSize = dlsym(handle, 'cublasMpGeadd_bufferSize')

        global __cublasMpGeadd
        __cublasMpGeadd = dlsym(RTLD_DEFAULT, 'cublasMpGeadd')
        if __cublasMpGeadd == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpGeadd = dlsym(handle, 'cublasMpGeadd')

        global __cublasMpTradd_bufferSize
        __cublasMpTradd_bufferSize = dlsym(RTLD_DEFAULT, 'cublasMpTradd_bufferSize')
        if __cublasMpTradd_bufferSize == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpTradd_bufferSize = dlsym(handle, 'cublasMpTradd_bufferSize')

        global __cublasMpTradd
        __cublasMpTradd = dlsym(RTLD_DEFAULT, 'cublasMpTradd')
        if __cublasMpTradd == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpTradd = dlsym(handle, 'cublasMpTradd')

        global __cublasMpSetEmulationStrategy
        __cublasMpSetEmulationStrategy = dlsym(RTLD_DEFAULT, 'cublasMpSetEmulationStrategy')
        if __cublasMpSetEmulationStrategy == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpSetEmulationStrategy = dlsym(handle, 'cublasMpSetEmulationStrategy')

        global __cublasMpGetEmulationStrategy
        __cublasMpGetEmulationStrategy = dlsym(RTLD_DEFAULT, 'cublasMpGetEmulationStrategy')
        if __cublasMpGetEmulationStrategy == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpGetEmulationStrategy = dlsym(handle, 'cublasMpGetEmulationStrategy')
        __py_cublasMp_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cublasMp()
    cdef dict data = {}

    global __cublasMpCreate
    data["__cublasMpCreate"] = <intptr_t>__cublasMpCreate

    global __cublasMpDestroy
    data["__cublasMpDestroy"] = <intptr_t>__cublasMpDestroy

    global __cublasMpStreamSet
    data["__cublasMpStreamSet"] = <intptr_t>__cublasMpStreamSet

    global __cublasMpStreamGet
    data["__cublasMpStreamGet"] = <intptr_t>__cublasMpStreamGet

    global __cublasMpGetVersion
    data["__cublasMpGetVersion"] = <intptr_t>__cublasMpGetVersion

    global __cublasMpGridCreate
    data["__cublasMpGridCreate"] = <intptr_t>__cublasMpGridCreate

    global __cublasMpGridDestroy
    data["__cublasMpGridDestroy"] = <intptr_t>__cublasMpGridDestroy

    global __cublasMpMatrixDescriptorCreate
    data["__cublasMpMatrixDescriptorCreate"] = <intptr_t>__cublasMpMatrixDescriptorCreate

    global __cublasMpMatrixDescriptorDestroy
    data["__cublasMpMatrixDescriptorDestroy"] = <intptr_t>__cublasMpMatrixDescriptorDestroy

    global __cublasMpMatrixDescriptorInit
    data["__cublasMpMatrixDescriptorInit"] = <intptr_t>__cublasMpMatrixDescriptorInit

    global __cublasMpMatmulDescriptorCreate
    data["__cublasMpMatmulDescriptorCreate"] = <intptr_t>__cublasMpMatmulDescriptorCreate

    global __cublasMpMatmulDescriptorDestroy
    data["__cublasMpMatmulDescriptorDestroy"] = <intptr_t>__cublasMpMatmulDescriptorDestroy

    global __cublasMpMatmulDescriptorInit
    data["__cublasMpMatmulDescriptorInit"] = <intptr_t>__cublasMpMatmulDescriptorInit

    global __cublasMpMatmulDescriptorAttributeSet
    data["__cublasMpMatmulDescriptorAttributeSet"] = <intptr_t>__cublasMpMatmulDescriptorAttributeSet

    global __cublasMpMatmulDescriptorAttributeGet
    data["__cublasMpMatmulDescriptorAttributeGet"] = <intptr_t>__cublasMpMatmulDescriptorAttributeGet

    global __cublasMpTrsm_bufferSize
    data["__cublasMpTrsm_bufferSize"] = <intptr_t>__cublasMpTrsm_bufferSize

    global __cublasMpTrsm
    data["__cublasMpTrsm"] = <intptr_t>__cublasMpTrsm

    global __cublasMpGemm_bufferSize
    data["__cublasMpGemm_bufferSize"] = <intptr_t>__cublasMpGemm_bufferSize

    global __cublasMpGemm
    data["__cublasMpGemm"] = <intptr_t>__cublasMpGemm

    global __cublasMpMatmul_bufferSize
    data["__cublasMpMatmul_bufferSize"] = <intptr_t>__cublasMpMatmul_bufferSize

    global __cublasMpMatmul
    data["__cublasMpMatmul"] = <intptr_t>__cublasMpMatmul

    global __cublasMpSyrk_bufferSize
    data["__cublasMpSyrk_bufferSize"] = <intptr_t>__cublasMpSyrk_bufferSize

    global __cublasMpSyrk
    data["__cublasMpSyrk"] = <intptr_t>__cublasMpSyrk

    global __cublasMpNumroc
    data["__cublasMpNumroc"] = <intptr_t>__cublasMpNumroc

    global __cublasMpGemr2D_bufferSize
    data["__cublasMpGemr2D_bufferSize"] = <intptr_t>__cublasMpGemr2D_bufferSize

    global __cublasMpGemr2D
    data["__cublasMpGemr2D"] = <intptr_t>__cublasMpGemr2D

    global __cublasMpTrmr2D_bufferSize
    data["__cublasMpTrmr2D_bufferSize"] = <intptr_t>__cublasMpTrmr2D_bufferSize

    global __cublasMpTrmr2D
    data["__cublasMpTrmr2D"] = <intptr_t>__cublasMpTrmr2D

    global __cublasMpGeadd_bufferSize
    data["__cublasMpGeadd_bufferSize"] = <intptr_t>__cublasMpGeadd_bufferSize

    global __cublasMpGeadd
    data["__cublasMpGeadd"] = <intptr_t>__cublasMpGeadd

    global __cublasMpTradd_bufferSize
    data["__cublasMpTradd_bufferSize"] = <intptr_t>__cublasMpTradd_bufferSize

    global __cublasMpTradd
    data["__cublasMpTradd"] = <intptr_t>__cublasMpTradd

    global __cublasMpSetEmulationStrategy
    data["__cublasMpSetEmulationStrategy"] = <intptr_t>__cublasMpSetEmulationStrategy

    global __cublasMpGetEmulationStrategy
    data["__cublasMpGetEmulationStrategy"] = <intptr_t>__cublasMpGetEmulationStrategy

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

cdef cublasMpStatus_t _cublasMpCreate(cublasMpHandle_t* handle, cudaStream_t stream) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpCreate
    _check_or_init_cublasMp()
    if __cublasMpCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpCreate is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t*, cudaStream_t) noexcept nogil>__cublasMpCreate)(
        handle, stream)


cdef cublasMpStatus_t _cublasMpDestroy(cublasMpHandle_t handle) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpDestroy
    _check_or_init_cublasMp()
    if __cublasMpDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpDestroy is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t) noexcept nogil>__cublasMpDestroy)(
        handle)


cdef cublasMpStatus_t _cublasMpStreamSet(cublasMpHandle_t handle, cudaStream_t stream) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpStreamSet
    _check_or_init_cublasMp()
    if __cublasMpStreamSet == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpStreamSet is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cudaStream_t) noexcept nogil>__cublasMpStreamSet)(
        handle, stream)


cdef cublasMpStatus_t _cublasMpStreamGet(cublasMpHandle_t handle, cudaStream_t* stream) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpStreamGet
    _check_or_init_cublasMp()
    if __cublasMpStreamGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpStreamGet is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cudaStream_t*) noexcept nogil>__cublasMpStreamGet)(
        handle, stream)


cdef cublasMpStatus_t _cublasMpGetVersion(int* version) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpGetVersion
    _check_or_init_cublasMp()
    if __cublasMpGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpGetVersion is not found")
    return (<cublasMpStatus_t (*)(int*) noexcept nogil>__cublasMpGetVersion)(
        version)


cdef cublasMpStatus_t _cublasMpGridCreate(int64_t nprow, int64_t npcol, cublasMpGridLayout_t layout, ncclComm_t comm, cublasMpGrid_t* grid) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpGridCreate
    _check_or_init_cublasMp()
    if __cublasMpGridCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpGridCreate is not found")
    return (<cublasMpStatus_t (*)(int64_t, int64_t, cublasMpGridLayout_t, ncclComm_t, cublasMpGrid_t*) noexcept nogil>__cublasMpGridCreate)(
        nprow, npcol, layout, comm, grid)


cdef cublasMpStatus_t _cublasMpGridDestroy(cublasMpGrid_t grid) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpGridDestroy
    _check_or_init_cublasMp()
    if __cublasMpGridDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpGridDestroy is not found")
    return (<cublasMpStatus_t (*)(cublasMpGrid_t) noexcept nogil>__cublasMpGridDestroy)(
        grid)


cdef cublasMpStatus_t _cublasMpMatrixDescriptorCreate(int64_t m, int64_t n, int64_t mb, int64_t nb, int64_t rsrc, int64_t csrc, int64_t lld, cudaDataType_t type, cublasMpGrid_t grid, cublasMpMatrixDescriptor_t* desc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpMatrixDescriptorCreate
    _check_or_init_cublasMp()
    if __cublasMpMatrixDescriptorCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpMatrixDescriptorCreate is not found")
    return (<cublasMpStatus_t (*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaDataType_t, cublasMpGrid_t, cublasMpMatrixDescriptor_t*) noexcept nogil>__cublasMpMatrixDescriptorCreate)(
        m, n, mb, nb, rsrc, csrc, lld, type, grid, desc)


cdef cublasMpStatus_t _cublasMpMatrixDescriptorDestroy(cublasMpMatrixDescriptor_t desc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpMatrixDescriptorDestroy
    _check_or_init_cublasMp()
    if __cublasMpMatrixDescriptorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpMatrixDescriptorDestroy is not found")
    return (<cublasMpStatus_t (*)(cublasMpMatrixDescriptor_t) noexcept nogil>__cublasMpMatrixDescriptorDestroy)(
        desc)


cdef cublasMpStatus_t _cublasMpMatrixDescriptorInit(int64_t m, int64_t n, int64_t mb, int64_t nb, int64_t rsrc, int64_t csrc, int64_t lld, cudaDataType_t type, cublasMpGrid_t grid, cublasMpMatrixDescriptor_t desc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpMatrixDescriptorInit
    _check_or_init_cublasMp()
    if __cublasMpMatrixDescriptorInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpMatrixDescriptorInit is not found")
    return (<cublasMpStatus_t (*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaDataType_t, cublasMpGrid_t, cublasMpMatrixDescriptor_t) noexcept nogil>__cublasMpMatrixDescriptorInit)(
        m, n, mb, nb, rsrc, csrc, lld, type, grid, desc)


cdef cublasMpStatus_t _cublasMpMatmulDescriptorCreate(cublasMpMatmulDescriptor_t* matmulDesc, cublasComputeType_t computeType) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpMatmulDescriptorCreate
    _check_or_init_cublasMp()
    if __cublasMpMatmulDescriptorCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpMatmulDescriptorCreate is not found")
    return (<cublasMpStatus_t (*)(cublasMpMatmulDescriptor_t*, cublasComputeType_t) noexcept nogil>__cublasMpMatmulDescriptorCreate)(
        matmulDesc, computeType)


cdef cublasMpStatus_t _cublasMpMatmulDescriptorDestroy(cublasMpMatmulDescriptor_t matmulDesc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpMatmulDescriptorDestroy
    _check_or_init_cublasMp()
    if __cublasMpMatmulDescriptorDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpMatmulDescriptorDestroy is not found")
    return (<cublasMpStatus_t (*)(cublasMpMatmulDescriptor_t) noexcept nogil>__cublasMpMatmulDescriptorDestroy)(
        matmulDesc)


cdef cublasMpStatus_t _cublasMpMatmulDescriptorInit(cublasMpMatmulDescriptor_t matmulDesc, cublasComputeType_t computeType) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpMatmulDescriptorInit
    _check_or_init_cublasMp()
    if __cublasMpMatmulDescriptorInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpMatmulDescriptorInit is not found")
    return (<cublasMpStatus_t (*)(cublasMpMatmulDescriptor_t, cublasComputeType_t) noexcept nogil>__cublasMpMatmulDescriptorInit)(
        matmulDesc, computeType)


cdef cublasMpStatus_t _cublasMpMatmulDescriptorAttributeSet(cublasMpMatmulDescriptor_t matmulDesc, cublasMpMatmulDescriptorAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpMatmulDescriptorAttributeSet
    _check_or_init_cublasMp()
    if __cublasMpMatmulDescriptorAttributeSet == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpMatmulDescriptorAttributeSet is not found")
    return (<cublasMpStatus_t (*)(cublasMpMatmulDescriptor_t, cublasMpMatmulDescriptorAttribute_t, const void*, size_t) noexcept nogil>__cublasMpMatmulDescriptorAttributeSet)(
        matmulDesc, attr, buf, sizeInBytes)


cdef cublasMpStatus_t _cublasMpMatmulDescriptorAttributeGet(cublasMpMatmulDescriptor_t matmulDesc, cublasMpMatmulDescriptorAttribute_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpMatmulDescriptorAttributeGet
    _check_or_init_cublasMp()
    if __cublasMpMatmulDescriptorAttributeGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpMatmulDescriptorAttributeGet is not found")
    return (<cublasMpStatus_t (*)(cublasMpMatmulDescriptor_t, cublasMpMatmulDescriptorAttribute_t, void*, size_t, size_t*) noexcept nogil>__cublasMpMatmulDescriptorAttributeGet)(
        matmulDesc, attr, buf, sizeInBytes, sizeWritten)


cdef cublasMpStatus_t _cublasMpTrsm_bufferSize(cublasMpHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, cublasComputeType_t computeType, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpTrsm_bufferSize
    _check_or_init_cublasMp()
    if __cublasMpTrsm_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpTrsm_bufferSize is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, cublasComputeType_t, size_t*, size_t*) noexcept nogil>__cublasMpTrsm_bufferSize)(
        handle, side, uplo, trans, diag, m, n, alpha, a, ia, ja, descA, b, ib, jb, descB, computeType, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpTrsm(cublasMpHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, cublasComputeType_t computeType, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpTrsm
    _check_or_init_cublasMp()
    if __cublasMpTrsm == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpTrsm is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, cublasComputeType_t, void*, size_t, void*, size_t) noexcept nogil>__cublasMpTrsm)(
        handle, side, uplo, trans, diag, m, n, alpha, a, ia, ja, descA, b, ib, jb, descB, computeType, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpGemm_bufferSize(cublasMpHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpGemm_bufferSize
    _check_or_init_cublasMp()
    if __cublasMpGemm_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpGemm_bufferSize is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, cublasComputeType_t, size_t*, size_t*) noexcept nogil>__cublasMpGemm_bufferSize)(
        handle, transA, transB, m, n, k, alpha, a, ia, ja, descA, b, ib, jb, descB, beta, c, ic, jc, descC, computeType, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpGemm(cublasMpHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpGemm
    _check_or_init_cublasMp()
    if __cublasMpGemm == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpGemm is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, cublasComputeType_t, void*, size_t, void*, size_t) noexcept nogil>__cublasMpGemm)(
        handle, transA, transB, m, n, k, alpha, a, ia, ja, descA, b, ib, jb, descB, beta, c, ic, jc, descC, computeType, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpMatmul_bufferSize(cublasMpHandle_t handle, cublasMpMatmulDescriptor_t matmulDesc, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, const void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d, int64_t id, int64_t jd, cublasMpMatrixDescriptor_t descD, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpMatmul_bufferSize
    _check_or_init_cublasMp()
    if __cublasMpMatmul_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpMatmul_bufferSize is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasMpMatmulDescriptor_t, int64_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, size_t*, size_t*) noexcept nogil>__cublasMpMatmul_bufferSize)(
        handle, matmulDesc, m, n, k, alpha, a, ia, ja, descA, b, ib, jb, descB, beta, c, ic, jc, descC, d, id, jd, descD, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpMatmul(cublasMpHandle_t handle, cublasMpMatmulDescriptor_t matmulDesc, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, const void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d, int64_t id, int64_t jd, cublasMpMatrixDescriptor_t descD, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpMatmul
    _check_or_init_cublasMp()
    if __cublasMpMatmul == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpMatmul is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasMpMatmulDescriptor_t, int64_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, size_t, void*, size_t) noexcept nogil>__cublasMpMatmul)(
        handle, matmulDesc, m, n, k, alpha, a, ia, ja, descA, b, ib, jb, descB, beta, c, ic, jc, descC, d, id, jd, descD, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpSyrk_bufferSize(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpSyrk_bufferSize
    _check_or_init_cublasMp()
    if __cublasMpSyrk_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpSyrk_bufferSize is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, cublasComputeType_t, size_t*, size_t*) noexcept nogil>__cublasMpSyrk_bufferSize)(
        handle, uplo, trans, n, k, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, computeType, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpSyrk(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpSyrk
    _check_or_init_cublasMp()
    if __cublasMpSyrk == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpSyrk is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, cublasComputeType_t, void*, size_t, void*, size_t) noexcept nogil>__cublasMpSyrk)(
        handle, uplo, trans, n, k, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, computeType, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef int64_t _cublasMpNumroc(int64_t n, int64_t nb, uint32_t iproc, uint32_t isrcproc, uint32_t nprocs) except?-42 nogil:
    global __cublasMpNumroc
    _check_or_init_cublasMp()
    if __cublasMpNumroc == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpNumroc is not found")
    return (<int64_t (*)(int64_t, int64_t, uint32_t, uint32_t, uint32_t) noexcept nogil>__cublasMpNumroc)(
        n, nb, iproc, isrcproc, nprocs)


cdef cublasMpStatus_t _cublasMpGemr2D_bufferSize(cublasMpHandle_t handle, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpGemr2D_bufferSize
    _check_or_init_cublasMp()
    if __cublasMpGemr2D_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpGemr2D_bufferSize is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, int64_t, int64_t, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, size_t*, size_t*, ncclComm_t) noexcept nogil>__cublasMpGemr2D_bufferSize)(
        handle, m, n, a, ia, ja, descA, b, ib, jb, descB, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost, global_comm)


cdef cublasMpStatus_t _cublasMpGemr2D(cublasMpHandle_t handle, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpGemr2D
    _check_or_init_cublasMp()
    if __cublasMpGemr2D == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpGemr2D is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, int64_t, int64_t, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, size_t, void*, size_t, ncclComm_t) noexcept nogil>__cublasMpGemr2D)(
        handle, m, n, a, ia, ja, descA, b, ib, jb, descB, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost, global_comm)


cdef cublasMpStatus_t _cublasMpTrmr2D_bufferSize(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpTrmr2D_bufferSize
    _check_or_init_cublasMp()
    if __cublasMpTrmr2D_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpTrmr2D_bufferSize is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasFillMode_t, cublasDiagType_t, int64_t, int64_t, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, size_t*, size_t*, ncclComm_t) noexcept nogil>__cublasMpTrmr2D_bufferSize)(
        handle, uplo, diag, m, n, a, ia, ja, descA, b, ib, jb, descB, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost, global_comm)


cdef cublasMpStatus_t _cublasMpTrmr2D(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpTrmr2D
    _check_or_init_cublasMp()
    if __cublasMpTrmr2D == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpTrmr2D is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasFillMode_t, cublasDiagType_t, int64_t, int64_t, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, size_t, void*, size_t, ncclComm_t) noexcept nogil>__cublasMpTrmr2D)(
        handle, uplo, diag, m, n, a, ia, ja, descA, b, ib, jb, descB, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost, global_comm)


cdef cublasMpStatus_t _cublasMpGeadd_bufferSize(cublasMpHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpGeadd_bufferSize
    _check_or_init_cublasMp()
    if __cublasMpGeadd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpGeadd_bufferSize is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasOperation_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, size_t*, size_t*) noexcept nogil>__cublasMpGeadd_bufferSize)(
        handle, trans, m, n, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpGeadd(cublasMpHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpGeadd
    _check_or_init_cublasMp()
    if __cublasMpGeadd == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpGeadd is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasOperation_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, size_t, void*, size_t) noexcept nogil>__cublasMpGeadd)(
        handle, trans, m, n, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpTradd_bufferSize(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpTradd_bufferSize
    _check_or_init_cublasMp()
    if __cublasMpTradd_bufferSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpTradd_bufferSize is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, size_t*, size_t*) noexcept nogil>__cublasMpTradd_bufferSize)(
        handle, uplo, trans, m, n, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpTradd(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpTradd
    _check_or_init_cublasMp()
    if __cublasMpTradd == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpTradd is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const void*, const void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, const void*, void*, int64_t, int64_t, cublasMpMatrixDescriptor_t, void*, size_t, void*, size_t) noexcept nogil>__cublasMpTradd)(
        handle, uplo, trans, m, n, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t _cublasMpSetEmulationStrategy(cublasMpHandle_t handle, cublasMpEmulationStrategy_t emulationStrategy) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpSetEmulationStrategy
    _check_or_init_cublasMp()
    if __cublasMpSetEmulationStrategy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpSetEmulationStrategy is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasMpEmulationStrategy_t) noexcept nogil>__cublasMpSetEmulationStrategy)(
        handle, emulationStrategy)


cdef cublasMpStatus_t _cublasMpGetEmulationStrategy(cublasMpHandle_t handle, cublasMpEmulationStrategy_t* emulationStrategy) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __cublasMpGetEmulationStrategy
    _check_or_init_cublasMp()
    if __cublasMpGetEmulationStrategy == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpGetEmulationStrategy is not found")
    return (<cublasMpStatus_t (*)(cublasMpHandle_t, cublasMpEmulationStrategy_t*) noexcept nogil>__cublasMpGetEmulationStrategy)(
        handle, emulationStrategy)
