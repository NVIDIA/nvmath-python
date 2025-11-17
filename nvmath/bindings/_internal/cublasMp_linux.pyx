# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.5.0 to 0.6.0. Do not modify it directly.

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
cdef void* __cublasMpGetVersion = NULL
cdef void* __cublasMpGridCreate = NULL
cdef void* __cublasMpGridDestroy = NULL
cdef void* __cublasMpMatrixDescriptorCreate = NULL
cdef void* __cublasMpMatrixDescriptorDestroy = NULL
cdef void* __cublasMpMatmulDescriptorCreate = NULL
cdef void* __cublasMpMatmulDescriptorDestroy = NULL
cdef void* __cublasMpMatmulDescriptorAttributeSet = NULL
cdef void* __cublasMpMatmulDescriptorAttributeGet = NULL
cdef void* __cublasMpMatmul_bufferSize = NULL
cdef void* __cublasMpMatmul = NULL
cdef void* __cublasMpNumroc = NULL


cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cublasmp")._handle_uint
    return <void*>handle


cdef int _check_or_init_cublasMp() except -1 nogil:
    global __py_cublasMp_init
    if __py_cublasMp_init:
        return 0

    cdef void* handle = NULL

    with gil, __symbol_lock:
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

        global __cublasMpNumroc
        __cublasMpNumroc = dlsym(RTLD_DEFAULT, 'cublasMpNumroc')
        if __cublasMpNumroc == NULL:
            if handle == NULL:
                handle = load_library()
            __cublasMpNumroc = dlsym(handle, 'cublasMpNumroc')
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

    global __cublasMpMatmulDescriptorCreate
    data["__cublasMpMatmulDescriptorCreate"] = <intptr_t>__cublasMpMatmulDescriptorCreate

    global __cublasMpMatmulDescriptorDestroy
    data["__cublasMpMatmulDescriptorDestroy"] = <intptr_t>__cublasMpMatmulDescriptorDestroy

    global __cublasMpMatmulDescriptorAttributeSet
    data["__cublasMpMatmulDescriptorAttributeSet"] = <intptr_t>__cublasMpMatmulDescriptorAttributeSet

    global __cublasMpMatmulDescriptorAttributeGet
    data["__cublasMpMatmulDescriptorAttributeGet"] = <intptr_t>__cublasMpMatmulDescriptorAttributeGet

    global __cublasMpMatmul_bufferSize
    data["__cublasMpMatmul_bufferSize"] = <intptr_t>__cublasMpMatmul_bufferSize

    global __cublasMpMatmul
    data["__cublasMpMatmul"] = <intptr_t>__cublasMpMatmul

    global __cublasMpNumroc
    data["__cublasMpNumroc"] = <intptr_t>__cublasMpNumroc

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


cdef int64_t _cublasMpNumroc(int64_t n, int64_t nb, uint32_t iproc, uint32_t isrcproc, uint32_t nprocs) except?-42 nogil:
    global __cublasMpNumroc
    _check_or_init_cublasMp()
    if __cublasMpNumroc == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasMpNumroc is not found")
    return (<int64_t (*)(int64_t, int64_t, uint32_t, uint32_t, uint32_t) noexcept nogil>__cublasMpNumroc)(
        n, nb, iproc, isrcproc, nprocs)
