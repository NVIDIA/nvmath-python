# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.0.1 to 13.1.0. Do not modify it directly.

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
cdef bint __py_curand_init = False

cdef void* __curandCreateGenerator = NULL
cdef void* __curandCreateGeneratorHost = NULL
cdef void* __curandDestroyGenerator = NULL
cdef void* __curandGetVersion = NULL
cdef void* __curandGetProperty = NULL
cdef void* __curandSetStream = NULL
cdef void* __curandSetPseudoRandomGeneratorSeed = NULL
cdef void* __curandSetGeneratorOffset = NULL
cdef void* __curandSetGeneratorOrdering = NULL
cdef void* __curandSetQuasiRandomGeneratorDimensions = NULL
cdef void* __curandGenerate = NULL
cdef void* __curandGenerateLongLong = NULL
cdef void* __curandGenerateUniform = NULL
cdef void* __curandGenerateUniformDouble = NULL
cdef void* __curandGenerateNormal = NULL
cdef void* __curandGenerateNormalDouble = NULL
cdef void* __curandGenerateLogNormal = NULL
cdef void* __curandGenerateLogNormalDouble = NULL
cdef void* __curandCreatePoissonDistribution = NULL
cdef void* __curandDestroyDistribution = NULL
cdef void* __curandGeneratePoisson = NULL
cdef void* __curandGeneratePoissonMethod = NULL
cdef void* __curandGenerateBinomial = NULL
cdef void* __curandGenerateBinomialMethod = NULL
cdef void* __curandGenerateSeeds = NULL
cdef void* __curandGetDirectionVectors32 = NULL
cdef void* __curandGetScrambleConstants32 = NULL
cdef void* __curandGetDirectionVectors64 = NULL
cdef void* __curandGetScrambleConstants64 = NULL


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()

cdef load_library(const int driver_ver):
    return load_nvidia_dynamic_lib("curand")._handle_uint

cdef int _check_or_init_curand() except -1 nogil:
    global __py_curand_init
    if __py_curand_init:
        return 0

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_curand_init:
            return 0

        driver_ver = get_cuda_version()

        # Load library
        handle = load_library(driver_ver)

        # Load function
        global __curandCreateGenerator
        __curandCreateGenerator = GetProcAddress(handle, 'curandCreateGenerator')

        global __curandCreateGeneratorHost
        __curandCreateGeneratorHost = GetProcAddress(handle, 'curandCreateGeneratorHost')

        global __curandDestroyGenerator
        __curandDestroyGenerator = GetProcAddress(handle, 'curandDestroyGenerator')

        global __curandGetVersion
        __curandGetVersion = GetProcAddress(handle, 'curandGetVersion')

        global __curandGetProperty
        __curandGetProperty = GetProcAddress(handle, 'curandGetProperty')

        global __curandSetStream
        __curandSetStream = GetProcAddress(handle, 'curandSetStream')

        global __curandSetPseudoRandomGeneratorSeed
        __curandSetPseudoRandomGeneratorSeed = GetProcAddress(handle, 'curandSetPseudoRandomGeneratorSeed')

        global __curandSetGeneratorOffset
        __curandSetGeneratorOffset = GetProcAddress(handle, 'curandSetGeneratorOffset')

        global __curandSetGeneratorOrdering
        __curandSetGeneratorOrdering = GetProcAddress(handle, 'curandSetGeneratorOrdering')

        global __curandSetQuasiRandomGeneratorDimensions
        __curandSetQuasiRandomGeneratorDimensions = GetProcAddress(handle, 'curandSetQuasiRandomGeneratorDimensions')

        global __curandGenerate
        __curandGenerate = GetProcAddress(handle, 'curandGenerate')

        global __curandGenerateLongLong
        __curandGenerateLongLong = GetProcAddress(handle, 'curandGenerateLongLong')

        global __curandGenerateUniform
        __curandGenerateUniform = GetProcAddress(handle, 'curandGenerateUniform')

        global __curandGenerateUniformDouble
        __curandGenerateUniformDouble = GetProcAddress(handle, 'curandGenerateUniformDouble')

        global __curandGenerateNormal
        __curandGenerateNormal = GetProcAddress(handle, 'curandGenerateNormal')

        global __curandGenerateNormalDouble
        __curandGenerateNormalDouble = GetProcAddress(handle, 'curandGenerateNormalDouble')

        global __curandGenerateLogNormal
        __curandGenerateLogNormal = GetProcAddress(handle, 'curandGenerateLogNormal')

        global __curandGenerateLogNormalDouble
        __curandGenerateLogNormalDouble = GetProcAddress(handle, 'curandGenerateLogNormalDouble')

        global __curandCreatePoissonDistribution
        __curandCreatePoissonDistribution = GetProcAddress(handle, 'curandCreatePoissonDistribution')

        global __curandDestroyDistribution
        __curandDestroyDistribution = GetProcAddress(handle, 'curandDestroyDistribution')

        global __curandGeneratePoisson
        __curandGeneratePoisson = GetProcAddress(handle, 'curandGeneratePoisson')

        global __curandGeneratePoissonMethod
        __curandGeneratePoissonMethod = GetProcAddress(handle, 'curandGeneratePoissonMethod')

        global __curandGenerateBinomial
        __curandGenerateBinomial = GetProcAddress(handle, 'curandGenerateBinomial')

        global __curandGenerateBinomialMethod
        __curandGenerateBinomialMethod = GetProcAddress(handle, 'curandGenerateBinomialMethod')

        global __curandGenerateSeeds
        __curandGenerateSeeds = GetProcAddress(handle, 'curandGenerateSeeds')

        global __curandGetDirectionVectors32
        __curandGetDirectionVectors32 = GetProcAddress(handle, 'curandGetDirectionVectors32')

        global __curandGetScrambleConstants32
        __curandGetScrambleConstants32 = GetProcAddress(handle, 'curandGetScrambleConstants32')

        global __curandGetDirectionVectors64
        __curandGetDirectionVectors64 = GetProcAddress(handle, 'curandGetDirectionVectors64')

        global __curandGetScrambleConstants64
        __curandGetScrambleConstants64 = GetProcAddress(handle, 'curandGetScrambleConstants64')

        __py_curand_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_curand()
    cdef dict data = {}

    global __curandCreateGenerator
    data["__curandCreateGenerator"] = <intptr_t>__curandCreateGenerator

    global __curandCreateGeneratorHost
    data["__curandCreateGeneratorHost"] = <intptr_t>__curandCreateGeneratorHost

    global __curandDestroyGenerator
    data["__curandDestroyGenerator"] = <intptr_t>__curandDestroyGenerator

    global __curandGetVersion
    data["__curandGetVersion"] = <intptr_t>__curandGetVersion

    global __curandGetProperty
    data["__curandGetProperty"] = <intptr_t>__curandGetProperty

    global __curandSetStream
    data["__curandSetStream"] = <intptr_t>__curandSetStream

    global __curandSetPseudoRandomGeneratorSeed
    data["__curandSetPseudoRandomGeneratorSeed"] = <intptr_t>__curandSetPseudoRandomGeneratorSeed

    global __curandSetGeneratorOffset
    data["__curandSetGeneratorOffset"] = <intptr_t>__curandSetGeneratorOffset

    global __curandSetGeneratorOrdering
    data["__curandSetGeneratorOrdering"] = <intptr_t>__curandSetGeneratorOrdering

    global __curandSetQuasiRandomGeneratorDimensions
    data["__curandSetQuasiRandomGeneratorDimensions"] = <intptr_t>__curandSetQuasiRandomGeneratorDimensions

    global __curandGenerate
    data["__curandGenerate"] = <intptr_t>__curandGenerate

    global __curandGenerateLongLong
    data["__curandGenerateLongLong"] = <intptr_t>__curandGenerateLongLong

    global __curandGenerateUniform
    data["__curandGenerateUniform"] = <intptr_t>__curandGenerateUniform

    global __curandGenerateUniformDouble
    data["__curandGenerateUniformDouble"] = <intptr_t>__curandGenerateUniformDouble

    global __curandGenerateNormal
    data["__curandGenerateNormal"] = <intptr_t>__curandGenerateNormal

    global __curandGenerateNormalDouble
    data["__curandGenerateNormalDouble"] = <intptr_t>__curandGenerateNormalDouble

    global __curandGenerateLogNormal
    data["__curandGenerateLogNormal"] = <intptr_t>__curandGenerateLogNormal

    global __curandGenerateLogNormalDouble
    data["__curandGenerateLogNormalDouble"] = <intptr_t>__curandGenerateLogNormalDouble

    global __curandCreatePoissonDistribution
    data["__curandCreatePoissonDistribution"] = <intptr_t>__curandCreatePoissonDistribution

    global __curandDestroyDistribution
    data["__curandDestroyDistribution"] = <intptr_t>__curandDestroyDistribution

    global __curandGeneratePoisson
    data["__curandGeneratePoisson"] = <intptr_t>__curandGeneratePoisson

    global __curandGeneratePoissonMethod
    data["__curandGeneratePoissonMethod"] = <intptr_t>__curandGeneratePoissonMethod

    global __curandGenerateBinomial
    data["__curandGenerateBinomial"] = <intptr_t>__curandGenerateBinomial

    global __curandGenerateBinomialMethod
    data["__curandGenerateBinomialMethod"] = <intptr_t>__curandGenerateBinomialMethod

    global __curandGenerateSeeds
    data["__curandGenerateSeeds"] = <intptr_t>__curandGenerateSeeds

    global __curandGetDirectionVectors32
    data["__curandGetDirectionVectors32"] = <intptr_t>__curandGetDirectionVectors32

    global __curandGetScrambleConstants32
    data["__curandGetScrambleConstants32"] = <intptr_t>__curandGetScrambleConstants32

    global __curandGetDirectionVectors64
    data["__curandGetDirectionVectors64"] = <intptr_t>__curandGetDirectionVectors64

    global __curandGetScrambleConstants64
    data["__curandGetScrambleConstants64"] = <intptr_t>__curandGetScrambleConstants64

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

cdef curandStatus_t _curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandCreateGenerator
    _check_or_init_curand()
    if __curandCreateGenerator == NULL:
        with gil:
            raise FunctionNotFoundError("function curandCreateGenerator is not found")
    return (<curandStatus_t (*)(curandGenerator_t*, curandRngType_t) noexcept nogil>__curandCreateGenerator)(
        generator, rng_type)


cdef curandStatus_t _curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandCreateGeneratorHost
    _check_or_init_curand()
    if __curandCreateGeneratorHost == NULL:
        with gil:
            raise FunctionNotFoundError("function curandCreateGeneratorHost is not found")
    return (<curandStatus_t (*)(curandGenerator_t*, curandRngType_t) noexcept nogil>__curandCreateGeneratorHost)(
        generator, rng_type)


cdef curandStatus_t _curandDestroyGenerator(curandGenerator_t generator) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandDestroyGenerator
    _check_or_init_curand()
    if __curandDestroyGenerator == NULL:
        with gil:
            raise FunctionNotFoundError("function curandDestroyGenerator is not found")
    return (<curandStatus_t (*)(curandGenerator_t) noexcept nogil>__curandDestroyGenerator)(
        generator)


cdef curandStatus_t _curandGetVersion(int* version) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGetVersion
    _check_or_init_curand()
    if __curandGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetVersion is not found")
    return (<curandStatus_t (*)(int*) noexcept nogil>__curandGetVersion)(
        version)


cdef curandStatus_t _curandGetProperty(libraryPropertyType type, int* value) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGetProperty
    _check_or_init_curand()
    if __curandGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetProperty is not found")
    return (<curandStatus_t (*)(libraryPropertyType, int*) noexcept nogil>__curandGetProperty)(
        type, value)


cdef curandStatus_t _curandSetStream(curandGenerator_t generator, cudaStream_t stream) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandSetStream
    _check_or_init_curand()
    if __curandSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function curandSetStream is not found")
    return (<curandStatus_t (*)(curandGenerator_t, cudaStream_t) noexcept nogil>__curandSetStream)(
        generator, stream)


cdef curandStatus_t _curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandSetPseudoRandomGeneratorSeed
    _check_or_init_curand()
    if __curandSetPseudoRandomGeneratorSeed == NULL:
        with gil:
            raise FunctionNotFoundError("function curandSetPseudoRandomGeneratorSeed is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned long long) noexcept nogil>__curandSetPseudoRandomGeneratorSeed)(
        generator, seed)


cdef curandStatus_t _curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandSetGeneratorOffset
    _check_or_init_curand()
    if __curandSetGeneratorOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function curandSetGeneratorOffset is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned long long) noexcept nogil>__curandSetGeneratorOffset)(
        generator, offset)


cdef curandStatus_t _curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandSetGeneratorOrdering
    _check_or_init_curand()
    if __curandSetGeneratorOrdering == NULL:
        with gil:
            raise FunctionNotFoundError("function curandSetGeneratorOrdering is not found")
    return (<curandStatus_t (*)(curandGenerator_t, curandOrdering_t) noexcept nogil>__curandSetGeneratorOrdering)(
        generator, order)


cdef curandStatus_t _curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandSetQuasiRandomGeneratorDimensions
    _check_or_init_curand()
    if __curandSetQuasiRandomGeneratorDimensions == NULL:
        with gil:
            raise FunctionNotFoundError("function curandSetQuasiRandomGeneratorDimensions is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int) noexcept nogil>__curandSetQuasiRandomGeneratorDimensions)(
        generator, num_dimensions)


cdef curandStatus_t _curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerate
    _check_or_init_curand()
    if __curandGenerate == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerate is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int*, size_t) noexcept nogil>__curandGenerate)(
        generator, outputPtr, num)


cdef curandStatus_t _curandGenerateLongLong(curandGenerator_t generator, unsigned long long* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerateLongLong
    _check_or_init_curand()
    if __curandGenerateLongLong == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateLongLong is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned long long*, size_t) noexcept nogil>__curandGenerateLongLong)(
        generator, outputPtr, num)


cdef curandStatus_t _curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerateUniform
    _check_or_init_curand()
    if __curandGenerateUniform == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateUniform is not found")
    return (<curandStatus_t (*)(curandGenerator_t, float*, size_t) noexcept nogil>__curandGenerateUniform)(
        generator, outputPtr, num)


cdef curandStatus_t _curandGenerateUniformDouble(curandGenerator_t generator, double* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerateUniformDouble
    _check_or_init_curand()
    if __curandGenerateUniformDouble == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateUniformDouble is not found")
    return (<curandStatus_t (*)(curandGenerator_t, double*, size_t) noexcept nogil>__curandGenerateUniformDouble)(
        generator, outputPtr, num)


cdef curandStatus_t _curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerateNormal
    _check_or_init_curand()
    if __curandGenerateNormal == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateNormal is not found")
    return (<curandStatus_t (*)(curandGenerator_t, float*, size_t, float, float) noexcept nogil>__curandGenerateNormal)(
        generator, outputPtr, n, mean, stddev)


cdef curandStatus_t _curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerateNormalDouble
    _check_or_init_curand()
    if __curandGenerateNormalDouble == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateNormalDouble is not found")
    return (<curandStatus_t (*)(curandGenerator_t, double*, size_t, double, double) noexcept nogil>__curandGenerateNormalDouble)(
        generator, outputPtr, n, mean, stddev)


cdef curandStatus_t _curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerateLogNormal
    _check_or_init_curand()
    if __curandGenerateLogNormal == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateLogNormal is not found")
    return (<curandStatus_t (*)(curandGenerator_t, float*, size_t, float, float) noexcept nogil>__curandGenerateLogNormal)(
        generator, outputPtr, n, mean, stddev)


cdef curandStatus_t _curandGenerateLogNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerateLogNormalDouble
    _check_or_init_curand()
    if __curandGenerateLogNormalDouble == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateLogNormalDouble is not found")
    return (<curandStatus_t (*)(curandGenerator_t, double*, size_t, double, double) noexcept nogil>__curandGenerateLogNormalDouble)(
        generator, outputPtr, n, mean, stddev)


cdef curandStatus_t _curandCreatePoissonDistribution(double lambda_, curandDiscreteDistribution_t* discrete_distribution) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandCreatePoissonDistribution
    _check_or_init_curand()
    if __curandCreatePoissonDistribution == NULL:
        with gil:
            raise FunctionNotFoundError("function curandCreatePoissonDistribution is not found")
    return (<curandStatus_t (*)(double, curandDiscreteDistribution_t*) noexcept nogil>__curandCreatePoissonDistribution)(
        lambda_, discrete_distribution)


cdef curandStatus_t _curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandDestroyDistribution
    _check_or_init_curand()
    if __curandDestroyDistribution == NULL:
        with gil:
            raise FunctionNotFoundError("function curandDestroyDistribution is not found")
    return (<curandStatus_t (*)(curandDiscreteDistribution_t) noexcept nogil>__curandDestroyDistribution)(
        discrete_distribution)


cdef curandStatus_t _curandGeneratePoisson(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda_) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGeneratePoisson
    _check_or_init_curand()
    if __curandGeneratePoisson == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGeneratePoisson is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int*, size_t, double) noexcept nogil>__curandGeneratePoisson)(
        generator, outputPtr, n, lambda_)


cdef curandStatus_t _curandGeneratePoissonMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda_, curandMethod_t method) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGeneratePoissonMethod
    _check_or_init_curand()
    if __curandGeneratePoissonMethod == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGeneratePoissonMethod is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int*, size_t, double, curandMethod_t) noexcept nogil>__curandGeneratePoissonMethod)(
        generator, outputPtr, n, lambda_, method)


cdef curandStatus_t _curandGenerateBinomial(curandGenerator_t generator, unsigned int* outputPtr, size_t num, unsigned int n, double p) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerateBinomial
    _check_or_init_curand()
    if __curandGenerateBinomial == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateBinomial is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int*, size_t, unsigned int, double) noexcept nogil>__curandGenerateBinomial)(
        generator, outputPtr, num, n, p)


cdef curandStatus_t _curandGenerateBinomialMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t num, unsigned int n, double p, curandMethod_t method) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerateBinomialMethod
    _check_or_init_curand()
    if __curandGenerateBinomialMethod == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateBinomialMethod is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int*, size_t, unsigned int, double, curandMethod_t) noexcept nogil>__curandGenerateBinomialMethod)(
        generator, outputPtr, num, n, p, method)


cdef curandStatus_t _curandGenerateSeeds(curandGenerator_t generator) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGenerateSeeds
    _check_or_init_curand()
    if __curandGenerateSeeds == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateSeeds is not found")
    return (<curandStatus_t (*)(curandGenerator_t) noexcept nogil>__curandGenerateSeeds)(
        generator)


cdef curandStatus_t _curandGetDirectionVectors32(curandDirectionVectors32_t* vectors[], curandDirectionVectorSet_t set) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGetDirectionVectors32
    _check_or_init_curand()
    if __curandGetDirectionVectors32 == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetDirectionVectors32 is not found")
    return (<curandStatus_t (*)(curandDirectionVectors32_t**, curandDirectionVectorSet_t) noexcept nogil>__curandGetDirectionVectors32)(
        vectors, set)


cdef curandStatus_t _curandGetScrambleConstants32(unsigned int** constants) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGetScrambleConstants32
    _check_or_init_curand()
    if __curandGetScrambleConstants32 == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetScrambleConstants32 is not found")
    return (<curandStatus_t (*)(unsigned int**) noexcept nogil>__curandGetScrambleConstants32)(
        constants)


cdef curandStatus_t _curandGetDirectionVectors64(curandDirectionVectors64_t* vectors[], curandDirectionVectorSet_t set) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGetDirectionVectors64
    _check_or_init_curand()
    if __curandGetDirectionVectors64 == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetDirectionVectors64 is not found")
    return (<curandStatus_t (*)(curandDirectionVectors64_t**, curandDirectionVectorSet_t) noexcept nogil>__curandGetDirectionVectors64)(
        vectors, set)


cdef curandStatus_t _curandGetScrambleConstants64(unsigned long long** constants) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    global __curandGetScrambleConstants64
    _check_or_init_curand()
    if __curandGetScrambleConstants64 == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetScrambleConstants64 is not found")
    return (<curandStatus_t (*)(unsigned long long**) noexcept nogil>__curandGetScrambleConstants64)(
        constants)
