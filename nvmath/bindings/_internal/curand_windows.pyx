# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_curand_dso_version_suffix

import os
import site

import win32api

from .utils import FunctionNotFoundError, NotSupportedError


###############################################################################
# Wrapper init
###############################################################################

LOAD_LIBRARY_SEARCH_SYSTEM32     = 0x00000800
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
cdef bint __py_curand_init = False
cdef void* __cuDriverGetVersion = NULL

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
    handle = 0

    for suffix in get_curand_dso_version_suffix(driver_ver):
        if len(suffix) == 0:
            continue
        dll_name = f"curand64_{suffix}.dll"

        # First check if the DLL has been loaded by 3rd parties
        try:
            handle = win32api.GetModuleHandle(dll_name)
        except:
            pass
        else:
            break

        # Next, check if DLLs are installed via pip
        for sp in get_site_packages():
            mod_path = os.path.join(sp, "nvidia", "curand", "bin")
            if not os.path.isdir(mod_path):
                continue
            os.add_dll_directory(mod_path)
        try:
            handle = win32api.LoadLibraryEx(
                # Note: LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR needs an abs path...
                os.path.join(mod_path, dll_name),
                0, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
        except:
            pass
        else:
            break

        # Finally, try default search
        try:
            handle = win32api.LoadLibrary(dll_name)
        except:
            pass
        else:
            break
    else:
        raise RuntimeError('Failed to load curand')

    assert handle != 0
    return handle


cdef int _check_or_init_curand() except -1 nogil:
    global __py_curand_init
    if __py_curand_init:
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
        err = (<int (*)(int*) nogil>__cuDriverGetVersion)(&driver_ver)
        if err != 0:
            raise RuntimeError('something went wrong')

        # Load library
        handle = load_library(driver_ver)

        # Load function
        global __curandCreateGenerator
        try:
            __curandCreateGenerator = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandCreateGenerator')
        except:
            pass

        global __curandCreateGeneratorHost
        try:
            __curandCreateGeneratorHost = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandCreateGeneratorHost')
        except:
            pass

        global __curandDestroyGenerator
        try:
            __curandDestroyGenerator = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandDestroyGenerator')
        except:
            pass

        global __curandGetVersion
        try:
            __curandGetVersion = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGetVersion')
        except:
            pass

        global __curandGetProperty
        try:
            __curandGetProperty = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGetProperty')
        except:
            pass

        global __curandSetStream
        try:
            __curandSetStream = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandSetStream')
        except:
            pass

        global __curandSetPseudoRandomGeneratorSeed
        try:
            __curandSetPseudoRandomGeneratorSeed = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandSetPseudoRandomGeneratorSeed')
        except:
            pass

        global __curandSetGeneratorOffset
        try:
            __curandSetGeneratorOffset = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandSetGeneratorOffset')
        except:
            pass

        global __curandSetGeneratorOrdering
        try:
            __curandSetGeneratorOrdering = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandSetGeneratorOrdering')
        except:
            pass

        global __curandSetQuasiRandomGeneratorDimensions
        try:
            __curandSetQuasiRandomGeneratorDimensions = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandSetQuasiRandomGeneratorDimensions')
        except:
            pass

        global __curandGenerate
        try:
            __curandGenerate = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerate')
        except:
            pass

        global __curandGenerateLongLong
        try:
            __curandGenerateLongLong = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerateLongLong')
        except:
            pass

        global __curandGenerateUniform
        try:
            __curandGenerateUniform = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerateUniform')
        except:
            pass

        global __curandGenerateUniformDouble
        try:
            __curandGenerateUniformDouble = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerateUniformDouble')
        except:
            pass

        global __curandGenerateNormal
        try:
            __curandGenerateNormal = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerateNormal')
        except:
            pass

        global __curandGenerateNormalDouble
        try:
            __curandGenerateNormalDouble = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerateNormalDouble')
        except:
            pass

        global __curandGenerateLogNormal
        try:
            __curandGenerateLogNormal = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerateLogNormal')
        except:
            pass

        global __curandGenerateLogNormalDouble
        try:
            __curandGenerateLogNormalDouble = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerateLogNormalDouble')
        except:
            pass

        global __curandCreatePoissonDistribution
        try:
            __curandCreatePoissonDistribution = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandCreatePoissonDistribution')
        except:
            pass

        global __curandDestroyDistribution
        try:
            __curandDestroyDistribution = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandDestroyDistribution')
        except:
            pass

        global __curandGeneratePoisson
        try:
            __curandGeneratePoisson = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGeneratePoisson')
        except:
            pass

        global __curandGeneratePoissonMethod
        try:
            __curandGeneratePoissonMethod = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGeneratePoissonMethod')
        except:
            pass

        global __curandGenerateBinomial
        try:
            __curandGenerateBinomial = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerateBinomial')
        except:
            pass

        global __curandGenerateBinomialMethod
        try:
            __curandGenerateBinomialMethod = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerateBinomialMethod')
        except:
            pass

        global __curandGenerateSeeds
        try:
            __curandGenerateSeeds = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGenerateSeeds')
        except:
            pass

        global __curandGetDirectionVectors32
        try:
            __curandGetDirectionVectors32 = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGetDirectionVectors32')
        except:
            pass

        global __curandGetScrambleConstants32
        try:
            __curandGetScrambleConstants32 = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGetScrambleConstants32')
        except:
            pass

        global __curandGetDirectionVectors64
        try:
            __curandGetDirectionVectors64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGetDirectionVectors64')
        except:
            pass

        global __curandGetScrambleConstants64
        try:
            __curandGetScrambleConstants64 = <void*><intptr_t>win32api.GetProcAddress(handle, 'curandGetScrambleConstants64')
        except:
            pass

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

cdef curandStatus_t _curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) except* nogil:
    global __curandCreateGenerator
    _check_or_init_curand()
    if __curandCreateGenerator == NULL:
        with gil:
            raise FunctionNotFoundError("function curandCreateGenerator is not found")
    return (<curandStatus_t (*)(curandGenerator_t*, curandRngType_t) nogil>__curandCreateGenerator)(
        generator, rng_type)


cdef curandStatus_t _curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type) except* nogil:
    global __curandCreateGeneratorHost
    _check_or_init_curand()
    if __curandCreateGeneratorHost == NULL:
        with gil:
            raise FunctionNotFoundError("function curandCreateGeneratorHost is not found")
    return (<curandStatus_t (*)(curandGenerator_t*, curandRngType_t) nogil>__curandCreateGeneratorHost)(
        generator, rng_type)


cdef curandStatus_t _curandDestroyGenerator(curandGenerator_t generator) except* nogil:
    global __curandDestroyGenerator
    _check_or_init_curand()
    if __curandDestroyGenerator == NULL:
        with gil:
            raise FunctionNotFoundError("function curandDestroyGenerator is not found")
    return (<curandStatus_t (*)(curandGenerator_t) nogil>__curandDestroyGenerator)(
        generator)


cdef curandStatus_t _curandGetVersion(int* version) except* nogil:
    global __curandGetVersion
    _check_or_init_curand()
    if __curandGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetVersion is not found")
    return (<curandStatus_t (*)(int*) nogil>__curandGetVersion)(
        version)


cdef curandStatus_t _curandGetProperty(libraryPropertyType type, int* value) except* nogil:
    global __curandGetProperty
    _check_or_init_curand()
    if __curandGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetProperty is not found")
    return (<curandStatus_t (*)(libraryPropertyType, int*) nogil>__curandGetProperty)(
        type, value)


cdef curandStatus_t _curandSetStream(curandGenerator_t generator, cudaStream_t stream) except* nogil:
    global __curandSetStream
    _check_or_init_curand()
    if __curandSetStream == NULL:
        with gil:
            raise FunctionNotFoundError("function curandSetStream is not found")
    return (<curandStatus_t (*)(curandGenerator_t, cudaStream_t) nogil>__curandSetStream)(
        generator, stream)


cdef curandStatus_t _curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) except* nogil:
    global __curandSetPseudoRandomGeneratorSeed
    _check_or_init_curand()
    if __curandSetPseudoRandomGeneratorSeed == NULL:
        with gil:
            raise FunctionNotFoundError("function curandSetPseudoRandomGeneratorSeed is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned long long) nogil>__curandSetPseudoRandomGeneratorSeed)(
        generator, seed)


cdef curandStatus_t _curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) except* nogil:
    global __curandSetGeneratorOffset
    _check_or_init_curand()
    if __curandSetGeneratorOffset == NULL:
        with gil:
            raise FunctionNotFoundError("function curandSetGeneratorOffset is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned long long) nogil>__curandSetGeneratorOffset)(
        generator, offset)


cdef curandStatus_t _curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) except* nogil:
    global __curandSetGeneratorOrdering
    _check_or_init_curand()
    if __curandSetGeneratorOrdering == NULL:
        with gil:
            raise FunctionNotFoundError("function curandSetGeneratorOrdering is not found")
    return (<curandStatus_t (*)(curandGenerator_t, curandOrdering_t) nogil>__curandSetGeneratorOrdering)(
        generator, order)


cdef curandStatus_t _curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions) except* nogil:
    global __curandSetQuasiRandomGeneratorDimensions
    _check_or_init_curand()
    if __curandSetQuasiRandomGeneratorDimensions == NULL:
        with gil:
            raise FunctionNotFoundError("function curandSetQuasiRandomGeneratorDimensions is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int) nogil>__curandSetQuasiRandomGeneratorDimensions)(
        generator, num_dimensions)


cdef curandStatus_t _curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num) except* nogil:
    global __curandGenerate
    _check_or_init_curand()
    if __curandGenerate == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerate is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int*, size_t) nogil>__curandGenerate)(
        generator, outputPtr, num)


cdef curandStatus_t _curandGenerateLongLong(curandGenerator_t generator, unsigned long long* outputPtr, size_t num) except* nogil:
    global __curandGenerateLongLong
    _check_or_init_curand()
    if __curandGenerateLongLong == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateLongLong is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned long long*, size_t) nogil>__curandGenerateLongLong)(
        generator, outputPtr, num)


cdef curandStatus_t _curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num) except* nogil:
    global __curandGenerateUniform
    _check_or_init_curand()
    if __curandGenerateUniform == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateUniform is not found")
    return (<curandStatus_t (*)(curandGenerator_t, float*, size_t) nogil>__curandGenerateUniform)(
        generator, outputPtr, num)


cdef curandStatus_t _curandGenerateUniformDouble(curandGenerator_t generator, double* outputPtr, size_t num) except* nogil:
    global __curandGenerateUniformDouble
    _check_or_init_curand()
    if __curandGenerateUniformDouble == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateUniformDouble is not found")
    return (<curandStatus_t (*)(curandGenerator_t, double*, size_t) nogil>__curandGenerateUniformDouble)(
        generator, outputPtr, num)


cdef curandStatus_t _curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) except* nogil:
    global __curandGenerateNormal
    _check_or_init_curand()
    if __curandGenerateNormal == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateNormal is not found")
    return (<curandStatus_t (*)(curandGenerator_t, float*, size_t, float, float) nogil>__curandGenerateNormal)(
        generator, outputPtr, n, mean, stddev)


cdef curandStatus_t _curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) except* nogil:
    global __curandGenerateNormalDouble
    _check_or_init_curand()
    if __curandGenerateNormalDouble == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateNormalDouble is not found")
    return (<curandStatus_t (*)(curandGenerator_t, double*, size_t, double, double) nogil>__curandGenerateNormalDouble)(
        generator, outputPtr, n, mean, stddev)


cdef curandStatus_t _curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) except* nogil:
    global __curandGenerateLogNormal
    _check_or_init_curand()
    if __curandGenerateLogNormal == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateLogNormal is not found")
    return (<curandStatus_t (*)(curandGenerator_t, float*, size_t, float, float) nogil>__curandGenerateLogNormal)(
        generator, outputPtr, n, mean, stddev)


cdef curandStatus_t _curandGenerateLogNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) except* nogil:
    global __curandGenerateLogNormalDouble
    _check_or_init_curand()
    if __curandGenerateLogNormalDouble == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateLogNormalDouble is not found")
    return (<curandStatus_t (*)(curandGenerator_t, double*, size_t, double, double) nogil>__curandGenerateLogNormalDouble)(
        generator, outputPtr, n, mean, stddev)


cdef curandStatus_t _curandCreatePoissonDistribution(double lambda_, curandDiscreteDistribution_t* discrete_distribution) except* nogil:
    global __curandCreatePoissonDistribution
    _check_or_init_curand()
    if __curandCreatePoissonDistribution == NULL:
        with gil:
            raise FunctionNotFoundError("function curandCreatePoissonDistribution is not found")
    return (<curandStatus_t (*)(double, curandDiscreteDistribution_t*) nogil>__curandCreatePoissonDistribution)(
        lambda_, discrete_distribution)


cdef curandStatus_t _curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) except* nogil:
    global __curandDestroyDistribution
    _check_or_init_curand()
    if __curandDestroyDistribution == NULL:
        with gil:
            raise FunctionNotFoundError("function curandDestroyDistribution is not found")
    return (<curandStatus_t (*)(curandDiscreteDistribution_t) nogil>__curandDestroyDistribution)(
        discrete_distribution)


cdef curandStatus_t _curandGeneratePoisson(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda_) except* nogil:
    global __curandGeneratePoisson
    _check_or_init_curand()
    if __curandGeneratePoisson == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGeneratePoisson is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int*, size_t, double) nogil>__curandGeneratePoisson)(
        generator, outputPtr, n, lambda_)


cdef curandStatus_t _curandGeneratePoissonMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda_, curandMethod_t method) except* nogil:
    global __curandGeneratePoissonMethod
    _check_or_init_curand()
    if __curandGeneratePoissonMethod == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGeneratePoissonMethod is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int*, size_t, double, curandMethod_t) nogil>__curandGeneratePoissonMethod)(
        generator, outputPtr, n, lambda_, method)


cdef curandStatus_t _curandGenerateBinomial(curandGenerator_t generator, unsigned int* outputPtr, size_t num, unsigned int n, double p) except* nogil:
    global __curandGenerateBinomial
    _check_or_init_curand()
    if __curandGenerateBinomial == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateBinomial is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int*, size_t, unsigned int, double) nogil>__curandGenerateBinomial)(
        generator, outputPtr, num, n, p)


cdef curandStatus_t _curandGenerateBinomialMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t num, unsigned int n, double p, curandMethod_t method) except* nogil:
    global __curandGenerateBinomialMethod
    _check_or_init_curand()
    if __curandGenerateBinomialMethod == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateBinomialMethod is not found")
    return (<curandStatus_t (*)(curandGenerator_t, unsigned int*, size_t, unsigned int, double, curandMethod_t) nogil>__curandGenerateBinomialMethod)(
        generator, outputPtr, num, n, p, method)


cdef curandStatus_t _curandGenerateSeeds(curandGenerator_t generator) except* nogil:
    global __curandGenerateSeeds
    _check_or_init_curand()
    if __curandGenerateSeeds == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGenerateSeeds is not found")
    return (<curandStatus_t (*)(curandGenerator_t) nogil>__curandGenerateSeeds)(
        generator)


cdef curandStatus_t _curandGetDirectionVectors32(curandDirectionVectors32_t* vectors[], curandDirectionVectorSet_t set) except* nogil:
    global __curandGetDirectionVectors32
    _check_or_init_curand()
    if __curandGetDirectionVectors32 == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetDirectionVectors32 is not found")
    return (<curandStatus_t (*)(curandDirectionVectors32_t**, curandDirectionVectorSet_t) nogil>__curandGetDirectionVectors32)(
        vectors, set)


cdef curandStatus_t _curandGetScrambleConstants32(unsigned int** constants) except* nogil:
    global __curandGetScrambleConstants32
    _check_or_init_curand()
    if __curandGetScrambleConstants32 == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetScrambleConstants32 is not found")
    return (<curandStatus_t (*)(unsigned int**) nogil>__curandGetScrambleConstants32)(
        constants)


cdef curandStatus_t _curandGetDirectionVectors64(curandDirectionVectors64_t* vectors[], curandDirectionVectorSet_t set) except* nogil:
    global __curandGetDirectionVectors64
    _check_or_init_curand()
    if __curandGetDirectionVectors64 == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetDirectionVectors64 is not found")
    return (<curandStatus_t (*)(curandDirectionVectors64_t**, curandDirectionVectorSet_t) nogil>__curandGetDirectionVectors64)(
        vectors, set)


cdef curandStatus_t _curandGetScrambleConstants64(unsigned long long** constants) except* nogil:
    global __curandGetScrambleConstants64
    _check_or_init_curand()
    if __curandGetScrambleConstants64 == NULL:
        with gil:
            raise FunctionNotFoundError("function curandGetScrambleConstants64 is not found")
    return (<curandStatus_t (*)(unsigned long long**) nogil>__curandGetScrambleConstants64)(
        constants)
