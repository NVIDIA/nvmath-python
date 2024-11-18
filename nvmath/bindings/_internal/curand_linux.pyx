# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_curand_dso_version_suffix

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


cdef void* load_library(const int driver_ver) except* with gil:
    cdef void* handle
    for suffix in get_curand_dso_version_suffix(driver_ver):
        so_name = "libcurand.so" + (f".{suffix}" if suffix else suffix)
        handle = dlopen(so_name.encode(), RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            break
    else:
        err_msg = dlerror()
        raise RuntimeError(f'Failed to dlopen libcurand ({err_msg.decode()})')
    return handle


cdef int _check_or_init_curand() except -1 nogil:
    global __py_curand_init
    if __py_curand_init:
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
    global __curandCreateGenerator
    __curandCreateGenerator = dlsym(RTLD_DEFAULT, 'curandCreateGenerator')
    if __curandCreateGenerator == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandCreateGenerator = dlsym(handle, 'curandCreateGenerator')

    global __curandCreateGeneratorHost
    __curandCreateGeneratorHost = dlsym(RTLD_DEFAULT, 'curandCreateGeneratorHost')
    if __curandCreateGeneratorHost == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandCreateGeneratorHost = dlsym(handle, 'curandCreateGeneratorHost')

    global __curandDestroyGenerator
    __curandDestroyGenerator = dlsym(RTLD_DEFAULT, 'curandDestroyGenerator')
    if __curandDestroyGenerator == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandDestroyGenerator = dlsym(handle, 'curandDestroyGenerator')

    global __curandGetVersion
    __curandGetVersion = dlsym(RTLD_DEFAULT, 'curandGetVersion')
    if __curandGetVersion == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGetVersion = dlsym(handle, 'curandGetVersion')

    global __curandGetProperty
    __curandGetProperty = dlsym(RTLD_DEFAULT, 'curandGetProperty')
    if __curandGetProperty == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGetProperty = dlsym(handle, 'curandGetProperty')

    global __curandSetStream
    __curandSetStream = dlsym(RTLD_DEFAULT, 'curandSetStream')
    if __curandSetStream == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandSetStream = dlsym(handle, 'curandSetStream')

    global __curandSetPseudoRandomGeneratorSeed
    __curandSetPseudoRandomGeneratorSeed = dlsym(RTLD_DEFAULT, 'curandSetPseudoRandomGeneratorSeed')
    if __curandSetPseudoRandomGeneratorSeed == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandSetPseudoRandomGeneratorSeed = dlsym(handle, 'curandSetPseudoRandomGeneratorSeed')

    global __curandSetGeneratorOffset
    __curandSetGeneratorOffset = dlsym(RTLD_DEFAULT, 'curandSetGeneratorOffset')
    if __curandSetGeneratorOffset == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandSetGeneratorOffset = dlsym(handle, 'curandSetGeneratorOffset')

    global __curandSetGeneratorOrdering
    __curandSetGeneratorOrdering = dlsym(RTLD_DEFAULT, 'curandSetGeneratorOrdering')
    if __curandSetGeneratorOrdering == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandSetGeneratorOrdering = dlsym(handle, 'curandSetGeneratorOrdering')

    global __curandSetQuasiRandomGeneratorDimensions
    __curandSetQuasiRandomGeneratorDimensions = dlsym(RTLD_DEFAULT, 'curandSetQuasiRandomGeneratorDimensions')
    if __curandSetQuasiRandomGeneratorDimensions == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandSetQuasiRandomGeneratorDimensions = dlsym(handle, 'curandSetQuasiRandomGeneratorDimensions')

    global __curandGenerate
    __curandGenerate = dlsym(RTLD_DEFAULT, 'curandGenerate')
    if __curandGenerate == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerate = dlsym(handle, 'curandGenerate')

    global __curandGenerateLongLong
    __curandGenerateLongLong = dlsym(RTLD_DEFAULT, 'curandGenerateLongLong')
    if __curandGenerateLongLong == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerateLongLong = dlsym(handle, 'curandGenerateLongLong')

    global __curandGenerateUniform
    __curandGenerateUniform = dlsym(RTLD_DEFAULT, 'curandGenerateUniform')
    if __curandGenerateUniform == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerateUniform = dlsym(handle, 'curandGenerateUniform')

    global __curandGenerateUniformDouble
    __curandGenerateUniformDouble = dlsym(RTLD_DEFAULT, 'curandGenerateUniformDouble')
    if __curandGenerateUniformDouble == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerateUniformDouble = dlsym(handle, 'curandGenerateUniformDouble')

    global __curandGenerateNormal
    __curandGenerateNormal = dlsym(RTLD_DEFAULT, 'curandGenerateNormal')
    if __curandGenerateNormal == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerateNormal = dlsym(handle, 'curandGenerateNormal')

    global __curandGenerateNormalDouble
    __curandGenerateNormalDouble = dlsym(RTLD_DEFAULT, 'curandGenerateNormalDouble')
    if __curandGenerateNormalDouble == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerateNormalDouble = dlsym(handle, 'curandGenerateNormalDouble')

    global __curandGenerateLogNormal
    __curandGenerateLogNormal = dlsym(RTLD_DEFAULT, 'curandGenerateLogNormal')
    if __curandGenerateLogNormal == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerateLogNormal = dlsym(handle, 'curandGenerateLogNormal')

    global __curandGenerateLogNormalDouble
    __curandGenerateLogNormalDouble = dlsym(RTLD_DEFAULT, 'curandGenerateLogNormalDouble')
    if __curandGenerateLogNormalDouble == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerateLogNormalDouble = dlsym(handle, 'curandGenerateLogNormalDouble')

    global __curandCreatePoissonDistribution
    __curandCreatePoissonDistribution = dlsym(RTLD_DEFAULT, 'curandCreatePoissonDistribution')
    if __curandCreatePoissonDistribution == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandCreatePoissonDistribution = dlsym(handle, 'curandCreatePoissonDistribution')

    global __curandDestroyDistribution
    __curandDestroyDistribution = dlsym(RTLD_DEFAULT, 'curandDestroyDistribution')
    if __curandDestroyDistribution == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandDestroyDistribution = dlsym(handle, 'curandDestroyDistribution')

    global __curandGeneratePoisson
    __curandGeneratePoisson = dlsym(RTLD_DEFAULT, 'curandGeneratePoisson')
    if __curandGeneratePoisson == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGeneratePoisson = dlsym(handle, 'curandGeneratePoisson')

    global __curandGeneratePoissonMethod
    __curandGeneratePoissonMethod = dlsym(RTLD_DEFAULT, 'curandGeneratePoissonMethod')
    if __curandGeneratePoissonMethod == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGeneratePoissonMethod = dlsym(handle, 'curandGeneratePoissonMethod')

    global __curandGenerateBinomial
    __curandGenerateBinomial = dlsym(RTLD_DEFAULT, 'curandGenerateBinomial')
    if __curandGenerateBinomial == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerateBinomial = dlsym(handle, 'curandGenerateBinomial')

    global __curandGenerateBinomialMethod
    __curandGenerateBinomialMethod = dlsym(RTLD_DEFAULT, 'curandGenerateBinomialMethod')
    if __curandGenerateBinomialMethod == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerateBinomialMethod = dlsym(handle, 'curandGenerateBinomialMethod')

    global __curandGenerateSeeds
    __curandGenerateSeeds = dlsym(RTLD_DEFAULT, 'curandGenerateSeeds')
    if __curandGenerateSeeds == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGenerateSeeds = dlsym(handle, 'curandGenerateSeeds')

    global __curandGetDirectionVectors32
    __curandGetDirectionVectors32 = dlsym(RTLD_DEFAULT, 'curandGetDirectionVectors32')
    if __curandGetDirectionVectors32 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGetDirectionVectors32 = dlsym(handle, 'curandGetDirectionVectors32')

    global __curandGetScrambleConstants32
    __curandGetScrambleConstants32 = dlsym(RTLD_DEFAULT, 'curandGetScrambleConstants32')
    if __curandGetScrambleConstants32 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGetScrambleConstants32 = dlsym(handle, 'curandGetScrambleConstants32')

    global __curandGetDirectionVectors64
    __curandGetDirectionVectors64 = dlsym(RTLD_DEFAULT, 'curandGetDirectionVectors64')
    if __curandGetDirectionVectors64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGetDirectionVectors64 = dlsym(handle, 'curandGetDirectionVectors64')

    global __curandGetScrambleConstants64
    __curandGetScrambleConstants64 = dlsym(RTLD_DEFAULT, 'curandGetScrambleConstants64')
    if __curandGetScrambleConstants64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __curandGetScrambleConstants64 = dlsym(handle, 'curandGetScrambleConstants64')

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
