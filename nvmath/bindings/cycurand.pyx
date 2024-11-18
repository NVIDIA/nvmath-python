# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from ._internal cimport curand as _curand


###############################################################################
# Wrapper functions
###############################################################################

cdef curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) except* nogil:
    return _curand._curandCreateGenerator(generator, rng_type)


cdef curandStatus_t curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type) except* nogil:
    return _curand._curandCreateGeneratorHost(generator, rng_type)


cdef curandStatus_t curandDestroyGenerator(curandGenerator_t generator) except* nogil:
    return _curand._curandDestroyGenerator(generator)


cdef curandStatus_t curandGetVersion(int* version) except* nogil:
    return _curand._curandGetVersion(version)


cdef curandStatus_t curandGetProperty(libraryPropertyType type, int* value) except* nogil:
    return _curand._curandGetProperty(type, value)


cdef curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream) except* nogil:
    return _curand._curandSetStream(generator, stream)


cdef curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) except* nogil:
    return _curand._curandSetPseudoRandomGeneratorSeed(generator, seed)


cdef curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) except* nogil:
    return _curand._curandSetGeneratorOffset(generator, offset)


cdef curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) except* nogil:
    return _curand._curandSetGeneratorOrdering(generator, order)


cdef curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions) except* nogil:
    return _curand._curandSetQuasiRandomGeneratorDimensions(generator, num_dimensions)


cdef curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num) except* nogil:
    return _curand._curandGenerate(generator, outputPtr, num)


cdef curandStatus_t curandGenerateLongLong(curandGenerator_t generator, unsigned long long* outputPtr, size_t num) except* nogil:
    return _curand._curandGenerateLongLong(generator, outputPtr, num)


cdef curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num) except* nogil:
    return _curand._curandGenerateUniform(generator, outputPtr, num)


cdef curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double* outputPtr, size_t num) except* nogil:
    return _curand._curandGenerateUniformDouble(generator, outputPtr, num)


cdef curandStatus_t curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) except* nogil:
    return _curand._curandGenerateNormal(generator, outputPtr, n, mean, stddev)


cdef curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) except* nogil:
    return _curand._curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev)


cdef curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) except* nogil:
    return _curand._curandGenerateLogNormal(generator, outputPtr, n, mean, stddev)


cdef curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) except* nogil:
    return _curand._curandGenerateLogNormalDouble(generator, outputPtr, n, mean, stddev)


cdef curandStatus_t curandCreatePoissonDistribution(double lambda_, curandDiscreteDistribution_t* discrete_distribution) except* nogil:
    return _curand._curandCreatePoissonDistribution(lambda_, discrete_distribution)


cdef curandStatus_t curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) except* nogil:
    return _curand._curandDestroyDistribution(discrete_distribution)


cdef curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda_) except* nogil:
    return _curand._curandGeneratePoisson(generator, outputPtr, n, lambda_)


cdef curandStatus_t curandGeneratePoissonMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda_, curandMethod_t method) except* nogil:
    return _curand._curandGeneratePoissonMethod(generator, outputPtr, n, lambda_, method)


cdef curandStatus_t curandGenerateBinomial(curandGenerator_t generator, unsigned int* outputPtr, size_t num, unsigned int n, double p) except* nogil:
    return _curand._curandGenerateBinomial(generator, outputPtr, num, n, p)


cdef curandStatus_t curandGenerateBinomialMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t num, unsigned int n, double p, curandMethod_t method) except* nogil:
    return _curand._curandGenerateBinomialMethod(generator, outputPtr, num, n, p, method)


cdef curandStatus_t curandGenerateSeeds(curandGenerator_t generator) except* nogil:
    return _curand._curandGenerateSeeds(generator)


cdef curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t* vectors[], curandDirectionVectorSet_t set) except* nogil:
    return _curand._curandGetDirectionVectors32(vectors, set)


cdef curandStatus_t curandGetScrambleConstants32(unsigned int** constants) except* nogil:
    return _curand._curandGetScrambleConstants32(constants)


cdef curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t* vectors[], curandDirectionVectorSet_t set) except* nogil:
    return _curand._curandGetDirectionVectors64(vectors, set)


cdef curandStatus_t curandGetScrambleConstants64(unsigned long long** constants) except* nogil:
    return _curand._curandGetScrambleConstants64(constants)
