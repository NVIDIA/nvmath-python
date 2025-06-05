# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from ..cycurand cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef curandStatus_t _curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandDestroyGenerator(curandGenerator_t generator) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGetVersion(int* version) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGetProperty(libraryPropertyType type, int* value) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandSetStream(curandGenerator_t generator, cudaStream_t stream) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerateLongLong(curandGenerator_t generator, unsigned long long* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerateUniformDouble(curandGenerator_t generator, double* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerateLogNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandCreatePoissonDistribution(double lambda_, curandDiscreteDistribution_t* discrete_distribution) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGeneratePoisson(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda_) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGeneratePoissonMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda_, curandMethod_t method) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerateBinomial(curandGenerator_t generator, unsigned int* outputPtr, size_t num, unsigned int n, double p) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerateBinomialMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t num, unsigned int n, double p, curandMethod_t method) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGenerateSeeds(curandGenerator_t generator) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGetDirectionVectors32(curandDirectionVectors32_t* vectors[], curandDirectionVectorSet_t set) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGetScrambleConstants32(unsigned int** constants) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGetDirectionVectors64(curandDirectionVectors64_t* vectors[], curandDirectionVectorSet_t set) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t _curandGetScrambleConstants64(unsigned long long** constants) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
