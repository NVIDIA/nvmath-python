# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int64_t
from libc.stdio cimport FILE


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum curandStatus "curandStatus":
    CURAND_STATUS_SUCCESS "CURAND_STATUS_SUCCESS" = 0
    CURAND_STATUS_VERSION_MISMATCH "CURAND_STATUS_VERSION_MISMATCH" = 100
    CURAND_STATUS_NOT_INITIALIZED "CURAND_STATUS_NOT_INITIALIZED" = 101
    CURAND_STATUS_ALLOCATION_FAILED "CURAND_STATUS_ALLOCATION_FAILED" = 102
    CURAND_STATUS_TYPE_ERROR "CURAND_STATUS_TYPE_ERROR" = 103
    CURAND_STATUS_OUT_OF_RANGE "CURAND_STATUS_OUT_OF_RANGE" = 104
    CURAND_STATUS_LENGTH_NOT_MULTIPLE "CURAND_STATUS_LENGTH_NOT_MULTIPLE" = 105
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED" = 106
    CURAND_STATUS_LAUNCH_FAILURE "CURAND_STATUS_LAUNCH_FAILURE" = 201
    CURAND_STATUS_PREEXISTING_FAILURE "CURAND_STATUS_PREEXISTING_FAILURE" = 202
    CURAND_STATUS_INITIALIZATION_FAILED "CURAND_STATUS_INITIALIZATION_FAILED" = 203
    CURAND_STATUS_ARCH_MISMATCH "CURAND_STATUS_ARCH_MISMATCH" = 204
    CURAND_STATUS_INTERNAL_ERROR "CURAND_STATUS_INTERNAL_ERROR" = 999
    _CURANDSTATUS_T_INTERNAL_LOADING_ERROR "_CURANDSTATUS_T_INTERNAL_LOADING_ERROR" = -42

ctypedef curandStatus curandStatus_t "curandStatus_t"

ctypedef enum curandRngType "curandRngType":
    CURAND_RNG_TEST "CURAND_RNG_TEST" = 0
    CURAND_RNG_PSEUDO_DEFAULT "CURAND_RNG_PSEUDO_DEFAULT" = 100
    CURAND_RNG_PSEUDO_XORWOW "CURAND_RNG_PSEUDO_XORWOW" = 101
    CURAND_RNG_PSEUDO_MRG32K3A "CURAND_RNG_PSEUDO_MRG32K3A" = 121
    CURAND_RNG_PSEUDO_MTGP32 "CURAND_RNG_PSEUDO_MTGP32" = 141
    CURAND_RNG_PSEUDO_MT19937 "CURAND_RNG_PSEUDO_MT19937" = 142
    CURAND_RNG_PSEUDO_PHILOX4_32_10 "CURAND_RNG_PSEUDO_PHILOX4_32_10" = 161
    CURAND_RNG_QUASI_DEFAULT "CURAND_RNG_QUASI_DEFAULT" = 200
    CURAND_RNG_QUASI_SOBOL32 "CURAND_RNG_QUASI_SOBOL32" = 201
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32" = 202
    CURAND_RNG_QUASI_SOBOL64 "CURAND_RNG_QUASI_SOBOL64" = 203
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64" = 204

ctypedef curandRngType curandRngType_t "curandRngType_t"

ctypedef enum curandOrdering "curandOrdering":
    CURAND_ORDERING_PSEUDO_BEST "CURAND_ORDERING_PSEUDO_BEST" = 100
    CURAND_ORDERING_PSEUDO_DEFAULT "CURAND_ORDERING_PSEUDO_DEFAULT" = 101
    CURAND_ORDERING_PSEUDO_SEEDED "CURAND_ORDERING_PSEUDO_SEEDED" = 102
    CURAND_ORDERING_PSEUDO_LEGACY "CURAND_ORDERING_PSEUDO_LEGACY" = 103
    CURAND_ORDERING_PSEUDO_DYNAMIC "CURAND_ORDERING_PSEUDO_DYNAMIC" = 104
    CURAND_ORDERING_QUASI_DEFAULT "CURAND_ORDERING_QUASI_DEFAULT" = 201

ctypedef curandOrdering curandOrdering_t "curandOrdering_t"

ctypedef enum curandDirectionVectorSet "curandDirectionVectorSet":
    CURAND_DIRECTION_VECTORS_32_JOEKUO6 "CURAND_DIRECTION_VECTORS_32_JOEKUO6" = 101
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 "CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6" = 102
    CURAND_DIRECTION_VECTORS_64_JOEKUO6 "CURAND_DIRECTION_VECTORS_64_JOEKUO6" = 103
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 "CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6" = 104

ctypedef curandDirectionVectorSet curandDirectionVectorSet_t "curandDirectionVectorSet_t"

ctypedef enum curandMethod "curandMethod":
    CURAND_CHOOSE_BEST "CURAND_CHOOSE_BEST" = 0
    CURAND_ITR "CURAND_ITR" = 1
    CURAND_KNUTH "CURAND_KNUTH" = 2
    CURAND_HITR "CURAND_HITR" = 3
    CURAND_M1 "CURAND_M1" = 4
    CURAND_M2 "CURAND_M2" = 5
    CURAND_BINARY_SEARCH "CURAND_BINARY_SEARCH" = 6
    CURAND_DISCRETE_GAUSS "CURAND_DISCRETE_GAUSS" = 7
    CURAND_REJECTION "CURAND_REJECTION" = 8
    CURAND_DEVICE_API "CURAND_DEVICE_API" = 9
    CURAND_FAST_REJECTION "CURAND_FAST_REJECTION" = 10
    CURAND_3RD "CURAND_3RD" = 11
    CURAND_DEFINITION "CURAND_DEFINITION" = 12
    CURAND_POISSON "CURAND_POISSON" = 13

ctypedef curandMethod curandMethod_t "curandMethod_t"


# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef int cudaDataType_t 'cudaDataType_t'
    ctypedef int cudaDataType 'cudaDataType'
    ctypedef int libraryPropertyType_t 'libraryPropertyType_t'
    ctypedef int libraryPropertyType 'libraryPropertyType'


ctypedef void* curandGenerator_t 'curandGenerator_t'
ctypedef void* curandDistribution_t 'curandDistribution_t'
ctypedef void* curandDistributionShift_t 'curandDistributionShift_t'
ctypedef void* curandDistributionM2Shift_t 'curandDistributionM2Shift_t'
ctypedef void* curandHistogramM2_t 'curandHistogramM2_t'
ctypedef void* curandHistogramM2K_t 'curandHistogramM2K_t'
ctypedef void* curandHistogramM2V_t 'curandHistogramM2V_t'
ctypedef void* curandDiscreteDistribution_t 'curandDiscreteDistribution_t'
ctypedef unsigned int curandDirectionVectors32_t[32]
ctypedef unsigned long long curandDirectionVectors64_t[64]


###############################################################################
# Functions
###############################################################################

cdef curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandDestroyGenerator(curandGenerator_t generator) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGetVersion(int* version) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGetProperty(libraryPropertyType type, int* value) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerateLongLong(curandGenerator_t generator, unsigned long long* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double* outputPtr, size_t num) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandCreatePoissonDistribution(double lambda_, curandDiscreteDistribution_t* discrete_distribution) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda_) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGeneratePoissonMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda_, curandMethod_t method) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerateBinomial(curandGenerator_t generator, unsigned int* outputPtr, size_t num, unsigned int n, double p) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerateBinomialMethod(curandGenerator_t generator, unsigned int* outputPtr, size_t num, unsigned int n, double p, curandMethod_t method) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGenerateSeeds(curandGenerator_t generator) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t* vectors[], curandDirectionVectorSet_t set) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGetScrambleConstants32(unsigned int** constants) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t* vectors[], curandDirectionVectorSet_t set) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef curandStatus_t curandGetScrambleConstants64(unsigned long long** constants) except?_CURANDSTATUS_T_INTERNAL_LOADING_ERROR nogil
