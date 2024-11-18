# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

cimport cython  # NOQA
cimport cpython
from cpython cimport memoryview as _memoryview

from enum import IntEnum as _IntEnum

import numpy as _numpy


###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """See `curandStatus`."""
    SUCCESS = CURAND_STATUS_SUCCESS
    VERSION_MISMATCH = CURAND_STATUS_VERSION_MISMATCH
    NOT_INITIALIZED = CURAND_STATUS_NOT_INITIALIZED
    ALLOCATION_FAILED = CURAND_STATUS_ALLOCATION_FAILED
    TYPE_ERROR = CURAND_STATUS_TYPE_ERROR
    OUT_OF_RANGE = CURAND_STATUS_OUT_OF_RANGE
    LENGTH_NOT_MULTIPLE = CURAND_STATUS_LENGTH_NOT_MULTIPLE
    DOUBLE_PRECISION_REQUIRED = CURAND_STATUS_DOUBLE_PRECISION_REQUIRED
    LAUNCH_FAILURE = CURAND_STATUS_LAUNCH_FAILURE
    PREEXISTING_FAILURE = CURAND_STATUS_PREEXISTING_FAILURE
    INITIALIZATION_FAILED = CURAND_STATUS_INITIALIZATION_FAILED
    ARCH_MISMATCH = CURAND_STATUS_ARCH_MISMATCH
    INTERNAL_ERROR = CURAND_STATUS_INTERNAL_ERROR

class RngType(_IntEnum):
    """See `curandRngType`."""
    TEST = CURAND_RNG_TEST
    PSEUDO_DEFAULT = CURAND_RNG_PSEUDO_DEFAULT
    PSEUDO_XORWOW = CURAND_RNG_PSEUDO_XORWOW
    PSEUDO_MRG32K3A = CURAND_RNG_PSEUDO_MRG32K3A
    PSEUDO_MTGP32 = CURAND_RNG_PSEUDO_MTGP32
    PSEUDO_MT19937 = CURAND_RNG_PSEUDO_MT19937
    PSEUDO_PHILOX4_32_10 = CURAND_RNG_PSEUDO_PHILOX4_32_10
    QUASI_DEFAULT = CURAND_RNG_QUASI_DEFAULT
    QUASI_SOBOL32 = CURAND_RNG_QUASI_SOBOL32
    QUASI_SCRAMBLED_SOBOL32 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
    QUASI_SOBOL64 = CURAND_RNG_QUASI_SOBOL64
    QUASI_SCRAMBLED_SOBOL64 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64

class Ordering(_IntEnum):
    """See `curandOrdering`."""
    PSEUDO_BEST = CURAND_ORDERING_PSEUDO_BEST
    PSEUDO_DEFAULT = CURAND_ORDERING_PSEUDO_DEFAULT
    PSEUDO_SEEDED = CURAND_ORDERING_PSEUDO_SEEDED
    PSEUDO_LEGACY = CURAND_ORDERING_PSEUDO_LEGACY
    PSEUDO_DYNAMIC = CURAND_ORDERING_PSEUDO_DYNAMIC
    QUASI_DEFAULT = CURAND_ORDERING_QUASI_DEFAULT

class DirectionVectorSet(_IntEnum):
    """See `curandDirectionVectorSet`."""
    DIRECTION_VECTORS_32_JOEKUO6 = CURAND_DIRECTION_VECTORS_32_JOEKUO6
    SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
    DIRECTION_VECTORS_64_JOEKUO6 = CURAND_DIRECTION_VECTORS_64_JOEKUO6
    SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6

class Method(_IntEnum):
    """See `curandMethod`."""
    METHOD_CHOOSE_BEST = CURAND_CHOOSE_BEST
    METHOD_ITR = CURAND_ITR
    METHOD_KNUTH = CURAND_KNUTH
    METHOD_HITR = CURAND_HITR
    METHOD_M1 = CURAND_M1
    METHOD_M2 = CURAND_M2
    METHOD_BINARY_SEARCH = CURAND_BINARY_SEARCH
    METHOD_DISCRETE_GAUSS = CURAND_DISCRETE_GAUSS
    METHOD_REJECTION = CURAND_REJECTION
    METHOD_DEVICE_API = CURAND_DEVICE_API
    METHOD_FAST_REJECTION = CURAND_FAST_REJECTION
    METHOD_3RD = CURAND_3RD
    METHOD_DEFINITION = CURAND_DEFINITION
    METHOD_POISSON = CURAND_POISSON


###############################################################################
# Error handling
###############################################################################

cdef class cuRANDError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value})"
        super(cuRANDError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuRANDError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create_generator(int rng_type) except? 0:
    """Create new random number generator.

    Args:
        rng_type (RngType): Type of generator to create.

    Returns:
        intptr_t: Pointer to generator.

    .. seealso:: `curandCreateGenerator`
    """
    cdef Generator generator
    with nogil:
        status = curandCreateGenerator(&generator, <_RngType>rng_type)
    check_status(status)
    return <intptr_t>generator


cpdef intptr_t create_generator_host(int rng_type) except? 0:
    """Create new host CPU random number generator.

    Args:
        rng_type (RngType): Type of generator to create.

    Returns:
        intptr_t: Pointer to generator.

    .. seealso:: `curandCreateGeneratorHost`
    """
    cdef Generator generator
    with nogil:
        status = curandCreateGeneratorHost(&generator, <_RngType>rng_type)
    check_status(status)
    return <intptr_t>generator


cpdef destroy_generator(intptr_t generator):
    """Destroy an existing generator.

    Args:
        generator (intptr_t): Generator to destroy.

    .. seealso:: `curandDestroyGenerator`
    """
    with nogil:
        status = curandDestroyGenerator(<Generator>generator)
    check_status(status)


cpdef int get_version() except? -1:
    """Return the version number of the library.

    Returns:
        int: CURAND library version.

    .. seealso:: `curandGetVersion`
    """
    cdef int version
    with nogil:
        status = curandGetVersion(&version)
    check_status(status)
    return version


cpdef int get_property(int type) except? -1:
    """Return the value of the curand property.

    Args:
        type (int): CUDA library property.

    Returns:
        int: integer value for the requested property.

    .. seealso:: `curandGetProperty`
    """
    cdef int value
    with nogil:
        status = curandGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


cpdef set_stream(intptr_t generator, intptr_t stream):
    """Set the current stream for CURAND kernel launches.

    Args:
        generator (intptr_t): Generator to modify.
        stream (intptr_t): Stream to use or ``NULL`` for null stream.

    .. seealso:: `curandSetStream`
    """
    with nogil:
        status = curandSetStream(<Generator>generator, <Stream>stream)
    check_status(status)


cpdef set_pseudo_random_generator_seed(intptr_t generator, unsigned long long seed):
    """Set the seed value of the pseudo-random number generator.

    Args:
        generator (intptr_t): Generator to modify.
        seed (unsigned long long): Seed value.

    .. seealso:: `curandSetPseudoRandomGeneratorSeed`
    """
    with nogil:
        status = curandSetPseudoRandomGeneratorSeed(<Generator>generator, seed)
    check_status(status)


cpdef set_generator_offset(intptr_t generator, unsigned long long offset):
    """Set the absolute offset of the pseudo or quasirandom number generator.

    Args:
        generator (intptr_t): Generator to modify.
        offset (unsigned long long): Absolute offset position.

    .. seealso:: `curandSetGeneratorOffset`
    """
    with nogil:
        status = curandSetGeneratorOffset(<Generator>generator, offset)
    check_status(status)


cpdef set_generator_ordering(intptr_t generator, int order):
    """Set the ordering of results of the pseudo or quasirandom number generator.

    Args:
        generator (intptr_t): Generator to modify.
        order (Ordering): Ordering of results.

    .. seealso:: `curandSetGeneratorOrdering`
    """
    with nogil:
        status = curandSetGeneratorOrdering(<Generator>generator, <_Ordering>order)
    check_status(status)


cpdef set_quasi_random_generator_dimensions(intptr_t generator, unsigned int num_dimensions):
    """Set the number of dimensions.

    Args:
        generator (intptr_t): Generator to modify.
        num_dimensions (unsigned int): Number of dimensions.

    .. seealso:: `curandSetQuasiRandomGeneratorDimensions`
    """
    with nogil:
        status = curandSetQuasiRandomGeneratorDimensions(<Generator>generator, num_dimensions)
    check_status(status)


cpdef generate(intptr_t generator, intptr_t output_ptr, size_t num):
    """Generate 32-bit pseudo or quasirandom numbers.

    Args:
        generator (intptr_t): Generator to use.
        output_ptr (intptr_t): Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results.
        num (size_t): Number of random 32-bit values to generate.

    .. seealso:: `curandGenerate`
    """
    with nogil:
        status = curandGenerate(<Generator>generator, <unsigned int*>output_ptr, num)
    check_status(status)


cpdef generate_long_long(intptr_t generator, intptr_t output_ptr, size_t num):
    """Generate 64-bit quasirandom numbers.

    Args:
        generator (intptr_t): Generator to use.
        output_ptr (intptr_t): Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results.
        num (size_t): Number of random 64-bit values to generate.

    .. seealso:: `curandGenerateLongLong`
    """
    with nogil:
        status = curandGenerateLongLong(<Generator>generator, <unsigned long long*>output_ptr, num)
    check_status(status)


cpdef generate_uniform(intptr_t generator, intptr_t output_ptr, size_t num):
    """Generate uniformly distributed floats.

    Args:
        generator (intptr_t): Generator to use.
        output_ptr (intptr_t): Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results.
        num (size_t): Number of floats to generate.

    .. seealso:: `curandGenerateUniform`
    """
    with nogil:
        status = curandGenerateUniform(<Generator>generator, <float*>output_ptr, num)
    check_status(status)


cpdef generate_uniform_double(intptr_t generator, intptr_t output_ptr, size_t num):
    """Generate uniformly distributed doubles.

    Args:
        generator (intptr_t): Generator to use.
        output_ptr (intptr_t): Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results.
        num (size_t): Number of doubles to generate.

    .. seealso:: `curandGenerateUniformDouble`
    """
    with nogil:
        status = curandGenerateUniformDouble(<Generator>generator, <double*>output_ptr, num)
    check_status(status)


cpdef generate_normal(intptr_t generator, intptr_t output_ptr, size_t n, float mean, float stddev):
    """Generate normally distributed doubles.

    Args:
        generator (intptr_t): Generator to use.
        output_ptr (intptr_t): Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results.
        n (size_t): Number of floats to generate.
        mean (float): Mean of normal distribution.
        stddev (float): Standard deviation of normal distribution.

    .. seealso:: `curandGenerateNormal`
    """
    with nogil:
        status = curandGenerateNormal(<Generator>generator, <float*>output_ptr, n, mean, stddev)
    check_status(status)


cpdef generate_normal_double(intptr_t generator, intptr_t output_ptr, size_t n, double mean, double stddev):
    """Generate normally distributed doubles.

    Args:
        generator (intptr_t): Generator to use.
        output_ptr (intptr_t): Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results.
        n (size_t): Number of doubles to generate.
        mean (double): Mean of normal distribution.
        stddev (double): Standard deviation of normal distribution.

    .. seealso:: `curandGenerateNormalDouble`
    """
    with nogil:
        status = curandGenerateNormalDouble(<Generator>generator, <double*>output_ptr, n, mean, stddev)
    check_status(status)


cpdef generate_log_normal(intptr_t generator, intptr_t output_ptr, size_t n, float mean, float stddev):
    """Generate log-normally distributed floats.

    Args:
        generator (intptr_t): Generator to use.
        output_ptr (intptr_t): Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results.
        n (size_t): Number of floats to generate.
        mean (float): Mean of associated normal distribution.
        stddev (float): Standard deviation of associated normal distribution.

    .. seealso:: `curandGenerateLogNormal`
    """
    with nogil:
        status = curandGenerateLogNormal(<Generator>generator, <float*>output_ptr, n, mean, stddev)
    check_status(status)


cpdef generate_log_normal_double(intptr_t generator, intptr_t output_ptr, size_t n, double mean, double stddev):
    """Generate log-normally distributed doubles.

    Args:
        generator (intptr_t): Generator to use.
        output_ptr (intptr_t): Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results.
        n (size_t): Number of doubles to generate.
        mean (double): Mean of normal distribution.
        stddev (double): Standard deviation of normal distribution.

    .. seealso:: `curandGenerateLogNormalDouble`
    """
    with nogil:
        status = curandGenerateLogNormalDouble(<Generator>generator, <double*>output_ptr, n, mean, stddev)
    check_status(status)


cpdef create_poisson_distribution(double lambda_, intptr_t discrete_distribution):
    """Construct the histogram array for a Poisson distribution.

    Args:
        lambda_ (double): lambda for the Poisson distribution.
        discrete_distribution (intptr_t): pointer to the histogram in device memory.

    .. seealso:: `curandCreatePoissonDistribution`
    """
    with nogil:
        status = curandCreatePoissonDistribution(lambda_, <DiscreteDistribution*>discrete_distribution)
    check_status(status)


cpdef destroy_distribution(intptr_t discrete_distribution):
    """Destroy the histogram array for a discrete distribution (e.g. Poisson).

    Args:
        discrete_distribution (intptr_t): pointer to device memory where the histogram is stored.

    .. seealso:: `curandDestroyDistribution`
    """
    with nogil:
        status = curandDestroyDistribution(<DiscreteDistribution>discrete_distribution)
    check_status(status)


cpdef generate_poisson(intptr_t generator, intptr_t output_ptr, size_t n, double lambda_):
    """Generate Poisson-distributed unsigned ints.

    Args:
        generator (intptr_t): Generator to use.
        output_ptr (intptr_t): Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results.
        n (size_t): Number of unsigned ints to generate.
        lambda_ (double): lambda for the Poisson distribution.

    .. seealso:: `curandGeneratePoisson`
    """
    with nogil:
        status = curandGeneratePoisson(<Generator>generator, <unsigned int*>output_ptr, n, lambda_)
    check_status(status)


cpdef generate_poisson_method(intptr_t generator, intptr_t output_ptr, size_t n, double lambda_, int method):
    with nogil:
        status = curandGeneratePoissonMethod(<Generator>generator, <unsigned int*>output_ptr, n, lambda_, <_Method>method)
    check_status(status)


cpdef generate_binomial(intptr_t generator, intptr_t output_ptr, size_t num, unsigned int n, double p):
    with nogil:
        status = curandGenerateBinomial(<Generator>generator, <unsigned int*>output_ptr, num, n, p)
    check_status(status)


cpdef generate_binomial_method(intptr_t generator, intptr_t output_ptr, size_t num, unsigned int n, double p, int method):
    with nogil:
        status = curandGenerateBinomialMethod(<Generator>generator, <unsigned int*>output_ptr, num, n, p, <_Method>method)
    check_status(status)


cpdef generate_seeds(intptr_t generator):
    """Setup starting states.

    Args:
        generator (intptr_t): Generator to update.

    .. seealso:: `curandGenerateSeeds`
    """
    with nogil:
        status = curandGenerateSeeds(<Generator>generator)
    check_status(status)


cpdef get_scramble_constants32(size_t size):
    """Get scramble constants for 32-bit scrambled Sobol' .

    Args:
        size(size_t): The number of dimensions.

    Returns:
        numpy.ndarray: a array of shape ``(size,)``.

    .. seealso:: `curandGetScrambleConstants32`
    """
    if size > 20000:
        raise ValueError("size cannot exceed 20,000 dimensions")
    cdef unsigned int* constants
    with nogil:
        status = curandGetScrambleConstants32(&constants)
    check_status(status)
    cdef object buf = _memoryview.PyMemoryView_FromMemory(
        <char*><intptr_t>constants, size * sizeof(unsigned int), cpython.PyBUF_READ)
    return _numpy.ndarray((size,), buffer=buf, dtype=_numpy.uint32)


cpdef get_scramble_constants64(size_t size):
    """Get scramble constants for 64-bit scrambled Sobol' .

    Args:
        size(size_t): The number of dimensions.

    Returns:
        numpy.ndarray: a array of shape ``(size,)``.

    .. seealso:: `curandGetScrambleConstants64`
    """
    if size > 20000:
        raise ValueError("size cannot exceed 20,000 dimensions")
    cdef unsigned long long* constants
    with nogil:
        status = curandGetScrambleConstants64(&constants)
    check_status(status)
    cdef object buf = _memoryview.PyMemoryView_FromMemory(
        <char*><intptr_t>constants, size * sizeof(unsigned long long), cpython.PyBUF_READ)
    return _numpy.ndarray((size,), buffer=buf, dtype=_numpy.uint64)


cpdef get_direction_vectors32(int set_, size_t size):
    """Get direction vectors for 32-bit quasirandom number generation.

    Args:
        set_(DirectionVectorSet): Which set of direction vectors to use.
        size(size_t): The number of dimensions.

    Returns:
        numpy.ndarray: a array of shape ``(size, 32)``.

    .. seealso:: `curandGetDirectionVectors32`
    """
    if size > 20000:
        raise ValueError("size cannot exceed 20,000 dimensions")
    cdef curandDirectionVectors32_t* vec
    with nogil:
        status = curandGetDirectionVectors32(&vec, <_DirectionVectorSet>set_)
    check_status(status)
    cdef object buf = _memoryview.PyMemoryView_FromMemory(
        <char*><intptr_t>vec, size * sizeof(curandDirectionVectors32_t), cpython.PyBUF_READ)
    return _numpy.ndarray((size, 32,), buffer=buf, dtype=_numpy.uint32)


cpdef get_direction_vectors64(int set_, size_t size):
    """Get direction vectors for 64-bit quasirandom number generation.

    Args:
        set_(DirectionVectorSet): Which set of direction vectors to use.
        size(size_t): The number of dimensions.

    Returns:
        numpy.ndarray: a array of shape ``(size, 64)``.

    .. seealso:: `curandGetDirectionVectors64`
    """
    if size > 20000:
        raise ValueError("size cannot exceed 20,000 dimensions")
    cdef curandDirectionVectors64_t* vec
    with nogil:
        status = curandGetDirectionVectors64(&vec, <_DirectionVectorSet>set_)
    check_status(status)
    cdef object buf = _memoryview.PyMemoryView_FromMemory(
        <char*><intptr_t>vec, size * sizeof(curandDirectionVectors64_t), cpython.PyBUF_READ)
    return _numpy.ndarray((size, 64,), buffer=buf, dtype=_numpy.uint64)
