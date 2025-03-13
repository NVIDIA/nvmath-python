# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

cimport cython

from libc.stdint cimport intptr_t

from .cycurand cimport *


###############################################################################
# Types
###############################################################################

ctypedef curandGenerator_t Generator
ctypedef curandDistribution_t Distribution
ctypedef curandDistributionShift_t DistributionShift
ctypedef curandDistributionM2Shift_t DistributionM2Shift
ctypedef curandHistogramM2_t HistogramM2
ctypedef curandHistogramM2K_t HistogramM2K
ctypedef curandHistogramM2V_t HistogramM2V
ctypedef curandDiscreteDistribution_t DiscreteDistribution

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef curandStatus_t _Status
ctypedef curandRngType_t _RngType
ctypedef curandOrdering_t _Ordering
ctypedef curandDirectionVectorSet_t _DirectionVectorSet
ctypedef curandMethod_t _Method


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create_generator(int rng_type) except? 0
cpdef intptr_t create_generator_host(int rng_type) except? 0
cpdef destroy_generator(intptr_t generator)
cpdef int get_version() except? -1
cpdef int get_property(int type) except? -1
cpdef set_stream(intptr_t generator, intptr_t stream)
cpdef set_pseudo_random_generator_seed(intptr_t generator, unsigned long long seed)
cpdef set_generator_offset(intptr_t generator, unsigned long long offset)
cpdef set_generator_ordering(intptr_t generator, int order)
cpdef set_quasi_random_generator_dimensions(intptr_t generator, unsigned int num_dimensions)
cpdef generate(intptr_t generator, intptr_t output_ptr, size_t num)
cpdef generate_long_long(intptr_t generator, intptr_t output_ptr, size_t num)
cpdef generate_uniform(intptr_t generator, intptr_t output_ptr, size_t num)
cpdef generate_uniform_double(intptr_t generator, intptr_t output_ptr, size_t num)
cpdef generate_normal(intptr_t generator, intptr_t output_ptr, size_t n, float mean, float stddev)
cpdef generate_normal_double(intptr_t generator, intptr_t output_ptr, size_t n, double mean, double stddev)
cpdef generate_log_normal(intptr_t generator, intptr_t output_ptr, size_t n, float mean, float stddev)
cpdef generate_log_normal_double(intptr_t generator, intptr_t output_ptr, size_t n, double mean, double stddev)
cpdef create_poisson_distribution(double lambda_, intptr_t discrete_distribution)
cpdef destroy_distribution(intptr_t discrete_distribution)
cpdef generate_poisson(intptr_t generator, intptr_t output_ptr, size_t n, double lambda_)
cpdef generate_poisson_method(intptr_t generator, intptr_t output_ptr, size_t n, double lambda_, int method)
cpdef generate_binomial(intptr_t generator, intptr_t output_ptr, size_t num, unsigned int n, double p)
cpdef generate_binomial_method(intptr_t generator, intptr_t output_ptr, size_t num, unsigned int n, double p, int method)
cpdef generate_seeds(intptr_t generator)
