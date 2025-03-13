# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.3.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cyfft cimport *


###############################################################################
# Types
###############################################################################




cdef class Plan:
    cdef intptr_t forward_handle
    cdef intptr_t inverse_handle
    cdef int precision
    cdef int kind


###############################################################################
# Enum
###############################################################################


# The `UNSPECIFIED` value is not valid sign in NVPL's header,
# but is used here to allow for planning C2C that can handle
# both directions
cpdef enum Sign:
    UNSPECIFIED = 0
    FORWARD = FFTW_FORWARD
    INVERSE = FFTW_INVERSE


cpdef enum PlannerFlags:
    ESTIMATE = FFTW_ESTIMATE
    MEASURE = FFTW_MEASURE
    PATIENT = FFTW_PATIENT
    EXHAUSTIVE = FFTW_EXHAUSTIVE
    WISDOM_ONLY = FFTW_WISDOM_ONLY


cpdef enum Precision:
    FLOAT = 32
    DOUBLE = 64


cpdef enum Kind:
    C2C = 1
    C2R = 2
    R2C = 3





###############################################################################
# Convenience wrappers/adapters
###############################################################################

cpdef int plan_with_nthreads(int precision, int num_threads) except -1
cpdef int planner_nthreads(int precision) except? -1
cpdef int init_threads(int precision) except -1
cpdef int cleanup_threads(int precision) except -1
cpdef Plan plan_many(int precision, int kind, int sign, int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags)
cpdef int destroy(Plan plan) except -1
cpdef int execute(Plan plan, intptr_t idata, intptr_t odata, int sign) except -1

###############################################################################
# Functions
###############################################################################

cpdef int get_version() except? -1
cpdef intptr_t plan_many_c2c_double(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, int sign, unsigned flags) except? 0
cpdef intptr_t plan_many_r2c_double(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags) except? 0
cpdef intptr_t plan_many_c2r_double(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags) except? 0
cpdef void execute_c2c_double(intptr_t plan, intptr_t idata, intptr_t odata) except*
cpdef void execute_r2c_double(intptr_t plan, intptr_t idata, intptr_t odata) except*
cpdef void execute_c2r_double(intptr_t plan, intptr_t idata, intptr_t odata) except*
cpdef intptr_t plan_many_c2c_float(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, int sign, unsigned flags) except? 0
cpdef intptr_t plan_many_r2c_float(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags) except? 0
cpdef intptr_t plan_many_c2r_float(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags) except? 0
cpdef void execute_c2c_float(intptr_t plan, intptr_t idata, intptr_t odata) except*
cpdef void execute_r2c_float(intptr_t plan, intptr_t idata, intptr_t odata) except*
cpdef void execute_c2r_float(intptr_t plan, intptr_t idata, intptr_t odata) except*
cpdef int init_threads_double() except 0
cpdef int init_threads_float() except 0
cpdef void plan_with_nthreads_double(int nthreads) except*
cpdef void plan_with_nthreads_float(int nthreads) except*
cpdef int planner_nthreads_double() except? 0
cpdef int planner_nthreads_float() except? 0
cpdef void cleanup_threads_double() except*
cpdef void cleanup_threads_float() except*
cpdef void destroy_plan_double(intptr_t plan) except*
cpdef void destroy_plan_float(intptr_t plan) except*
