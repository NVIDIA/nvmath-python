# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.3.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums



# types
cdef extern from *:
    """
    // Transform direction
    #define FFTW_FORWARD -1
    #define FFTW_INVERSE  1
    #define FFTW_BACKWARD 1

    // Planner flags
    #define FFTW_ESTIMATE           0x01
    #define FFTW_MEASURE            0x02
    #define FFTW_PATIENT            0x03
    #define FFTW_EXHAUSTIVE         0x04
    #define FFTW_WISDOM_ONLY        0x05

    // Algorithm restriction flags
    #define FFTW_DESTROY_INPUT      0x08
    #define FFTW_PRESERVE_INPUT     0x0C
    #define FFTW_UNALIGNED          0x10

    typedef double fftw_complex[2] __attribute__ ((aligned (16)));
    typedef float fftwf_complex[2] __attribute__ ((aligned (8)));
    """

    cdef const int FFTW_FORWARD
    cdef const int FFTW_INVERSE

    cdef const int FFTW_ESTIMATE
    cdef const int FFTW_MEASURE
    cdef const int FFTW_PATIENT
    cdef const int FFTW_EXHAUSTIVE
    cdef const int FFTW_WISDOM_ONLY

    cdef const int FFTW_DESTROY_INPUT
    cdef const int FFTW_PRESERVE_INPUT
    cdef const int FFTW_UNALIGNED

    ctypedef double fftw_complex[2]
    ctypedef float fftwf_complex[2]


ctypedef void* fftw_plan 'fftw_plan'
ctypedef void* fftwf_plan 'fftwf_plan'


###############################################################################
# Functions
###############################################################################

cdef int nvpl_fft_get_version() except?-42 nogil
cdef fftw_plan fftw_plan_many_dft(int rank, const int* n, int batch, fftw_complex* in_, const int* inembed, int istride, int idist, fftw_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags) except?NULL nogil
cdef fftw_plan fftw_plan_many_dft_r2c(int rank, const int* n, int batch, double* in_, const int* inembed, int istride, int idist, fftw_complex* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil
cdef fftw_plan fftw_plan_many_dft_c2r(int rank, const int* n, int batch, fftw_complex* in_, const int* inembed, int istride, int idist, double* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil
cdef void fftw_execute_dft(const fftw_plan plan, fftw_complex* idata, fftw_complex* odata) except* nogil
cdef void fftw_execute_dft_r2c(const fftw_plan plan, double* idata, fftw_complex* odata) except* nogil
cdef void fftw_execute_dft_c2r(const fftw_plan plan, fftw_complex* idata, double* odata) except* nogil
cdef fftwf_plan fftwf_plan_many_dft(int rank, const int* n, int batch, fftwf_complex* in_, const int* inembed, int istride, int idist, fftwf_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags) except?NULL nogil
cdef fftwf_plan fftwf_plan_many_dft_r2c(int rank, const int* n, int batch, float* in_, const int* inembed, int istride, int idist, fftwf_complex* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil
cdef fftwf_plan fftwf_plan_many_dft_c2r(int rank, const int* n, int batch, fftwf_complex* in_, const int* inembed, int istride, int idist, float* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil
cdef void fftwf_execute_dft(const fftwf_plan plan, fftwf_complex* idata, fftwf_complex* odata) except* nogil
cdef void fftwf_execute_dft_r2c(const fftwf_plan plan, float* idata, fftwf_complex* odata) except* nogil
cdef void fftwf_execute_dft_c2r(const fftwf_plan plan, fftwf_complex* idata, float* odata) except* nogil
cdef int fftw_init_threads() except?-42 nogil
cdef int fftwf_init_threads() except?-42 nogil
cdef void fftw_plan_with_nthreads(int nthreads) except* nogil
cdef void fftwf_plan_with_nthreads(int nthreads) except* nogil
cdef int fftw_planner_nthreads() except?-42 nogil
cdef int fftwf_planner_nthreads() except?-42 nogil
cdef void fftw_cleanup_threads() except* nogil
cdef void fftwf_cleanup_threads() except* nogil
cdef void fftw_destroy_plan(fftw_plan plan) except* nogil
cdef void fftwf_destroy_plan(fftwf_plan plan) except* nogil
