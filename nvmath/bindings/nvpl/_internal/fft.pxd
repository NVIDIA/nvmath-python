# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.3.0. Do not modify it directly.

from ..cyfft cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef int _nvpl_fft_get_version() except?-42 nogil
cdef fftw_plan _fftw_plan_many_dft(int rank, const int* n, int batch, fftw_complex* in_, const int* inembed, int istride, int idist, fftw_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags) except?NULL nogil
cdef fftw_plan _fftw_plan_many_dft_r2c(int rank, const int* n, int batch, double* in_, const int* inembed, int istride, int idist, fftw_complex* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil
cdef fftw_plan _fftw_plan_many_dft_c2r(int rank, const int* n, int batch, fftw_complex* in_, const int* inembed, int istride, int idist, double* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil
cdef void _fftw_execute_dft(const fftw_plan plan, fftw_complex* idata, fftw_complex* odata) except* nogil
cdef void _fftw_execute_dft_r2c(const fftw_plan plan, double* idata, fftw_complex* odata) except* nogil
cdef void _fftw_execute_dft_c2r(const fftw_plan plan, fftw_complex* idata, double* odata) except* nogil
cdef fftwf_plan _fftwf_plan_many_dft(int rank, const int* n, int batch, fftwf_complex* in_, const int* inembed, int istride, int idist, fftwf_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags) except?NULL nogil
cdef fftwf_plan _fftwf_plan_many_dft_r2c(int rank, const int* n, int batch, float* in_, const int* inembed, int istride, int idist, fftwf_complex* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil
cdef fftwf_plan _fftwf_plan_many_dft_c2r(int rank, const int* n, int batch, fftwf_complex* in_, const int* inembed, int istride, int idist, float* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil
cdef void _fftwf_execute_dft(const fftwf_plan plan, fftwf_complex* idata, fftwf_complex* odata) except* nogil
cdef void _fftwf_execute_dft_r2c(const fftwf_plan plan, float* idata, fftwf_complex* odata) except* nogil
cdef void _fftwf_execute_dft_c2r(const fftwf_plan plan, fftwf_complex* idata, float* odata) except* nogil
cdef int _fftw_init_threads() except?-42 nogil
cdef int _fftwf_init_threads() except?-42 nogil
cdef void _fftw_plan_with_nthreads(int nthreads) except* nogil
cdef void _fftwf_plan_with_nthreads(int nthreads) except* nogil
cdef int _fftw_planner_nthreads() except?-42 nogil
cdef int _fftwf_planner_nthreads() except?-42 nogil
cdef void _fftw_cleanup_threads() except* nogil
cdef void _fftwf_cleanup_threads() except* nogil
cdef void _fftw_destroy_plan(fftw_plan plan) except* nogil
cdef void _fftwf_destroy_plan(fftwf_plan plan) except* nogil
