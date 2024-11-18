# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.3.0. Do not modify it directly.

from ._internal cimport fft as _nvpl_fft


###############################################################################
# Wrapper functions
###############################################################################

cdef int nvpl_fft_get_version() except* nogil:
    return _nvpl_fft._nvpl_fft_get_version()


cdef fftw_plan fftw_plan_many_dft(int rank, const int* n, int batch, fftw_complex* in_, const int* inembed, int istride, int idist, fftw_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags) except* nogil:
    return _nvpl_fft._fftw_plan_many_dft(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, sign, flags)


cdef fftw_plan fftw_plan_many_dft_r2c(int rank, const int* n, int batch, double* in_, const int* inembed, int istride, int idist, fftw_complex* out, const int* onembed, int ostride, int odist, unsigned flags) except* nogil:
    return _nvpl_fft._fftw_plan_many_dft_r2c(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


cdef fftw_plan fftw_plan_many_dft_c2r(int rank, const int* n, int batch, fftw_complex* in_, const int* inembed, int istride, int idist, double* out, const int* onembed, int ostride, int odist, unsigned flags) except* nogil:
    return _nvpl_fft._fftw_plan_many_dft_c2r(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


cdef void fftw_execute_dft(const fftw_plan plan, fftw_complex* idata, fftw_complex* odata) except* nogil:
    _nvpl_fft._fftw_execute_dft(plan, idata, odata)


cdef void fftw_execute_dft_r2c(const fftw_plan plan, double* idata, fftw_complex* odata) except* nogil:
    _nvpl_fft._fftw_execute_dft_r2c(plan, idata, odata)


cdef void fftw_execute_dft_c2r(const fftw_plan plan, fftw_complex* idata, double* odata) except* nogil:
    _nvpl_fft._fftw_execute_dft_c2r(plan, idata, odata)


cdef fftwf_plan fftwf_plan_many_dft(int rank, const int* n, int batch, fftwf_complex* in_, const int* inembed, int istride, int idist, fftwf_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags) except* nogil:
    return _nvpl_fft._fftwf_plan_many_dft(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, sign, flags)


cdef fftwf_plan fftwf_plan_many_dft_r2c(int rank, const int* n, int batch, float* in_, const int* inembed, int istride, int idist, fftwf_complex* out, const int* onembed, int ostride, int odist, unsigned flags) except* nogil:
    return _nvpl_fft._fftwf_plan_many_dft_r2c(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


cdef fftwf_plan fftwf_plan_many_dft_c2r(int rank, const int* n, int batch, fftwf_complex* in_, const int* inembed, int istride, int idist, float* out, const int* onembed, int ostride, int odist, unsigned flags) except* nogil:
    return _nvpl_fft._fftwf_plan_many_dft_c2r(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


cdef void fftwf_execute_dft(const fftwf_plan plan, fftwf_complex* idata, fftwf_complex* odata) except* nogil:
    _nvpl_fft._fftwf_execute_dft(plan, idata, odata)


cdef void fftwf_execute_dft_r2c(const fftwf_plan plan, float* idata, fftwf_complex* odata) except* nogil:
    _nvpl_fft._fftwf_execute_dft_r2c(plan, idata, odata)


cdef void fftwf_execute_dft_c2r(const fftwf_plan plan, fftwf_complex* idata, float* odata) except* nogil:
    _nvpl_fft._fftwf_execute_dft_c2r(plan, idata, odata)


cdef int fftw_init_threads() except* nogil:
    return _nvpl_fft._fftw_init_threads()


cdef int fftwf_init_threads() except* nogil:
    return _nvpl_fft._fftwf_init_threads()


cdef void fftw_plan_with_nthreads(int nthreads) except* nogil:
    _nvpl_fft._fftw_plan_with_nthreads(nthreads)


cdef void fftwf_plan_with_nthreads(int nthreads) except* nogil:
    _nvpl_fft._fftwf_plan_with_nthreads(nthreads)


cdef int fftw_planner_nthreads() except* nogil:
    return _nvpl_fft._fftw_planner_nthreads()


cdef int fftwf_planner_nthreads() except* nogil:
    return _nvpl_fft._fftwf_planner_nthreads()


cdef void fftw_cleanup_threads() except* nogil:
    _nvpl_fft._fftw_cleanup_threads()


cdef void fftwf_cleanup_threads() except* nogil:
    _nvpl_fft._fftwf_cleanup_threads()


cdef void fftw_destroy_plan(fftw_plan plan) except* nogil:
    _nvpl_fft._fftw_destroy_plan(plan)


cdef void fftwf_destroy_plan(fftwf_plan plan) except* nogil:
    _nvpl_fft._fftwf_destroy_plan(plan)
