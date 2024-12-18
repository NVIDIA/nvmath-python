# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.3.0. Do not modify it directly.

cimport cython  # NOQA
from libc.stdint cimport int64_t
from libcpp.vector cimport vector

from .._internal.utils cimport get_resource_ptr, nullable_unique_ptr

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################




###############################################################################
# Types
###############################################################################

cdef class Plan:

    def __cinit__(Plan self, intptr_t forward_handle, intptr_t inverse_handle, int precision, int kind):
        self.forward_handle = forward_handle
        self.inverse_handle = inverse_handle
        self.precision = precision
        self.kind = kind

    def __repr__(Plan self):
        return (
            f"Plan({self.forward_handle}, {self.inverse_handle}, "
            f"{self.precision}, {self.kind})"
        )

    # In principle, it does not make much sense to be picklable, because
    # passing the plan to another process won't work as the handles will be,
    # most likely, invalid. On the other hand, we don't make such
    # restrictions when the handle is a plain integer
    def __reduce__(Plan self):
        return (type(self), (self.forward_handle, self.inverse_handle, self.precision, self.kind))


###############################################################################
# Error handling
###############################################################################

class FFTWError(Exception):
    pass


class FFTWUnaligned(FFTWError):
    pass


@cython.profile(False)
cdef inline check_plan(intptr_t plan):
    if plan == 0:
        raise FFTWError("Planning failed")


@cython.profile(False)
cdef inline check_nthreads(int nthreads):
    if nthreads <= 0:
        raise FFTWError(
            f"The number of threads available for the plan execution "
            f"was reported to be {nthreads}, expected a positive integer."
        )


@cython.profile(False)
cdef inline check_init_threads(intptr_t nthreads):
    if nthreads == 0:
        raise FFTWError(f"Initialization of FFT threading failed")


@cython.profile(False)
cdef inline intptr_t get_ptr_alignment(intptr_t ptr):
    return ptr & (~(ptr - 1))


@cython.profile(False)
cdef inline check_alignment(intptr_t in_ptr, intptr_t out_ptr, int alignment):
    if in_ptr != 0 and get_ptr_alignment(in_ptr) < alignment:
        raise FFTWUnaligned(
            f"The input tensor's underlying memory pointer must be "
            f"aligned to at least {alignment} bytes. "
            f"The address {in_ptr} is not aligned enough."
        )

    if out_ptr != 0 and get_ptr_alignment(out_ptr) < alignment:
        raise FFTWUnaligned(
            f"The output tensor's underlying memory pointer must be "
            f"aligned to at least {alignment} bytes. "
            f"The address {out_ptr} is not aligned enough."
        )


###############################################################################
# Convenience wrappers/adapters
###############################################################################


cdef inline intptr_t plan_many_c2c(int precision, int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, int sign, unsigned flags) except? 0:
    if precision == Precision.FLOAT:
        return plan_many_c2c_float(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, sign, flags)
    elif precision == Precision.DOUBLE:
        return plan_many_c2c_double(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, sign, flags)
    else:
        raise FFTWError("Unsupported precision")


cdef inline intptr_t plan_many_r2c(int precision, int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags) except? 0:
    if precision == Precision.FLOAT:
        return plan_many_r2c_float(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)
    elif precision == Precision.DOUBLE:
        return plan_many_r2c_double(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)
    else:
        raise FFTWError("Unsupported precision")


cdef inline intptr_t plan_many_c2r(int precision, int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags) except? 0:
    if precision == Precision.FLOAT:
        return plan_many_c2r_float(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)
    elif precision == Precision.DOUBLE:
        return plan_many_c2r_double(rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)
    else:
        raise FFTWError("Unsupported precision")


cdef inline int execute_c2c(int precision, intptr_t plan, intptr_t idata, intptr_t odata) except -1:
    if plan == 0:
        raise FFTWError("Invalid plan handle")
    if precision == Precision.FLOAT:
        check_alignment(idata, odata, 8)
        execute_c2c_float(plan, idata, odata)
    elif precision == Precision.DOUBLE:
        check_alignment(idata, odata, 16)
        execute_c2c_double(plan, idata, odata)
    else:
        raise FFTWError("Unsupported precision")
    return 0


cdef inline int execute_c2r(int precision, intptr_t plan, intptr_t idata, intptr_t odata) except -1:
    if plan == 0:
        raise FFTWError("Invalid plan handle")
    if precision == Precision.FLOAT:
        check_alignment(idata, odata, 8)
        execute_c2r_float(plan, idata, odata)
    elif precision == Precision.DOUBLE:
        check_alignment(idata, odata, 16)
        execute_c2r_double(plan, idata, odata)
    else:
        raise FFTWError("Unsupported precision")
    return 0


cdef inline int execute_r2c(int precision, intptr_t plan, intptr_t idata, intptr_t odata) except -1:
    if plan == 0:
        raise FFTWError("Invalid plan handle")
    if precision == Precision.FLOAT:
        check_alignment(idata, odata, 8)
        execute_r2c_float(plan, idata, odata)
    elif precision == Precision.DOUBLE:
        check_alignment(idata, odata, 16)
        execute_r2c_double(plan, idata, odata)
    else:
        raise FFTWError("Unsupported precision")
    return 0


cdef inline int destroy_plan(int precision, intptr_t plan) except -1:
    if precision == Precision.FLOAT:
        destroy_plan_float(plan)
    elif precision == Precision.DOUBLE:
        destroy_plan_double(plan)
    else:
        raise FFTWError("Unsupported precision")
    return 0


##### Python accessible wrappers #######

cpdef int plan_with_nthreads(int precision, int num_threads) except -1:
    if precision == Precision.FLOAT:
        plan_with_nthreads_float(num_threads)
    elif precision == Precision.DOUBLE:
        plan_with_nthreads_double(num_threads)
    else:
        raise FFTWError("Unsupported precision")
    return 0


cpdef int planner_nthreads(int precision) except? -1:
    if precision == Precision.FLOAT:
        return planner_nthreads_float()
    elif precision == Precision.DOUBLE:
        return planner_nthreads_double()
    else:
        raise FFTWError("Unsupported precision")


cpdef int init_threads(int precision) except -1:
    if precision == Precision.FLOAT:
        init_threads_float()
    elif precision == Precision.DOUBLE:
        init_threads_double()
    else:
        raise FFTWError("Unsupported precision")
    return 0


cpdef int cleanup_threads(int precision) except -1:
    if precision == Precision.FLOAT:
        cleanup_threads_float()
    elif precision == Precision.DOUBLE:
        cleanup_threads_double()
    else:
        raise FFTWError("Unsupported precision")
    return 0


cpdef Plan plan_many(int precision, int kind, int sign, int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags):
    cdef Plan plan = Plan(0, 0, precision, kind)
    if kind == Kind.C2C:
        if sign == Sign.UNSPECIFIED:
            plan.forward_handle = plan_many_c2c(precision, rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, Sign.FORWARD, flags)
            try:
                plan.inverse_handle = plan_many_c2c(precision, rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, Sign.INVERSE, flags)
            except:
                destroy_plan(precision, plan.forward_handle)
                plan.forward_handle = 0
                raise
        elif sign == Sign.FORWARD:
            plan.forward_handle = plan_many_c2c(precision, rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, Sign.FORWARD, flags)
        elif sign == Sign.INVERSE:
            plan.inverse_handle = plan_many_c2c(precision, rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, Sign.INVERSE, flags)
        else:
            raise FFTWError("Incorrect sign/direction specified")
    elif kind == Kind.R2C:
        if sign == Sign.UNSPECIFIED or sign == Sign.FORWARD:
            plan.forward_handle = plan_many_r2c(precision, rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)
        else:
            raise FFTWError("The R2C sign must be unspecified or forward")
    elif kind == Kind.C2R:
        if sign == Sign.UNSPECIFIED or sign == Sign.INVERSE:
            plan.inverse_handle = plan_many_c2r(precision, rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)
        else:
            raise FFTWError("The C2R sign must be unspecified or inverse/backward")
    else:
        raise FFTWError("Unsupported FFT type")
    return plan


cpdef int execute(Plan plan, intptr_t idata, intptr_t odata, int sign) except -1:
    cdef int kind = plan.kind
    if kind == Kind.C2C:
        if sign == Sign.FORWARD:
            return execute_c2c(plan.precision, plan.forward_handle, idata, odata)
        elif sign == Sign.INVERSE:
            return execute_c2c(plan.precision, plan.inverse_handle, idata, odata)
        else:
            raise FFTWError("The FFT sign/direction must be specified: either forward or inverse")
    elif kind == Kind.R2C:
        if sign == Sign.FORWARD or sign == Sign.UNSPECIFIED:
            return execute_r2c(plan.precision, plan.forward_handle, idata, odata)
        else:
            raise FFTWError("The R2C sign must be unspecified or forward")
    elif kind == Kind.C2R:
        if sign == Sign.INVERSE or sign == Sign.UNSPECIFIED:
            return execute_c2r(plan.precision, plan.inverse_handle, idata, odata)
        else:
            raise FFTWError("The C2R sign must be unspecified or invserse/backward")
    else:
        raise FFTWError("Unsupported FFT type")


cpdef int destroy(Plan plan) except -1:
    if plan.inverse_handle != 0:
        destroy_plan(plan.precision, plan.inverse_handle)
        plan.inverse_handle = 0
    if plan.forward_handle != 0:
        destroy_plan(plan.precision, plan.forward_handle)
        plan.forward_handle = 0
    return 0


###############################################################################
# Wrapper functions
###############################################################################

cpdef int get_version() except? -1:
    """See `nvpl_fft_get_version`."""
    return nvpl_fft_get_version()


cpdef intptr_t plan_many_c2c_double(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, int sign, unsigned flags) except? 0:
    """See `fftw_plan_many_dft`."""
    cdef nullable_unique_ptr[ vector[int] ] _n_
    get_resource_ptr[int](_n_, n, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _inembed_
    get_resource_ptr[int](_inembed_, inembed, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _onembed_
    get_resource_ptr[int](_onembed_, onembed, <int*>NULL)
    cdef intptr_t ret
    with nogil:
        ret = <intptr_t>fftw_plan_many_dft(rank, <const int*>(_n_.data()), batch, <fftw_complex*>in_, <const int*>(_inembed_.data()), istride, idist, <fftw_complex*>out, <const int*>(_onembed_.data()), ostride, odist, sign, flags)
    check_plan(<intptr_t>ret)
    return ret


cpdef intptr_t plan_many_r2c_double(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags) except? 0:
    """See `fftw_plan_many_dft_r2c`."""
    cdef nullable_unique_ptr[ vector[int] ] _n_
    get_resource_ptr[int](_n_, n, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _inembed_
    get_resource_ptr[int](_inembed_, inembed, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _onembed_
    get_resource_ptr[int](_onembed_, onembed, <int*>NULL)
    cdef intptr_t ret
    with nogil:
        ret = <intptr_t>fftw_plan_many_dft_r2c(rank, <const int*>(_n_.data()), batch, <double*>in_, <const int*>(_inembed_.data()), istride, idist, <fftw_complex*>out, <const int*>(_onembed_.data()), ostride, odist, flags)
    check_plan(<intptr_t>ret)
    return ret


cpdef intptr_t plan_many_c2r_double(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags) except? 0:
    """See `fftw_plan_many_dft_c2r`."""
    cdef nullable_unique_ptr[ vector[int] ] _n_
    get_resource_ptr[int](_n_, n, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _inembed_
    get_resource_ptr[int](_inembed_, inembed, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _onembed_
    get_resource_ptr[int](_onembed_, onembed, <int*>NULL)
    cdef intptr_t ret
    with nogil:
        ret = <intptr_t>fftw_plan_many_dft_c2r(rank, <const int*>(_n_.data()), batch, <fftw_complex*>in_, <const int*>(_inembed_.data()), istride, idist, <double*>out, <const int*>(_onembed_.data()), ostride, odist, flags)
    check_plan(<intptr_t>ret)
    return ret


cpdef void execute_c2c_double(intptr_t plan, intptr_t idata, intptr_t odata) except*:
    """See `fftw_execute_dft`."""
    fftw_execute_dft(<const fftw_plan>plan, <fftw_complex*>idata, <fftw_complex*>odata)


cpdef void execute_r2c_double(intptr_t plan, intptr_t idata, intptr_t odata) except*:
    """See `fftw_execute_dft_r2c`."""
    fftw_execute_dft_r2c(<const fftw_plan>plan, <double*>idata, <fftw_complex*>odata)


cpdef void execute_c2r_double(intptr_t plan, intptr_t idata, intptr_t odata) except*:
    """See `fftw_execute_dft_c2r`."""
    fftw_execute_dft_c2r(<const fftw_plan>plan, <fftw_complex*>idata, <double*>odata)


cpdef intptr_t plan_many_c2c_float(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, int sign, unsigned flags) except? 0:
    """See `fftwf_plan_many_dft`."""
    cdef nullable_unique_ptr[ vector[int] ] _n_
    get_resource_ptr[int](_n_, n, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _inembed_
    get_resource_ptr[int](_inembed_, inembed, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _onembed_
    get_resource_ptr[int](_onembed_, onembed, <int*>NULL)
    cdef intptr_t ret
    with nogil:
        ret = <intptr_t>fftwf_plan_many_dft(rank, <const int*>(_n_.data()), batch, <fftwf_complex*>in_, <const int*>(_inembed_.data()), istride, idist, <fftwf_complex*>out, <const int*>(_onembed_.data()), ostride, odist, sign, flags)
    check_plan(<intptr_t>ret)
    return ret


cpdef intptr_t plan_many_r2c_float(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags) except? 0:
    """See `fftwf_plan_many_dft_r2c`."""
    cdef nullable_unique_ptr[ vector[int] ] _n_
    get_resource_ptr[int](_n_, n, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _inembed_
    get_resource_ptr[int](_inembed_, inembed, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _onembed_
    get_resource_ptr[int](_onembed_, onembed, <int*>NULL)
    cdef intptr_t ret
    with nogil:
        ret = <intptr_t>fftwf_plan_many_dft_r2c(rank, <const int*>(_n_.data()), batch, <float*>in_, <const int*>(_inembed_.data()), istride, idist, <fftwf_complex*>out, <const int*>(_onembed_.data()), ostride, odist, flags)
    check_plan(<intptr_t>ret)
    return ret


cpdef intptr_t plan_many_c2r_float(int rank, n, int batch, intptr_t in_, inembed, int istride, int idist, intptr_t out, onembed, int ostride, int odist, unsigned flags) except? 0:
    """See `fftwf_plan_many_dft_c2r`."""
    cdef nullable_unique_ptr[ vector[int] ] _n_
    get_resource_ptr[int](_n_, n, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _inembed_
    get_resource_ptr[int](_inembed_, inembed, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _onembed_
    get_resource_ptr[int](_onembed_, onembed, <int*>NULL)
    cdef intptr_t ret
    with nogil:
        ret = <intptr_t>fftwf_plan_many_dft_c2r(rank, <const int*>(_n_.data()), batch, <fftwf_complex*>in_, <const int*>(_inembed_.data()), istride, idist, <float*>out, <const int*>(_onembed_.data()), ostride, odist, flags)
    check_plan(<intptr_t>ret)
    return ret


cpdef void execute_c2c_float(intptr_t plan, intptr_t idata, intptr_t odata) except*:
    """See `fftwf_execute_dft`."""
    fftwf_execute_dft(<const fftwf_plan>plan, <fftwf_complex*>idata, <fftwf_complex*>odata)


cpdef void execute_r2c_float(intptr_t plan, intptr_t idata, intptr_t odata) except*:
    """See `fftwf_execute_dft_r2c`."""
    fftwf_execute_dft_r2c(<const fftwf_plan>plan, <float*>idata, <fftwf_complex*>odata)


cpdef void execute_c2r_float(intptr_t plan, intptr_t idata, intptr_t odata) except*:
    """See `fftwf_execute_dft_c2r`."""
    fftwf_execute_dft_c2r(<const fftwf_plan>plan, <fftwf_complex*>idata, <float*>odata)


cpdef int init_threads_double() except 0:
    """See `fftw_init_threads`."""
    cdef intptr_t ret
    with nogil:
        ret = fftw_init_threads()
    check_init_threads(ret)
    return ret


cpdef int init_threads_float() except 0:
    """See `fftwf_init_threads`."""
    cdef intptr_t ret
    with nogil:
        ret = fftwf_init_threads()
    check_init_threads(ret)
    return ret


cpdef void plan_with_nthreads_double(int nthreads) except*:
    """See `fftw_plan_with_nthreads`."""
    fftw_plan_with_nthreads(nthreads)


cpdef void plan_with_nthreads_float(int nthreads) except*:
    """See `fftwf_plan_with_nthreads`."""
    fftwf_plan_with_nthreads(nthreads)


cpdef int planner_nthreads_double() except? 0:
    """See `fftw_planner_nthreads`."""
    cdef intptr_t ret
    with nogil:
        ret = fftw_planner_nthreads()
    check_nthreads(ret)
    return ret


cpdef int planner_nthreads_float() except? 0:
    """See `fftwf_planner_nthreads`."""
    cdef intptr_t ret
    with nogil:
        ret = fftwf_planner_nthreads()
    check_nthreads(ret)
    return ret


cpdef void cleanup_threads_double() except*:
    """See `fftw_cleanup_threads`."""
    fftw_cleanup_threads()


cpdef void cleanup_threads_float() except*:
    """See `fftwf_cleanup_threads`."""
    fftwf_cleanup_threads()


cpdef void destroy_plan_double(intptr_t plan) except*:
    """See `fftw_destroy_plan`."""
    fftw_destroy_plan(<fftw_plan>plan)


cpdef void destroy_plan_float(intptr_t plan) except*:
    """See `fftwf_destroy_plan`."""
    fftwf_destroy_plan(<fftwf_plan>plan)
