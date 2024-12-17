# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

cimport cython  # NOQA
from libc.stdint cimport int64_t
from libcpp.vector cimport vector

from ._internal.utils cimport (get_resource_ptr, get_resource_ptrs, nullable_unique_ptr,
                               get_buffer_pointer,)

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################

class LibFormat(_IntEnum):
    """See `libFormat_t`."""
    CUFFT = LIB_FORMAT_CUFFT
    UNDEFINED = LIB_FORMAT_UNDEFINED

class Result(_IntEnum):
    """See `cufftResult`."""
    SUCCESS = CUFFT_SUCCESS
    INVALID_PLAN = CUFFT_INVALID_PLAN
    ALLOC_FAILED = CUFFT_ALLOC_FAILED
    INVALID_TYPE = CUFFT_INVALID_TYPE
    INVALID_VALUE = CUFFT_INVALID_VALUE
    INTERNAL_ERROR = CUFFT_INTERNAL_ERROR
    EXEC_FAILED = CUFFT_EXEC_FAILED
    SETUP_FAILED = CUFFT_SETUP_FAILED
    INVALID_SIZE = CUFFT_INVALID_SIZE
    UNALIGNED_DATA = CUFFT_UNALIGNED_DATA
    INCOMPLETE_PARAMETER_LIST = CUFFT_INCOMPLETE_PARAMETER_LIST
    INVALID_DEVICE = CUFFT_INVALID_DEVICE
    PARSE_ERROR = CUFFT_PARSE_ERROR
    NO_WORKSPACE = CUFFT_NO_WORKSPACE
    NOT_IMPLEMENTED = CUFFT_NOT_IMPLEMENTED
    LICENSE_ERROR = CUFFT_LICENSE_ERROR
    NOT_SUPPORTED = CUFFT_NOT_SUPPORTED

class Type(_IntEnum):
    """See `cufftType`."""
    R2C = CUFFT_R2C
    C2R = CUFFT_C2R
    C2C = CUFFT_C2C
    D2Z = CUFFT_D2Z
    Z2D = CUFFT_Z2D
    Z2Z = CUFFT_Z2Z

class Compatibility(_IntEnum):
    """See `cufftCompatibility`."""
    FFTW_PADDING = CUFFT_COMPATIBILITY_FFTW_PADDING

class XtSubFormat(_IntEnum):
    """See `cufftXtSubFormat`."""
    FORMAT_INPUT = CUFFT_XT_FORMAT_INPUT
    FORMAT_OUTPUT = CUFFT_XT_FORMAT_OUTPUT
    FORMAT_INPLACE = CUFFT_XT_FORMAT_INPLACE
    FORMAT_INPLACE_SHUFFLED = CUFFT_XT_FORMAT_INPLACE_SHUFFLED
    FORMAT_1D_INPUT_SHUFFLED = CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED
    FORMAT_DISTRIBUTED_INPUT = CUFFT_XT_FORMAT_DISTRIBUTED_INPUT
    FORMAT_DISTRIBUTED_OUTPUT = CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT
    FORMAT_FORMAT_UNDEFINED = CUFFT_FORMAT_UNDEFINED

class XtCopyType(_IntEnum):
    """See `cufftXtCopyType`."""
    HOST_TO_DEVICE = CUFFT_COPY_HOST_TO_DEVICE
    DEVICE_TO_HOST = CUFFT_COPY_DEVICE_TO_HOST
    DEVICE_TO_DEVICE = CUFFT_COPY_DEVICE_TO_DEVICE
    UNDEFINED = CUFFT_COPY_UNDEFINED

class XtQueryType(_IntEnum):
    """See `cufftXtQueryType`."""
    QUERY_1D_FACTORS = CUFFT_QUERY_1D_FACTORS
    QUERY_UNDEFINED = CUFFT_QUERY_UNDEFINED

class XtWorkAreaPolicy(_IntEnum):
    """See `cufftXtWorkAreaPolicy`."""
    MINIMAL = CUFFT_WORKAREA_MINIMAL
    USER = CUFFT_WORKAREA_USER
    PERFORMANCE = CUFFT_WORKAREA_PERFORMANCE

class XtCallbackType(_IntEnum):
    """See `cufftXtCallbackType`."""
    LD_COMPLEX = CUFFT_CB_LD_COMPLEX
    LD_COMPLEX_DOUBLE = CUFFT_CB_LD_COMPLEX_DOUBLE
    LD_REAL = CUFFT_CB_LD_REAL
    LD_REAL_DOUBLE = CUFFT_CB_LD_REAL_DOUBLE
    ST_COMPLEX = CUFFT_CB_ST_COMPLEX
    ST_COMPLEX_DOUBLE = CUFFT_CB_ST_COMPLEX_DOUBLE
    ST_REAL = CUFFT_CB_ST_REAL
    ST_REAL_DOUBLE = CUFFT_CB_ST_REAL_DOUBLE
    UNDEFINED = CUFFT_CB_UNDEFINED

class Property(_IntEnum):
    """See `cufftProperty`."""
    PATIENT_JIT = NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT
    MAX_NUM_HOST_THREADS = NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS


###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
    CUFFT_SUCCESS                   : 'CUFFT_SUCCESS',
    CUFFT_INVALID_PLAN              : 'CUFFT_INVALID_PLAN',
    CUFFT_ALLOC_FAILED              : 'CUFFT_ALLOC_FAILED',
    CUFFT_INVALID_TYPE              : 'CUFFT_INVALID_TYPE',
    CUFFT_INVALID_VALUE             : 'CUFFT_INVALID_VALUE',
    CUFFT_INTERNAL_ERROR            : 'CUFFT_INTERNAL_ERROR',
    CUFFT_EXEC_FAILED               : 'CUFFT_EXEC_FAILED',
    CUFFT_SETUP_FAILED              : 'CUFFT_SETUP_FAILED',
    CUFFT_INVALID_SIZE              : 'CUFFT_INVALID_SIZE',
    CUFFT_UNALIGNED_DATA            : 'CUFFT_UNALIGNED_DATA',
    CUFFT_INCOMPLETE_PARAMETER_LIST : 'CUFFT_INCOMPLETE_PARAMETER_LIST',
    CUFFT_INVALID_DEVICE            : 'CUFFT_INVALID_DEVICE',
    CUFFT_PARSE_ERROR               : 'CUFFT_PARSE_ERROR',
    CUFFT_NO_WORKSPACE              : 'CUFFT_NO_WORKSPACE',
    CUFFT_NOT_IMPLEMENTED           : 'CUFFT_NOT_IMPLEMENTED',
    CUFFT_LICENSE_ERROR             : 'CUFFT_LICENSE_ERROR',
    CUFFT_NOT_SUPPORTED             : 'CUFFT_NOT_SUPPORTED',
}


class cuFFTError(Exception):

    def __init__(self, status):
        self.status = status
        cdef str err = STATUS[status]
        super(cuFFTError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuFFTError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef plan1d(intptr_t plan, int nx, int type, int batch):
    """See `cufftPlan1d`."""
    with nogil:
        status = cufftPlan1d(<cufftHandle*>plan, nx, <_Type>type, batch)
    check_status(status)


cpdef plan2d(intptr_t plan, int nx, int ny, int type):
    """See `cufftPlan2d`."""
    with nogil:
        status = cufftPlan2d(<cufftHandle*>plan, nx, ny, <_Type>type)
    check_status(status)


cpdef plan3d(intptr_t plan, int nx, int ny, int nz, int type):
    """See `cufftPlan3d`."""
    with nogil:
        status = cufftPlan3d(<cufftHandle*>plan, nx, ny, nz, <_Type>type)
    check_status(status)


cpdef plan_many(intptr_t plan, int rank, n, inembed, int istride, int idist, onembed, int ostride, int odist, int type, int batch):
    """See `cufftPlanMany`."""
    cdef nullable_unique_ptr[ vector[int] ] _n_
    get_resource_ptr[int](_n_, n, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _inembed_
    get_resource_ptr[int](_inembed_, inembed, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _onembed_
    get_resource_ptr[int](_onembed_, onembed, <int*>NULL)
    with nogil:
        status = cufftPlanMany(<cufftHandle*>plan, rank, <int*>(_n_.data()), <int*>(_inembed_.data()), istride, idist, <int*>(_onembed_.data()), ostride, odist, <_Type>type, batch)
    check_status(status)


cpdef size_t make_plan1d(int plan, int nx, int type, int batch) except? -1:
    """See `cufftMakePlan1d`."""
    cdef size_t work_size
    with nogil:
        status = cufftMakePlan1d(<cufftHandle>plan, nx, <_Type>type, batch, &work_size)
    check_status(status)
    return work_size


cpdef size_t make_plan2d(int plan, int nx, int ny, int type) except? -1:
    """See `cufftMakePlan2d`."""
    cdef size_t work_size
    with nogil:
        status = cufftMakePlan2d(<cufftHandle>plan, nx, ny, <_Type>type, &work_size)
    check_status(status)
    return work_size


cpdef size_t make_plan3d(int plan, int nx, int ny, int nz, int type) except? -1:
    """See `cufftMakePlan3d`."""
    cdef size_t work_size
    with nogil:
        status = cufftMakePlan3d(<cufftHandle>plan, nx, ny, nz, <_Type>type, &work_size)
    check_status(status)
    return work_size


cpdef size_t make_plan_many(int plan, int rank, n, inembed, int istride, int idist, onembed, int ostride, int odist, int type, int batch) except? -1:
    """See `cufftMakePlanMany`."""
    cdef nullable_unique_ptr[ vector[int] ] _n_
    get_resource_ptr[int](_n_, n, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _inembed_
    get_resource_ptr[int](_inembed_, inembed, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _onembed_
    get_resource_ptr[int](_onembed_, onembed, <int*>NULL)
    cdef size_t work_size
    with nogil:
        status = cufftMakePlanMany(<cufftHandle>plan, rank, <int*>(_n_.data()), <int*>(_inembed_.data()), istride, idist, <int*>(_onembed_.data()), ostride, odist, <_Type>type, batch, &work_size)
    check_status(status)
    return work_size


cpdef size_t make_plan_many64(int plan, int rank, n, inembed, long long int istride, long long int idist, onembed, long long int ostride, long long int odist, int type, long long int batch) except? -1:
    """See `cufftMakePlanMany64`."""
    cdef nullable_unique_ptr[ vector[int64_t] ] _n_
    get_resource_ptr[int64_t](_n_, n, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _inembed_
    get_resource_ptr[int64_t](_inembed_, inembed, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _onembed_
    get_resource_ptr[int64_t](_onembed_, onembed, <int64_t*>NULL)
    cdef size_t work_size
    with nogil:
        status = cufftMakePlanMany64(<cufftHandle>plan, rank, <long long int*>(_n_.data()), <long long int*>(_inembed_.data()), istride, idist, <long long int*>(_onembed_.data()), ostride, odist, <_Type>type, batch, &work_size)
    check_status(status)
    return work_size


cpdef size_t get_size_many64(int plan, int rank, n, inembed, long long int istride, long long int idist, onembed, long long int ostride, long long int odist, int type, long long int batch) except? -1:
    """See `cufftGetSizeMany64`."""
    cdef nullable_unique_ptr[ vector[int64_t] ] _n_
    get_resource_ptr[int64_t](_n_, n, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _inembed_
    get_resource_ptr[int64_t](_inembed_, inembed, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _onembed_
    get_resource_ptr[int64_t](_onembed_, onembed, <int64_t*>NULL)
    cdef size_t work_size
    with nogil:
        status = cufftGetSizeMany64(<cufftHandle>plan, rank, <long long int*>(_n_.data()), <long long int*>(_inembed_.data()), istride, idist, <long long int*>(_onembed_.data()), ostride, odist, <_Type>type, batch, &work_size)
    check_status(status)
    return work_size


cpdef size_t estimate1d(int nx, int type, int batch) except? -1:
    """See `cufftEstimate1d`."""
    cdef size_t work_size
    with nogil:
        status = cufftEstimate1d(nx, <_Type>type, batch, &work_size)
    check_status(status)
    return work_size


cpdef size_t estimate2d(int nx, int ny, int type) except? -1:
    """See `cufftEstimate2d`."""
    cdef size_t work_size
    with nogil:
        status = cufftEstimate2d(nx, ny, <_Type>type, &work_size)
    check_status(status)
    return work_size


cpdef size_t estimate3d(int nx, int ny, int nz, int type) except? -1:
    """See `cufftEstimate3d`."""
    cdef size_t work_size
    with nogil:
        status = cufftEstimate3d(nx, ny, nz, <_Type>type, &work_size)
    check_status(status)
    return work_size


cpdef size_t estimate_many(int rank, n, inembed, int istride, int idist, onembed, int ostride, int odist, int type, int batch) except? -1:
    """See `cufftEstimateMany`."""
    cdef nullable_unique_ptr[ vector[int] ] _n_
    get_resource_ptr[int](_n_, n, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _inembed_
    get_resource_ptr[int](_inembed_, inembed, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _onembed_
    get_resource_ptr[int](_onembed_, onembed, <int*>NULL)
    cdef size_t work_size
    with nogil:
        status = cufftEstimateMany(rank, <int*>(_n_.data()), <int*>(_inembed_.data()), istride, idist, <int*>(_onembed_.data()), ostride, odist, <_Type>type, batch, &work_size)
    check_status(status)
    return work_size


cpdef int create() except? -1:
    """See `cufftCreate`."""
    cdef cufftHandle handle
    with nogil:
        status = cufftCreate(&handle)
    check_status(status)
    return <int>handle


cpdef size_t get_size1d(int handle, int nx, int type, int batch) except? -1:
    """See `cufftGetSize1d`."""
    cdef size_t work_size
    with nogil:
        status = cufftGetSize1d(<cufftHandle>handle, nx, <_Type>type, batch, &work_size)
    check_status(status)
    return work_size


cpdef size_t get_size2d(int handle, int nx, int ny, int type) except? -1:
    """See `cufftGetSize2d`."""
    cdef size_t work_size
    with nogil:
        status = cufftGetSize2d(<cufftHandle>handle, nx, ny, <_Type>type, &work_size)
    check_status(status)
    return work_size


cpdef size_t get_size3d(int handle, int nx, int ny, int nz, int type) except? -1:
    """See `cufftGetSize3d`."""
    cdef size_t work_size
    with nogil:
        status = cufftGetSize3d(<cufftHandle>handle, nx, ny, nz, <_Type>type, &work_size)
    check_status(status)
    return work_size


cpdef size_t get_size_many(int handle, int rank, intptr_t n, intptr_t inembed, int istride, int idist, intptr_t onembed, int ostride, int odist, int type, int batch) except? -1:
    """See `cufftGetSizeMany`."""
    cdef size_t work_area
    with nogil:
        status = cufftGetSizeMany(<cufftHandle>handle, rank, <int*>n, <int*>inembed, istride, idist, <int*>onembed, ostride, odist, <_Type>type, batch, &work_area)
    check_status(status)
    return work_area


cpdef size_t get_size(int handle) except? -1:
    """See `cufftGetSize`."""
    cdef size_t work_size
    with nogil:
        status = cufftGetSize(<cufftHandle>handle, &work_size)
    check_status(status)
    return work_size


cpdef set_work_area(int plan, intptr_t work_area):
    """See `cufftSetWorkArea`."""
    with nogil:
        status = cufftSetWorkArea(<cufftHandle>plan, <void*>work_area)
    check_status(status)


cpdef set_auto_allocation(int plan, int auto_allocate):
    """See `cufftSetAutoAllocation`."""
    with nogil:
        status = cufftSetAutoAllocation(<cufftHandle>plan, auto_allocate)
    check_status(status)


cpdef exec_c2c(int plan, intptr_t idata, intptr_t odata, int direction):
    """See `cufftExecC2C`."""
    with nogil:
        status = cufftExecC2C(<cufftHandle>plan, <cufftComplex*>idata, <cufftComplex*>odata, direction)
    check_status(status)


cpdef exec_r2c(int plan, intptr_t idata, intptr_t odata):
    """See `cufftExecR2C`."""
    with nogil:
        status = cufftExecR2C(<cufftHandle>plan, <cufftReal*>idata, <cufftComplex*>odata)
    check_status(status)


cpdef exec_c2r(int plan, intptr_t idata, intptr_t odata):
    """See `cufftExecC2R`."""
    with nogil:
        status = cufftExecC2R(<cufftHandle>plan, <cufftComplex*>idata, <cufftReal*>odata)
    check_status(status)


cpdef exec_z2z(int plan, intptr_t idata, intptr_t odata, int direction):
    """See `cufftExecZ2Z`."""
    with nogil:
        status = cufftExecZ2Z(<cufftHandle>plan, <cufftDoubleComplex*>idata, <cufftDoubleComplex*>odata, direction)
    check_status(status)


cpdef exec_d2z(int plan, intptr_t idata, intptr_t odata):
    """See `cufftExecD2Z`."""
    with nogil:
        status = cufftExecD2Z(<cufftHandle>plan, <cufftDoubleReal*>idata, <cufftDoubleComplex*>odata)
    check_status(status)


cpdef exec_z2d(int plan, intptr_t idata, intptr_t odata):
    """See `cufftExecZ2D`."""
    with nogil:
        status = cufftExecZ2D(<cufftHandle>plan, <cufftDoubleComplex*>idata, <cufftDoubleReal*>odata)
    check_status(status)


cpdef set_stream(int plan, intptr_t stream):
    """See `cufftSetStream`."""
    with nogil:
        status = cufftSetStream(<cufftHandle>plan, <Stream>stream)
    check_status(status)


cpdef destroy(int plan):
    """See `cufftDestroy`."""
    with nogil:
        status = cufftDestroy(<cufftHandle>plan)
    check_status(status)


cpdef int get_version() except? -1:
    """See `cufftGetVersion`."""
    cdef int version
    with nogil:
        status = cufftGetVersion(&version)
    check_status(status)
    return version


cpdef int get_property(int type) except? -1:
    """See `cufftGetProperty`."""
    cdef int value
    with nogil:
        status = cufftGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


cpdef xt_set_gpus(int handle, int n_gpus, which_gpus):
    """See `cufftXtSetGPUs`."""
    cdef nullable_unique_ptr[ vector[int] ] _which_gpus_
    get_resource_ptr[int](_which_gpus_, which_gpus, <int*>NULL)
    with nogil:
        status = cufftXtSetGPUs(<cufftHandle>handle, n_gpus, <int*>(_which_gpus_.data()))
    check_status(status)


cpdef intptr_t xt_malloc(int plan, int format) except? -1:
    """See `cufftXtMalloc`."""
    cdef cudaLibXtDesc* descriptor
    with nogil:
        status = cufftXtMalloc(<cufftHandle>plan, &descriptor, <_XtSubFormat>format)
    check_status(status)
    return <intptr_t>descriptor


cpdef xt_memcpy(int plan, intptr_t dst_pointer, intptr_t src_pointer, int type):
    """See `cufftXtMemcpy`."""
    with nogil:
        status = cufftXtMemcpy(<cufftHandle>plan, <void*>dst_pointer, <void*>src_pointer, <_XtCopyType>type)
    check_status(status)


cpdef xt_free(intptr_t descriptor):
    """See `cufftXtFree`."""
    with nogil:
        status = cufftXtFree(<cudaLibXtDesc*>descriptor)
    check_status(status)


cpdef xt_set_work_area(int plan, intptr_t work_area):
    """See `cufftXtSetWorkArea`."""
    with nogil:
        status = cufftXtSetWorkArea(<cufftHandle>plan, <void**>work_area)
    check_status(status)


cpdef xt_exec_descriptor_c2c(int plan, intptr_t input, intptr_t output, int direction):
    """See `cufftXtExecDescriptorC2C`."""
    with nogil:
        status = cufftXtExecDescriptorC2C(<cufftHandle>plan, <cudaLibXtDesc*>input, <cudaLibXtDesc*>output, direction)
    check_status(status)


cpdef xt_exec_descriptor_r2c(int plan, intptr_t input, intptr_t output):
    """See `cufftXtExecDescriptorR2C`."""
    with nogil:
        status = cufftXtExecDescriptorR2C(<cufftHandle>plan, <cudaLibXtDesc*>input, <cudaLibXtDesc*>output)
    check_status(status)


cpdef xt_exec_descriptor_c2r(int plan, intptr_t input, intptr_t output):
    """See `cufftXtExecDescriptorC2R`."""
    with nogil:
        status = cufftXtExecDescriptorC2R(<cufftHandle>plan, <cudaLibXtDesc*>input, <cudaLibXtDesc*>output)
    check_status(status)


cpdef xt_exec_descriptor_z2z(int plan, intptr_t input, intptr_t output, int direction):
    """See `cufftXtExecDescriptorZ2Z`."""
    with nogil:
        status = cufftXtExecDescriptorZ2Z(<cufftHandle>plan, <cudaLibXtDesc*>input, <cudaLibXtDesc*>output, direction)
    check_status(status)


cpdef xt_exec_descriptor_d2z(int plan, intptr_t input, intptr_t output):
    """See `cufftXtExecDescriptorD2Z`."""
    with nogil:
        status = cufftXtExecDescriptorD2Z(<cufftHandle>plan, <cudaLibXtDesc*>input, <cudaLibXtDesc*>output)
    check_status(status)


cpdef xt_exec_descriptor_z2d(int plan, intptr_t input, intptr_t output):
    """See `cufftXtExecDescriptorZ2D`."""
    with nogil:
        status = cufftXtExecDescriptorZ2D(<cufftHandle>plan, <cudaLibXtDesc*>input, <cudaLibXtDesc*>output)
    check_status(status)


cpdef xt_query_plan(int plan, intptr_t query_struct, int query_type):
    """See `cufftXtQueryPlan`."""
    with nogil:
        status = cufftXtQueryPlan(<cufftHandle>plan, <void*>query_struct, <_XtQueryType>query_type)
    check_status(status)


cpdef xt_clear_callback(int plan, int cb_type):
    """See `cufftXtClearCallback`."""
    with nogil:
        status = cufftXtClearCallback(<cufftHandle>plan, <_XtCallbackType>cb_type)
    check_status(status)


cpdef xt_set_callback_shared_size(int plan, int cb_type, size_t shared_size):
    """See `cufftXtSetCallbackSharedSize`."""
    with nogil:
        status = cufftXtSetCallbackSharedSize(<cufftHandle>plan, <_XtCallbackType>cb_type, shared_size)
    check_status(status)


cpdef size_t xt_make_plan_many(int plan, int rank, n, inembed, long long int istride, long long int idist, int inputtype, onembed, long long int ostride, long long int odist, int outputtype, long long int batch, int executiontype) except? 0:
    """See `cufftXtMakePlanMany`."""
    cdef nullable_unique_ptr[ vector[int64_t] ] _n_
    get_resource_ptr[int64_t](_n_, n, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _inembed_
    get_resource_ptr[int64_t](_inembed_, inembed, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _onembed_
    get_resource_ptr[int64_t](_onembed_, onembed, <int64_t*>NULL)
    cdef size_t work_size
    with nogil:
        status = cufftXtMakePlanMany(<cufftHandle>plan, rank, <long long int*>(_n_.data()), <long long int*>(_inembed_.data()), istride, idist, <DataType>inputtype, <long long int*>(_onembed_.data()), ostride, odist, <DataType>outputtype, batch, &work_size, <DataType>executiontype)
    check_status(status)
    return work_size


cpdef size_t xt_get_size_many(int plan, int rank, n, inembed, long long int istride, long long int idist, int inputtype, onembed, long long int ostride, long long int odist, int outputtype, long long int batch, int executiontype) except? 0:
    """See `cufftXtGetSizeMany`."""
    cdef nullable_unique_ptr[ vector[int64_t] ] _n_
    get_resource_ptr[int64_t](_n_, n, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _inembed_
    get_resource_ptr[int64_t](_inembed_, inembed, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _onembed_
    get_resource_ptr[int64_t](_onembed_, onembed, <int64_t*>NULL)
    cdef size_t work_size
    with nogil:
        status = cufftXtGetSizeMany(<cufftHandle>plan, rank, <long long int*>(_n_.data()), <long long int*>(_inembed_.data()), istride, idist, <DataType>inputtype, <long long int*>(_onembed_.data()), ostride, odist, <DataType>outputtype, batch, &work_size, <DataType>executiontype)
    check_status(status)
    return work_size


cpdef xt_exec(int plan, intptr_t input, intptr_t output, int direction):
    """See `cufftXtExec`."""
    with nogil:
        status = cufftXtExec(<cufftHandle>plan, <void*>input, <void*>output, direction)
    check_status(status)


cpdef xt_exec_descriptor(int plan, intptr_t input, intptr_t output, int direction):
    """See `cufftXtExecDescriptor`."""
    with nogil:
        status = cufftXtExecDescriptor(<cufftHandle>plan, <cudaLibXtDesc*>input, <cudaLibXtDesc*>output, direction)
    check_status(status)


cpdef xt_set_work_area_policy(int plan, int policy, intptr_t work_size):
    """See `cufftXtSetWorkAreaPolicy`."""
    with nogil:
        status = cufftXtSetWorkAreaPolicy(<cufftHandle>plan, <_XtWorkAreaPolicy>policy, <size_t*>work_size)
    check_status(status)


cpdef xt_set_jit_callback(int plan, lto_callback_fatbin, size_t lto_callback_fatbin_size, int type, caller_info):
    """See `cufftXtSetJITCallback`."""
    cdef void* _lto_callback_fatbin_ = get_buffer_pointer(lto_callback_fatbin, lto_callback_fatbin_size, readonly=True)
    cdef nullable_unique_ptr[ vector[void*] ] _caller_info_
    get_resource_ptrs[void](_caller_info_, caller_info, <void*>NULL)
    with nogil:
        status = cufftXtSetJITCallback(<cufftHandle>plan, <const void*>_lto_callback_fatbin_, lto_callback_fatbin_size, <_XtCallbackType>type, <void**>(_caller_info_.data()))
    check_status(status)


cpdef xt_set_subformat_default(int plan, int subformat_forward, int subformat_inverse):
    """See `cufftXtSetSubformatDefault`."""
    with nogil:
        status = cufftXtSetSubformatDefault(<cufftHandle>plan, <_XtSubFormat>subformat_forward, <_XtSubFormat>subformat_inverse)
    check_status(status)


cpdef set_plan_property_int64(int plan, int property, long long int input_value_int):
    """See `cufftSetPlanPropertyInt64`."""
    with nogil:
        status = cufftSetPlanPropertyInt64(<cufftHandle>plan, <_Property>property, <const long long int>input_value_int)
    check_status(status)


cpdef long long int get_plan_property_int64(int plan, int property) except? -1:
    """See `cufftGetPlanPropertyInt64`."""
    cdef long long int return_ptr_value
    with nogil:
        status = cufftGetPlanPropertyInt64(<cufftHandle>plan, <_Property>property, &return_ptr_value)
    check_status(status)
    return return_ptr_value


cpdef reset_plan_property(int plan, int property):
    """See `cufftResetPlanProperty`."""
    with nogil:
        status = cufftResetPlanProperty(<cufftHandle>plan, <_Property>property)
    check_status(status)
