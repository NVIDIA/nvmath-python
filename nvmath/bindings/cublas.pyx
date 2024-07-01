# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.4.1. Do not modify it directly.

cimport cython  # NOQA

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """See `cublasStatus_t`."""
    SUCCESS = CUBLAS_STATUS_SUCCESS
    NOT_INITIALIZED = CUBLAS_STATUS_NOT_INITIALIZED
    ALLOC_FAILED = CUBLAS_STATUS_ALLOC_FAILED
    INVALID_VALUE = CUBLAS_STATUS_INVALID_VALUE
    ARCH_MISMATCH = CUBLAS_STATUS_ARCH_MISMATCH
    MAPPING_ERROR = CUBLAS_STATUS_MAPPING_ERROR
    EXECUTION_FAILED = CUBLAS_STATUS_EXECUTION_FAILED
    INTERNAL_ERROR = CUBLAS_STATUS_INTERNAL_ERROR
    NOT_SUPPORTED = CUBLAS_STATUS_NOT_SUPPORTED
    LICENSE_ERROR = CUBLAS_STATUS_LICENSE_ERROR

class FillMode(_IntEnum):
    """See `cublasFillMode_t`."""
    LOWER = CUBLAS_FILL_MODE_LOWER
    UPPER = CUBLAS_FILL_MODE_UPPER
    FULL = CUBLAS_FILL_MODE_FULL

class DiagType(_IntEnum):
    """See `cublasDiagType_t`."""
    NON_UNIT = CUBLAS_DIAG_NON_UNIT
    UNIT = CUBLAS_DIAG_UNIT

class SideMode(_IntEnum):
    """See `cublasSideMode_t`."""
    LEFT = CUBLAS_SIDE_LEFT
    RIGHT = CUBLAS_SIDE_RIGHT

class Operation(_IntEnum):
    """See `cublasOperation_t`."""
    N = CUBLAS_OP_N
    T = CUBLAS_OP_T
    C = CUBLAS_OP_C
    HERMITAN = CUBLAS_OP_HERMITAN
    CONJG = CUBLAS_OP_CONJG

class PointerMode(_IntEnum):
    """See `cublasPointerMode_t`."""
    HOST = CUBLAS_POINTER_MODE_HOST
    DEVICE = CUBLAS_POINTER_MODE_DEVICE

class AtomicsMode(_IntEnum):
    """See `cublasAtomicsMode_t`."""
    NOT_ALLOWED = CUBLAS_ATOMICS_NOT_ALLOWED
    ALLOWED = CUBLAS_ATOMICS_ALLOWED

class GemmAlgo(_IntEnum):
    """See `cublasGemmAlgo_t`."""
    DFALT = CUBLAS_GEMM_DFALT
    DEFAULT = CUBLAS_GEMM_DEFAULT
    ALGO0 = CUBLAS_GEMM_ALGO0
    ALGO1 = CUBLAS_GEMM_ALGO1
    ALGO2 = CUBLAS_GEMM_ALGO2
    ALGO3 = CUBLAS_GEMM_ALGO3
    ALGO4 = CUBLAS_GEMM_ALGO4
    ALGO5 = CUBLAS_GEMM_ALGO5
    ALGO6 = CUBLAS_GEMM_ALGO6
    ALGO7 = CUBLAS_GEMM_ALGO7
    ALGO8 = CUBLAS_GEMM_ALGO8
    ALGO9 = CUBLAS_GEMM_ALGO9
    ALGO10 = CUBLAS_GEMM_ALGO10
    ALGO11 = CUBLAS_GEMM_ALGO11
    ALGO12 = CUBLAS_GEMM_ALGO12
    ALGO13 = CUBLAS_GEMM_ALGO13
    ALGO14 = CUBLAS_GEMM_ALGO14
    ALGO15 = CUBLAS_GEMM_ALGO15
    ALGO16 = CUBLAS_GEMM_ALGO16
    ALGO17 = CUBLAS_GEMM_ALGO17
    ALGO18 = CUBLAS_GEMM_ALGO18
    ALGO19 = CUBLAS_GEMM_ALGO19
    ALGO20 = CUBLAS_GEMM_ALGO20
    ALGO21 = CUBLAS_GEMM_ALGO21
    ALGO22 = CUBLAS_GEMM_ALGO22
    ALGO23 = CUBLAS_GEMM_ALGO23
    DEFAULT_TENSOR_OP = CUBLAS_GEMM_DEFAULT_TENSOR_OP
    DFALT_TENSOR_OP = CUBLAS_GEMM_DFALT_TENSOR_OP
    ALGO0_TENSOR_OP = CUBLAS_GEMM_ALGO0_TENSOR_OP
    ALGO1_TENSOR_OP = CUBLAS_GEMM_ALGO1_TENSOR_OP
    ALGO2_TENSOR_OP = CUBLAS_GEMM_ALGO2_TENSOR_OP
    ALGO3_TENSOR_OP = CUBLAS_GEMM_ALGO3_TENSOR_OP
    ALGO4_TENSOR_OP = CUBLAS_GEMM_ALGO4_TENSOR_OP
    ALGO5_TENSOR_OP = CUBLAS_GEMM_ALGO5_TENSOR_OP
    ALGO6_TENSOR_OP = CUBLAS_GEMM_ALGO6_TENSOR_OP
    ALGO7_TENSOR_OP = CUBLAS_GEMM_ALGO7_TENSOR_OP
    ALGO8_TENSOR_OP = CUBLAS_GEMM_ALGO8_TENSOR_OP
    ALGO9_TENSOR_OP = CUBLAS_GEMM_ALGO9_TENSOR_OP
    ALGO10_TENSOR_OP = CUBLAS_GEMM_ALGO10_TENSOR_OP
    ALGO11_TENSOR_OP = CUBLAS_GEMM_ALGO11_TENSOR_OP
    ALGO12_TENSOR_OP = CUBLAS_GEMM_ALGO12_TENSOR_OP
    ALGO13_TENSOR_OP = CUBLAS_GEMM_ALGO13_TENSOR_OP
    ALGO14_TENSOR_OP = CUBLAS_GEMM_ALGO14_TENSOR_OP
    ALGO15_TENSOR_OP = CUBLAS_GEMM_ALGO15_TENSOR_OP

class Math(_IntEnum):
    """See `cublasMath_t`."""
    DEFAULT_MATH = CUBLAS_DEFAULT_MATH
    TENSOR_OP_MATH = CUBLAS_TENSOR_OP_MATH
    PEDANTIC_MATH = CUBLAS_PEDANTIC_MATH
    TF32_TENSOR_OP_MATH = CUBLAS_TF32_TENSOR_OP_MATH
    DISALLOW_REDUCED_PRECISION_REDUCTION = CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION

class ComputeType(_IntEnum):
    """See `cublasComputeType_t`."""
    COMPUTE_16F = CUBLAS_COMPUTE_16F
    COMPUTE_16F_PEDANTIC = CUBLAS_COMPUTE_16F_PEDANTIC
    COMPUTE_32F = CUBLAS_COMPUTE_32F
    COMPUTE_32F_PEDANTIC = CUBLAS_COMPUTE_32F_PEDANTIC
    COMPUTE_32F_FAST_16F = CUBLAS_COMPUTE_32F_FAST_16F
    COMPUTE_32F_FAST_16BF = CUBLAS_COMPUTE_32F_FAST_16BF
    COMPUTE_32F_FAST_TF32 = CUBLAS_COMPUTE_32F_FAST_TF32
    COMPUTE_64F = CUBLAS_COMPUTE_64F
    COMPUTE_64F_PEDANTIC = CUBLAS_COMPUTE_64F_PEDANTIC
    COMPUTE_32I = CUBLAS_COMPUTE_32I
    COMPUTE_32I_PEDANTIC = CUBLAS_COMPUTE_32I_PEDANTIC


###############################################################################
# Error handling
###############################################################################

class cuBLASError(Exception):

    def __init__(self, status):
        from ._internal.cublas import _inspect_function_pointer
        self.status = status
        cdef str err
        if (_inspect_function_pointer("__cublasGetStatusName") != 0
                and _inspect_function_pointer("__cublasGetStatusString") != 0):
            err = f"{get_status_string(status)} ({get_status_name(status)})"
        else:
            s = Status(status)
            err = f"{s.name} ({s.value})"
        super(cuBLASError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuBLASError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create() except? 0:
    """See `cublasCreate`."""
    cdef Handle handle
    with nogil:
        status = cublasCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """See `cublasDestroy`."""
    with nogil:
        status = cublasDestroy(<Handle>handle)
    check_status(status)


cpdef int get_version(intptr_t handle) except? -1:
    """See `cublasGetVersion`."""
    cdef int version
    with nogil:
        status = cublasGetVersion(<Handle>handle, &version)
    check_status(status)
    return version


cpdef int get_property(int type) except? -1:
    """See `cublasGetProperty`."""
    cdef int value
    with nogil:
        status = cublasGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


cpdef size_t get_cudart_version():
    """See `cublasGetCudartVersion`."""
    return cublasGetCudartVersion()


cpdef set_workspace(intptr_t handle, intptr_t workspace, size_t workspace_size_in_bytes):
    """See `cublasSetWorkspace`."""
    with nogil:
        status = cublasSetWorkspace(<Handle>handle, <void*>workspace, workspace_size_in_bytes)
    check_status(status)


cpdef set_stream(intptr_t handle, intptr_t stream_id):
    """See `cublasSetStream`."""
    with nogil:
        status = cublasSetStream(<Handle>handle, <Stream>stream_id)
    check_status(status)


cpdef intptr_t get_stream(intptr_t handle) except? 0:
    """See `cublasGetStream`."""
    cdef Stream stream_id
    with nogil:
        status = cublasGetStream(<Handle>handle, &stream_id)
    check_status(status)
    return <intptr_t>stream_id


cpdef int get_pointer_mode(intptr_t handle) except? -1:
    """See `cublasGetPointerMode`."""
    cdef _PointerMode mode
    with nogil:
        status = cublasGetPointerMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


cpdef set_pointer_mode(intptr_t handle, int mode):
    """See `cublasSetPointerMode`."""
    with nogil:
        status = cublasSetPointerMode(<Handle>handle, <_PointerMode>mode)
    check_status(status)


cpdef int get_atomics_mode(intptr_t handle) except? -1:
    """See `cublasGetAtomicsMode`."""
    cdef _AtomicsMode mode
    with nogil:
        status = cublasGetAtomicsMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


cpdef set_atomics_mode(intptr_t handle, int mode):
    """See `cublasSetAtomicsMode`."""
    with nogil:
        status = cublasSetAtomicsMode(<Handle>handle, <_AtomicsMode>mode)
    check_status(status)


cpdef int get_math_mode(intptr_t handle) except? -1:
    """See `cublasGetMathMode`."""
    cdef _Math mode
    with nogil:
        status = cublasGetMathMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


cpdef set_math_mode(intptr_t handle, int mode):
    """See `cublasSetMathMode`."""
    with nogil:
        status = cublasSetMathMode(<Handle>handle, <_Math>mode)
    check_status(status)


cpdef logger_configure(int log_is_on, int log_to_std_out, int log_to_std_err, log_file_name):
    """See `cublasLoggerConfigure`."""
    if not isinstance(log_file_name, str):
        raise TypeError("log_file_name must be a Python str")
    cdef bytes _temp_log_file_name_ = (<str>log_file_name).encode()
    cdef char* _log_file_name_ = _temp_log_file_name_
    with nogil:
        status = cublasLoggerConfigure(log_is_on, log_to_std_out, log_to_std_err, <const char*>_log_file_name_)
    check_status(status)


cpdef set_vector(int n, int elem_size, intptr_t x, int incx, intptr_t device_ptr, int incy):
    """See `cublasSetVector`."""
    with nogil:
        status = cublasSetVector(n, elem_size, <const void*>x, incx, <void*>device_ptr, incy)
    check_status(status)


cpdef get_vector(int n, int elem_size, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasGetVector`."""
    with nogil:
        status = cublasGetVector(n, elem_size, <const void*>x, incx, <void*>y, incy)
    check_status(status)


cpdef set_matrix(int rows, int cols, int elem_size, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasSetMatrix`."""
    with nogil:
        status = cublasSetMatrix(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb)
    check_status(status)


cpdef get_matrix(int rows, int cols, int elem_size, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasGetMatrix`."""
    with nogil:
        status = cublasGetMatrix(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb)
    check_status(status)


cpdef set_vector_async(int n, int elem_size, intptr_t host_ptr, int incx, intptr_t device_ptr, int incy, intptr_t stream):
    """See `cublasSetVectorAsync`."""
    with nogil:
        status = cublasSetVectorAsync(n, elem_size, <const void*>host_ptr, incx, <void*>device_ptr, incy, <Stream>stream)
    check_status(status)


cpdef get_vector_async(int n, int elem_size, intptr_t device_ptr, int incx, intptr_t host_ptr, int incy, intptr_t stream):
    """See `cublasGetVectorAsync`."""
    with nogil:
        status = cublasGetVectorAsync(n, elem_size, <const void*>device_ptr, incx, <void*>host_ptr, incy, <Stream>stream)
    check_status(status)


cpdef set_matrix_async(int rows, int cols, int elem_size, intptr_t a, int lda, intptr_t b, int ldb, intptr_t stream):
    """See `cublasSetMatrixAsync`."""
    with nogil:
        status = cublasSetMatrixAsync(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb, <Stream>stream)
    check_status(status)


cpdef get_matrix_async(int rows, int cols, int elem_size, intptr_t a, int lda, intptr_t b, int ldb, intptr_t stream):
    """See `cublasGetMatrixAsync`."""
    with nogil:
        status = cublasGetMatrixAsync(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb, <Stream>stream)
    check_status(status)


cpdef nrm2_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t result, int result_type, int execution_type):
    """See `cublasNrm2Ex`."""
    with nogil:
        status = cublasNrm2Ex(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(status)


cpdef snrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasSnrm2`."""
    with nogil:
        status = cublasSnrm2(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(status)


cpdef dnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasDnrm2`."""
    with nogil:
        status = cublasDnrm2(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(status)


cpdef scnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasScnrm2`."""
    with nogil:
        status = cublasScnrm2(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(status)


cpdef dznrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasDznrm2`."""
    with nogil:
        status = cublasDznrm2(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(status)


cpdef dot_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy, intptr_t result, int result_type, int execution_type):
    """See `cublasDotEx`."""
    with nogil:
        status = cublasDotEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <const void*>y, <DataType>y_type, incy, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(status)


cpdef dotc_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy, intptr_t result, int result_type, int execution_type):
    """See `cublasDotcEx`."""
    with nogil:
        status = cublasDotcEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <const void*>y, <DataType>y_type, incy, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(status)


cpdef sdot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasSdot`."""
    with nogil:
        status = cublasSdot(<Handle>handle, n, <const float*>x, incx, <const float*>y, incy, <float*>result)
    check_status(status)


cpdef ddot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasDdot`."""
    with nogil:
        status = cublasDdot(<Handle>handle, n, <const double*>x, incx, <const double*>y, incy, <double*>result)
    check_status(status)


cpdef cdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasCdotu`."""
    with nogil:
        status = cublasCdotu(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(status)


cpdef cdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasCdotc`."""
    with nogil:
        status = cublasCdotc(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(status)


cpdef zdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasZdotu`."""
    with nogil:
        status = cublasZdotu(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)


cpdef zdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasZdotc`."""
    with nogil:
        status = cublasZdotc(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)


cpdef scal_ex(intptr_t handle, int n, intptr_t alpha, int alpha_type, intptr_t x, int x_type, int incx, int execution_type):
    """See `cublasScalEx`."""
    with nogil:
        status = cublasScalEx(<Handle>handle, n, <const void*>alpha, <DataType>alpha_type, <void*>x, <DataType>x_type, incx, <DataType>execution_type)
    check_status(status)


cpdef sscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasSscal`."""
    with nogil:
        status = cublasSscal(<Handle>handle, n, <const float*>alpha, <float*>x, incx)
    check_status(status)


cpdef dscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasDscal`."""
    with nogil:
        status = cublasDscal(<Handle>handle, n, <const double*>alpha, <double*>x, incx)
    check_status(status)


cpdef cscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasCscal`."""
    with nogil:
        status = cublasCscal(<Handle>handle, n, <const cuComplex*>alpha, <cuComplex*>x, incx)
    check_status(status)


cpdef csscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasCsscal`."""
    with nogil:
        status = cublasCsscal(<Handle>handle, n, <const float*>alpha, <cuComplex*>x, incx)
    check_status(status)


cpdef zscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasZscal`."""
    with nogil:
        status = cublasZscal(<Handle>handle, n, <const cuDoubleComplex*>alpha, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef zdscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasZdscal`."""
    with nogil:
        status = cublasZdscal(<Handle>handle, n, <const double*>alpha, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef axpy_ex(intptr_t handle, int n, intptr_t alpha, int alpha_type, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy, int executiontype):
    """See `cublasAxpyEx`."""
    with nogil:
        status = cublasAxpyEx(<Handle>handle, n, <const void*>alpha, <DataType>alpha_type, <const void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <DataType>executiontype)
    check_status(status)


cpdef saxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasSaxpy`."""
    with nogil:
        status = cublasSaxpy(<Handle>handle, n, <const float*>alpha, <const float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef daxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasDaxpy`."""
    with nogil:
        status = cublasDaxpy(<Handle>handle, n, <const double*>alpha, <const double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef caxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasCaxpy`."""
    with nogil:
        status = cublasCaxpy(<Handle>handle, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(status)


cpdef zaxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasZaxpy`."""
    with nogil:
        status = cublasZaxpy(<Handle>handle, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef copy_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy):
    """See `cublasCopyEx`."""
    with nogil:
        status = cublasCopyEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy)
    check_status(status)


cpdef scopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasScopy`."""
    with nogil:
        status = cublasScopy(<Handle>handle, n, <const float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef dcopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasDcopy`."""
    with nogil:
        status = cublasDcopy(<Handle>handle, n, <const double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef ccopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasCcopy`."""
    with nogil:
        status = cublasCcopy(<Handle>handle, n, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(status)


cpdef zcopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasZcopy`."""
    with nogil:
        status = cublasZcopy(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasSswap`."""
    with nogil:
        status = cublasSswap(<Handle>handle, n, <float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef dswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasDswap`."""
    with nogil:
        status = cublasDswap(<Handle>handle, n, <double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef cswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasCswap`."""
    with nogil:
        status = cublasCswap(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(status)


cpdef zswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasZswap`."""
    with nogil:
        status = cublasZswap(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef swap_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy):
    """See `cublasSwapEx`."""
    with nogil:
        status = cublasSwapEx(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy)
    check_status(status)


cpdef isamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIsamax`."""
    with nogil:
        status = cublasIsamax(<Handle>handle, n, <const float*>x, incx, <int*>result)
    check_status(status)


cpdef idamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIdamax`."""
    with nogil:
        status = cublasIdamax(<Handle>handle, n, <const double*>x, incx, <int*>result)
    check_status(status)


cpdef icamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIcamax`."""
    with nogil:
        status = cublasIcamax(<Handle>handle, n, <const cuComplex*>x, incx, <int*>result)
    check_status(status)


cpdef izamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIzamax`."""
    with nogil:
        status = cublasIzamax(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int*>result)
    check_status(status)


cpdef iamax_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t result):
    """See `cublasIamaxEx`."""
    with nogil:
        status = cublasIamaxEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <int*>result)
    check_status(status)


cpdef isamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIsamin`."""
    with nogil:
        status = cublasIsamin(<Handle>handle, n, <const float*>x, incx, <int*>result)
    check_status(status)


cpdef idamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIdamin`."""
    with nogil:
        status = cublasIdamin(<Handle>handle, n, <const double*>x, incx, <int*>result)
    check_status(status)


cpdef icamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIcamin`."""
    with nogil:
        status = cublasIcamin(<Handle>handle, n, <const cuComplex*>x, incx, <int*>result)
    check_status(status)


cpdef izamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIzamin`."""
    with nogil:
        status = cublasIzamin(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int*>result)
    check_status(status)


cpdef iamin_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t result):
    """See `cublasIaminEx`."""
    with nogil:
        status = cublasIaminEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <int*>result)
    check_status(status)


cpdef asum_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t result, int result_type, int executiontype):
    """See `cublasAsumEx`."""
    with nogil:
        status = cublasAsumEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>result, <DataType>result_type, <DataType>executiontype)
    check_status(status)


cpdef sasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasSasum`."""
    with nogil:
        status = cublasSasum(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(status)


cpdef dasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasDasum`."""
    with nogil:
        status = cublasDasum(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(status)


cpdef scasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasScasum`."""
    with nogil:
        status = cublasScasum(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(status)


cpdef dzasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasDzasum`."""
    with nogil:
        status = cublasDzasum(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(status)


cpdef srot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasSrot`."""
    with nogil:
        status = cublasSrot(<Handle>handle, n, <float*>x, incx, <float*>y, incy, <const float*>c, <const float*>s)
    check_status(status)


cpdef drot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasDrot`."""
    with nogil:
        status = cublasDrot(<Handle>handle, n, <double*>x, incx, <double*>y, incy, <const double*>c, <const double*>s)
    check_status(status)


cpdef crot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasCrot`."""
    with nogil:
        status = cublasCrot(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy, <const float*>c, <const cuComplex*>s)
    check_status(status)


cpdef csrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasCsrot`."""
    with nogil:
        status = cublasCsrot(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy, <const float*>c, <const float*>s)
    check_status(status)


cpdef zrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasZrot`."""
    with nogil:
        status = cublasZrot(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy, <const double*>c, <const cuDoubleComplex*>s)
    check_status(status)


cpdef zdrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasZdrot`."""
    with nogil:
        status = cublasZdrot(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy, <const double*>c, <const double*>s)
    check_status(status)


cpdef rot_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy, intptr_t c, intptr_t s, int cs_type, int executiontype):
    """See `cublasRotEx`."""
    with nogil:
        status = cublasRotEx(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <const void*>c, <const void*>s, <DataType>cs_type, <DataType>executiontype)
    check_status(status)


cpdef srotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    """See `cublasSrotg`."""
    with nogil:
        status = cublasSrotg(<Handle>handle, <float*>a, <float*>b, <float*>c, <float*>s)
    check_status(status)


cpdef drotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    """See `cublasDrotg`."""
    with nogil:
        status = cublasDrotg(<Handle>handle, <double*>a, <double*>b, <double*>c, <double*>s)
    check_status(status)


cpdef crotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    """See `cublasCrotg`."""
    with nogil:
        status = cublasCrotg(<Handle>handle, <cuComplex*>a, <cuComplex*>b, <float*>c, <cuComplex*>s)
    check_status(status)


cpdef zrotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    """See `cublasZrotg`."""
    with nogil:
        status = cublasZrotg(<Handle>handle, <cuDoubleComplex*>a, <cuDoubleComplex*>b, <double*>c, <cuDoubleComplex*>s)
    check_status(status)


cpdef rotg_ex(intptr_t handle, intptr_t a, intptr_t b, int ab_type, intptr_t c, intptr_t s, int cs_type, int executiontype):
    """See `cublasRotgEx`."""
    with nogil:
        status = cublasRotgEx(<Handle>handle, <void*>a, <void*>b, <DataType>ab_type, <void*>c, <void*>s, <DataType>cs_type, <DataType>executiontype)
    check_status(status)


cpdef srotm(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t param):
    """See `cublasSrotm`."""
    with nogil:
        status = cublasSrotm(<Handle>handle, n, <float*>x, incx, <float*>y, incy, <const float*>param)
    check_status(status)


cpdef drotm(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t param):
    """See `cublasDrotm`."""
    with nogil:
        status = cublasDrotm(<Handle>handle, n, <double*>x, incx, <double*>y, incy, <const double*>param)
    check_status(status)


cpdef rotm_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy, intptr_t param, int param_type, int executiontype):
    """See `cublasRotmEx`."""
    with nogil:
        status = cublasRotmEx(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <const void*>param, <DataType>param_type, <DataType>executiontype)
    check_status(status)


cpdef srotmg(intptr_t handle, intptr_t d1, intptr_t d2, intptr_t x1, intptr_t y1, intptr_t param):
    """See `cublasSrotmg`."""
    with nogil:
        status = cublasSrotmg(<Handle>handle, <float*>d1, <float*>d2, <float*>x1, <const float*>y1, <float*>param)
    check_status(status)


cpdef drotmg(intptr_t handle, intptr_t d1, intptr_t d2, intptr_t x1, intptr_t y1, intptr_t param):
    """See `cublasDrotmg`."""
    with nogil:
        status = cublasDrotmg(<Handle>handle, <double*>d1, <double*>d2, <double*>x1, <const double*>y1, <double*>param)
    check_status(status)


cpdef rotmg_ex(intptr_t handle, intptr_t d1, int d1type, intptr_t d2, int d2type, intptr_t x1, int x1type, intptr_t y1, int y1type, intptr_t param, int param_type, int executiontype):
    """See `cublasRotmgEx`."""
    with nogil:
        status = cublasRotmgEx(<Handle>handle, <void*>d1, <DataType>d1type, <void*>d2, <DataType>d2type, <void*>x1, <DataType>x1type, <const void*>y1, <DataType>y1type, <void*>param, <DataType>param_type, <DataType>executiontype)
    check_status(status)


cpdef sgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasSgemv`."""
    with nogil:
        status = cublasSgemv(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasDgemv`."""
    with nogil:
        status = cublasDgemv(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef cgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasCgemv`."""
    with nogil:
        status = cublasCgemv(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZgemv`."""
    with nogil:
        status = cublasZgemv(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasSgbmv`."""
    with nogil:
        status = cublasSgbmv(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasDgbmv`."""
    with nogil:
        status = cublasDgbmv(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef cgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasCgbmv`."""
    with nogil:
        status = cublasCgbmv(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZgbmv`."""
    with nogil:
        status = cublasZgbmv(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef strmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasStrmv`."""
    with nogil:
        status = cublasStrmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>a, lda, <float*>x, incx)
    check_status(status)


cpdef dtrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasDtrmv`."""
    with nogil:
        status = cublasDtrmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>a, lda, <double*>x, incx)
    check_status(status)


cpdef ctrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasCtrmv`."""
    with nogil:
        status = cublasCtrmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasZtrmv`."""
    with nogil:
        status = cublasZtrmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasStbmv`."""
    with nogil:
        status = cublasStbmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const float*>a, lda, <float*>x, incx)
    check_status(status)


cpdef dtbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasDtbmv`."""
    with nogil:
        status = cublasDtbmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const double*>a, lda, <double*>x, incx)
    check_status(status)


cpdef ctbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasCtbmv`."""
    with nogil:
        status = cublasCtbmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasZtbmv`."""
    with nogil:
        status = cublasZtbmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasStpmv`."""
    with nogil:
        status = cublasStpmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>ap, <float*>x, incx)
    check_status(status)


cpdef dtpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasDtpmv`."""
    with nogil:
        status = cublasDtpmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>ap, <double*>x, incx)
    check_status(status)


cpdef ctpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasCtpmv`."""
    with nogil:
        status = cublasCtpmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>ap, <cuComplex*>x, incx)
    check_status(status)


cpdef ztpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasZtpmv`."""
    with nogil:
        status = cublasZtpmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>ap, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef strsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasStrsv`."""
    with nogil:
        status = cublasStrsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>a, lda, <float*>x, incx)
    check_status(status)


cpdef dtrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasDtrsv`."""
    with nogil:
        status = cublasDtrsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>a, lda, <double*>x, incx)
    check_status(status)


cpdef ctrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasCtrsv`."""
    with nogil:
        status = cublasCtrsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasZtrsv`."""
    with nogil:
        status = cublasZtrsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasStpsv`."""
    with nogil:
        status = cublasStpsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>ap, <float*>x, incx)
    check_status(status)


cpdef dtpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasDtpsv`."""
    with nogil:
        status = cublasDtpsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>ap, <double*>x, incx)
    check_status(status)


cpdef ctpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasCtpsv`."""
    with nogil:
        status = cublasCtpsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>ap, <cuComplex*>x, incx)
    check_status(status)


cpdef ztpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasZtpsv`."""
    with nogil:
        status = cublasZtpsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>ap, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasStbsv`."""
    with nogil:
        status = cublasStbsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const float*>a, lda, <float*>x, incx)
    check_status(status)


cpdef dtbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasDtbsv`."""
    with nogil:
        status = cublasDtbsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const double*>a, lda, <double*>x, incx)
    check_status(status)


cpdef ctbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasCtbsv`."""
    with nogil:
        status = cublasCtbsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasZtbsv`."""
    with nogil:
        status = cublasZtbsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef ssymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasSsymv`."""
    with nogil:
        status = cublasSsymv(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dsymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasDsymv`."""
    with nogil:
        status = cublasDsymv(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef csymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasCsymv`."""
    with nogil:
        status = cublasCsymv(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zsymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZsymv`."""
    with nogil:
        status = cublasZsymv(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef chemv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasChemv`."""
    with nogil:
        status = cublasChemv(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zhemv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZhemv`."""
    with nogil:
        status = cublasZhemv(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef ssbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasSsbmv`."""
    with nogil:
        status = cublasSsbmv(<Handle>handle, <_FillMode>uplo, n, k, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dsbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasDsbmv`."""
    with nogil:
        status = cublasDsbmv(<Handle>handle, <_FillMode>uplo, n, k, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef chbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasChbmv`."""
    with nogil:
        status = cublasChbmv(<Handle>handle, <_FillMode>uplo, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zhbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZhbmv`."""
    with nogil:
        status = cublasZhbmv(<Handle>handle, <_FillMode>uplo, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sspmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t ap, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasSspmv`."""
    with nogil:
        status = cublasSspmv(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>ap, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dspmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t ap, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasDspmv`."""
    with nogil:
        status = cublasDspmv(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>ap, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef chpmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t ap, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasChpmv`."""
    with nogil:
        status = cublasChpmv(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>ap, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zhpmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t ap, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZhpmv`."""
    with nogil:
        status = cublasZhpmv(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>ap, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasSger`."""
    with nogil:
        status = cublasSger(<Handle>handle, m, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>a, lda)
    check_status(status)


cpdef dger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasDger`."""
    with nogil:
        status = cublasDger(<Handle>handle, m, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>a, lda)
    check_status(status)


cpdef cgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasCgeru`."""
    with nogil:
        status = cublasCgeru(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(status)


cpdef cgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasCgerc`."""
    with nogil:
        status = cublasCgerc(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(status)


cpdef zgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasZgeru`."""
    with nogil:
        status = cublasZgeru(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef zgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasZgerc`."""
    with nogil:
        status = cublasZgerc(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef ssyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasSsyr`."""
    with nogil:
        status = cublasSsyr(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <float*>a, lda)
    check_status(status)


cpdef dsyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasDsyr`."""
    with nogil:
        status = cublasDsyr(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <double*>a, lda)
    check_status(status)


cpdef csyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasCsyr`."""
    with nogil:
        status = cublasCsyr(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>a, lda)
    check_status(status)


cpdef zsyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasZsyr`."""
    with nogil:
        status = cublasZsyr(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef cher(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasCher`."""
    with nogil:
        status = cublasCher(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const cuComplex*>x, incx, <cuComplex*>a, lda)
    check_status(status)


cpdef zher(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasZher`."""
    with nogil:
        status = cublasZher(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef sspr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t ap):
    """See `cublasSspr`."""
    with nogil:
        status = cublasSspr(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <float*>ap)
    check_status(status)


cpdef dspr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t ap):
    """See `cublasDspr`."""
    with nogil:
        status = cublasDspr(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <double*>ap)
    check_status(status)


cpdef chpr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t ap):
    """See `cublasChpr`."""
    with nogil:
        status = cublasChpr(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const cuComplex*>x, incx, <cuComplex*>ap)
    check_status(status)


cpdef zhpr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t ap):
    """See `cublasZhpr`."""
    with nogil:
        status = cublasZhpr(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>ap)
    check_status(status)


cpdef ssyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasSsyr2`."""
    with nogil:
        status = cublasSsyr2(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>a, lda)
    check_status(status)


cpdef dsyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasDsyr2`."""
    with nogil:
        status = cublasDsyr2(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>a, lda)
    check_status(status)


cpdef csyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasCsyr2`."""
    with nogil:
        status = cublasCsyr2(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(status)


cpdef zsyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasZsyr2`."""
    with nogil:
        status = cublasZsyr2(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef cher2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasCher2`."""
    with nogil:
        status = cublasCher2(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(status)


cpdef zher2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasZher2`."""
    with nogil:
        status = cublasZher2(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef sspr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t ap):
    """See `cublasSspr2`."""
    with nogil:
        status = cublasSspr2(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>ap)
    check_status(status)


cpdef dspr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t ap):
    """See `cublasDspr2`."""
    with nogil:
        status = cublasDspr2(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>ap)
    check_status(status)


cpdef chpr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t ap):
    """See `cublasChpr2`."""
    with nogil:
        status = cublasChpr2(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>ap)
    check_status(status)


cpdef zhpr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t ap):
    """See `cublasZhpr2`."""
    with nogil:
        status = cublasZhpr2(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>ap)
    check_status(status)


cpdef sgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasSgemm`."""
    with nogil:
        status = cublasSgemm(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasDgemm`."""
    with nogil:
        status = cublasDgemm(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef cgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCgemm`."""
    with nogil:
        status = cublasCgemm(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef cgemm3m(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCgemm3m`."""
    with nogil:
        status = cublasCgemm3m(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef cgemm3m_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t b, int btype, int ldb, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCgemm3mEx`."""
    with nogil:
        status = cublasCgemm3mEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef zgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZgemm`."""
    with nogil:
        status = cublasZgemm(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef zgemm3m(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZgemm3m`."""
    with nogil:
        status = cublasZgemm3m(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef sgemm_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t b, int btype, int ldb, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasSgemmEx`."""
    with nogil:
        status = cublasSgemmEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef gemm_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t b, int btype, int ldb, intptr_t beta, intptr_t c, int ctype, int ldc, int compute_type, int algo):
    """See `cublasGemmEx`."""
    with nogil:
        status = cublasGemmEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const void*>beta, <void*>c, <DataType>ctype, ldc, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(status)


cpdef cgemm_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t b, int btype, int ldb, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCgemmEx`."""
    with nogil:
        status = cublasCgemmEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef uint8gemm_bias(intptr_t handle, int transa, int transb, int transc, int m, int n, int k, intptr_t a, int a_bias, int lda, intptr_t b, int b_bias, int ldb, intptr_t c, int c_bias, int ldc, int c_mult, int c_shift):
    """See `cublasUint8gemmBias`."""
    with nogil:
        status = cublasUint8gemmBias(<Handle>handle, <_Operation>transa, <_Operation>transb, <_Operation>transc, m, n, k, <const unsigned char*>a, a_bias, lda, <const unsigned char*>b, b_bias, ldb, <unsigned char*>c, c_bias, ldc, c_mult, c_shift)
    check_status(status)


cpdef ssyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasSsyrk`."""
    with nogil:
        status = cublasSsyrk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dsyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasDsyrk`."""
    with nogil:
        status = cublasDsyrk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef csyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCsyrk`."""
    with nogil:
        status = cublasCsyrk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zsyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZsyrk`."""
    with nogil:
        status = cublasZsyrk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef csyrk_ex(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCsyrkEx`."""
    with nogil:
        status = cublasCsyrkEx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef csyrk3m_ex(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCsyrk3mEx`."""
    with nogil:
        status = cublasCsyrk3mEx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef cherk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCherk`."""
    with nogil:
        status = cublasCherk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const cuComplex*>a, lda, <const float*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zherk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZherk`."""
    with nogil:
        status = cublasZherk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const cuDoubleComplex*>a, lda, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef cherk_ex(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCherkEx`."""
    with nogil:
        status = cublasCherkEx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef cherk3m_ex(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCherk3mEx`."""
    with nogil:
        status = cublasCherk3mEx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef ssyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasSsyr2k`."""
    with nogil:
        status = cublasSsyr2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dsyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasDsyr2k`."""
    with nogil:
        status = cublasDsyr2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef csyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCsyr2k`."""
    with nogil:
        status = cublasCsyr2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zsyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZsyr2k`."""
    with nogil:
        status = cublasZsyr2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef cher2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCher2k`."""
    with nogil:
        status = cublasCher2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const float*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zher2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZher2k`."""
    with nogil:
        status = cublasZher2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef ssyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasSsyrkx`."""
    with nogil:
        status = cublasSsyrkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dsyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasDsyrkx`."""
    with nogil:
        status = cublasDsyrkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef csyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCsyrkx`."""
    with nogil:
        status = cublasCsyrkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zsyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZsyrkx`."""
    with nogil:
        status = cublasZsyrkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef cherkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCherkx`."""
    with nogil:
        status = cublasCherkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const float*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zherkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZherkx`."""
    with nogil:
        status = cublasZherkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef ssymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasSsymm`."""
    with nogil:
        status = cublasSsymm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dsymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasDsymm`."""
    with nogil:
        status = cublasDsymm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef csymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCsymm`."""
    with nogil:
        status = cublasCsymm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zsymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZsymm`."""
    with nogil:
        status = cublasZsymm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef chemm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasChemm`."""
    with nogil:
        status = cublasChemm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zhemm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZhemm`."""
    with nogil:
        status = cublasZhemm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef strsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasStrsm`."""
    with nogil:
        status = cublasStrsm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float*>a, lda, <float*>b, ldb)
    check_status(status)


cpdef dtrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasDtrsm`."""
    with nogil:
        status = cublasDtrsm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double*>a, lda, <double*>b, ldb)
    check_status(status)


cpdef ctrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasCtrsm`."""
    with nogil:
        status = cublasCtrsm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <cuComplex*>b, ldb)
    check_status(status)


cpdef ztrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasZtrsm`."""
    with nogil:
        status = cublasZtrsm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>b, ldb)
    check_status(status)


cpdef strmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasStrmm`."""
    with nogil:
        status = cublasStrmm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <float*>c, ldc)
    check_status(status)


cpdef dtrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasDtrmm`."""
    with nogil:
        status = cublasDtrmm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <double*>c, ldc)
    check_status(status)


cpdef ctrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasCtrmm`."""
    with nogil:
        status = cublasCtrmm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <cuComplex*>c, ldc)
    check_status(status)


cpdef ztrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasZtrmm`."""
    with nogil:
        status = cublasZtrmm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef sgemm_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int lda, intptr_t barray, int ldb, intptr_t beta, intptr_t carray, int ldc, int batch_count):
    """See `cublasSgemmBatched`."""
    with nogil:
        status = cublasSgemmBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float* const*>aarray, lda, <const float* const*>barray, ldb, <const float*>beta, <float* const*>carray, ldc, batch_count)
    check_status(status)


cpdef dgemm_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int lda, intptr_t barray, int ldb, intptr_t beta, intptr_t carray, int ldc, int batch_count):
    """See `cublasDgemmBatched`."""
    with nogil:
        status = cublasDgemmBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double* const*>aarray, lda, <const double* const*>barray, ldb, <const double*>beta, <double* const*>carray, ldc, batch_count)
    check_status(status)


cpdef cgemm_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int lda, intptr_t barray, int ldb, intptr_t beta, intptr_t carray, int ldc, int batch_count):
    """See `cublasCgemmBatched`."""
    with nogil:
        status = cublasCgemmBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>barray, ldb, <const cuComplex*>beta, <cuComplex* const*>carray, ldc, batch_count)
    check_status(status)


cpdef cgemm3m_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int lda, intptr_t barray, int ldb, intptr_t beta, intptr_t carray, int ldc, int batch_count):
    """See `cublasCgemm3mBatched`."""
    with nogil:
        status = cublasCgemm3mBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>barray, ldb, <const cuComplex*>beta, <cuComplex* const*>carray, ldc, batch_count)
    check_status(status)


cpdef zgemm_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int lda, intptr_t barray, int ldb, intptr_t beta, intptr_t carray, int ldc, int batch_count):
    """See `cublasZgemmBatched`."""
    with nogil:
        status = cublasZgemmBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>aarray, lda, <const cuDoubleComplex* const*>barray, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex* const*>carray, ldc, batch_count)
    check_status(status)


cpdef gemm_batched_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int atype, int lda, intptr_t barray, int btype, int ldb, intptr_t beta, intptr_t carray, int ctype, int ldc, int batch_count, int compute_type, int algo):
    """See `cublasGemmBatchedEx`."""
    with nogil:
        status = cublasGemmBatchedEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void* const*>aarray, <DataType>atype, lda, <const void* const*>barray, <DataType>btype, ldb, <const void*>beta, <void* const*>carray, <DataType>ctype, ldc, batch_count, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(status)


cpdef gemm_strided_batched_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, long long int stride_a, intptr_t b, int btype, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ctype, int ldc, long long int stride_c, int batch_count, int compute_type, int algo):
    """See `cublasGemmStridedBatchedEx`."""
    with nogil:
        status = cublasGemmStridedBatchedEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void*>a, <DataType>atype, lda, stride_a, <const void*>b, <DataType>btype, ldb, stride_b, <const void*>beta, <void*>c, <DataType>ctype, ldc, stride_c, batch_count, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(status)


cpdef sgemm_strided_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t b, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ldc, long long int stride_c, int batch_count):
    """See `cublasSgemmStridedBatched`."""
    with nogil:
        status = cublasSgemmStridedBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float*>a, lda, stride_a, <const float*>b, ldb, stride_b, <const float*>beta, <float*>c, ldc, stride_c, batch_count)
    check_status(status)


cpdef dgemm_strided_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t b, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ldc, long long int stride_c, int batch_count):
    """See `cublasDgemmStridedBatched`."""
    with nogil:
        status = cublasDgemmStridedBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double*>a, lda, stride_a, <const double*>b, ldb, stride_b, <const double*>beta, <double*>c, ldc, stride_c, batch_count)
    check_status(status)


cpdef cgemm_strided_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t b, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ldc, long long int stride_c, int batch_count):
    """See `cublasCgemmStridedBatched`."""
    with nogil:
        status = cublasCgemmStridedBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>b, ldb, stride_b, <const cuComplex*>beta, <cuComplex*>c, ldc, stride_c, batch_count)
    check_status(status)


cpdef cgemm3m_strided_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t b, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ldc, long long int stride_c, int batch_count):
    """See `cublasCgemm3mStridedBatched`."""
    with nogil:
        status = cublasCgemm3mStridedBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>b, ldb, stride_b, <const cuComplex*>beta, <cuComplex*>c, ldc, stride_c, batch_count)
    check_status(status)


cpdef zgemm_strided_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t b, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ldc, long long int stride_c, int batch_count):
    """See `cublasZgemmStridedBatched`."""
    with nogil:
        status = cublasZgemmStridedBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, stride_a, <const cuDoubleComplex*>b, ldb, stride_b, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc, stride_c, batch_count)
    check_status(status)


cpdef sgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasSgeam`."""
    with nogil:
        status = cublasSgeam(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const float*>alpha, <const float*>a, lda, <const float*>beta, <const float*>b, ldb, <float*>c, ldc)
    check_status(status)


cpdef dgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasDgeam`."""
    with nogil:
        status = cublasDgeam(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const double*>alpha, <const double*>a, lda, <const double*>beta, <const double*>b, ldb, <double*>c, ldc)
    check_status(status)


cpdef cgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasCgeam`."""
    with nogil:
        status = cublasCgeam(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>beta, <const cuComplex*>b, ldb, <cuComplex*>c, ldc)
    check_status(status)


cpdef zgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasZgeam`."""
    with nogil:
        status = cublasZgeam(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>beta, <const cuDoubleComplex*>b, ldb, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef sgetrf_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t info, int batch_size):
    """See `cublasSgetrfBatched`."""
    with nogil:
        status = cublasSgetrfBatched(<Handle>handle, n, <float* const*>a, lda, <int*>p, <int*>info, batch_size)
    check_status(status)


cpdef dgetrf_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t info, int batch_size):
    """See `cublasDgetrfBatched`."""
    with nogil:
        status = cublasDgetrfBatched(<Handle>handle, n, <double* const*>a, lda, <int*>p, <int*>info, batch_size)
    check_status(status)


cpdef cgetrf_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t info, int batch_size):
    """See `cublasCgetrfBatched`."""
    with nogil:
        status = cublasCgetrfBatched(<Handle>handle, n, <cuComplex* const*>a, lda, <int*>p, <int*>info, batch_size)
    check_status(status)


cpdef zgetrf_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t info, int batch_size):
    """See `cublasZgetrfBatched`."""
    with nogil:
        status = cublasZgetrfBatched(<Handle>handle, n, <cuDoubleComplex* const*>a, lda, <int*>p, <int*>info, batch_size)
    check_status(status)


cpdef sgetri_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t c, int ldc, intptr_t info, int batch_size):
    """See `cublasSgetriBatched`."""
    with nogil:
        status = cublasSgetriBatched(<Handle>handle, n, <const float* const*>a, lda, <const int*>p, <float* const*>c, ldc, <int*>info, batch_size)
    check_status(status)


cpdef dgetri_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t c, int ldc, intptr_t info, int batch_size):
    """See `cublasDgetriBatched`."""
    with nogil:
        status = cublasDgetriBatched(<Handle>handle, n, <const double* const*>a, lda, <const int*>p, <double* const*>c, ldc, <int*>info, batch_size)
    check_status(status)


cpdef cgetri_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t c, int ldc, intptr_t info, int batch_size):
    """See `cublasCgetriBatched`."""
    with nogil:
        status = cublasCgetriBatched(<Handle>handle, n, <const cuComplex* const*>a, lda, <const int*>p, <cuComplex* const*>c, ldc, <int*>info, batch_size)
    check_status(status)


cpdef zgetri_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t c, int ldc, intptr_t info, int batch_size):
    """See `cublasZgetriBatched`."""
    with nogil:
        status = cublasZgetriBatched(<Handle>handle, n, <const cuDoubleComplex* const*>a, lda, <const int*>p, <cuDoubleComplex* const*>c, ldc, <int*>info, batch_size)
    check_status(status)


cpdef sgetrs_batched(intptr_t handle, int trans, int n, int nrhs, intptr_t aarray, int lda, intptr_t dev_ipiv, intptr_t barray, int ldb, intptr_t info, int batch_size):
    """See `cublasSgetrsBatched`."""
    with nogil:
        status = cublasSgetrsBatched(<Handle>handle, <_Operation>trans, n, nrhs, <const float* const*>aarray, lda, <const int*>dev_ipiv, <float* const*>barray, ldb, <int*>info, batch_size)
    check_status(status)


cpdef dgetrs_batched(intptr_t handle, int trans, int n, int nrhs, intptr_t aarray, int lda, intptr_t dev_ipiv, intptr_t barray, int ldb, intptr_t info, int batch_size):
    """See `cublasDgetrsBatched`."""
    with nogil:
        status = cublasDgetrsBatched(<Handle>handle, <_Operation>trans, n, nrhs, <const double* const*>aarray, lda, <const int*>dev_ipiv, <double* const*>barray, ldb, <int*>info, batch_size)
    check_status(status)


cpdef cgetrs_batched(intptr_t handle, int trans, int n, int nrhs, intptr_t aarray, int lda, intptr_t dev_ipiv, intptr_t barray, int ldb, intptr_t info, int batch_size):
    """See `cublasCgetrsBatched`."""
    with nogil:
        status = cublasCgetrsBatched(<Handle>handle, <_Operation>trans, n, nrhs, <const cuComplex* const*>aarray, lda, <const int*>dev_ipiv, <cuComplex* const*>barray, ldb, <int*>info, batch_size)
    check_status(status)


cpdef zgetrs_batched(intptr_t handle, int trans, int n, int nrhs, intptr_t aarray, int lda, intptr_t dev_ipiv, intptr_t barray, int ldb, intptr_t info, int batch_size):
    """See `cublasZgetrsBatched`."""
    with nogil:
        status = cublasZgetrsBatched(<Handle>handle, <_Operation>trans, n, nrhs, <const cuDoubleComplex* const*>aarray, lda, <const int*>dev_ipiv, <cuDoubleComplex* const*>barray, ldb, <int*>info, batch_size)
    check_status(status)


cpdef strsm_batched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, int batch_count):
    """See `cublasStrsmBatched`."""
    with nogil:
        status = cublasStrsmBatched(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float* const*>a, lda, <float* const*>b, ldb, batch_count)
    check_status(status)


cpdef dtrsm_batched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, int batch_count):
    """See `cublasDtrsmBatched`."""
    with nogil:
        status = cublasDtrsmBatched(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double* const*>a, lda, <double* const*>b, ldb, batch_count)
    check_status(status)


cpdef ctrsm_batched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, int batch_count):
    """See `cublasCtrsmBatched`."""
    with nogil:
        status = cublasCtrsmBatched(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex* const*>a, lda, <cuComplex* const*>b, ldb, batch_count)
    check_status(status)


cpdef ztrsm_batched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, int batch_count):
    """See `cublasZtrsmBatched`."""
    with nogil:
        status = cublasZtrsmBatched(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>a, lda, <cuDoubleComplex* const*>b, ldb, batch_count)
    check_status(status)


cpdef smatinv_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t ainv, int lda_inv, intptr_t info, int batch_size):
    """See `cublasSmatinvBatched`."""
    with nogil:
        status = cublasSmatinvBatched(<Handle>handle, n, <const float* const*>a, lda, <float* const*>ainv, lda_inv, <int*>info, batch_size)
    check_status(status)


cpdef dmatinv_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t ainv, int lda_inv, intptr_t info, int batch_size):
    """See `cublasDmatinvBatched`."""
    with nogil:
        status = cublasDmatinvBatched(<Handle>handle, n, <const double* const*>a, lda, <double* const*>ainv, lda_inv, <int*>info, batch_size)
    check_status(status)


cpdef cmatinv_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t ainv, int lda_inv, intptr_t info, int batch_size):
    """See `cublasCmatinvBatched`."""
    with nogil:
        status = cublasCmatinvBatched(<Handle>handle, n, <const cuComplex* const*>a, lda, <cuComplex* const*>ainv, lda_inv, <int*>info, batch_size)
    check_status(status)


cpdef zmatinv_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t ainv, int lda_inv, intptr_t info, int batch_size):
    """See `cublasZmatinvBatched`."""
    with nogil:
        status = cublasZmatinvBatched(<Handle>handle, n, <const cuDoubleComplex* const*>a, lda, <cuDoubleComplex* const*>ainv, lda_inv, <int*>info, batch_size)
    check_status(status)


cpdef sgeqrf_batched(intptr_t handle, int m, int n, intptr_t aarray, int lda, intptr_t tau_array, intptr_t info, int batch_size):
    """See `cublasSgeqrfBatched`."""
    with nogil:
        status = cublasSgeqrfBatched(<Handle>handle, m, n, <float* const*>aarray, lda, <float* const*>tau_array, <int*>info, batch_size)
    check_status(status)


cpdef dgeqrf_batched(intptr_t handle, int m, int n, intptr_t aarray, int lda, intptr_t tau_array, intptr_t info, int batch_size):
    """See `cublasDgeqrfBatched`."""
    with nogil:
        status = cublasDgeqrfBatched(<Handle>handle, m, n, <double* const*>aarray, lda, <double* const*>tau_array, <int*>info, batch_size)
    check_status(status)


cpdef cgeqrf_batched(intptr_t handle, int m, int n, intptr_t aarray, int lda, intptr_t tau_array, intptr_t info, int batch_size):
    """See `cublasCgeqrfBatched`."""
    with nogil:
        status = cublasCgeqrfBatched(<Handle>handle, m, n, <cuComplex* const*>aarray, lda, <cuComplex* const*>tau_array, <int*>info, batch_size)
    check_status(status)


cpdef zgeqrf_batched(intptr_t handle, int m, int n, intptr_t aarray, int lda, intptr_t tau_array, intptr_t info, int batch_size):
    """See `cublasZgeqrfBatched`."""
    with nogil:
        status = cublasZgeqrfBatched(<Handle>handle, m, n, <cuDoubleComplex* const*>aarray, lda, <cuDoubleComplex* const*>tau_array, <int*>info, batch_size)
    check_status(status)


cpdef sgels_batched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t aarray, int lda, intptr_t carray, int ldc, intptr_t info, intptr_t dev_info_array, int batch_size):
    """See `cublasSgelsBatched`."""
    with nogil:
        status = cublasSgelsBatched(<Handle>handle, <_Operation>trans, m, n, nrhs, <float* const*>aarray, lda, <float* const*>carray, ldc, <int*>info, <int*>dev_info_array, batch_size)
    check_status(status)


cpdef dgels_batched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t aarray, int lda, intptr_t carray, int ldc, intptr_t info, intptr_t dev_info_array, int batch_size):
    """See `cublasDgelsBatched`."""
    with nogil:
        status = cublasDgelsBatched(<Handle>handle, <_Operation>trans, m, n, nrhs, <double* const*>aarray, lda, <double* const*>carray, ldc, <int*>info, <int*>dev_info_array, batch_size)
    check_status(status)


cpdef cgels_batched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t aarray, int lda, intptr_t carray, int ldc, intptr_t info, intptr_t dev_info_array, int batch_size):
    """See `cublasCgelsBatched`."""
    with nogil:
        status = cublasCgelsBatched(<Handle>handle, <_Operation>trans, m, n, nrhs, <cuComplex* const*>aarray, lda, <cuComplex* const*>carray, ldc, <int*>info, <int*>dev_info_array, batch_size)
    check_status(status)


cpdef zgels_batched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t aarray, int lda, intptr_t carray, int ldc, intptr_t info, intptr_t dev_info_array, int batch_size):
    """See `cublasZgelsBatched`."""
    with nogil:
        status = cublasZgelsBatched(<Handle>handle, <_Operation>trans, m, n, nrhs, <cuDoubleComplex* const*>aarray, lda, <cuDoubleComplex* const*>carray, ldc, <int*>info, <int*>dev_info_array, batch_size)
    check_status(status)


cpdef sdgmm(intptr_t handle, int mode, int m, int n, intptr_t a, int lda, intptr_t x, int incx, intptr_t c, int ldc):
    """See `cublasSdgmm`."""
    with nogil:
        status = cublasSdgmm(<Handle>handle, <_SideMode>mode, m, n, <const float*>a, lda, <const float*>x, incx, <float*>c, ldc)
    check_status(status)


cpdef ddgmm(intptr_t handle, int mode, int m, int n, intptr_t a, int lda, intptr_t x, int incx, intptr_t c, int ldc):
    """See `cublasDdgmm`."""
    with nogil:
        status = cublasDdgmm(<Handle>handle, <_SideMode>mode, m, n, <const double*>a, lda, <const double*>x, incx, <double*>c, ldc)
    check_status(status)


cpdef cdgmm(intptr_t handle, int mode, int m, int n, intptr_t a, int lda, intptr_t x, int incx, intptr_t c, int ldc):
    """See `cublasCdgmm`."""
    with nogil:
        status = cublasCdgmm(<Handle>handle, <_SideMode>mode, m, n, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <cuComplex*>c, ldc)
    check_status(status)


cpdef zdgmm(intptr_t handle, int mode, int m, int n, intptr_t a, int lda, intptr_t x, int incx, intptr_t c, int ldc):
    """See `cublasZdgmm`."""
    with nogil:
        status = cublasZdgmm(<Handle>handle, <_SideMode>mode, m, n, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef stpttr(intptr_t handle, int uplo, int n, intptr_t ap, intptr_t a, int lda):
    """See `cublasStpttr`."""
    with nogil:
        status = cublasStpttr(<Handle>handle, <_FillMode>uplo, n, <const float*>ap, <float*>a, lda)
    check_status(status)


cpdef dtpttr(intptr_t handle, int uplo, int n, intptr_t ap, intptr_t a, int lda):
    """See `cublasDtpttr`."""
    with nogil:
        status = cublasDtpttr(<Handle>handle, <_FillMode>uplo, n, <const double*>ap, <double*>a, lda)
    check_status(status)


cpdef ctpttr(intptr_t handle, int uplo, int n, intptr_t ap, intptr_t a, int lda):
    """See `cublasCtpttr`."""
    with nogil:
        status = cublasCtpttr(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>ap, <cuComplex*>a, lda)
    check_status(status)


cpdef ztpttr(intptr_t handle, int uplo, int n, intptr_t ap, intptr_t a, int lda):
    """See `cublasZtpttr`."""
    with nogil:
        status = cublasZtpttr(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>ap, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef strttp(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ap):
    """See `cublasStrttp`."""
    with nogil:
        status = cublasStrttp(<Handle>handle, <_FillMode>uplo, n, <const float*>a, lda, <float*>ap)
    check_status(status)


cpdef dtrttp(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ap):
    """See `cublasDtrttp`."""
    with nogil:
        status = cublasDtrttp(<Handle>handle, <_FillMode>uplo, n, <const double*>a, lda, <double*>ap)
    check_status(status)


cpdef ctrttp(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ap):
    """See `cublasCtrttp`."""
    with nogil:
        status = cublasCtrttp(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>a, lda, <cuComplex*>ap)
    check_status(status)


cpdef ztrttp(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ap):
    """See `cublasZtrttp`."""
    with nogil:
        status = cublasZtrttp(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>ap)
    check_status(status)


cpdef int get_sm_count_target(intptr_t handle) except? -1:
    """See `cublasGetSmCountTarget`."""
    cdef int sm_count_target
    with nogil:
        status = cublasGetSmCountTarget(<Handle>handle, &sm_count_target)
    check_status(status)
    return sm_count_target


cpdef set_sm_count_target(intptr_t handle, int sm_count_target):
    """See `cublasSetSmCountTarget`."""
    with nogil:
        status = cublasSetSmCountTarget(<Handle>handle, sm_count_target)
    check_status(status)


cpdef str get_status_name(int status):
    """See `cublasGetStatusName`."""
    cdef bytes _output_
    _output_ = cublasGetStatusName(<_Status>status)
    return _output_.decode()


cpdef str get_status_string(int status):
    """See `cublasGetStatusString`."""
    cdef bytes _output_
    _output_ = cublasGetStatusString(<_Status>status)
    return _output_.decode()


cpdef sgemv_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t aarray, int lda, intptr_t xarray, int incx, intptr_t beta, intptr_t yarray, int incy, int batch_count):
    """See `cublasSgemvBatched`."""
    with nogil:
        status = cublasSgemvBatched(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float* const*>aarray, lda, <const float* const*>xarray, incx, <const float*>beta, <float* const*>yarray, incy, batch_count)
    check_status(status)


cpdef dgemv_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t aarray, int lda, intptr_t xarray, int incx, intptr_t beta, intptr_t yarray, int incy, int batch_count):
    """See `cublasDgemvBatched`."""
    with nogil:
        status = cublasDgemvBatched(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double* const*>aarray, lda, <const double* const*>xarray, incx, <const double*>beta, <double* const*>yarray, incy, batch_count)
    check_status(status)


cpdef cgemv_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t aarray, int lda, intptr_t xarray, int incx, intptr_t beta, intptr_t yarray, int incy, int batch_count):
    """See `cublasCgemvBatched`."""
    with nogil:
        status = cublasCgemvBatched(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>xarray, incx, <const cuComplex*>beta, <cuComplex* const*>yarray, incy, batch_count)
    check_status(status)


cpdef zgemv_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t aarray, int lda, intptr_t xarray, int incx, intptr_t beta, intptr_t yarray, int incy, int batch_count):
    """See `cublasZgemvBatched`."""
    with nogil:
        status = cublasZgemvBatched(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>aarray, lda, <const cuDoubleComplex* const*>xarray, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex* const*>yarray, incy, batch_count)
    check_status(status)


cpdef sgemv_strided_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t x, int incx, long long int strid_ex, intptr_t beta, intptr_t y, int incy, long long int stridey, int batch_count):
    """See `cublasSgemvStridedBatched`."""
    with nogil:
        status = cublasSgemvStridedBatched(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float*>a, lda, stride_a, <const float*>x, incx, strid_ex, <const float*>beta, <float*>y, incy, stridey, batch_count)
    check_status(status)


cpdef dgemv_strided_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t x, int incx, long long int strid_ex, intptr_t beta, intptr_t y, int incy, long long int stridey, int batch_count):
    """See `cublasDgemvStridedBatched`."""
    with nogil:
        status = cublasDgemvStridedBatched(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double*>a, lda, stride_a, <const double*>x, incx, strid_ex, <const double*>beta, <double*>y, incy, stridey, batch_count)
    check_status(status)


cpdef cgemv_strided_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t x, int incx, long long int strid_ex, intptr_t beta, intptr_t y, int incy, long long int stridey, int batch_count):
    """See `cublasCgemvStridedBatched`."""
    with nogil:
        status = cublasCgemvStridedBatched(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>x, incx, strid_ex, <const cuComplex*>beta, <cuComplex*>y, incy, stridey, batch_count)
    check_status(status)


cpdef zgemv_strided_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t x, int incx, long long int strid_ex, intptr_t beta, intptr_t y, int incy, long long int stridey, int batch_count):
    """See `cublasZgemvStridedBatched`."""
    with nogil:
        status = cublasZgemvStridedBatched(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, stride_a, <const cuDoubleComplex*>x, incx, strid_ex, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy, stridey, batch_count)
    check_status(status)


cpdef set_vector_64(int64_t n, int64_t elem_size, intptr_t x, int64_t incx, intptr_t device_ptr, int64_t incy):
    """See `cublasSetVector_64`."""
    with nogil:
        status = cublasSetVector_64(n, elem_size, <const void*>x, incx, <void*>device_ptr, incy)
    check_status(status)


cpdef get_vector_64(int64_t n, int64_t elem_size, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasGetVector_64`."""
    with nogil:
        status = cublasGetVector_64(n, elem_size, <const void*>x, incx, <void*>y, incy)
    check_status(status)


cpdef set_matrix_64(int64_t rows, int64_t cols, int64_t elem_size, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasSetMatrix_64`."""
    with nogil:
        status = cublasSetMatrix_64(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb)
    check_status(status)


cpdef get_matrix_64(int64_t rows, int64_t cols, int64_t elem_size, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasGetMatrix_64`."""
    with nogil:
        status = cublasGetMatrix_64(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb)
    check_status(status)


cpdef set_vector_async_64(int64_t n, int64_t elem_size, intptr_t host_ptr, int64_t incx, intptr_t device_ptr, int64_t incy, intptr_t stream):
    """See `cublasSetVectorAsync_64`."""
    with nogil:
        status = cublasSetVectorAsync_64(n, elem_size, <const void*>host_ptr, incx, <void*>device_ptr, incy, <Stream>stream)
    check_status(status)


cpdef get_vector_async_64(int64_t n, int64_t elem_size, intptr_t device_ptr, int64_t incx, intptr_t host_ptr, int64_t incy, intptr_t stream):
    """See `cublasGetVectorAsync_64`."""
    with nogil:
        status = cublasGetVectorAsync_64(n, elem_size, <const void*>device_ptr, incx, <void*>host_ptr, incy, <Stream>stream)
    check_status(status)


cpdef set_matrix_async_64(int64_t rows, int64_t cols, int64_t elem_size, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t stream):
    """See `cublasSetMatrixAsync_64`."""
    with nogil:
        status = cublasSetMatrixAsync_64(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb, <Stream>stream)
    check_status(status)


cpdef get_matrix_async_64(int64_t rows, int64_t cols, int64_t elem_size, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t stream):
    """See `cublasGetMatrixAsync_64`."""
    with nogil:
        status = cublasGetMatrixAsync_64(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb, <Stream>stream)
    check_status(status)


cpdef nrm2ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t result, int result_type, int execution_type):
    """See `cublasNrm2Ex_64`."""
    with nogil:
        status = cublasNrm2Ex_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(status)


cpdef snrm2_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasSnrm2_64`."""
    with nogil:
        status = cublasSnrm2_64(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(status)


cpdef dnrm2_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasDnrm2_64`."""
    with nogil:
        status = cublasDnrm2_64(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(status)


cpdef scnrm2_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasScnrm2_64`."""
    with nogil:
        status = cublasScnrm2_64(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(status)


cpdef dznrm2_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasDznrm2_64`."""
    with nogil:
        status = cublasDznrm2_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(status)


cpdef dot_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy, intptr_t result, int result_type, int execution_type):
    """See `cublasDotEx_64`."""
    with nogil:
        status = cublasDotEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <const void*>y, <DataType>y_type, incy, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(status)


cpdef dotc_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy, intptr_t result, int result_type, int execution_type):
    """See `cublasDotcEx_64`."""
    with nogil:
        status = cublasDotcEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <const void*>y, <DataType>y_type, incy, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(status)


cpdef sdot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasSdot_64`."""
    with nogil:
        status = cublasSdot_64(<Handle>handle, n, <const float*>x, incx, <const float*>y, incy, <float*>result)
    check_status(status)


cpdef ddot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasDdot_64`."""
    with nogil:
        status = cublasDdot_64(<Handle>handle, n, <const double*>x, incx, <const double*>y, incy, <double*>result)
    check_status(status)


cpdef cdotu_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasCdotu_64`."""
    with nogil:
        status = cublasCdotu_64(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(status)


cpdef cdotc_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasCdotc_64`."""
    with nogil:
        status = cublasCdotc_64(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(status)


cpdef zdotu_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasZdotu_64`."""
    with nogil:
        status = cublasZdotu_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)


cpdef zdotc_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasZdotc_64`."""
    with nogil:
        status = cublasZdotc_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)


cpdef scal_ex_64(intptr_t handle, int64_t n, intptr_t alpha, int alpha_type, intptr_t x, int x_type, int64_t incx, int execution_type):
    """See `cublasScalEx_64`."""
    with nogil:
        status = cublasScalEx_64(<Handle>handle, n, <const void*>alpha, <DataType>alpha_type, <void*>x, <DataType>x_type, incx, <DataType>execution_type)
    check_status(status)


cpdef sscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasSscal_64`."""
    with nogil:
        status = cublasSscal_64(<Handle>handle, n, <const float*>alpha, <float*>x, incx)
    check_status(status)


cpdef dscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasDscal_64`."""
    with nogil:
        status = cublasDscal_64(<Handle>handle, n, <const double*>alpha, <double*>x, incx)
    check_status(status)


cpdef cscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasCscal_64`."""
    with nogil:
        status = cublasCscal_64(<Handle>handle, n, <const cuComplex*>alpha, <cuComplex*>x, incx)
    check_status(status)


cpdef csscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasCsscal_64`."""
    with nogil:
        status = cublasCsscal_64(<Handle>handle, n, <const float*>alpha, <cuComplex*>x, incx)
    check_status(status)


cpdef zscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasZscal_64`."""
    with nogil:
        status = cublasZscal_64(<Handle>handle, n, <const cuDoubleComplex*>alpha, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef zdscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasZdscal_64`."""
    with nogil:
        status = cublasZdscal_64(<Handle>handle, n, <const double*>alpha, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef axpy_ex_64(intptr_t handle, int64_t n, intptr_t alpha, int alpha_type, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy, int executiontype):
    """See `cublasAxpyEx_64`."""
    with nogil:
        status = cublasAxpyEx_64(<Handle>handle, n, <const void*>alpha, <DataType>alpha_type, <const void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <DataType>executiontype)
    check_status(status)


cpdef saxpy_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasSaxpy_64`."""
    with nogil:
        status = cublasSaxpy_64(<Handle>handle, n, <const float*>alpha, <const float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef daxpy_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasDaxpy_64`."""
    with nogil:
        status = cublasDaxpy_64(<Handle>handle, n, <const double*>alpha, <const double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef caxpy_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasCaxpy_64`."""
    with nogil:
        status = cublasCaxpy_64(<Handle>handle, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(status)


cpdef zaxpy_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasZaxpy_64`."""
    with nogil:
        status = cublasZaxpy_64(<Handle>handle, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef copy_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy):
    """See `cublasCopyEx_64`."""
    with nogil:
        status = cublasCopyEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy)
    check_status(status)


cpdef scopy_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasScopy_64`."""
    with nogil:
        status = cublasScopy_64(<Handle>handle, n, <const float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef dcopy_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasDcopy_64`."""
    with nogil:
        status = cublasDcopy_64(<Handle>handle, n, <const double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef ccopy_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasCcopy_64`."""
    with nogil:
        status = cublasCcopy_64(<Handle>handle, n, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(status)


cpdef zcopy_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasZcopy_64`."""
    with nogil:
        status = cublasZcopy_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sswap_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasSswap_64`."""
    with nogil:
        status = cublasSswap_64(<Handle>handle, n, <float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef dswap_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasDswap_64`."""
    with nogil:
        status = cublasDswap_64(<Handle>handle, n, <double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef cswap_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasCswap_64`."""
    with nogil:
        status = cublasCswap_64(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(status)


cpdef zswap_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasZswap_64`."""
    with nogil:
        status = cublasZswap_64(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef swap_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy):
    """See `cublasSwapEx_64`."""
    with nogil:
        status = cublasSwapEx_64(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy)
    check_status(status)


cpdef isamax_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIsamax_64`."""
    with nogil:
        status = cublasIsamax_64(<Handle>handle, n, <const float*>x, incx, <int64_t*>result)
    check_status(status)


cpdef idamax_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIdamax_64`."""
    with nogil:
        status = cublasIdamax_64(<Handle>handle, n, <const double*>x, incx, <int64_t*>result)
    check_status(status)


cpdef icamax_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIcamax_64`."""
    with nogil:
        status = cublasIcamax_64(<Handle>handle, n, <const cuComplex*>x, incx, <int64_t*>result)
    check_status(status)


cpdef izamax_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIzamax_64`."""
    with nogil:
        status = cublasIzamax_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int64_t*>result)
    check_status(status)


cpdef iamax_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t result):
    """See `cublasIamaxEx_64`."""
    with nogil:
        status = cublasIamaxEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <int64_t*>result)
    check_status(status)


cpdef isamin_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIsamin_64`."""
    with nogil:
        status = cublasIsamin_64(<Handle>handle, n, <const float*>x, incx, <int64_t*>result)
    check_status(status)


cpdef idamin_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIdamin_64`."""
    with nogil:
        status = cublasIdamin_64(<Handle>handle, n, <const double*>x, incx, <int64_t*>result)
    check_status(status)


cpdef icamin_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIcamin_64`."""
    with nogil:
        status = cublasIcamin_64(<Handle>handle, n, <const cuComplex*>x, incx, <int64_t*>result)
    check_status(status)


cpdef izamin_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIzamin_64`."""
    with nogil:
        status = cublasIzamin_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int64_t*>result)
    check_status(status)


cpdef iamin_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t result):
    """See `cublasIaminEx_64`."""
    with nogil:
        status = cublasIaminEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <int64_t*>result)
    check_status(status)


cpdef asum_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t result, int result_type, int executiontype):
    """See `cublasAsumEx_64`."""
    with nogil:
        status = cublasAsumEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>result, <DataType>result_type, <DataType>executiontype)
    check_status(status)


cpdef sasum_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasSasum_64`."""
    with nogil:
        status = cublasSasum_64(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(status)


cpdef dasum_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasDasum_64`."""
    with nogil:
        status = cublasDasum_64(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(status)


cpdef scasum_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasScasum_64`."""
    with nogil:
        status = cublasScasum_64(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(status)


cpdef dzasum_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasDzasum_64`."""
    with nogil:
        status = cublasDzasum_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(status)


cpdef srot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasSrot_64`."""
    with nogil:
        status = cublasSrot_64(<Handle>handle, n, <float*>x, incx, <float*>y, incy, <const float*>c, <const float*>s)
    check_status(status)


cpdef drot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasDrot_64`."""
    with nogil:
        status = cublasDrot_64(<Handle>handle, n, <double*>x, incx, <double*>y, incy, <const double*>c, <const double*>s)
    check_status(status)


cpdef crot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasCrot_64`."""
    with nogil:
        status = cublasCrot_64(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy, <const float*>c, <const cuComplex*>s)
    check_status(status)


cpdef csrot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasCsrot_64`."""
    with nogil:
        status = cublasCsrot_64(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy, <const float*>c, <const float*>s)
    check_status(status)


cpdef zrot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasZrot_64`."""
    with nogil:
        status = cublasZrot_64(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy, <const double*>c, <const cuDoubleComplex*>s)
    check_status(status)


cpdef zdrot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasZdrot_64`."""
    with nogil:
        status = cublasZdrot_64(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy, <const double*>c, <const double*>s)
    check_status(status)


cpdef rot_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy, intptr_t c, intptr_t s, int cs_type, int executiontype):
    """See `cublasRotEx_64`."""
    with nogil:
        status = cublasRotEx_64(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <const void*>c, <const void*>s, <DataType>cs_type, <DataType>executiontype)
    check_status(status)


cpdef srotm_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t param):
    """See `cublasSrotm_64`."""
    with nogil:
        status = cublasSrotm_64(<Handle>handle, n, <float*>x, incx, <float*>y, incy, <const float*>param)
    check_status(status)


cpdef drotm_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t param):
    """See `cublasDrotm_64`."""
    with nogil:
        status = cublasDrotm_64(<Handle>handle, n, <double*>x, incx, <double*>y, incy, <const double*>param)
    check_status(status)


cpdef rotm_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy, intptr_t param, int param_type, int executiontype):
    """See `cublasRotmEx_64`."""
    with nogil:
        status = cublasRotmEx_64(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <const void*>param, <DataType>param_type, <DataType>executiontype)
    check_status(status)


cpdef sgemv_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasSgemv_64`."""
    with nogil:
        status = cublasSgemv_64(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dgemv_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasDgemv_64`."""
    with nogil:
        status = cublasDgemv_64(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef cgemv_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasCgemv_64`."""
    with nogil:
        status = cublasCgemv_64(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zgemv_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZgemv_64`."""
    with nogil:
        status = cublasZgemv_64(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sgbmv_64(intptr_t handle, int trans, int64_t m, int64_t n, int64_t kl, int64_t ku, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasSgbmv_64`."""
    with nogil:
        status = cublasSgbmv_64(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dgbmv_64(intptr_t handle, int trans, int64_t m, int64_t n, int64_t kl, int64_t ku, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasDgbmv_64`."""
    with nogil:
        status = cublasDgbmv_64(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef cgbmv_64(intptr_t handle, int trans, int64_t m, int64_t n, int64_t kl, int64_t ku, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasCgbmv_64`."""
    with nogil:
        status = cublasCgbmv_64(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zgbmv_64(intptr_t handle, int trans, int64_t m, int64_t n, int64_t kl, int64_t ku, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZgbmv_64`."""
    with nogil:
        status = cublasZgbmv_64(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef strmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasStrmv_64`."""
    with nogil:
        status = cublasStrmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>a, lda, <float*>x, incx)
    check_status(status)


cpdef dtrmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasDtrmv_64`."""
    with nogil:
        status = cublasDtrmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>a, lda, <double*>x, incx)
    check_status(status)


cpdef ctrmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasCtrmv_64`."""
    with nogil:
        status = cublasCtrmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztrmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasZtrmv_64`."""
    with nogil:
        status = cublasZtrmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stbmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasStbmv_64`."""
    with nogil:
        status = cublasStbmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const float*>a, lda, <float*>x, incx)
    check_status(status)


cpdef dtbmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasDtbmv_64`."""
    with nogil:
        status = cublasDtbmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const double*>a, lda, <double*>x, incx)
    check_status(status)


cpdef ctbmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasCtbmv_64`."""
    with nogil:
        status = cublasCtbmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztbmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasZtbmv_64`."""
    with nogil:
        status = cublasZtbmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stpmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasStpmv_64`."""
    with nogil:
        status = cublasStpmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>ap, <float*>x, incx)
    check_status(status)


cpdef dtpmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasDtpmv_64`."""
    with nogil:
        status = cublasDtpmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>ap, <double*>x, incx)
    check_status(status)


cpdef ctpmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasCtpmv_64`."""
    with nogil:
        status = cublasCtpmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>ap, <cuComplex*>x, incx)
    check_status(status)


cpdef ztpmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasZtpmv_64`."""
    with nogil:
        status = cublasZtpmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>ap, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef strsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasStrsv_64`."""
    with nogil:
        status = cublasStrsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>a, lda, <float*>x, incx)
    check_status(status)


cpdef dtrsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasDtrsv_64`."""
    with nogil:
        status = cublasDtrsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>a, lda, <double*>x, incx)
    check_status(status)


cpdef ctrsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasCtrsv_64`."""
    with nogil:
        status = cublasCtrsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztrsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasZtrsv_64`."""
    with nogil:
        status = cublasZtrsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stpsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasStpsv_64`."""
    with nogil:
        status = cublasStpsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>ap, <float*>x, incx)
    check_status(status)


cpdef dtpsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasDtpsv_64`."""
    with nogil:
        status = cublasDtpsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>ap, <double*>x, incx)
    check_status(status)


cpdef ctpsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasCtpsv_64`."""
    with nogil:
        status = cublasCtpsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>ap, <cuComplex*>x, incx)
    check_status(status)


cpdef ztpsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasZtpsv_64`."""
    with nogil:
        status = cublasZtpsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>ap, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stbsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasStbsv_64`."""
    with nogil:
        status = cublasStbsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const float*>a, lda, <float*>x, incx)
    check_status(status)


cpdef dtbsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasDtbsv_64`."""
    with nogil:
        status = cublasDtbsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const double*>a, lda, <double*>x, incx)
    check_status(status)


cpdef ctbsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasCtbsv_64`."""
    with nogil:
        status = cublasCtbsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztbsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasZtbsv_64`."""
    with nogil:
        status = cublasZtbsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef ssymv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasSsymv_64`."""
    with nogil:
        status = cublasSsymv_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dsymv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasDsymv_64`."""
    with nogil:
        status = cublasDsymv_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef csymv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasCsymv_64`."""
    with nogil:
        status = cublasCsymv_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zsymv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZsymv_64`."""
    with nogil:
        status = cublasZsymv_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef chemv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasChemv_64`."""
    with nogil:
        status = cublasChemv_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zhemv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZhemv_64`."""
    with nogil:
        status = cublasZhemv_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef ssbmv_64(intptr_t handle, int uplo, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasSsbmv_64`."""
    with nogil:
        status = cublasSsbmv_64(<Handle>handle, <_FillMode>uplo, n, k, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dsbmv_64(intptr_t handle, int uplo, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasDsbmv_64`."""
    with nogil:
        status = cublasDsbmv_64(<Handle>handle, <_FillMode>uplo, n, k, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef chbmv_64(intptr_t handle, int uplo, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasChbmv_64`."""
    with nogil:
        status = cublasChbmv_64(<Handle>handle, <_FillMode>uplo, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zhbmv_64(intptr_t handle, int uplo, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZhbmv_64`."""
    with nogil:
        status = cublasZhbmv_64(<Handle>handle, <_FillMode>uplo, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sspmv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t ap, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasSspmv_64`."""
    with nogil:
        status = cublasSspmv_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>ap, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dspmv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t ap, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasDspmv_64`."""
    with nogil:
        status = cublasDspmv_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>ap, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef chpmv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t ap, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasChpmv_64`."""
    with nogil:
        status = cublasChpmv_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>ap, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zhpmv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t ap, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZhpmv_64`."""
    with nogil:
        status = cublasZhpmv_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>ap, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sger_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasSger_64`."""
    with nogil:
        status = cublasSger_64(<Handle>handle, m, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>a, lda)
    check_status(status)


cpdef dger_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasDger_64`."""
    with nogil:
        status = cublasDger_64(<Handle>handle, m, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>a, lda)
    check_status(status)


cpdef cgeru_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasCgeru_64`."""
    with nogil:
        status = cublasCgeru_64(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(status)


cpdef cgerc_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasCgerc_64`."""
    with nogil:
        status = cublasCgerc_64(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(status)


cpdef zgeru_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasZgeru_64`."""
    with nogil:
        status = cublasZgeru_64(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef zgerc_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasZgerc_64`."""
    with nogil:
        status = cublasZgerc_64(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef ssyr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasSsyr_64`."""
    with nogil:
        status = cublasSsyr_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <float*>a, lda)
    check_status(status)


cpdef dsyr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasDsyr_64`."""
    with nogil:
        status = cublasDsyr_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <double*>a, lda)
    check_status(status)


cpdef csyr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasCsyr_64`."""
    with nogil:
        status = cublasCsyr_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>a, lda)
    check_status(status)


cpdef zsyr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasZsyr_64`."""
    with nogil:
        status = cublasZsyr_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef cher_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasCher_64`."""
    with nogil:
        status = cublasCher_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const cuComplex*>x, incx, <cuComplex*>a, lda)
    check_status(status)


cpdef zher_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasZher_64`."""
    with nogil:
        status = cublasZher_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef sspr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t ap):
    """See `cublasSspr_64`."""
    with nogil:
        status = cublasSspr_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <float*>ap)
    check_status(status)


cpdef dspr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t ap):
    """See `cublasDspr_64`."""
    with nogil:
        status = cublasDspr_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <double*>ap)
    check_status(status)


cpdef chpr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t ap):
    """See `cublasChpr_64`."""
    with nogil:
        status = cublasChpr_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const cuComplex*>x, incx, <cuComplex*>ap)
    check_status(status)


cpdef zhpr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t ap):
    """See `cublasZhpr_64`."""
    with nogil:
        status = cublasZhpr_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>ap)
    check_status(status)


cpdef ssyr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasSsyr2_64`."""
    with nogil:
        status = cublasSsyr2_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>a, lda)
    check_status(status)


cpdef dsyr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasDsyr2_64`."""
    with nogil:
        status = cublasDsyr2_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>a, lda)
    check_status(status)


cpdef csyr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasCsyr2_64`."""
    with nogil:
        status = cublasCsyr2_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(status)


cpdef zsyr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasZsyr2_64`."""
    with nogil:
        status = cublasZsyr2_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef cher2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasCher2_64`."""
    with nogil:
        status = cublasCher2_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(status)


cpdef zher2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasZher2_64`."""
    with nogil:
        status = cublasZher2_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(status)


cpdef sspr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t ap):
    """See `cublasSspr2_64`."""
    with nogil:
        status = cublasSspr2_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>ap)
    check_status(status)


cpdef dspr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t ap):
    """See `cublasDspr2_64`."""
    with nogil:
        status = cublasDspr2_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>ap)
    check_status(status)


cpdef chpr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t ap):
    """See `cublasChpr2_64`."""
    with nogil:
        status = cublasChpr2_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>ap)
    check_status(status)


cpdef zhpr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t ap):
    """See `cublasZhpr2_64`."""
    with nogil:
        status = cublasZhpr2_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>ap)
    check_status(status)


cpdef sgemv_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t xarray, int64_t incx, intptr_t beta, intptr_t yarray, int64_t incy, int64_t batch_count):
    """See `cublasSgemvBatched_64`."""
    with nogil:
        status = cublasSgemvBatched_64(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float* const*>aarray, lda, <const float* const*>xarray, incx, <const float*>beta, <float* const*>yarray, incy, batch_count)
    check_status(status)


cpdef dgemv_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t xarray, int64_t incx, intptr_t beta, intptr_t yarray, int64_t incy, int64_t batch_count):
    """See `cublasDgemvBatched_64`."""
    with nogil:
        status = cublasDgemvBatched_64(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double* const*>aarray, lda, <const double* const*>xarray, incx, <const double*>beta, <double* const*>yarray, incy, batch_count)
    check_status(status)


cpdef cgemv_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t xarray, int64_t incx, intptr_t beta, intptr_t yarray, int64_t incy, int64_t batch_count):
    """See `cublasCgemvBatched_64`."""
    with nogil:
        status = cublasCgemvBatched_64(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>xarray, incx, <const cuComplex*>beta, <cuComplex* const*>yarray, incy, batch_count)
    check_status(status)


cpdef zgemv_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t xarray, int64_t incx, intptr_t beta, intptr_t yarray, int64_t incy, int64_t batch_count):
    """See `cublasZgemvBatched_64`."""
    with nogil:
        status = cublasZgemvBatched_64(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>aarray, lda, <const cuDoubleComplex* const*>xarray, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex* const*>yarray, incy, batch_count)
    check_status(status)


cpdef sgemv_strided_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t x, int64_t incx, long long int strid_ex, intptr_t beta, intptr_t y, int64_t incy, long long int stridey, int64_t batch_count):
    """See `cublasSgemvStridedBatched_64`."""
    with nogil:
        status = cublasSgemvStridedBatched_64(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float*>a, lda, stride_a, <const float*>x, incx, strid_ex, <const float*>beta, <float*>y, incy, stridey, batch_count)
    check_status(status)


cpdef dgemv_strided_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t x, int64_t incx, long long int strid_ex, intptr_t beta, intptr_t y, int64_t incy, long long int stridey, int64_t batch_count):
    """See `cublasDgemvStridedBatched_64`."""
    with nogil:
        status = cublasDgemvStridedBatched_64(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double*>a, lda, stride_a, <const double*>x, incx, strid_ex, <const double*>beta, <double*>y, incy, stridey, batch_count)
    check_status(status)


cpdef cgemv_strided_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t x, int64_t incx, long long int strid_ex, intptr_t beta, intptr_t y, int64_t incy, long long int stridey, int64_t batch_count):
    """See `cublasCgemvStridedBatched_64`."""
    with nogil:
        status = cublasCgemvStridedBatched_64(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>x, incx, strid_ex, <const cuComplex*>beta, <cuComplex*>y, incy, stridey, batch_count)
    check_status(status)


cpdef zgemv_strided_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t x, int64_t incx, long long int strid_ex, intptr_t beta, intptr_t y, int64_t incy, long long int stridey, int64_t batch_count):
    """See `cublasZgemvStridedBatched_64`."""
    with nogil:
        status = cublasZgemvStridedBatched_64(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, stride_a, <const cuDoubleComplex*>x, incx, strid_ex, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy, stridey, batch_count)
    check_status(status)


cpdef sgemm_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasSgemm_64`."""
    with nogil:
        status = cublasSgemm_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dgemm_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasDgemm_64`."""
    with nogil:
        status = cublasDgemm_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef cgemm_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCgemm_64`."""
    with nogil:
        status = cublasCgemm_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef cgemm3m_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCgemm3m_64`."""
    with nogil:
        status = cublasCgemm3m_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef cgemm3m_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t b, int btype, int64_t ldb, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCgemm3mEx_64`."""
    with nogil:
        status = cublasCgemm3mEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef zgemm_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZgemm_64`."""
    with nogil:
        status = cublasZgemm_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef zgemm3m_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZgemm3m_64`."""
    with nogil:
        status = cublasZgemm3m_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef sgemm_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t b, int btype, int64_t ldb, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasSgemmEx_64`."""
    with nogil:
        status = cublasSgemmEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef gemm_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t b, int btype, int64_t ldb, intptr_t beta, intptr_t c, int ctype, int64_t ldc, int compute_type, int algo):
    """See `cublasGemmEx_64`."""
    with nogil:
        status = cublasGemmEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const void*>beta, <void*>c, <DataType>ctype, ldc, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(status)


cpdef cgemm_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t b, int btype, int64_t ldb, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCgemmEx_64`."""
    with nogil:
        status = cublasCgemmEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef ssyrk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasSsyrk_64`."""
    with nogil:
        status = cublasSsyrk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dsyrk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasDsyrk_64`."""
    with nogil:
        status = cublasDsyrk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef csyrk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCsyrk_64`."""
    with nogil:
        status = cublasCsyrk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zsyrk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZsyrk_64`."""
    with nogil:
        status = cublasZsyrk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef csyrk_ex_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCsyrkEx_64`."""
    with nogil:
        status = cublasCsyrkEx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef csyrk3m_ex_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCsyrk3mEx_64`."""
    with nogil:
        status = cublasCsyrk3mEx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef cherk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCherk_64`."""
    with nogil:
        status = cublasCherk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const cuComplex*>a, lda, <const float*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zherk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZherk_64`."""
    with nogil:
        status = cublasZherk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const cuDoubleComplex*>a, lda, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef cherk_ex_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCherkEx_64`."""
    with nogil:
        status = cublasCherkEx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef cherk3m_ex_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCherk3mEx_64`."""
    with nogil:
        status = cublasCherk3mEx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(status)


cpdef ssyr2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasSsyr2k_64`."""
    with nogil:
        status = cublasSsyr2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dsyr2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasDsyr2k_64`."""
    with nogil:
        status = cublasDsyr2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef csyr2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCsyr2k_64`."""
    with nogil:
        status = cublasCsyr2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zsyr2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZsyr2k_64`."""
    with nogil:
        status = cublasZsyr2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef cher2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCher2k_64`."""
    with nogil:
        status = cublasCher2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const float*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zher2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZher2k_64`."""
    with nogil:
        status = cublasZher2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef ssyrkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasSsyrkx_64`."""
    with nogil:
        status = cublasSsyrkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dsyrkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasDsyrkx_64`."""
    with nogil:
        status = cublasDsyrkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef csyrkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCsyrkx_64`."""
    with nogil:
        status = cublasCsyrkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zsyrkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZsyrkx_64`."""
    with nogil:
        status = cublasZsyrkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef cherkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCherkx_64`."""
    with nogil:
        status = cublasCherkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const float*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zherkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZherkx_64`."""
    with nogil:
        status = cublasZherkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef ssymm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasSsymm_64`."""
    with nogil:
        status = cublasSsymm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(status)


cpdef dsymm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasDsymm_64`."""
    with nogil:
        status = cublasDsymm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(status)


cpdef csymm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCsymm_64`."""
    with nogil:
        status = cublasCsymm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zsymm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZsymm_64`."""
    with nogil:
        status = cublasZsymm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef chemm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasChemm_64`."""
    with nogil:
        status = cublasChemm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(status)


cpdef zhemm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZhemm_64`."""
    with nogil:
        status = cublasZhemm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef strsm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasStrsm_64`."""
    with nogil:
        status = cublasStrsm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float*>a, lda, <float*>b, ldb)
    check_status(status)


cpdef dtrsm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasDtrsm_64`."""
    with nogil:
        status = cublasDtrsm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double*>a, lda, <double*>b, ldb)
    check_status(status)


cpdef ctrsm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasCtrsm_64`."""
    with nogil:
        status = cublasCtrsm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <cuComplex*>b, ldb)
    check_status(status)


cpdef ztrsm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasZtrsm_64`."""
    with nogil:
        status = cublasZtrsm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>b, ldb)
    check_status(status)


cpdef strmm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasStrmm_64`."""
    with nogil:
        status = cublasStrmm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <float*>c, ldc)
    check_status(status)


cpdef dtrmm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasDtrmm_64`."""
    with nogil:
        status = cublasDtrmm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <double*>c, ldc)
    check_status(status)


cpdef ctrmm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasCtrmm_64`."""
    with nogil:
        status = cublasCtrmm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <cuComplex*>c, ldc)
    check_status(status)


cpdef ztrmm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasZtrmm_64`."""
    with nogil:
        status = cublasZtrmm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef sgemm_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t barray, int64_t ldb, intptr_t beta, intptr_t carray, int64_t ldc, int64_t batch_count):
    """See `cublasSgemmBatched_64`."""
    with nogil:
        status = cublasSgemmBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float* const*>aarray, lda, <const float* const*>barray, ldb, <const float*>beta, <float* const*>carray, ldc, batch_count)
    check_status(status)


cpdef dgemm_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t barray, int64_t ldb, intptr_t beta, intptr_t carray, int64_t ldc, int64_t batch_count):
    """See `cublasDgemmBatched_64`."""
    with nogil:
        status = cublasDgemmBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double* const*>aarray, lda, <const double* const*>barray, ldb, <const double*>beta, <double* const*>carray, ldc, batch_count)
    check_status(status)


cpdef cgemm_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t barray, int64_t ldb, intptr_t beta, intptr_t carray, int64_t ldc, int64_t batch_count):
    """See `cublasCgemmBatched_64`."""
    with nogil:
        status = cublasCgemmBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>barray, ldb, <const cuComplex*>beta, <cuComplex* const*>carray, ldc, batch_count)
    check_status(status)


cpdef cgemm3m_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t barray, int64_t ldb, intptr_t beta, intptr_t carray, int64_t ldc, int64_t batch_count):
    """See `cublasCgemm3mBatched_64`."""
    with nogil:
        status = cublasCgemm3mBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>barray, ldb, <const cuComplex*>beta, <cuComplex* const*>carray, ldc, batch_count)
    check_status(status)


cpdef zgemm_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t barray, int64_t ldb, intptr_t beta, intptr_t carray, int64_t ldc, int64_t batch_count):
    """See `cublasZgemmBatched_64`."""
    with nogil:
        status = cublasZgemmBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>aarray, lda, <const cuDoubleComplex* const*>barray, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex* const*>carray, ldc, batch_count)
    check_status(status)


cpdef sgemm_strided_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t b, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int64_t ldc, long long int stride_c, int64_t batch_count):
    """See `cublasSgemmStridedBatched_64`."""
    with nogil:
        status = cublasSgemmStridedBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float*>a, lda, stride_a, <const float*>b, ldb, stride_b, <const float*>beta, <float*>c, ldc, stride_c, batch_count)
    check_status(status)


cpdef dgemm_strided_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t b, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int64_t ldc, long long int stride_c, int64_t batch_count):
    """See `cublasDgemmStridedBatched_64`."""
    with nogil:
        status = cublasDgemmStridedBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double*>a, lda, stride_a, <const double*>b, ldb, stride_b, <const double*>beta, <double*>c, ldc, stride_c, batch_count)
    check_status(status)


cpdef cgemm_strided_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t b, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int64_t ldc, long long int stride_c, int64_t batch_count):
    """See `cublasCgemmStridedBatched_64`."""
    with nogil:
        status = cublasCgemmStridedBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>b, ldb, stride_b, <const cuComplex*>beta, <cuComplex*>c, ldc, stride_c, batch_count)
    check_status(status)


cpdef cgemm3m_strided_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t b, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int64_t ldc, long long int stride_c, int64_t batch_count):
    """See `cublasCgemm3mStridedBatched_64`."""
    with nogil:
        status = cublasCgemm3mStridedBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>b, ldb, stride_b, <const cuComplex*>beta, <cuComplex*>c, ldc, stride_c, batch_count)
    check_status(status)


cpdef zgemm_strided_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t b, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int64_t ldc, long long int stride_c, int64_t batch_count):
    """See `cublasZgemmStridedBatched_64`."""
    with nogil:
        status = cublasZgemmStridedBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, stride_a, <const cuDoubleComplex*>b, ldb, stride_b, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc, stride_c, batch_count)
    check_status(status)


cpdef gemm_batched_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int atype, int64_t lda, intptr_t barray, int btype, int64_t ldb, intptr_t beta, intptr_t carray, int ctype, int64_t ldc, int64_t batch_count, int compute_type, int algo):
    """See `cublasGemmBatchedEx_64`."""
    with nogil:
        status = cublasGemmBatchedEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void* const*>aarray, <DataType>atype, lda, <const void* const*>barray, <DataType>btype, ldb, <const void*>beta, <void* const*>carray, <DataType>ctype, ldc, batch_count, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(status)


cpdef gemm_strided_batched_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, long long int stride_a, intptr_t b, int btype, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int ctype, int64_t ldc, long long int stride_c, int64_t batch_count, int compute_type, int algo):
    """See `cublasGemmStridedBatchedEx_64`."""
    with nogil:
        status = cublasGemmStridedBatchedEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void*>a, <DataType>atype, lda, stride_a, <const void*>b, <DataType>btype, ldb, stride_b, <const void*>beta, <void*>c, <DataType>ctype, ldc, stride_c, batch_count, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(status)


cpdef sgeam_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasSgeam_64`."""
    with nogil:
        status = cublasSgeam_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const float*>alpha, <const float*>a, lda, <const float*>beta, <const float*>b, ldb, <float*>c, ldc)
    check_status(status)


cpdef dgeam_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasDgeam_64`."""
    with nogil:
        status = cublasDgeam_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const double*>alpha, <const double*>a, lda, <const double*>beta, <const double*>b, ldb, <double*>c, ldc)
    check_status(status)


cpdef cgeam_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasCgeam_64`."""
    with nogil:
        status = cublasCgeam_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>beta, <const cuComplex*>b, ldb, <cuComplex*>c, ldc)
    check_status(status)


cpdef zgeam_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasZgeam_64`."""
    with nogil:
        status = cublasZgeam_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>beta, <const cuDoubleComplex*>b, ldb, <cuDoubleComplex*>c, ldc)
    check_status(status)


cpdef strsm_batched_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, int64_t batch_count):
    """See `cublasStrsmBatched_64`."""
    with nogil:
        status = cublasStrsmBatched_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float* const*>a, lda, <float* const*>b, ldb, batch_count)
    check_status(status)


cpdef dtrsm_batched_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, int64_t batch_count):
    """See `cublasDtrsmBatched_64`."""
    with nogil:
        status = cublasDtrsmBatched_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double* const*>a, lda, <double* const*>b, ldb, batch_count)
    check_status(status)


cpdef ctrsm_batched_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, int64_t batch_count):
    """See `cublasCtrsmBatched_64`."""
    with nogil:
        status = cublasCtrsmBatched_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex* const*>a, lda, <cuComplex* const*>b, ldb, batch_count)
    check_status(status)


cpdef ztrsm_batched_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, int64_t batch_count):
    """See `cublasZtrsmBatched_64`."""
    with nogil:
        status = cublasZtrsmBatched_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>a, lda, <cuDoubleComplex* const*>b, ldb, batch_count)
    check_status(status)


cpdef sdgmm_64(intptr_t handle, int mode, int64_t m, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t c, int64_t ldc):
    """See `cublasSdgmm_64`."""
    with nogil:
        status = cublasSdgmm_64(<Handle>handle, <_SideMode>mode, m, n, <const float*>a, lda, <const float*>x, incx, <float*>c, ldc)
    check_status(status)


cpdef ddgmm_64(intptr_t handle, int mode, int64_t m, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t c, int64_t ldc):
    """See `cublasDdgmm_64`."""
    with nogil:
        status = cublasDdgmm_64(<Handle>handle, <_SideMode>mode, m, n, <const double*>a, lda, <const double*>x, incx, <double*>c, ldc)
    check_status(status)


cpdef cdgmm_64(intptr_t handle, int mode, int64_t m, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t c, int64_t ldc):
    """See `cublasCdgmm_64`."""
    with nogil:
        status = cublasCdgmm_64(<Handle>handle, <_SideMode>mode, m, n, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <cuComplex*>c, ldc)
    check_status(status)


cpdef zdgmm_64(intptr_t handle, int mode, int64_t m, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t c, int64_t ldc):
    """See `cublasZdgmm_64`."""
    with nogil:
        status = cublasZdgmm_64(<Handle>handle, <_SideMode>mode, m, n, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>c, ldc)
    check_status(status)
