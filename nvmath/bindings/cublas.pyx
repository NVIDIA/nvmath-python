# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 13.2.0, generator version 0.3.1.dev1301+g7215ac36e. Do not modify it directly.

cimport cython  # NOQA
from libcpp.vector cimport vector

from ._internal.utils cimport (get_resource_ptr, get_resource_ptrs, nullable_unique_ptr,
                               get_buffer_pointer,)

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """
    See `cublasStatus_t`.
    """
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
    """
    See `cublasFillMode_t`.
    """
    LOWER = CUBLAS_FILL_MODE_LOWER
    UPPER = CUBLAS_FILL_MODE_UPPER
    FULL = CUBLAS_FILL_MODE_FULL

class DiagType(_IntEnum):
    """
    See `cublasDiagType_t`.
    """
    NON_UNIT = CUBLAS_DIAG_NON_UNIT
    UNIT = CUBLAS_DIAG_UNIT

class SideMode(_IntEnum):
    """
    See `cublasSideMode_t`.
    """
    LEFT = CUBLAS_SIDE_LEFT
    RIGHT = CUBLAS_SIDE_RIGHT

class Operation(_IntEnum):
    """
    See `cublasOperation_t`.
    """
    N = CUBLAS_OP_N
    T = CUBLAS_OP_T
    C = CUBLAS_OP_C
    HERMITAN = CUBLAS_OP_HERMITAN
    CONJG = CUBLAS_OP_CONJG

class PointerMode(_IntEnum):
    """
    See `cublasPointerMode_t`.
    """
    HOST = CUBLAS_POINTER_MODE_HOST
    DEVICE = CUBLAS_POINTER_MODE_DEVICE

class AtomicsMode(_IntEnum):
    """
    See `cublasAtomicsMode_t`.
    """
    NOT_ALLOWED = CUBLAS_ATOMICS_NOT_ALLOWED
    ALLOWED = CUBLAS_ATOMICS_ALLOWED

class GemmAlgo(_IntEnum):
    """
    See `cublasGemmAlgo_t`.
    """
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
    AUTOTUNE = CUBLAS_GEMM_AUTOTUNE

class Math(_IntEnum):
    """
    See `cublasMath_t`.
    """
    DEFAULT_MATH = CUBLAS_DEFAULT_MATH
    TENSOR_OP_MATH = CUBLAS_TENSOR_OP_MATH
    PEDANTIC_MATH = CUBLAS_PEDANTIC_MATH
    TF32_TENSOR_OP_MATH = CUBLAS_TF32_TENSOR_OP_MATH
    FP32_EMULATED_BF16X9_MATH = CUBLAS_FP32_EMULATED_BF16X9_MATH
    FP64_EMULATED_FIXEDPOINT_MATH = CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH
    DISALLOW_REDUCED_PRECISION_REDUCTION = CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION

class ComputeType(_IntEnum):
    """
    See `cublasComputeType_t`.
    """
    COMPUTE_16F = CUBLAS_COMPUTE_16F
    COMPUTE_16F_PEDANTIC = CUBLAS_COMPUTE_16F_PEDANTIC
    COMPUTE_32F = CUBLAS_COMPUTE_32F
    COMPUTE_32F_PEDANTIC = CUBLAS_COMPUTE_32F_PEDANTIC
    COMPUTE_32F_FAST_16F = CUBLAS_COMPUTE_32F_FAST_16F
    COMPUTE_32F_FAST_16BF = CUBLAS_COMPUTE_32F_FAST_16BF
    COMPUTE_32F_FAST_TF32 = CUBLAS_COMPUTE_32F_FAST_TF32
    COMPUTE_32F_EMULATED_16BFX9 = CUBLAS_COMPUTE_32F_EMULATED_16BFX9
    COMPUTE_64F = CUBLAS_COMPUTE_64F
    COMPUTE_64F_PEDANTIC = CUBLAS_COMPUTE_64F_PEDANTIC
    COMPUTE_64F_EMULATED_FIXEDPOINT = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT
    COMPUTE_32I = CUBLAS_COMPUTE_32I
    COMPUTE_32I_PEDANTIC = CUBLAS_COMPUTE_32I_PEDANTIC

class EmulationStrategy(_IntEnum):
    """
    See `cublasEmulationStrategy_t`.
    """
    DEFAULT = CUBLAS_EMULATION_STRATEGY_DEFAULT
    PERFORMANT = CUBLAS_EMULATION_STRATEGY_PERFORMANT
    EAGER = CUBLAS_EMULATION_STRATEGY_EAGER


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
# Convenience wrappers/adapters
###############################################################################


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create() except? 0:
    """See `cublasCreate`."""
    cdef Handle handle
    with nogil:
        __status__ = cublasCreate(&handle)
    check_status(__status__)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """See `cublasDestroy`."""
    with nogil:
        __status__ = cublasDestroy(<Handle>handle)
    check_status(__status__)


cpdef int get_version(intptr_t handle) except? -1:
    """See `cublasGetVersion`."""
    cdef int version
    with nogil:
        __status__ = cublasGetVersion(<Handle>handle, &version)
    check_status(__status__)
    return version


cpdef int get_property(int type) except? -1:
    """See `cublasGetProperty`."""
    cdef int value
    with nogil:
        __status__ = cublasGetProperty(<LibraryPropertyType>type, &value)
    check_status(__status__)
    return value


cpdef size_t get_cudart_version() except? 0:
    """See `cublasGetCudartVersion`."""
    return cublasGetCudartVersion()


cpdef set_workspace(intptr_t handle, intptr_t workspace, size_t workspace_size_in_bytes):
    """See `cublasSetWorkspace`."""
    with nogil:
        __status__ = cublasSetWorkspace(<Handle>handle, <void*>workspace, workspace_size_in_bytes)
    check_status(__status__)


cpdef set_stream(intptr_t handle, intptr_t stream_id):
    """See `cublasSetStream`."""
    with nogil:
        __status__ = cublasSetStream(<Handle>handle, <Stream>stream_id)
    check_status(__status__)


cpdef intptr_t get_stream(intptr_t handle) except? 0:
    """See `cublasGetStream`."""
    cdef Stream stream_id
    with nogil:
        __status__ = cublasGetStream(<Handle>handle, &stream_id)
    check_status(__status__)
    return <intptr_t>stream_id


cpdef int get_pointer_mode(intptr_t handle) except? -1:
    """See `cublasGetPointerMode`."""
    cdef _PointerMode mode
    with nogil:
        __status__ = cublasGetPointerMode(<Handle>handle, &mode)
    check_status(__status__)
    return <int>mode


cpdef set_pointer_mode(intptr_t handle, int mode):
    """See `cublasSetPointerMode`."""
    with nogil:
        __status__ = cublasSetPointerMode(<Handle>handle, <_PointerMode>mode)
    check_status(__status__)


cpdef int get_atomics_mode(intptr_t handle) except? -1:
    """See `cublasGetAtomicsMode`."""
    cdef _AtomicsMode mode
    with nogil:
        __status__ = cublasGetAtomicsMode(<Handle>handle, &mode)
    check_status(__status__)
    return <int>mode


cpdef set_atomics_mode(intptr_t handle, int mode):
    """See `cublasSetAtomicsMode`."""
    with nogil:
        __status__ = cublasSetAtomicsMode(<Handle>handle, <_AtomicsMode>mode)
    check_status(__status__)


cpdef int get_math_mode(intptr_t handle) except? -1:
    """See `cublasGetMathMode`."""
    cdef _Math mode
    with nogil:
        __status__ = cublasGetMathMode(<Handle>handle, &mode)
    check_status(__status__)
    return <int>mode


cpdef set_math_mode(intptr_t handle, int mode):
    """See `cublasSetMathMode`."""
    with nogil:
        __status__ = cublasSetMathMode(<Handle>handle, <_Math>mode)
    check_status(__status__)


cpdef logger_configure(int log_is_on, int log_to_std_out, int log_to_std_err, log_file_name):
    """See `cublasLoggerConfigure`."""
    if not isinstance(log_file_name, str):
        raise TypeError("log_file_name must be a Python str")
    cdef bytes _temp_log_file_name_ = (<str>log_file_name).encode()
    cdef char* _log_file_name_ = _temp_log_file_name_
    with nogil:
        __status__ = cublasLoggerConfigure(log_is_on, log_to_std_out, log_to_std_err, <const char*>_log_file_name_)
    check_status(__status__)


cpdef set_vector(int n, int elem_size, intptr_t x, int incx, intptr_t device_ptr, int incy):
    """See `cublasSetVector`."""
    with nogil:
        __status__ = cublasSetVector(n, elem_size, <const void*>x, incx, <void*>device_ptr, incy)
    check_status(__status__)


cpdef get_vector(int n, int elem_size, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasGetVector`."""
    with nogil:
        __status__ = cublasGetVector(n, elem_size, <const void*>x, incx, <void*>y, incy)
    check_status(__status__)


cpdef set_matrix(int rows, int cols, int elem_size, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasSetMatrix`."""
    with nogil:
        __status__ = cublasSetMatrix(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb)
    check_status(__status__)


cpdef get_matrix(int rows, int cols, int elem_size, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasGetMatrix`."""
    with nogil:
        __status__ = cublasGetMatrix(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb)
    check_status(__status__)


cpdef set_vector_async(int n, int elem_size, intptr_t host_ptr, int incx, intptr_t device_ptr, int incy, intptr_t stream):
    """See `cublasSetVectorAsync`."""
    with nogil:
        __status__ = cublasSetVectorAsync(n, elem_size, <const void*>host_ptr, incx, <void*>device_ptr, incy, <Stream>stream)
    check_status(__status__)


cpdef get_vector_async(int n, int elem_size, intptr_t device_ptr, int incx, intptr_t host_ptr, int incy, intptr_t stream):
    """See `cublasGetVectorAsync`."""
    with nogil:
        __status__ = cublasGetVectorAsync(n, elem_size, <const void*>device_ptr, incx, <void*>host_ptr, incy, <Stream>stream)
    check_status(__status__)


cpdef set_matrix_async(int rows, int cols, int elem_size, intptr_t a, int lda, intptr_t b, int ldb, intptr_t stream):
    """See `cublasSetMatrixAsync`."""
    with nogil:
        __status__ = cublasSetMatrixAsync(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb, <Stream>stream)
    check_status(__status__)


cpdef get_matrix_async(int rows, int cols, int elem_size, intptr_t a, int lda, intptr_t b, int ldb, intptr_t stream):
    """See `cublasGetMatrixAsync`."""
    with nogil:
        __status__ = cublasGetMatrixAsync(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb, <Stream>stream)
    check_status(__status__)


cpdef nrm2_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t result, int result_type, int execution_type):
    """See `cublasNrm2Ex`."""
    with nogil:
        __status__ = cublasNrm2Ex(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(__status__)


cpdef snrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasSnrm2`."""
    with nogil:
        __status__ = cublasSnrm2(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(__status__)


cpdef dnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasDnrm2`."""
    with nogil:
        __status__ = cublasDnrm2(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(__status__)


cpdef scnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasScnrm2`."""
    with nogil:
        __status__ = cublasScnrm2(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(__status__)


cpdef dznrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasDznrm2`."""
    with nogil:
        __status__ = cublasDznrm2(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(__status__)


cpdef dot_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy, intptr_t result, int result_type, int execution_type):
    """See `cublasDotEx`."""
    with nogil:
        __status__ = cublasDotEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <const void*>y, <DataType>y_type, incy, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(__status__)


cpdef dotc_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy, intptr_t result, int result_type, int execution_type):
    """See `cublasDotcEx`."""
    with nogil:
        __status__ = cublasDotcEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <const void*>y, <DataType>y_type, incy, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(__status__)


cpdef sdot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasSdot`."""
    with nogil:
        __status__ = cublasSdot(<Handle>handle, n, <const float*>x, incx, <const float*>y, incy, <float*>result)
    check_status(__status__)


cpdef ddot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasDdot`."""
    with nogil:
        __status__ = cublasDdot(<Handle>handle, n, <const double*>x, incx, <const double*>y, incy, <double*>result)
    check_status(__status__)


cpdef cdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasCdotu`."""
    with nogil:
        __status__ = cublasCdotu(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(__status__)


cpdef cdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasCdotc`."""
    with nogil:
        __status__ = cublasCdotc(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(__status__)


cpdef zdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasZdotu`."""
    with nogil:
        __status__ = cublasZdotu(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(__status__)


cpdef zdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    """See `cublasZdotc`."""
    with nogil:
        __status__ = cublasZdotc(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(__status__)


cpdef scal_ex(intptr_t handle, int n, intptr_t alpha, int alpha_type, intptr_t x, int x_type, int incx, int execution_type):
    """See `cublasScalEx`."""
    with nogil:
        __status__ = cublasScalEx(<Handle>handle, n, <const void*>alpha, <DataType>alpha_type, <void*>x, <DataType>x_type, incx, <DataType>execution_type)
    check_status(__status__)


cpdef sscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasSscal`."""
    with nogil:
        __status__ = cublasSscal(<Handle>handle, n, <const float*>alpha, <float*>x, incx)
    check_status(__status__)


cpdef dscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasDscal`."""
    with nogil:
        __status__ = cublasDscal(<Handle>handle, n, <const double*>alpha, <double*>x, incx)
    check_status(__status__)


cpdef cscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasCscal`."""
    with nogil:
        __status__ = cublasCscal(<Handle>handle, n, <const cuComplex*>alpha, <cuComplex*>x, incx)
    check_status(__status__)


cpdef csscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasCsscal`."""
    with nogil:
        __status__ = cublasCsscal(<Handle>handle, n, <const float*>alpha, <cuComplex*>x, incx)
    check_status(__status__)


cpdef zscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasZscal`."""
    with nogil:
        __status__ = cublasZscal(<Handle>handle, n, <const cuDoubleComplex*>alpha, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef zdscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    """See `cublasZdscal`."""
    with nogil:
        __status__ = cublasZdscal(<Handle>handle, n, <const double*>alpha, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef axpy_ex(intptr_t handle, int n, intptr_t alpha, int alpha_type, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy, int executiontype):
    """See `cublasAxpyEx`."""
    with nogil:
        __status__ = cublasAxpyEx(<Handle>handle, n, <const void*>alpha, <DataType>alpha_type, <const void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <DataType>executiontype)
    check_status(__status__)


cpdef saxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasSaxpy`."""
    with nogil:
        __status__ = cublasSaxpy(<Handle>handle, n, <const float*>alpha, <const float*>x, incx, <float*>y, incy)
    check_status(__status__)


cpdef daxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasDaxpy`."""
    with nogil:
        __status__ = cublasDaxpy(<Handle>handle, n, <const double*>alpha, <const double*>x, incx, <double*>y, incy)
    check_status(__status__)


cpdef caxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasCaxpy`."""
    with nogil:
        __status__ = cublasCaxpy(<Handle>handle, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zaxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasZaxpy`."""
    with nogil:
        __status__ = cublasZaxpy(<Handle>handle, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef copy_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy):
    """See `cublasCopyEx`."""
    with nogil:
        __status__ = cublasCopyEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy)
    check_status(__status__)


cpdef scopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasScopy`."""
    with nogil:
        __status__ = cublasScopy(<Handle>handle, n, <const float*>x, incx, <float*>y, incy)
    check_status(__status__)


cpdef dcopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasDcopy`."""
    with nogil:
        __status__ = cublasDcopy(<Handle>handle, n, <const double*>x, incx, <double*>y, incy)
    check_status(__status__)


cpdef ccopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasCcopy`."""
    with nogil:
        __status__ = cublasCcopy(<Handle>handle, n, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zcopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasZcopy`."""
    with nogil:
        __status__ = cublasZcopy(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef sswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasSswap`."""
    with nogil:
        __status__ = cublasSswap(<Handle>handle, n, <float*>x, incx, <float*>y, incy)
    check_status(__status__)


cpdef dswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasDswap`."""
    with nogil:
        __status__ = cublasDswap(<Handle>handle, n, <double*>x, incx, <double*>y, incy)
    check_status(__status__)


cpdef cswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasCswap`."""
    with nogil:
        __status__ = cublasCswap(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    """See `cublasZswap`."""
    with nogil:
        __status__ = cublasZswap(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef swap_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy):
    """See `cublasSwapEx`."""
    with nogil:
        __status__ = cublasSwapEx(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy)
    check_status(__status__)


cpdef isamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIsamax`."""
    with nogil:
        __status__ = cublasIsamax(<Handle>handle, n, <const float*>x, incx, <int*>result)
    check_status(__status__)


cpdef idamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIdamax`."""
    with nogil:
        __status__ = cublasIdamax(<Handle>handle, n, <const double*>x, incx, <int*>result)
    check_status(__status__)


cpdef icamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIcamax`."""
    with nogil:
        __status__ = cublasIcamax(<Handle>handle, n, <const cuComplex*>x, incx, <int*>result)
    check_status(__status__)


cpdef izamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIzamax`."""
    with nogil:
        __status__ = cublasIzamax(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int*>result)
    check_status(__status__)


cpdef iamax_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t result):
    """See `cublasIamaxEx`."""
    with nogil:
        __status__ = cublasIamaxEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <int*>result)
    check_status(__status__)


cpdef isamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIsamin`."""
    with nogil:
        __status__ = cublasIsamin(<Handle>handle, n, <const float*>x, incx, <int*>result)
    check_status(__status__)


cpdef idamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIdamin`."""
    with nogil:
        __status__ = cublasIdamin(<Handle>handle, n, <const double*>x, incx, <int*>result)
    check_status(__status__)


cpdef icamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIcamin`."""
    with nogil:
        __status__ = cublasIcamin(<Handle>handle, n, <const cuComplex*>x, incx, <int*>result)
    check_status(__status__)


cpdef izamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasIzamin`."""
    with nogil:
        __status__ = cublasIzamin(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int*>result)
    check_status(__status__)


cpdef iamin_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t result):
    """See `cublasIaminEx`."""
    with nogil:
        __status__ = cublasIaminEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <int*>result)
    check_status(__status__)


cpdef asum_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t result, int result_type, int executiontype):
    """See `cublasAsumEx`."""
    with nogil:
        __status__ = cublasAsumEx(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>result, <DataType>result_type, <DataType>executiontype)
    check_status(__status__)


cpdef sasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasSasum`."""
    with nogil:
        __status__ = cublasSasum(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(__status__)


cpdef dasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasDasum`."""
    with nogil:
        __status__ = cublasDasum(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(__status__)


cpdef scasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasScasum`."""
    with nogil:
        __status__ = cublasScasum(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(__status__)


cpdef dzasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    """See `cublasDzasum`."""
    with nogil:
        __status__ = cublasDzasum(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(__status__)


cpdef srot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasSrot`."""
    with nogil:
        __status__ = cublasSrot(<Handle>handle, n, <float*>x, incx, <float*>y, incy, <const float*>c, <const float*>s)
    check_status(__status__)


cpdef drot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasDrot`."""
    with nogil:
        __status__ = cublasDrot(<Handle>handle, n, <double*>x, incx, <double*>y, incy, <const double*>c, <const double*>s)
    check_status(__status__)


cpdef crot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasCrot`."""
    with nogil:
        __status__ = cublasCrot(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy, <const float*>c, <const cuComplex*>s)
    check_status(__status__)


cpdef csrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasCsrot`."""
    with nogil:
        __status__ = cublasCsrot(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy, <const float*>c, <const float*>s)
    check_status(__status__)


cpdef zrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasZrot`."""
    with nogil:
        __status__ = cublasZrot(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy, <const double*>c, <const cuDoubleComplex*>s)
    check_status(__status__)


cpdef zdrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    """See `cublasZdrot`."""
    with nogil:
        __status__ = cublasZdrot(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy, <const double*>c, <const double*>s)
    check_status(__status__)


cpdef rot_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy, intptr_t c, intptr_t s, int cs_type, int executiontype):
    """See `cublasRotEx`."""
    with nogil:
        __status__ = cublasRotEx(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <const void*>c, <const void*>s, <DataType>cs_type, <DataType>executiontype)
    check_status(__status__)


cpdef srotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    """See `cublasSrotg`."""
    with nogil:
        __status__ = cublasSrotg(<Handle>handle, <float*>a, <float*>b, <float*>c, <float*>s)
    check_status(__status__)


cpdef drotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    """See `cublasDrotg`."""
    with nogil:
        __status__ = cublasDrotg(<Handle>handle, <double*>a, <double*>b, <double*>c, <double*>s)
    check_status(__status__)


cpdef crotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    """See `cublasCrotg`."""
    with nogil:
        __status__ = cublasCrotg(<Handle>handle, <cuComplex*>a, <cuComplex*>b, <float*>c, <cuComplex*>s)
    check_status(__status__)


cpdef zrotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    """See `cublasZrotg`."""
    with nogil:
        __status__ = cublasZrotg(<Handle>handle, <cuDoubleComplex*>a, <cuDoubleComplex*>b, <double*>c, <cuDoubleComplex*>s)
    check_status(__status__)


cpdef rotg_ex(intptr_t handle, intptr_t a, intptr_t b, int ab_type, intptr_t c, intptr_t s, int cs_type, int executiontype):
    """See `cublasRotgEx`."""
    with nogil:
        __status__ = cublasRotgEx(<Handle>handle, <void*>a, <void*>b, <DataType>ab_type, <void*>c, <void*>s, <DataType>cs_type, <DataType>executiontype)
    check_status(__status__)


cpdef srotm(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t param):
    """See `cublasSrotm`."""
    with nogil:
        __status__ = cublasSrotm(<Handle>handle, n, <float*>x, incx, <float*>y, incy, <const float*>param)
    check_status(__status__)


cpdef drotm(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t param):
    """See `cublasDrotm`."""
    with nogil:
        __status__ = cublasDrotm(<Handle>handle, n, <double*>x, incx, <double*>y, incy, <const double*>param)
    check_status(__status__)


cpdef rotm_ex(intptr_t handle, int n, intptr_t x, int x_type, int incx, intptr_t y, int y_type, int incy, intptr_t param, int param_type, int executiontype):
    """See `cublasRotmEx`."""
    with nogil:
        __status__ = cublasRotmEx(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <const void*>param, <DataType>param_type, <DataType>executiontype)
    check_status(__status__)


cpdef srotmg(intptr_t handle, intptr_t d1, intptr_t d2, intptr_t x1, intptr_t y1, intptr_t param):
    """See `cublasSrotmg`."""
    with nogil:
        __status__ = cublasSrotmg(<Handle>handle, <float*>d1, <float*>d2, <float*>x1, <const float*>y1, <float*>param)
    check_status(__status__)


cpdef drotmg(intptr_t handle, intptr_t d1, intptr_t d2, intptr_t x1, intptr_t y1, intptr_t param):
    """See `cublasDrotmg`."""
    with nogil:
        __status__ = cublasDrotmg(<Handle>handle, <double*>d1, <double*>d2, <double*>x1, <const double*>y1, <double*>param)
    check_status(__status__)


cpdef rotmg_ex(intptr_t handle, intptr_t d1, int d1type, intptr_t d2, int d2type, intptr_t x1, int x1type, intptr_t y1, int y1type, intptr_t param, int param_type, int executiontype):
    """See `cublasRotmgEx`."""
    with nogil:
        __status__ = cublasRotmgEx(<Handle>handle, <void*>d1, <DataType>d1type, <void*>d2, <DataType>d2type, <void*>x1, <DataType>x1type, <const void*>y1, <DataType>y1type, <void*>param, <DataType>param_type, <DataType>executiontype)
    check_status(__status__)


cpdef sgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasSgemv`."""
    with nogil:
        __status__ = cublasSgemv(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(__status__)


cpdef dgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasDgemv`."""
    with nogil:
        __status__ = cublasDgemv(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(__status__)


cpdef cgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasCgemv`."""
    with nogil:
        __status__ = cublasCgemv(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZgemv`."""
    with nogil:
        __status__ = cublasZgemv(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef sgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasSgbmv`."""
    with nogil:
        __status__ = cublasSgbmv(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(__status__)


cpdef dgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasDgbmv`."""
    with nogil:
        __status__ = cublasDgbmv(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(__status__)


cpdef cgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasCgbmv`."""
    with nogil:
        __status__ = cublasCgbmv(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZgbmv`."""
    with nogil:
        __status__ = cublasZgbmv(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef strmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasStrmv`."""
    with nogil:
        __status__ = cublasStrmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>a, lda, <float*>x, incx)
    check_status(__status__)


cpdef dtrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasDtrmv`."""
    with nogil:
        __status__ = cublasDtrmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>a, lda, <double*>x, incx)
    check_status(__status__)


cpdef ctrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasCtrmv`."""
    with nogil:
        __status__ = cublasCtrmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasZtrmv`."""
    with nogil:
        __status__ = cublasZtrmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef stbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasStbmv`."""
    with nogil:
        __status__ = cublasStbmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const float*>a, lda, <float*>x, incx)
    check_status(__status__)


cpdef dtbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasDtbmv`."""
    with nogil:
        __status__ = cublasDtbmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const double*>a, lda, <double*>x, incx)
    check_status(__status__)


cpdef ctbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasCtbmv`."""
    with nogil:
        __status__ = cublasCtbmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasZtbmv`."""
    with nogil:
        __status__ = cublasZtbmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef stpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasStpmv`."""
    with nogil:
        __status__ = cublasStpmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>ap, <float*>x, incx)
    check_status(__status__)


cpdef dtpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasDtpmv`."""
    with nogil:
        __status__ = cublasDtpmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>ap, <double*>x, incx)
    check_status(__status__)


cpdef ctpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasCtpmv`."""
    with nogil:
        __status__ = cublasCtpmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>ap, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasZtpmv`."""
    with nogil:
        __status__ = cublasZtpmv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>ap, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef strsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasStrsv`."""
    with nogil:
        __status__ = cublasStrsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>a, lda, <float*>x, incx)
    check_status(__status__)


cpdef dtrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasDtrsv`."""
    with nogil:
        __status__ = cublasDtrsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>a, lda, <double*>x, incx)
    check_status(__status__)


cpdef ctrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasCtrsv`."""
    with nogil:
        __status__ = cublasCtrsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasZtrsv`."""
    with nogil:
        __status__ = cublasZtrsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef stpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasStpsv`."""
    with nogil:
        __status__ = cublasStpsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>ap, <float*>x, incx)
    check_status(__status__)


cpdef dtpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasDtpsv`."""
    with nogil:
        __status__ = cublasDtpsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>ap, <double*>x, incx)
    check_status(__status__)


cpdef ctpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasCtpsv`."""
    with nogil:
        __status__ = cublasCtpsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>ap, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t ap, intptr_t x, int incx):
    """See `cublasZtpsv`."""
    with nogil:
        __status__ = cublasZtpsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>ap, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef stbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasStbsv`."""
    with nogil:
        __status__ = cublasStbsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const float*>a, lda, <float*>x, incx)
    check_status(__status__)


cpdef dtbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasDtbsv`."""
    with nogil:
        __status__ = cublasDtbsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const double*>a, lda, <double*>x, incx)
    check_status(__status__)


cpdef ctbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasCtbsv`."""
    with nogil:
        __status__ = cublasCtbsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t a, int lda, intptr_t x, int incx):
    """See `cublasZtbsv`."""
    with nogil:
        __status__ = cublasZtbsv(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef ssymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasSsymv`."""
    with nogil:
        __status__ = cublasSsymv(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(__status__)


cpdef dsymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasDsymv`."""
    with nogil:
        __status__ = cublasDsymv(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(__status__)


cpdef csymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasCsymv`."""
    with nogil:
        __status__ = cublasCsymv(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zsymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZsymv`."""
    with nogil:
        __status__ = cublasZsymv(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef chemv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasChemv`."""
    with nogil:
        __status__ = cublasChemv(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zhemv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZhemv`."""
    with nogil:
        __status__ = cublasZhemv(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef ssbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasSsbmv`."""
    with nogil:
        __status__ = cublasSsbmv(<Handle>handle, <_FillMode>uplo, n, k, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(__status__)


cpdef dsbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasDsbmv`."""
    with nogil:
        __status__ = cublasDsbmv(<Handle>handle, <_FillMode>uplo, n, k, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(__status__)


cpdef chbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasChbmv`."""
    with nogil:
        __status__ = cublasChbmv(<Handle>handle, <_FillMode>uplo, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zhbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZhbmv`."""
    with nogil:
        __status__ = cublasZhbmv(<Handle>handle, <_FillMode>uplo, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef sspmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t ap, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasSspmv`."""
    with nogil:
        __status__ = cublasSspmv(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>ap, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(__status__)


cpdef dspmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t ap, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasDspmv`."""
    with nogil:
        __status__ = cublasDspmv(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>ap, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(__status__)


cpdef chpmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t ap, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasChpmv`."""
    with nogil:
        __status__ = cublasChpmv(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>ap, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zhpmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t ap, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    """See `cublasZhpmv`."""
    with nogil:
        __status__ = cublasZhpmv(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>ap, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef sger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasSger`."""
    with nogil:
        __status__ = cublasSger(<Handle>handle, m, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>a, lda)
    check_status(__status__)


cpdef dger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasDger`."""
    with nogil:
        __status__ = cublasDger(<Handle>handle, m, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>a, lda)
    check_status(__status__)


cpdef cgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasCgeru`."""
    with nogil:
        __status__ = cublasCgeru(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(__status__)


cpdef cgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasCgerc`."""
    with nogil:
        __status__ = cublasCgerc(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(__status__)


cpdef zgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasZgeru`."""
    with nogil:
        __status__ = cublasZgeru(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef zgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasZgerc`."""
    with nogil:
        __status__ = cublasZgerc(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef ssyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasSsyr`."""
    with nogil:
        __status__ = cublasSsyr(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <float*>a, lda)
    check_status(__status__)


cpdef dsyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasDsyr`."""
    with nogil:
        __status__ = cublasDsyr(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <double*>a, lda)
    check_status(__status__)


cpdef csyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasCsyr`."""
    with nogil:
        __status__ = cublasCsyr(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>a, lda)
    check_status(__status__)


cpdef zsyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasZsyr`."""
    with nogil:
        __status__ = cublasZsyr(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef cher(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasCher`."""
    with nogil:
        __status__ = cublasCher(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const cuComplex*>x, incx, <cuComplex*>a, lda)
    check_status(__status__)


cpdef zher(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t a, int lda):
    """See `cublasZher`."""
    with nogil:
        __status__ = cublasZher(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef sspr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t ap):
    """See `cublasSspr`."""
    with nogil:
        __status__ = cublasSspr(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <float*>ap)
    check_status(__status__)


cpdef dspr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t ap):
    """See `cublasDspr`."""
    with nogil:
        __status__ = cublasDspr(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <double*>ap)
    check_status(__status__)


cpdef chpr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t ap):
    """See `cublasChpr`."""
    with nogil:
        __status__ = cublasChpr(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const cuComplex*>x, incx, <cuComplex*>ap)
    check_status(__status__)


cpdef zhpr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t ap):
    """See `cublasZhpr`."""
    with nogil:
        __status__ = cublasZhpr(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>ap)
    check_status(__status__)


cpdef ssyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasSsyr2`."""
    with nogil:
        __status__ = cublasSsyr2(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>a, lda)
    check_status(__status__)


cpdef dsyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasDsyr2`."""
    with nogil:
        __status__ = cublasDsyr2(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>a, lda)
    check_status(__status__)


cpdef csyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasCsyr2`."""
    with nogil:
        __status__ = cublasCsyr2(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(__status__)


cpdef zsyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasZsyr2`."""
    with nogil:
        __status__ = cublasZsyr2(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef cher2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasCher2`."""
    with nogil:
        __status__ = cublasCher2(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(__status__)


cpdef zher2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t a, int lda):
    """See `cublasZher2`."""
    with nogil:
        __status__ = cublasZher2(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef sspr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t ap):
    """See `cublasSspr2`."""
    with nogil:
        __status__ = cublasSspr2(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>ap)
    check_status(__status__)


cpdef dspr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t ap):
    """See `cublasDspr2`."""
    with nogil:
        __status__ = cublasDspr2(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>ap)
    check_status(__status__)


cpdef chpr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t ap):
    """See `cublasChpr2`."""
    with nogil:
        __status__ = cublasChpr2(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>ap)
    check_status(__status__)


cpdef zhpr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t ap):
    """See `cublasZhpr2`."""
    with nogil:
        __status__ = cublasZhpr2(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>ap)
    check_status(__status__)


cpdef sgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasSgemm`."""
    with nogil:
        __status__ = cublasSgemm(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(__status__)


cpdef dgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasDgemm`."""
    with nogil:
        __status__ = cublasDgemm(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(__status__)


cpdef cgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCgemm`."""
    with nogil:
        __status__ = cublasCgemm(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef cgemm3m(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCgemm3m`."""
    with nogil:
        __status__ = cublasCgemm3m(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef cgemm3m_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t b, int btype, int ldb, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCgemm3mEx`."""
    with nogil:
        __status__ = cublasCgemm3mEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef zgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZgemm`."""
    with nogil:
        __status__ = cublasZgemm(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef zgemm3m(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZgemm3m`."""
    with nogil:
        __status__ = cublasZgemm3m(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef sgemm_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t b, int btype, int ldb, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasSgemmEx`."""
    with nogil:
        __status__ = cublasSgemmEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef gemm_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t b, int btype, int ldb, intptr_t beta, intptr_t c, int ctype, int ldc, int compute_type, int algo):
    """See `cublasGemmEx`."""
    with nogil:
        __status__ = cublasGemmEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const void*>beta, <void*>c, <DataType>ctype, ldc, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(__status__)


cpdef cgemm_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t b, int btype, int ldb, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCgemmEx`."""
    with nogil:
        __status__ = cublasCgemmEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef uint8gemm_bias(intptr_t handle, int transa, int transb, int transc, int m, int n, int k, intptr_t a, int a_bias, int lda, intptr_t b, int b_bias, int ldb, intptr_t c, int c_bias, int ldc, int c_mult, int c_shift):
    """See `cublasUint8gemmBias`."""
    with nogil:
        __status__ = cublasUint8gemmBias(<Handle>handle, <_Operation>transa, <_Operation>transb, <_Operation>transc, m, n, k, <const unsigned char*>a, a_bias, lda, <const unsigned char*>b, b_bias, ldb, <unsigned char*>c, c_bias, ldc, c_mult, c_shift)
    check_status(__status__)


cpdef ssyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasSsyrk`."""
    with nogil:
        __status__ = cublasSsyrk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>beta, <float*>c, ldc)
    check_status(__status__)


cpdef dsyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasDsyrk`."""
    with nogil:
        __status__ = cublasDsyrk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>beta, <double*>c, ldc)
    check_status(__status__)


cpdef csyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCsyrk`."""
    with nogil:
        __status__ = cublasCsyrk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zsyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZsyrk`."""
    with nogil:
        __status__ = cublasZsyrk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef csyrk_ex(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCsyrkEx`."""
    with nogil:
        __status__ = cublasCsyrkEx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef csyrk3m_ex(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCsyrk3mEx`."""
    with nogil:
        __status__ = cublasCsyrk3mEx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef cherk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCherk`."""
    with nogil:
        __status__ = cublasCherk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const cuComplex*>a, lda, <const float*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zherk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZherk`."""
    with nogil:
        __status__ = cublasZherk(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const cuDoubleComplex*>a, lda, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef cherk_ex(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCherkEx`."""
    with nogil:
        __status__ = cublasCherkEx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef cherk3m_ex(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, intptr_t beta, intptr_t c, int ctype, int ldc):
    """See `cublasCherk3mEx`."""
    with nogil:
        __status__ = cublasCherk3mEx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef ssyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasSsyr2k`."""
    with nogil:
        __status__ = cublasSsyr2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(__status__)


cpdef dsyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasDsyr2k`."""
    with nogil:
        __status__ = cublasDsyr2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(__status__)


cpdef csyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCsyr2k`."""
    with nogil:
        __status__ = cublasCsyr2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zsyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZsyr2k`."""
    with nogil:
        __status__ = cublasZsyr2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef cher2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCher2k`."""
    with nogil:
        __status__ = cublasCher2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const float*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zher2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZher2k`."""
    with nogil:
        __status__ = cublasZher2k(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef ssyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasSsyrkx`."""
    with nogil:
        __status__ = cublasSsyrkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(__status__)


cpdef dsyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasDsyrkx`."""
    with nogil:
        __status__ = cublasDsyrkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(__status__)


cpdef csyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCsyrkx`."""
    with nogil:
        __status__ = cublasCsyrkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zsyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZsyrkx`."""
    with nogil:
        __status__ = cublasZsyrkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef cherkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCherkx`."""
    with nogil:
        __status__ = cublasCherkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const float*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zherkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZherkx`."""
    with nogil:
        __status__ = cublasZherkx(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef ssymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasSsymm`."""
    with nogil:
        __status__ = cublasSsymm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(__status__)


cpdef dsymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasDsymm`."""
    with nogil:
        __status__ = cublasDsymm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(__status__)


cpdef csymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasCsymm`."""
    with nogil:
        __status__ = cublasCsymm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zsymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZsymm`."""
    with nogil:
        __status__ = cublasZsymm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef chemm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasChemm`."""
    with nogil:
        __status__ = cublasChemm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zhemm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t beta, intptr_t c, int ldc):
    """See `cublasZhemm`."""
    with nogil:
        __status__ = cublasZhemm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef strsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasStrsm`."""
    with nogil:
        __status__ = cublasStrsm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float*>a, lda, <float*>b, ldb)
    check_status(__status__)


cpdef dtrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasDtrsm`."""
    with nogil:
        __status__ = cublasDtrsm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double*>a, lda, <double*>b, ldb)
    check_status(__status__)


cpdef ctrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasCtrsm`."""
    with nogil:
        __status__ = cublasCtrsm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <cuComplex*>b, ldb)
    check_status(__status__)


cpdef ztrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb):
    """See `cublasZtrsm`."""
    with nogil:
        __status__ = cublasZtrsm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>b, ldb)
    check_status(__status__)


cpdef strmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasStrmm`."""
    with nogil:
        __status__ = cublasStrmm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <float*>c, ldc)
    check_status(__status__)


cpdef dtrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasDtrmm`."""
    with nogil:
        __status__ = cublasDtrmm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <double*>c, ldc)
    check_status(__status__)


cpdef ctrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasCtrmm`."""
    with nogil:
        __status__ = cublasCtrmm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef ztrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasZtrmm`."""
    with nogil:
        __status__ = cublasZtrmm(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef sgemm_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int lda, intptr_t barray, int ldb, intptr_t beta, intptr_t carray, int ldc, int batch_count):
    """See `cublasSgemmBatched`."""
    with nogil:
        __status__ = cublasSgemmBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float* const*>aarray, lda, <const float* const*>barray, ldb, <const float*>beta, <float* const*>carray, ldc, batch_count)
    check_status(__status__)


cpdef dgemm_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int lda, intptr_t barray, int ldb, intptr_t beta, intptr_t carray, int ldc, int batch_count):
    """See `cublasDgemmBatched`."""
    with nogil:
        __status__ = cublasDgemmBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double* const*>aarray, lda, <const double* const*>barray, ldb, <const double*>beta, <double* const*>carray, ldc, batch_count)
    check_status(__status__)


cpdef cgemm_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int lda, intptr_t barray, int ldb, intptr_t beta, intptr_t carray, int ldc, int batch_count):
    """See `cublasCgemmBatched`."""
    with nogil:
        __status__ = cublasCgemmBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>barray, ldb, <const cuComplex*>beta, <cuComplex* const*>carray, ldc, batch_count)
    check_status(__status__)


cpdef cgemm3m_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int lda, intptr_t barray, int ldb, intptr_t beta, intptr_t carray, int ldc, int batch_count):
    """See `cublasCgemm3mBatched`."""
    with nogil:
        __status__ = cublasCgemm3mBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>barray, ldb, <const cuComplex*>beta, <cuComplex* const*>carray, ldc, batch_count)
    check_status(__status__)


cpdef zgemm_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int lda, intptr_t barray, int ldb, intptr_t beta, intptr_t carray, int ldc, int batch_count):
    """See `cublasZgemmBatched`."""
    with nogil:
        __status__ = cublasZgemmBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>aarray, lda, <const cuDoubleComplex* const*>barray, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex* const*>carray, ldc, batch_count)
    check_status(__status__)


cpdef gemm_batched_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t aarray, int atype, int lda, intptr_t barray, int btype, int ldb, intptr_t beta, intptr_t carray, int ctype, int ldc, int batch_count, int compute_type, int algo):
    """See `cublasGemmBatchedEx`."""
    with nogil:
        __status__ = cublasGemmBatchedEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void* const*>aarray, <DataType>atype, lda, <const void* const*>barray, <DataType>btype, ldb, <const void*>beta, <void* const*>carray, <DataType>ctype, ldc, batch_count, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(__status__)


cpdef gemm_strided_batched_ex(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int atype, int lda, long long int stride_a, intptr_t b, int btype, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ctype, int ldc, long long int stride_c, int batch_count, int compute_type, int algo):
    """See `cublasGemmStridedBatchedEx`."""
    with nogil:
        __status__ = cublasGemmStridedBatchedEx(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void*>a, <DataType>atype, lda, stride_a, <const void*>b, <DataType>btype, ldb, stride_b, <const void*>beta, <void*>c, <DataType>ctype, ldc, stride_c, batch_count, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(__status__)


cpdef sgemm_strided_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t b, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ldc, long long int stride_c, int batch_count):
    """See `cublasSgemmStridedBatched`."""
    with nogil:
        __status__ = cublasSgemmStridedBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float*>a, lda, stride_a, <const float*>b, ldb, stride_b, <const float*>beta, <float*>c, ldc, stride_c, batch_count)
    check_status(__status__)


cpdef dgemm_strided_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t b, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ldc, long long int stride_c, int batch_count):
    """See `cublasDgemmStridedBatched`."""
    with nogil:
        __status__ = cublasDgemmStridedBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double*>a, lda, stride_a, <const double*>b, ldb, stride_b, <const double*>beta, <double*>c, ldc, stride_c, batch_count)
    check_status(__status__)


cpdef cgemm_strided_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t b, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ldc, long long int stride_c, int batch_count):
    """See `cublasCgemmStridedBatched`."""
    with nogil:
        __status__ = cublasCgemmStridedBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>b, ldb, stride_b, <const cuComplex*>beta, <cuComplex*>c, ldc, stride_c, batch_count)
    check_status(__status__)


cpdef cgemm3m_strided_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t b, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ldc, long long int stride_c, int batch_count):
    """See `cublasCgemm3mStridedBatched`."""
    with nogil:
        __status__ = cublasCgemm3mStridedBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>b, ldb, stride_b, <const cuComplex*>beta, <cuComplex*>c, ldc, stride_c, batch_count)
    check_status(__status__)


cpdef zgemm_strided_batched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t b, int ldb, long long int stride_b, intptr_t beta, intptr_t c, int ldc, long long int stride_c, int batch_count):
    """See `cublasZgemmStridedBatched`."""
    with nogil:
        __status__ = cublasZgemmStridedBatched(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, stride_a, <const cuDoubleComplex*>b, ldb, stride_b, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc, stride_c, batch_count)
    check_status(__status__)


cpdef sgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasSgeam`."""
    with nogil:
        __status__ = cublasSgeam(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const float*>alpha, <const float*>a, lda, <const float*>beta, <const float*>b, ldb, <float*>c, ldc)
    check_status(__status__)


cpdef dgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasDgeam`."""
    with nogil:
        __status__ = cublasDgeam(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const double*>alpha, <const double*>a, lda, <const double*>beta, <const double*>b, ldb, <double*>c, ldc)
    check_status(__status__)


cpdef cgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasCgeam`."""
    with nogil:
        __status__ = cublasCgeam(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>beta, <const cuComplex*>b, ldb, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t beta, intptr_t b, int ldb, intptr_t c, int ldc):
    """See `cublasZgeam`."""
    with nogil:
        __status__ = cublasZgeam(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>beta, <const cuDoubleComplex*>b, ldb, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef sgetrf_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t info, int batch_size):
    """See `cublasSgetrfBatched`."""
    with nogil:
        __status__ = cublasSgetrfBatched(<Handle>handle, n, <float* const*>a, lda, <int*>p, <int*>info, batch_size)
    check_status(__status__)


cpdef dgetrf_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t info, int batch_size):
    """See `cublasDgetrfBatched`."""
    with nogil:
        __status__ = cublasDgetrfBatched(<Handle>handle, n, <double* const*>a, lda, <int*>p, <int*>info, batch_size)
    check_status(__status__)


cpdef cgetrf_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t info, int batch_size):
    """See `cublasCgetrfBatched`."""
    with nogil:
        __status__ = cublasCgetrfBatched(<Handle>handle, n, <cuComplex* const*>a, lda, <int*>p, <int*>info, batch_size)
    check_status(__status__)


cpdef zgetrf_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t info, int batch_size):
    """See `cublasZgetrfBatched`."""
    with nogil:
        __status__ = cublasZgetrfBatched(<Handle>handle, n, <cuDoubleComplex* const*>a, lda, <int*>p, <int*>info, batch_size)
    check_status(__status__)


cpdef sgetri_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t c, int ldc, intptr_t info, int batch_size):
    """See `cublasSgetriBatched`."""
    with nogil:
        __status__ = cublasSgetriBatched(<Handle>handle, n, <const float* const*>a, lda, <const int*>p, <float* const*>c, ldc, <int*>info, batch_size)
    check_status(__status__)


cpdef dgetri_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t c, int ldc, intptr_t info, int batch_size):
    """See `cublasDgetriBatched`."""
    with nogil:
        __status__ = cublasDgetriBatched(<Handle>handle, n, <const double* const*>a, lda, <const int*>p, <double* const*>c, ldc, <int*>info, batch_size)
    check_status(__status__)


cpdef cgetri_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t c, int ldc, intptr_t info, int batch_size):
    """See `cublasCgetriBatched`."""
    with nogil:
        __status__ = cublasCgetriBatched(<Handle>handle, n, <const cuComplex* const*>a, lda, <const int*>p, <cuComplex* const*>c, ldc, <int*>info, batch_size)
    check_status(__status__)


cpdef zgetri_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t p, intptr_t c, int ldc, intptr_t info, int batch_size):
    """See `cublasZgetriBatched`."""
    with nogil:
        __status__ = cublasZgetriBatched(<Handle>handle, n, <const cuDoubleComplex* const*>a, lda, <const int*>p, <cuDoubleComplex* const*>c, ldc, <int*>info, batch_size)
    check_status(__status__)


cpdef sgetrs_batched(intptr_t handle, int trans, int n, int nrhs, intptr_t aarray, int lda, intptr_t dev_ipiv, intptr_t barray, int ldb, intptr_t info, int batch_size):
    """See `cublasSgetrsBatched`."""
    with nogil:
        __status__ = cublasSgetrsBatched(<Handle>handle, <_Operation>trans, n, nrhs, <const float* const*>aarray, lda, <const int*>dev_ipiv, <float* const*>barray, ldb, <int*>info, batch_size)
    check_status(__status__)


cpdef dgetrs_batched(intptr_t handle, int trans, int n, int nrhs, intptr_t aarray, int lda, intptr_t dev_ipiv, intptr_t barray, int ldb, intptr_t info, int batch_size):
    """See `cublasDgetrsBatched`."""
    with nogil:
        __status__ = cublasDgetrsBatched(<Handle>handle, <_Operation>trans, n, nrhs, <const double* const*>aarray, lda, <const int*>dev_ipiv, <double* const*>barray, ldb, <int*>info, batch_size)
    check_status(__status__)


cpdef cgetrs_batched(intptr_t handle, int trans, int n, int nrhs, intptr_t aarray, int lda, intptr_t dev_ipiv, intptr_t barray, int ldb, intptr_t info, int batch_size):
    """See `cublasCgetrsBatched`."""
    with nogil:
        __status__ = cublasCgetrsBatched(<Handle>handle, <_Operation>trans, n, nrhs, <const cuComplex* const*>aarray, lda, <const int*>dev_ipiv, <cuComplex* const*>barray, ldb, <int*>info, batch_size)
    check_status(__status__)


cpdef zgetrs_batched(intptr_t handle, int trans, int n, int nrhs, intptr_t aarray, int lda, intptr_t dev_ipiv, intptr_t barray, int ldb, intptr_t info, int batch_size):
    """See `cublasZgetrsBatched`."""
    with nogil:
        __status__ = cublasZgetrsBatched(<Handle>handle, <_Operation>trans, n, nrhs, <const cuDoubleComplex* const*>aarray, lda, <const int*>dev_ipiv, <cuDoubleComplex* const*>barray, ldb, <int*>info, batch_size)
    check_status(__status__)


cpdef strsm_batched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, int batch_count):
    """See `cublasStrsmBatched`."""
    with nogil:
        __status__ = cublasStrsmBatched(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float* const*>a, lda, <float* const*>b, ldb, batch_count)
    check_status(__status__)


cpdef dtrsm_batched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, int batch_count):
    """See `cublasDtrsmBatched`."""
    with nogil:
        __status__ = cublasDtrsmBatched(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double* const*>a, lda, <double* const*>b, ldb, batch_count)
    check_status(__status__)


cpdef ctrsm_batched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, int batch_count):
    """See `cublasCtrsmBatched`."""
    with nogil:
        __status__ = cublasCtrsmBatched(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex* const*>a, lda, <cuComplex* const*>b, ldb, batch_count)
    check_status(__status__)


cpdef ztrsm_batched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t a, int lda, intptr_t b, int ldb, int batch_count):
    """See `cublasZtrsmBatched`."""
    with nogil:
        __status__ = cublasZtrsmBatched(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>a, lda, <cuDoubleComplex* const*>b, ldb, batch_count)
    check_status(__status__)


cpdef smatinv_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t ainv, int lda_inv, intptr_t info, int batch_size):
    """See `cublasSmatinvBatched`."""
    with nogil:
        __status__ = cublasSmatinvBatched(<Handle>handle, n, <const float* const*>a, lda, <float* const*>ainv, lda_inv, <int*>info, batch_size)
    check_status(__status__)


cpdef dmatinv_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t ainv, int lda_inv, intptr_t info, int batch_size):
    """See `cublasDmatinvBatched`."""
    with nogil:
        __status__ = cublasDmatinvBatched(<Handle>handle, n, <const double* const*>a, lda, <double* const*>ainv, lda_inv, <int*>info, batch_size)
    check_status(__status__)


cpdef cmatinv_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t ainv, int lda_inv, intptr_t info, int batch_size):
    """See `cublasCmatinvBatched`."""
    with nogil:
        __status__ = cublasCmatinvBatched(<Handle>handle, n, <const cuComplex* const*>a, lda, <cuComplex* const*>ainv, lda_inv, <int*>info, batch_size)
    check_status(__status__)


cpdef zmatinv_batched(intptr_t handle, int n, intptr_t a, int lda, intptr_t ainv, int lda_inv, intptr_t info, int batch_size):
    """See `cublasZmatinvBatched`."""
    with nogil:
        __status__ = cublasZmatinvBatched(<Handle>handle, n, <const cuDoubleComplex* const*>a, lda, <cuDoubleComplex* const*>ainv, lda_inv, <int*>info, batch_size)
    check_status(__status__)


cpdef sgeqrf_batched(intptr_t handle, int m, int n, intptr_t aarray, int lda, intptr_t tau_array, intptr_t info, int batch_size):
    """See `cublasSgeqrfBatched`."""
    with nogil:
        __status__ = cublasSgeqrfBatched(<Handle>handle, m, n, <float* const*>aarray, lda, <float* const*>tau_array, <int*>info, batch_size)
    check_status(__status__)


cpdef dgeqrf_batched(intptr_t handle, int m, int n, intptr_t aarray, int lda, intptr_t tau_array, intptr_t info, int batch_size):
    """See `cublasDgeqrfBatched`."""
    with nogil:
        __status__ = cublasDgeqrfBatched(<Handle>handle, m, n, <double* const*>aarray, lda, <double* const*>tau_array, <int*>info, batch_size)
    check_status(__status__)


cpdef cgeqrf_batched(intptr_t handle, int m, int n, intptr_t aarray, int lda, intptr_t tau_array, intptr_t info, int batch_size):
    """See `cublasCgeqrfBatched`."""
    with nogil:
        __status__ = cublasCgeqrfBatched(<Handle>handle, m, n, <cuComplex* const*>aarray, lda, <cuComplex* const*>tau_array, <int*>info, batch_size)
    check_status(__status__)


cpdef zgeqrf_batched(intptr_t handle, int m, int n, intptr_t aarray, int lda, intptr_t tau_array, intptr_t info, int batch_size):
    """See `cublasZgeqrfBatched`."""
    with nogil:
        __status__ = cublasZgeqrfBatched(<Handle>handle, m, n, <cuDoubleComplex* const*>aarray, lda, <cuDoubleComplex* const*>tau_array, <int*>info, batch_size)
    check_status(__status__)


cpdef sgels_batched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t aarray, int lda, intptr_t carray, int ldc, intptr_t info, intptr_t dev_info_array, int batch_size):
    """See `cublasSgelsBatched`."""
    with nogil:
        __status__ = cublasSgelsBatched(<Handle>handle, <_Operation>trans, m, n, nrhs, <float* const*>aarray, lda, <float* const*>carray, ldc, <int*>info, <int*>dev_info_array, batch_size)
    check_status(__status__)


cpdef dgels_batched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t aarray, int lda, intptr_t carray, int ldc, intptr_t info, intptr_t dev_info_array, int batch_size):
    """See `cublasDgelsBatched`."""
    with nogil:
        __status__ = cublasDgelsBatched(<Handle>handle, <_Operation>trans, m, n, nrhs, <double* const*>aarray, lda, <double* const*>carray, ldc, <int*>info, <int*>dev_info_array, batch_size)
    check_status(__status__)


cpdef cgels_batched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t aarray, int lda, intptr_t carray, int ldc, intptr_t info, intptr_t dev_info_array, int batch_size):
    """See `cublasCgelsBatched`."""
    with nogil:
        __status__ = cublasCgelsBatched(<Handle>handle, <_Operation>trans, m, n, nrhs, <cuComplex* const*>aarray, lda, <cuComplex* const*>carray, ldc, <int*>info, <int*>dev_info_array, batch_size)
    check_status(__status__)


cpdef zgels_batched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t aarray, int lda, intptr_t carray, int ldc, intptr_t info, intptr_t dev_info_array, int batch_size):
    """See `cublasZgelsBatched`."""
    with nogil:
        __status__ = cublasZgelsBatched(<Handle>handle, <_Operation>trans, m, n, nrhs, <cuDoubleComplex* const*>aarray, lda, <cuDoubleComplex* const*>carray, ldc, <int*>info, <int*>dev_info_array, batch_size)
    check_status(__status__)


cpdef sdgmm(intptr_t handle, int mode, int m, int n, intptr_t a, int lda, intptr_t x, int incx, intptr_t c, int ldc):
    """See `cublasSdgmm`."""
    with nogil:
        __status__ = cublasSdgmm(<Handle>handle, <_SideMode>mode, m, n, <const float*>a, lda, <const float*>x, incx, <float*>c, ldc)
    check_status(__status__)


cpdef ddgmm(intptr_t handle, int mode, int m, int n, intptr_t a, int lda, intptr_t x, int incx, intptr_t c, int ldc):
    """See `cublasDdgmm`."""
    with nogil:
        __status__ = cublasDdgmm(<Handle>handle, <_SideMode>mode, m, n, <const double*>a, lda, <const double*>x, incx, <double*>c, ldc)
    check_status(__status__)


cpdef cdgmm(intptr_t handle, int mode, int m, int n, intptr_t a, int lda, intptr_t x, int incx, intptr_t c, int ldc):
    """See `cublasCdgmm`."""
    with nogil:
        __status__ = cublasCdgmm(<Handle>handle, <_SideMode>mode, m, n, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zdgmm(intptr_t handle, int mode, int m, int n, intptr_t a, int lda, intptr_t x, int incx, intptr_t c, int ldc):
    """See `cublasZdgmm`."""
    with nogil:
        __status__ = cublasZdgmm(<Handle>handle, <_SideMode>mode, m, n, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef stpttr(intptr_t handle, int uplo, int n, intptr_t ap, intptr_t a, int lda):
    """See `cublasStpttr`."""
    with nogil:
        __status__ = cublasStpttr(<Handle>handle, <_FillMode>uplo, n, <const float*>ap, <float*>a, lda)
    check_status(__status__)


cpdef dtpttr(intptr_t handle, int uplo, int n, intptr_t ap, intptr_t a, int lda):
    """See `cublasDtpttr`."""
    with nogil:
        __status__ = cublasDtpttr(<Handle>handle, <_FillMode>uplo, n, <const double*>ap, <double*>a, lda)
    check_status(__status__)


cpdef ctpttr(intptr_t handle, int uplo, int n, intptr_t ap, intptr_t a, int lda):
    """See `cublasCtpttr`."""
    with nogil:
        __status__ = cublasCtpttr(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>ap, <cuComplex*>a, lda)
    check_status(__status__)


cpdef ztpttr(intptr_t handle, int uplo, int n, intptr_t ap, intptr_t a, int lda):
    """See `cublasZtpttr`."""
    with nogil:
        __status__ = cublasZtpttr(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>ap, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef strttp(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ap):
    """See `cublasStrttp`."""
    with nogil:
        __status__ = cublasStrttp(<Handle>handle, <_FillMode>uplo, n, <const float*>a, lda, <float*>ap)
    check_status(__status__)


cpdef dtrttp(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ap):
    """See `cublasDtrttp`."""
    with nogil:
        __status__ = cublasDtrttp(<Handle>handle, <_FillMode>uplo, n, <const double*>a, lda, <double*>ap)
    check_status(__status__)


cpdef ctrttp(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ap):
    """See `cublasCtrttp`."""
    with nogil:
        __status__ = cublasCtrttp(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>a, lda, <cuComplex*>ap)
    check_status(__status__)


cpdef ztrttp(intptr_t handle, int uplo, int n, intptr_t a, int lda, intptr_t ap):
    """See `cublasZtrttp`."""
    with nogil:
        __status__ = cublasZtrttp(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>ap)
    check_status(__status__)


cpdef int get_sm_count_target(intptr_t handle) except? -1:
    """See `cublasGetSmCountTarget`."""
    cdef int sm_count_target
    with nogil:
        __status__ = cublasGetSmCountTarget(<Handle>handle, &sm_count_target)
    check_status(__status__)
    return sm_count_target


cpdef set_sm_count_target(intptr_t handle, int sm_count_target):
    """See `cublasSetSmCountTarget`."""
    with nogil:
        __status__ = cublasSetSmCountTarget(<Handle>handle, sm_count_target)
    check_status(__status__)


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
        __status__ = cublasSgemvBatched(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float* const*>aarray, lda, <const float* const*>xarray, incx, <const float*>beta, <float* const*>yarray, incy, batch_count)
    check_status(__status__)


cpdef dgemv_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t aarray, int lda, intptr_t xarray, int incx, intptr_t beta, intptr_t yarray, int incy, int batch_count):
    """See `cublasDgemvBatched`."""
    with nogil:
        __status__ = cublasDgemvBatched(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double* const*>aarray, lda, <const double* const*>xarray, incx, <const double*>beta, <double* const*>yarray, incy, batch_count)
    check_status(__status__)


cpdef cgemv_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t aarray, int lda, intptr_t xarray, int incx, intptr_t beta, intptr_t yarray, int incy, int batch_count):
    """See `cublasCgemvBatched`."""
    with nogil:
        __status__ = cublasCgemvBatched(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>xarray, incx, <const cuComplex*>beta, <cuComplex* const*>yarray, incy, batch_count)
    check_status(__status__)


cpdef zgemv_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t aarray, int lda, intptr_t xarray, int incx, intptr_t beta, intptr_t yarray, int incy, int batch_count):
    """See `cublasZgemvBatched`."""
    with nogil:
        __status__ = cublasZgemvBatched(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>aarray, lda, <const cuDoubleComplex* const*>xarray, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex* const*>yarray, incy, batch_count)
    check_status(__status__)


cpdef sgemv_strided_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t x, int incx, long long int strid_ex, intptr_t beta, intptr_t y, int incy, long long int stridey, int batch_count):
    """See `cublasSgemvStridedBatched`."""
    with nogil:
        __status__ = cublasSgemvStridedBatched(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float*>a, lda, stride_a, <const float*>x, incx, strid_ex, <const float*>beta, <float*>y, incy, stridey, batch_count)
    check_status(__status__)


cpdef dgemv_strided_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t x, int incx, long long int strid_ex, intptr_t beta, intptr_t y, int incy, long long int stridey, int batch_count):
    """See `cublasDgemvStridedBatched`."""
    with nogil:
        __status__ = cublasDgemvStridedBatched(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double*>a, lda, stride_a, <const double*>x, incx, strid_ex, <const double*>beta, <double*>y, incy, stridey, batch_count)
    check_status(__status__)


cpdef cgemv_strided_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t x, int incx, long long int strid_ex, intptr_t beta, intptr_t y, int incy, long long int stridey, int batch_count):
    """See `cublasCgemvStridedBatched`."""
    with nogil:
        __status__ = cublasCgemvStridedBatched(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>x, incx, strid_ex, <const cuComplex*>beta, <cuComplex*>y, incy, stridey, batch_count)
    check_status(__status__)


cpdef zgemv_strided_batched(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t a, int lda, long long int stride_a, intptr_t x, int incx, long long int strid_ex, intptr_t beta, intptr_t y, int incy, long long int stridey, int batch_count):
    """See `cublasZgemvStridedBatched`."""
    with nogil:
        __status__ = cublasZgemvStridedBatched(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, stride_a, <const cuDoubleComplex*>x, incx, strid_ex, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy, stridey, batch_count)
    check_status(__status__)


cpdef set_vector_64(int64_t n, int64_t elem_size, intptr_t x, int64_t incx, intptr_t device_ptr, int64_t incy):
    """See `cublasSetVector_64`."""
    with nogil:
        __status__ = cublasSetVector_64(n, elem_size, <const void*>x, incx, <void*>device_ptr, incy)
    check_status(__status__)


cpdef get_vector_64(int64_t n, int64_t elem_size, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasGetVector_64`."""
    with nogil:
        __status__ = cublasGetVector_64(n, elem_size, <const void*>x, incx, <void*>y, incy)
    check_status(__status__)


cpdef set_matrix_64(int64_t rows, int64_t cols, int64_t elem_size, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasSetMatrix_64`."""
    with nogil:
        __status__ = cublasSetMatrix_64(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb)
    check_status(__status__)


cpdef get_matrix_64(int64_t rows, int64_t cols, int64_t elem_size, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasGetMatrix_64`."""
    with nogil:
        __status__ = cublasGetMatrix_64(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb)
    check_status(__status__)


cpdef set_vector_async_64(int64_t n, int64_t elem_size, intptr_t host_ptr, int64_t incx, intptr_t device_ptr, int64_t incy, intptr_t stream):
    """See `cublasSetVectorAsync_64`."""
    with nogil:
        __status__ = cublasSetVectorAsync_64(n, elem_size, <const void*>host_ptr, incx, <void*>device_ptr, incy, <Stream>stream)
    check_status(__status__)


cpdef get_vector_async_64(int64_t n, int64_t elem_size, intptr_t device_ptr, int64_t incx, intptr_t host_ptr, int64_t incy, intptr_t stream):
    """See `cublasGetVectorAsync_64`."""
    with nogil:
        __status__ = cublasGetVectorAsync_64(n, elem_size, <const void*>device_ptr, incx, <void*>host_ptr, incy, <Stream>stream)
    check_status(__status__)


cpdef set_matrix_async_64(int64_t rows, int64_t cols, int64_t elem_size, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t stream):
    """See `cublasSetMatrixAsync_64`."""
    with nogil:
        __status__ = cublasSetMatrixAsync_64(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb, <Stream>stream)
    check_status(__status__)


cpdef get_matrix_async_64(int64_t rows, int64_t cols, int64_t elem_size, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t stream):
    """See `cublasGetMatrixAsync_64`."""
    with nogil:
        __status__ = cublasGetMatrixAsync_64(rows, cols, elem_size, <const void*>a, lda, <void*>b, ldb, <Stream>stream)
    check_status(__status__)


cpdef nrm2ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t result, int result_type, int execution_type):
    """See `cublasNrm2Ex_64`."""
    with nogil:
        __status__ = cublasNrm2Ex_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(__status__)


cpdef snrm2_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasSnrm2_64`."""
    with nogil:
        __status__ = cublasSnrm2_64(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(__status__)


cpdef dnrm2_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasDnrm2_64`."""
    with nogil:
        __status__ = cublasDnrm2_64(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(__status__)


cpdef scnrm2_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasScnrm2_64`."""
    with nogil:
        __status__ = cublasScnrm2_64(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(__status__)


cpdef dznrm2_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasDznrm2_64`."""
    with nogil:
        __status__ = cublasDznrm2_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(__status__)


cpdef dot_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy, intptr_t result, int result_type, int execution_type):
    """See `cublasDotEx_64`."""
    with nogil:
        __status__ = cublasDotEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <const void*>y, <DataType>y_type, incy, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(__status__)


cpdef dotc_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy, intptr_t result, int result_type, int execution_type):
    """See `cublasDotcEx_64`."""
    with nogil:
        __status__ = cublasDotcEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <const void*>y, <DataType>y_type, incy, <void*>result, <DataType>result_type, <DataType>execution_type)
    check_status(__status__)


cpdef sdot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasSdot_64`."""
    with nogil:
        __status__ = cublasSdot_64(<Handle>handle, n, <const float*>x, incx, <const float*>y, incy, <float*>result)
    check_status(__status__)


cpdef ddot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasDdot_64`."""
    with nogil:
        __status__ = cublasDdot_64(<Handle>handle, n, <const double*>x, incx, <const double*>y, incy, <double*>result)
    check_status(__status__)


cpdef cdotu_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasCdotu_64`."""
    with nogil:
        __status__ = cublasCdotu_64(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(__status__)


cpdef cdotc_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasCdotc_64`."""
    with nogil:
        __status__ = cublasCdotc_64(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(__status__)


cpdef zdotu_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasZdotu_64`."""
    with nogil:
        __status__ = cublasZdotu_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(__status__)


cpdef zdotc_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t result):
    """See `cublasZdotc_64`."""
    with nogil:
        __status__ = cublasZdotc_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(__status__)


cpdef scal_ex_64(intptr_t handle, int64_t n, intptr_t alpha, int alpha_type, intptr_t x, int x_type, int64_t incx, int execution_type):
    """See `cublasScalEx_64`."""
    with nogil:
        __status__ = cublasScalEx_64(<Handle>handle, n, <const void*>alpha, <DataType>alpha_type, <void*>x, <DataType>x_type, incx, <DataType>execution_type)
    check_status(__status__)


cpdef sscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasSscal_64`."""
    with nogil:
        __status__ = cublasSscal_64(<Handle>handle, n, <const float*>alpha, <float*>x, incx)
    check_status(__status__)


cpdef dscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasDscal_64`."""
    with nogil:
        __status__ = cublasDscal_64(<Handle>handle, n, <const double*>alpha, <double*>x, incx)
    check_status(__status__)


cpdef cscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasCscal_64`."""
    with nogil:
        __status__ = cublasCscal_64(<Handle>handle, n, <const cuComplex*>alpha, <cuComplex*>x, incx)
    check_status(__status__)


cpdef csscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasCsscal_64`."""
    with nogil:
        __status__ = cublasCsscal_64(<Handle>handle, n, <const float*>alpha, <cuComplex*>x, incx)
    check_status(__status__)


cpdef zscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasZscal_64`."""
    with nogil:
        __status__ = cublasZscal_64(<Handle>handle, n, <const cuDoubleComplex*>alpha, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef zdscal_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx):
    """See `cublasZdscal_64`."""
    with nogil:
        __status__ = cublasZdscal_64(<Handle>handle, n, <const double*>alpha, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef axpy_ex_64(intptr_t handle, int64_t n, intptr_t alpha, int alpha_type, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy, int executiontype):
    """See `cublasAxpyEx_64`."""
    with nogil:
        __status__ = cublasAxpyEx_64(<Handle>handle, n, <const void*>alpha, <DataType>alpha_type, <const void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <DataType>executiontype)
    check_status(__status__)


cpdef saxpy_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasSaxpy_64`."""
    with nogil:
        __status__ = cublasSaxpy_64(<Handle>handle, n, <const float*>alpha, <const float*>x, incx, <float*>y, incy)
    check_status(__status__)


cpdef daxpy_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasDaxpy_64`."""
    with nogil:
        __status__ = cublasDaxpy_64(<Handle>handle, n, <const double*>alpha, <const double*>x, incx, <double*>y, incy)
    check_status(__status__)


cpdef caxpy_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasCaxpy_64`."""
    with nogil:
        __status__ = cublasCaxpy_64(<Handle>handle, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zaxpy_64(intptr_t handle, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasZaxpy_64`."""
    with nogil:
        __status__ = cublasZaxpy_64(<Handle>handle, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef copy_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy):
    """See `cublasCopyEx_64`."""
    with nogil:
        __status__ = cublasCopyEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy)
    check_status(__status__)


cpdef scopy_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasScopy_64`."""
    with nogil:
        __status__ = cublasScopy_64(<Handle>handle, n, <const float*>x, incx, <float*>y, incy)
    check_status(__status__)


cpdef dcopy_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasDcopy_64`."""
    with nogil:
        __status__ = cublasDcopy_64(<Handle>handle, n, <const double*>x, incx, <double*>y, incy)
    check_status(__status__)


cpdef ccopy_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasCcopy_64`."""
    with nogil:
        __status__ = cublasCcopy_64(<Handle>handle, n, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zcopy_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasZcopy_64`."""
    with nogil:
        __status__ = cublasZcopy_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef sswap_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasSswap_64`."""
    with nogil:
        __status__ = cublasSswap_64(<Handle>handle, n, <float*>x, incx, <float*>y, incy)
    check_status(__status__)


cpdef dswap_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasDswap_64`."""
    with nogil:
        __status__ = cublasDswap_64(<Handle>handle, n, <double*>x, incx, <double*>y, incy)
    check_status(__status__)


cpdef cswap_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasCswap_64`."""
    with nogil:
        __status__ = cublasCswap_64(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zswap_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy):
    """See `cublasZswap_64`."""
    with nogil:
        __status__ = cublasZswap_64(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef swap_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy):
    """See `cublasSwapEx_64`."""
    with nogil:
        __status__ = cublasSwapEx_64(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy)
    check_status(__status__)


cpdef isamax_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIsamax_64`."""
    with nogil:
        __status__ = cublasIsamax_64(<Handle>handle, n, <const float*>x, incx, <int64_t*>result)
    check_status(__status__)


cpdef idamax_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIdamax_64`."""
    with nogil:
        __status__ = cublasIdamax_64(<Handle>handle, n, <const double*>x, incx, <int64_t*>result)
    check_status(__status__)


cpdef icamax_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIcamax_64`."""
    with nogil:
        __status__ = cublasIcamax_64(<Handle>handle, n, <const cuComplex*>x, incx, <int64_t*>result)
    check_status(__status__)


cpdef izamax_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIzamax_64`."""
    with nogil:
        __status__ = cublasIzamax_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int64_t*>result)
    check_status(__status__)


cpdef iamax_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t result):
    """See `cublasIamaxEx_64`."""
    with nogil:
        __status__ = cublasIamaxEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <int64_t*>result)
    check_status(__status__)


cpdef isamin_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIsamin_64`."""
    with nogil:
        __status__ = cublasIsamin_64(<Handle>handle, n, <const float*>x, incx, <int64_t*>result)
    check_status(__status__)


cpdef idamin_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIdamin_64`."""
    with nogil:
        __status__ = cublasIdamin_64(<Handle>handle, n, <const double*>x, incx, <int64_t*>result)
    check_status(__status__)


cpdef icamin_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIcamin_64`."""
    with nogil:
        __status__ = cublasIcamin_64(<Handle>handle, n, <const cuComplex*>x, incx, <int64_t*>result)
    check_status(__status__)


cpdef izamin_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasIzamin_64`."""
    with nogil:
        __status__ = cublasIzamin_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int64_t*>result)
    check_status(__status__)


cpdef iamin_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t result):
    """See `cublasIaminEx_64`."""
    with nogil:
        __status__ = cublasIaminEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <int64_t*>result)
    check_status(__status__)


cpdef asum_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t result, int result_type, int executiontype):
    """See `cublasAsumEx_64`."""
    with nogil:
        __status__ = cublasAsumEx_64(<Handle>handle, n, <const void*>x, <DataType>x_type, incx, <void*>result, <DataType>result_type, <DataType>executiontype)
    check_status(__status__)


cpdef sasum_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasSasum_64`."""
    with nogil:
        __status__ = cublasSasum_64(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(__status__)


cpdef dasum_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasDasum_64`."""
    with nogil:
        __status__ = cublasDasum_64(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(__status__)


cpdef scasum_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasScasum_64`."""
    with nogil:
        __status__ = cublasScasum_64(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(__status__)


cpdef dzasum_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t result):
    """See `cublasDzasum_64`."""
    with nogil:
        __status__ = cublasDzasum_64(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(__status__)


cpdef srot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasSrot_64`."""
    with nogil:
        __status__ = cublasSrot_64(<Handle>handle, n, <float*>x, incx, <float*>y, incy, <const float*>c, <const float*>s)
    check_status(__status__)


cpdef drot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasDrot_64`."""
    with nogil:
        __status__ = cublasDrot_64(<Handle>handle, n, <double*>x, incx, <double*>y, incy, <const double*>c, <const double*>s)
    check_status(__status__)


cpdef crot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasCrot_64`."""
    with nogil:
        __status__ = cublasCrot_64(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy, <const float*>c, <const cuComplex*>s)
    check_status(__status__)


cpdef csrot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasCsrot_64`."""
    with nogil:
        __status__ = cublasCsrot_64(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy, <const float*>c, <const float*>s)
    check_status(__status__)


cpdef zrot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasZrot_64`."""
    with nogil:
        __status__ = cublasZrot_64(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy, <const double*>c, <const cuDoubleComplex*>s)
    check_status(__status__)


cpdef zdrot_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t c, intptr_t s):
    """See `cublasZdrot_64`."""
    with nogil:
        __status__ = cublasZdrot_64(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy, <const double*>c, <const double*>s)
    check_status(__status__)


cpdef rot_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy, intptr_t c, intptr_t s, int cs_type, int executiontype):
    """See `cublasRotEx_64`."""
    with nogil:
        __status__ = cublasRotEx_64(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <const void*>c, <const void*>s, <DataType>cs_type, <DataType>executiontype)
    check_status(__status__)


cpdef srotm_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t param):
    """See `cublasSrotm_64`."""
    with nogil:
        __status__ = cublasSrotm_64(<Handle>handle, n, <float*>x, incx, <float*>y, incy, <const float*>param)
    check_status(__status__)


cpdef drotm_64(intptr_t handle, int64_t n, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t param):
    """See `cublasDrotm_64`."""
    with nogil:
        __status__ = cublasDrotm_64(<Handle>handle, n, <double*>x, incx, <double*>y, incy, <const double*>param)
    check_status(__status__)


cpdef rotm_ex_64(intptr_t handle, int64_t n, intptr_t x, int x_type, int64_t incx, intptr_t y, int y_type, int64_t incy, intptr_t param, int param_type, int executiontype):
    """See `cublasRotmEx_64`."""
    with nogil:
        __status__ = cublasRotmEx_64(<Handle>handle, n, <void*>x, <DataType>x_type, incx, <void*>y, <DataType>y_type, incy, <const void*>param, <DataType>param_type, <DataType>executiontype)
    check_status(__status__)


cpdef sgemv_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasSgemv_64`."""
    with nogil:
        __status__ = cublasSgemv_64(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(__status__)


cpdef dgemv_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasDgemv_64`."""
    with nogil:
        __status__ = cublasDgemv_64(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(__status__)


cpdef cgemv_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasCgemv_64`."""
    with nogil:
        __status__ = cublasCgemv_64(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zgemv_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZgemv_64`."""
    with nogil:
        __status__ = cublasZgemv_64(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef sgbmv_64(intptr_t handle, int trans, int64_t m, int64_t n, int64_t kl, int64_t ku, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasSgbmv_64`."""
    with nogil:
        __status__ = cublasSgbmv_64(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(__status__)


cpdef dgbmv_64(intptr_t handle, int trans, int64_t m, int64_t n, int64_t kl, int64_t ku, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasDgbmv_64`."""
    with nogil:
        __status__ = cublasDgbmv_64(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(__status__)


cpdef cgbmv_64(intptr_t handle, int trans, int64_t m, int64_t n, int64_t kl, int64_t ku, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasCgbmv_64`."""
    with nogil:
        __status__ = cublasCgbmv_64(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zgbmv_64(intptr_t handle, int trans, int64_t m, int64_t n, int64_t kl, int64_t ku, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZgbmv_64`."""
    with nogil:
        __status__ = cublasZgbmv_64(<Handle>handle, <_Operation>trans, m, n, kl, ku, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef strmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasStrmv_64`."""
    with nogil:
        __status__ = cublasStrmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>a, lda, <float*>x, incx)
    check_status(__status__)


cpdef dtrmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasDtrmv_64`."""
    with nogil:
        __status__ = cublasDtrmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>a, lda, <double*>x, incx)
    check_status(__status__)


cpdef ctrmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasCtrmv_64`."""
    with nogil:
        __status__ = cublasCtrmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztrmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasZtrmv_64`."""
    with nogil:
        __status__ = cublasZtrmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef stbmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasStbmv_64`."""
    with nogil:
        __status__ = cublasStbmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const float*>a, lda, <float*>x, incx)
    check_status(__status__)


cpdef dtbmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasDtbmv_64`."""
    with nogil:
        __status__ = cublasDtbmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const double*>a, lda, <double*>x, incx)
    check_status(__status__)


cpdef ctbmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasCtbmv_64`."""
    with nogil:
        __status__ = cublasCtbmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztbmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasZtbmv_64`."""
    with nogil:
        __status__ = cublasZtbmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef stpmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasStpmv_64`."""
    with nogil:
        __status__ = cublasStpmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>ap, <float*>x, incx)
    check_status(__status__)


cpdef dtpmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasDtpmv_64`."""
    with nogil:
        __status__ = cublasDtpmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>ap, <double*>x, incx)
    check_status(__status__)


cpdef ctpmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasCtpmv_64`."""
    with nogil:
        __status__ = cublasCtpmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>ap, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztpmv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasZtpmv_64`."""
    with nogil:
        __status__ = cublasZtpmv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>ap, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef strsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasStrsv_64`."""
    with nogil:
        __status__ = cublasStrsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>a, lda, <float*>x, incx)
    check_status(__status__)


cpdef dtrsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasDtrsv_64`."""
    with nogil:
        __status__ = cublasDtrsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>a, lda, <double*>x, incx)
    check_status(__status__)


cpdef ctrsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasCtrsv_64`."""
    with nogil:
        __status__ = cublasCtrsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztrsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasZtrsv_64`."""
    with nogil:
        __status__ = cublasZtrsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef stpsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasStpsv_64`."""
    with nogil:
        __status__ = cublasStpsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const float*>ap, <float*>x, incx)
    check_status(__status__)


cpdef dtpsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasDtpsv_64`."""
    with nogil:
        __status__ = cublasDtpsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const double*>ap, <double*>x, incx)
    check_status(__status__)


cpdef ctpsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasCtpsv_64`."""
    with nogil:
        __status__ = cublasCtpsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuComplex*>ap, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztpsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, intptr_t ap, intptr_t x, int64_t incx):
    """See `cublasZtpsv_64`."""
    with nogil:
        __status__ = cublasZtpsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, <const cuDoubleComplex*>ap, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef stbsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasStbsv_64`."""
    with nogil:
        __status__ = cublasStbsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const float*>a, lda, <float*>x, incx)
    check_status(__status__)


cpdef dtbsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasDtbsv_64`."""
    with nogil:
        __status__ = cublasDtbsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const double*>a, lda, <double*>x, incx)
    check_status(__status__)


cpdef ctbsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasCtbsv_64`."""
    with nogil:
        __status__ = cublasCtbsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuComplex*>a, lda, <cuComplex*>x, incx)
    check_status(__status__)


cpdef ztbsv_64(intptr_t handle, int uplo, int trans, int diag, int64_t n, int64_t k, intptr_t a, int64_t lda, intptr_t x, int64_t incx):
    """See `cublasZtbsv_64`."""
    with nogil:
        __status__ = cublasZtbsv_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, n, k, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>x, incx)
    check_status(__status__)


cpdef ssymv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasSsymv_64`."""
    with nogil:
        __status__ = cublasSsymv_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(__status__)


cpdef dsymv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasDsymv_64`."""
    with nogil:
        __status__ = cublasDsymv_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(__status__)


cpdef csymv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasCsymv_64`."""
    with nogil:
        __status__ = cublasCsymv_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zsymv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZsymv_64`."""
    with nogil:
        __status__ = cublasZsymv_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef chemv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasChemv_64`."""
    with nogil:
        __status__ = cublasChemv_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zhemv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZhemv_64`."""
    with nogil:
        __status__ = cublasZhemv_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef ssbmv_64(intptr_t handle, int uplo, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasSsbmv_64`."""
    with nogil:
        __status__ = cublasSsbmv_64(<Handle>handle, <_FillMode>uplo, n, k, <const float*>alpha, <const float*>a, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(__status__)


cpdef dsbmv_64(intptr_t handle, int uplo, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasDsbmv_64`."""
    with nogil:
        __status__ = cublasDsbmv_64(<Handle>handle, <_FillMode>uplo, n, k, <const double*>alpha, <const double*>a, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(__status__)


cpdef chbmv_64(intptr_t handle, int uplo, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasChbmv_64`."""
    with nogil:
        __status__ = cublasChbmv_64(<Handle>handle, <_FillMode>uplo, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zhbmv_64(intptr_t handle, int uplo, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZhbmv_64`."""
    with nogil:
        __status__ = cublasZhbmv_64(<Handle>handle, <_FillMode>uplo, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef sspmv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t ap, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasSspmv_64`."""
    with nogil:
        __status__ = cublasSspmv_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>ap, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(__status__)


cpdef dspmv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t ap, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasDspmv_64`."""
    with nogil:
        __status__ = cublasDspmv_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>ap, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(__status__)


cpdef chpmv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t ap, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasChpmv_64`."""
    with nogil:
        __status__ = cublasChpmv_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>ap, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(__status__)


cpdef zhpmv_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t ap, intptr_t x, int64_t incx, intptr_t beta, intptr_t y, int64_t incy):
    """See `cublasZhpmv_64`."""
    with nogil:
        __status__ = cublasZhpmv_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>ap, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(__status__)


cpdef sger_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasSger_64`."""
    with nogil:
        __status__ = cublasSger_64(<Handle>handle, m, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>a, lda)
    check_status(__status__)


cpdef dger_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasDger_64`."""
    with nogil:
        __status__ = cublasDger_64(<Handle>handle, m, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>a, lda)
    check_status(__status__)


cpdef cgeru_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasCgeru_64`."""
    with nogil:
        __status__ = cublasCgeru_64(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(__status__)


cpdef cgerc_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasCgerc_64`."""
    with nogil:
        __status__ = cublasCgerc_64(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(__status__)


cpdef zgeru_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasZgeru_64`."""
    with nogil:
        __status__ = cublasZgeru_64(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef zgerc_64(intptr_t handle, int64_t m, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasZgerc_64`."""
    with nogil:
        __status__ = cublasZgerc_64(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef ssyr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasSsyr_64`."""
    with nogil:
        __status__ = cublasSsyr_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <float*>a, lda)
    check_status(__status__)


cpdef dsyr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasDsyr_64`."""
    with nogil:
        __status__ = cublasDsyr_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <double*>a, lda)
    check_status(__status__)


cpdef csyr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasCsyr_64`."""
    with nogil:
        __status__ = cublasCsyr_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>a, lda)
    check_status(__status__)


cpdef zsyr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasZsyr_64`."""
    with nogil:
        __status__ = cublasZsyr_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef cher_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasCher_64`."""
    with nogil:
        __status__ = cublasCher_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const cuComplex*>x, incx, <cuComplex*>a, lda)
    check_status(__status__)


cpdef zher_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t a, int64_t lda):
    """See `cublasZher_64`."""
    with nogil:
        __status__ = cublasZher_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef sspr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t ap):
    """See `cublasSspr_64`."""
    with nogil:
        __status__ = cublasSspr_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <float*>ap)
    check_status(__status__)


cpdef dspr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t ap):
    """See `cublasDspr_64`."""
    with nogil:
        __status__ = cublasDspr_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <double*>ap)
    check_status(__status__)


cpdef chpr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t ap):
    """See `cublasChpr_64`."""
    with nogil:
        __status__ = cublasChpr_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const cuComplex*>x, incx, <cuComplex*>ap)
    check_status(__status__)


cpdef zhpr_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t ap):
    """See `cublasZhpr_64`."""
    with nogil:
        __status__ = cublasZhpr_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>ap)
    check_status(__status__)


cpdef ssyr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasSsyr2_64`."""
    with nogil:
        __status__ = cublasSsyr2_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>a, lda)
    check_status(__status__)


cpdef dsyr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasDsyr2_64`."""
    with nogil:
        __status__ = cublasDsyr2_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>a, lda)
    check_status(__status__)


cpdef csyr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasCsyr2_64`."""
    with nogil:
        __status__ = cublasCsyr2_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(__status__)


cpdef zsyr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasZsyr2_64`."""
    with nogil:
        __status__ = cublasZsyr2_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef cher2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasCher2_64`."""
    with nogil:
        __status__ = cublasCher2_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>a, lda)
    check_status(__status__)


cpdef zher2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t a, int64_t lda):
    """See `cublasZher2_64`."""
    with nogil:
        __status__ = cublasZher2_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>a, lda)
    check_status(__status__)


cpdef sspr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t ap):
    """See `cublasSspr2_64`."""
    with nogil:
        __status__ = cublasSspr2_64(<Handle>handle, <_FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>ap)
    check_status(__status__)


cpdef dspr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t ap):
    """See `cublasDspr2_64`."""
    with nogil:
        __status__ = cublasDspr2_64(<Handle>handle, <_FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>ap)
    check_status(__status__)


cpdef chpr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t ap):
    """See `cublasChpr2_64`."""
    with nogil:
        __status__ = cublasChpr2_64(<Handle>handle, <_FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>ap)
    check_status(__status__)


cpdef zhpr2_64(intptr_t handle, int uplo, int64_t n, intptr_t alpha, intptr_t x, int64_t incx, intptr_t y, int64_t incy, intptr_t ap):
    """See `cublasZhpr2_64`."""
    with nogil:
        __status__ = cublasZhpr2_64(<Handle>handle, <_FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>ap)
    check_status(__status__)


cpdef sgemv_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t xarray, int64_t incx, intptr_t beta, intptr_t yarray, int64_t incy, int64_t batch_count):
    """See `cublasSgemvBatched_64`."""
    with nogil:
        __status__ = cublasSgemvBatched_64(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float* const*>aarray, lda, <const float* const*>xarray, incx, <const float*>beta, <float* const*>yarray, incy, batch_count)
    check_status(__status__)


cpdef dgemv_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t xarray, int64_t incx, intptr_t beta, intptr_t yarray, int64_t incy, int64_t batch_count):
    """See `cublasDgemvBatched_64`."""
    with nogil:
        __status__ = cublasDgemvBatched_64(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double* const*>aarray, lda, <const double* const*>xarray, incx, <const double*>beta, <double* const*>yarray, incy, batch_count)
    check_status(__status__)


cpdef cgemv_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t xarray, int64_t incx, intptr_t beta, intptr_t yarray, int64_t incy, int64_t batch_count):
    """See `cublasCgemvBatched_64`."""
    with nogil:
        __status__ = cublasCgemvBatched_64(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>xarray, incx, <const cuComplex*>beta, <cuComplex* const*>yarray, incy, batch_count)
    check_status(__status__)


cpdef zgemv_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t xarray, int64_t incx, intptr_t beta, intptr_t yarray, int64_t incy, int64_t batch_count):
    """See `cublasZgemvBatched_64`."""
    with nogil:
        __status__ = cublasZgemvBatched_64(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>aarray, lda, <const cuDoubleComplex* const*>xarray, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex* const*>yarray, incy, batch_count)
    check_status(__status__)


cpdef sgemv_strided_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t x, int64_t incx, long long int strid_ex, intptr_t beta, intptr_t y, int64_t incy, long long int stridey, int64_t batch_count):
    """See `cublasSgemvStridedBatched_64`."""
    with nogil:
        __status__ = cublasSgemvStridedBatched_64(<Handle>handle, <_Operation>trans, m, n, <const float*>alpha, <const float*>a, lda, stride_a, <const float*>x, incx, strid_ex, <const float*>beta, <float*>y, incy, stridey, batch_count)
    check_status(__status__)


cpdef dgemv_strided_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t x, int64_t incx, long long int strid_ex, intptr_t beta, intptr_t y, int64_t incy, long long int stridey, int64_t batch_count):
    """See `cublasDgemvStridedBatched_64`."""
    with nogil:
        __status__ = cublasDgemvStridedBatched_64(<Handle>handle, <_Operation>trans, m, n, <const double*>alpha, <const double*>a, lda, stride_a, <const double*>x, incx, strid_ex, <const double*>beta, <double*>y, incy, stridey, batch_count)
    check_status(__status__)


cpdef cgemv_strided_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t x, int64_t incx, long long int strid_ex, intptr_t beta, intptr_t y, int64_t incy, long long int stridey, int64_t batch_count):
    """See `cublasCgemvStridedBatched_64`."""
    with nogil:
        __status__ = cublasCgemvStridedBatched_64(<Handle>handle, <_Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>x, incx, strid_ex, <const cuComplex*>beta, <cuComplex*>y, incy, stridey, batch_count)
    check_status(__status__)


cpdef zgemv_strided_batched_64(intptr_t handle, int trans, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t x, int64_t incx, long long int strid_ex, intptr_t beta, intptr_t y, int64_t incy, long long int stridey, int64_t batch_count):
    """See `cublasZgemvStridedBatched_64`."""
    with nogil:
        __status__ = cublasZgemvStridedBatched_64(<Handle>handle, <_Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, stride_a, <const cuDoubleComplex*>x, incx, strid_ex, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy, stridey, batch_count)
    check_status(__status__)


cpdef sgemm_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasSgemm_64`."""
    with nogil:
        __status__ = cublasSgemm_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(__status__)


cpdef dgemm_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasDgemm_64`."""
    with nogil:
        __status__ = cublasDgemm_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(__status__)


cpdef cgemm_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCgemm_64`."""
    with nogil:
        __status__ = cublasCgemm_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef cgemm3m_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCgemm3m_64`."""
    with nogil:
        __status__ = cublasCgemm3m_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef cgemm3m_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t b, int btype, int64_t ldb, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCgemm3mEx_64`."""
    with nogil:
        __status__ = cublasCgemm3mEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef zgemm_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZgemm_64`."""
    with nogil:
        __status__ = cublasZgemm_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef zgemm3m_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZgemm3m_64`."""
    with nogil:
        __status__ = cublasZgemm3m_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef sgemm_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t b, int btype, int64_t ldb, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasSgemmEx_64`."""
    with nogil:
        __status__ = cublasSgemmEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef gemm_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t b, int btype, int64_t ldb, intptr_t beta, intptr_t c, int ctype, int64_t ldc, int compute_type, int algo):
    """See `cublasGemmEx_64`."""
    with nogil:
        __status__ = cublasGemmEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const void*>beta, <void*>c, <DataType>ctype, ldc, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(__status__)


cpdef cgemm_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t b, int btype, int64_t ldb, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCgemmEx_64`."""
    with nogil:
        __status__ = cublasCgemmEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const void*>b, <DataType>btype, ldb, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef ssyrk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasSsyrk_64`."""
    with nogil:
        __status__ = cublasSsyrk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>beta, <float*>c, ldc)
    check_status(__status__)


cpdef dsyrk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasDsyrk_64`."""
    with nogil:
        __status__ = cublasDsyrk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>beta, <double*>c, ldc)
    check_status(__status__)


cpdef csyrk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCsyrk_64`."""
    with nogil:
        __status__ = cublasCsyrk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zsyrk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZsyrk_64`."""
    with nogil:
        __status__ = cublasZsyrk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef csyrk_ex_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCsyrkEx_64`."""
    with nogil:
        __status__ = cublasCsyrkEx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef csyrk3m_ex_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCsyrk3mEx_64`."""
    with nogil:
        __status__ = cublasCsyrk3mEx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const void*>a, <DataType>atype, lda, <const cuComplex*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef cherk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCherk_64`."""
    with nogil:
        __status__ = cublasCherk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const cuComplex*>a, lda, <const float*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zherk_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZherk_64`."""
    with nogil:
        __status__ = cublasZherk_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const cuDoubleComplex*>a, lda, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef cherk_ex_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCherkEx_64`."""
    with nogil:
        __status__ = cublasCherkEx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef cherk3m_ex_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, intptr_t beta, intptr_t c, int ctype, int64_t ldc):
    """See `cublasCherk3mEx_64`."""
    with nogil:
        __status__ = cublasCherk3mEx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const void*>a, <DataType>atype, lda, <const float*>beta, <void*>c, <DataType>ctype, ldc)
    check_status(__status__)


cpdef ssyr2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasSsyr2k_64`."""
    with nogil:
        __status__ = cublasSsyr2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(__status__)


cpdef dsyr2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasDsyr2k_64`."""
    with nogil:
        __status__ = cublasDsyr2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(__status__)


cpdef csyr2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCsyr2k_64`."""
    with nogil:
        __status__ = cublasCsyr2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zsyr2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZsyr2k_64`."""
    with nogil:
        __status__ = cublasZsyr2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef cher2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCher2k_64`."""
    with nogil:
        __status__ = cublasCher2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const float*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zher2k_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZher2k_64`."""
    with nogil:
        __status__ = cublasZher2k_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef ssyrkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasSsyrkx_64`."""
    with nogil:
        __status__ = cublasSsyrkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(__status__)


cpdef dsyrkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasDsyrkx_64`."""
    with nogil:
        __status__ = cublasDsyrkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(__status__)


cpdef csyrkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCsyrkx_64`."""
    with nogil:
        __status__ = cublasCsyrkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zsyrkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZsyrkx_64`."""
    with nogil:
        __status__ = cublasZsyrkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef cherkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCherkx_64`."""
    with nogil:
        __status__ = cublasCherkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const float*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zherkx_64(intptr_t handle, int uplo, int trans, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZherkx_64`."""
    with nogil:
        __status__ = cublasZherkx_64(<Handle>handle, <_FillMode>uplo, <_Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const double*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef ssymm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasSsymm_64`."""
    with nogil:
        __status__ = cublasSsymm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <const float*>beta, <float*>c, ldc)
    check_status(__status__)


cpdef dsymm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasDsymm_64`."""
    with nogil:
        __status__ = cublasDsymm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <const double*>beta, <double*>c, ldc)
    check_status(__status__)


cpdef csymm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasCsymm_64`."""
    with nogil:
        __status__ = cublasCsymm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zsymm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZsymm_64`."""
    with nogil:
        __status__ = cublasZsymm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef chemm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasChemm_64`."""
    with nogil:
        __status__ = cublasChemm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <const cuComplex*>beta, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zhemm_64(intptr_t handle, int side, int uplo, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t beta, intptr_t c, int64_t ldc):
    """See `cublasZhemm_64`."""
    with nogil:
        __status__ = cublasZhemm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef strsm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasStrsm_64`."""
    with nogil:
        __status__ = cublasStrsm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float*>a, lda, <float*>b, ldb)
    check_status(__status__)


cpdef dtrsm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasDtrsm_64`."""
    with nogil:
        __status__ = cublasDtrsm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double*>a, lda, <double*>b, ldb)
    check_status(__status__)


cpdef ctrsm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasCtrsm_64`."""
    with nogil:
        __status__ = cublasCtrsm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <cuComplex*>b, ldb)
    check_status(__status__)


cpdef ztrsm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb):
    """See `cublasZtrsm_64`."""
    with nogil:
        __status__ = cublasZtrsm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <cuDoubleComplex*>b, ldb)
    check_status(__status__)


cpdef strmm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasStrmm_64`."""
    with nogil:
        __status__ = cublasStrmm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float*>a, lda, <const float*>b, ldb, <float*>c, ldc)
    check_status(__status__)


cpdef dtrmm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasDtrmm_64`."""
    with nogil:
        __status__ = cublasDtrmm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double*>a, lda, <const double*>b, ldb, <double*>c, ldc)
    check_status(__status__)


cpdef ctrmm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasCtrmm_64`."""
    with nogil:
        __status__ = cublasCtrmm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>b, ldb, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef ztrmm_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasZtrmm_64`."""
    with nogil:
        __status__ = cublasZtrmm_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>b, ldb, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef sgemm_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t barray, int64_t ldb, intptr_t beta, intptr_t carray, int64_t ldc, int64_t batch_count):
    """See `cublasSgemmBatched_64`."""
    with nogil:
        __status__ = cublasSgemmBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float* const*>aarray, lda, <const float* const*>barray, ldb, <const float*>beta, <float* const*>carray, ldc, batch_count)
    check_status(__status__)


cpdef dgemm_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t barray, int64_t ldb, intptr_t beta, intptr_t carray, int64_t ldc, int64_t batch_count):
    """See `cublasDgemmBatched_64`."""
    with nogil:
        __status__ = cublasDgemmBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double* const*>aarray, lda, <const double* const*>barray, ldb, <const double*>beta, <double* const*>carray, ldc, batch_count)
    check_status(__status__)


cpdef cgemm_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t barray, int64_t ldb, intptr_t beta, intptr_t carray, int64_t ldc, int64_t batch_count):
    """See `cublasCgemmBatched_64`."""
    with nogil:
        __status__ = cublasCgemmBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>barray, ldb, <const cuComplex*>beta, <cuComplex* const*>carray, ldc, batch_count)
    check_status(__status__)


cpdef cgemm3m_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t barray, int64_t ldb, intptr_t beta, intptr_t carray, int64_t ldc, int64_t batch_count):
    """See `cublasCgemm3mBatched_64`."""
    with nogil:
        __status__ = cublasCgemm3mBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex* const*>aarray, lda, <const cuComplex* const*>barray, ldb, <const cuComplex*>beta, <cuComplex* const*>carray, ldc, batch_count)
    check_status(__status__)


cpdef zgemm_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int64_t lda, intptr_t barray, int64_t ldb, intptr_t beta, intptr_t carray, int64_t ldc, int64_t batch_count):
    """See `cublasZgemmBatched_64`."""
    with nogil:
        __status__ = cublasZgemmBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>aarray, lda, <const cuDoubleComplex* const*>barray, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex* const*>carray, ldc, batch_count)
    check_status(__status__)


cpdef sgemm_strided_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t b, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int64_t ldc, long long int stride_c, int64_t batch_count):
    """See `cublasSgemmStridedBatched_64`."""
    with nogil:
        __status__ = cublasSgemmStridedBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const float*>alpha, <const float*>a, lda, stride_a, <const float*>b, ldb, stride_b, <const float*>beta, <float*>c, ldc, stride_c, batch_count)
    check_status(__status__)


cpdef dgemm_strided_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t b, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int64_t ldc, long long int stride_c, int64_t batch_count):
    """See `cublasDgemmStridedBatched_64`."""
    with nogil:
        __status__ = cublasDgemmStridedBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const double*>alpha, <const double*>a, lda, stride_a, <const double*>b, ldb, stride_b, <const double*>beta, <double*>c, ldc, stride_c, batch_count)
    check_status(__status__)


cpdef cgemm_strided_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t b, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int64_t ldc, long long int stride_c, int64_t batch_count):
    """See `cublasCgemmStridedBatched_64`."""
    with nogil:
        __status__ = cublasCgemmStridedBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>b, ldb, stride_b, <const cuComplex*>beta, <cuComplex*>c, ldc, stride_c, batch_count)
    check_status(__status__)


cpdef cgemm3m_strided_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t b, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int64_t ldc, long long int stride_c, int64_t batch_count):
    """See `cublasCgemm3mStridedBatched_64`."""
    with nogil:
        __status__ = cublasCgemm3mStridedBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>a, lda, stride_a, <const cuComplex*>b, ldb, stride_b, <const cuComplex*>beta, <cuComplex*>c, ldc, stride_c, batch_count)
    check_status(__status__)


cpdef zgemm_strided_batched_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t lda, long long int stride_a, intptr_t b, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int64_t ldc, long long int stride_c, int64_t batch_count):
    """See `cublasZgemmStridedBatched_64`."""
    with nogil:
        __status__ = cublasZgemmStridedBatched_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, stride_a, <const cuDoubleComplex*>b, ldb, stride_b, <const cuDoubleComplex*>beta, <cuDoubleComplex*>c, ldc, stride_c, batch_count)
    check_status(__status__)


cpdef gemm_batched_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t aarray, int atype, int64_t lda, intptr_t barray, int btype, int64_t ldb, intptr_t beta, intptr_t carray, int ctype, int64_t ldc, int64_t batch_count, int compute_type, int algo):
    """See `cublasGemmBatchedEx_64`."""
    with nogil:
        __status__ = cublasGemmBatchedEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void* const*>aarray, <DataType>atype, lda, <const void* const*>barray, <DataType>btype, ldb, <const void*>beta, <void* const*>carray, <DataType>ctype, ldc, batch_count, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(__status__)


cpdef gemm_strided_batched_ex_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int atype, int64_t lda, long long int stride_a, intptr_t b, int btype, int64_t ldb, long long int stride_b, intptr_t beta, intptr_t c, int ctype, int64_t ldc, long long int stride_c, int64_t batch_count, int compute_type, int algo):
    """See `cublasGemmStridedBatchedEx_64`."""
    with nogil:
        __status__ = cublasGemmStridedBatchedEx_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, k, <const void*>alpha, <const void*>a, <DataType>atype, lda, stride_a, <const void*>b, <DataType>btype, ldb, stride_b, <const void*>beta, <void*>c, <DataType>ctype, ldc, stride_c, batch_count, <_ComputeType>compute_type, <_GemmAlgo>algo)
    check_status(__status__)


cpdef sgeam_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasSgeam_64`."""
    with nogil:
        __status__ = cublasSgeam_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const float*>alpha, <const float*>a, lda, <const float*>beta, <const float*>b, ldb, <float*>c, ldc)
    check_status(__status__)


cpdef dgeam_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasDgeam_64`."""
    with nogil:
        __status__ = cublasDgeam_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const double*>alpha, <const double*>a, lda, <const double*>beta, <const double*>b, ldb, <double*>c, ldc)
    check_status(__status__)


cpdef cgeam_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasCgeam_64`."""
    with nogil:
        __status__ = cublasCgeam_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const cuComplex*>alpha, <const cuComplex*>a, lda, <const cuComplex*>beta, <const cuComplex*>b, ldb, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zgeam_64(intptr_t handle, int transa, int transb, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t beta, intptr_t b, int64_t ldb, intptr_t c, int64_t ldc):
    """See `cublasZgeam_64`."""
    with nogil:
        __status__ = cublasZgeam_64(<Handle>handle, <_Operation>transa, <_Operation>transb, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>beta, <const cuDoubleComplex*>b, ldb, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef strsm_batched_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, int64_t batch_count):
    """See `cublasStrsmBatched_64`."""
    with nogil:
        __status__ = cublasStrsmBatched_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const float*>alpha, <const float* const*>a, lda, <float* const*>b, ldb, batch_count)
    check_status(__status__)


cpdef dtrsm_batched_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, int64_t batch_count):
    """See `cublasDtrsmBatched_64`."""
    with nogil:
        __status__ = cublasDtrsmBatched_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const double*>alpha, <const double* const*>a, lda, <double* const*>b, ldb, batch_count)
    check_status(__status__)


cpdef ctrsm_batched_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, int64_t batch_count):
    """See `cublasCtrsmBatched_64`."""
    with nogil:
        __status__ = cublasCtrsmBatched_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex* const*>a, lda, <cuComplex* const*>b, ldb, batch_count)
    check_status(__status__)


cpdef ztrsm_batched_64(intptr_t handle, int side, int uplo, int trans, int diag, int64_t m, int64_t n, intptr_t alpha, intptr_t a, int64_t lda, intptr_t b, int64_t ldb, int64_t batch_count):
    """See `cublasZtrsmBatched_64`."""
    with nogil:
        __status__ = cublasZtrsmBatched_64(<Handle>handle, <_SideMode>side, <_FillMode>uplo, <_Operation>trans, <_DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>a, lda, <cuDoubleComplex* const*>b, ldb, batch_count)
    check_status(__status__)


cpdef sdgmm_64(intptr_t handle, int mode, int64_t m, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t c, int64_t ldc):
    """See `cublasSdgmm_64`."""
    with nogil:
        __status__ = cublasSdgmm_64(<Handle>handle, <_SideMode>mode, m, n, <const float*>a, lda, <const float*>x, incx, <float*>c, ldc)
    check_status(__status__)


cpdef ddgmm_64(intptr_t handle, int mode, int64_t m, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t c, int64_t ldc):
    """See `cublasDdgmm_64`."""
    with nogil:
        __status__ = cublasDdgmm_64(<Handle>handle, <_SideMode>mode, m, n, <const double*>a, lda, <const double*>x, incx, <double*>c, ldc)
    check_status(__status__)


cpdef cdgmm_64(intptr_t handle, int mode, int64_t m, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t c, int64_t ldc):
    """See `cublasCdgmm_64`."""
    with nogil:
        __status__ = cublasCdgmm_64(<Handle>handle, <_SideMode>mode, m, n, <const cuComplex*>a, lda, <const cuComplex*>x, incx, <cuComplex*>c, ldc)
    check_status(__status__)


cpdef zdgmm_64(intptr_t handle, int mode, int64_t m, int64_t n, intptr_t a, int64_t lda, intptr_t x, int64_t incx, intptr_t c, int64_t ldc):
    """See `cublasZdgmm_64`."""
    with nogil:
        __status__ = cublasZdgmm_64(<Handle>handle, <_SideMode>mode, m, n, <const cuDoubleComplex*>a, lda, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>c, ldc)
    check_status(__status__)


cpdef sgemm_grouped_batched(intptr_t handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, intptr_t aarray, lda_array, intptr_t barray, ldb_array, beta_array, intptr_t carray, ldc_array, int group_count, group_size):
    """See `cublasSgemmGroupedBatched`."""
    cdef nullable_unique_ptr[ vector[int] ] _transa_array_
    get_resource_ptr[int](_transa_array_, transa_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _transb_array_
    get_resource_ptr[int](_transb_array_, transb_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _m_array_
    get_resource_ptr[int](_m_array_, m_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _n_array_
    get_resource_ptr[int](_n_array_, n_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _k_array_
    get_resource_ptr[int](_k_array_, k_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[float] ] _alpha_array_
    get_resource_ptr[float](_alpha_array_, alpha_array, <float*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _lda_array_
    get_resource_ptr[int](_lda_array_, lda_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _ldb_array_
    get_resource_ptr[int](_ldb_array_, ldb_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[float] ] _beta_array_
    get_resource_ptr[float](_beta_array_, beta_array, <float*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _ldc_array_
    get_resource_ptr[int](_ldc_array_, ldc_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _group_size_
    get_resource_ptr[int](_group_size_, group_size, <int*>NULL)
    with nogil:
        __status__ = cublasSgemmGroupedBatched(<Handle>handle, <const _Operation*>(_transa_array_.data()), <const _Operation*>(_transb_array_.data()), <const int*>(_m_array_.data()), <const int*>(_n_array_.data()), <const int*>(_k_array_.data()), <const float*>(_alpha_array_.data()), <const float* const*>aarray, <const int*>(_lda_array_.data()), <const float* const*>barray, <const int*>(_ldb_array_.data()), <const float*>(_beta_array_.data()), <float* const*>carray, <const int*>(_ldc_array_.data()), group_count, <const int*>(_group_size_.data()))
    check_status(__status__)


cpdef sgemm_grouped_batched_64(intptr_t handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, intptr_t aarray, lda_array, intptr_t barray, ldb_array, beta_array, intptr_t carray, ldc_array, int64_t group_count, group_size):
    """See `cublasSgemmGroupedBatched_64`."""
    cdef nullable_unique_ptr[ vector[int] ] _transa_array_
    get_resource_ptr[int](_transa_array_, transa_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _transb_array_
    get_resource_ptr[int](_transb_array_, transb_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _m_array_
    get_resource_ptr[int64_t](_m_array_, m_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _n_array_
    get_resource_ptr[int64_t](_n_array_, n_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _k_array_
    get_resource_ptr[int64_t](_k_array_, k_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[float] ] _alpha_array_
    get_resource_ptr[float](_alpha_array_, alpha_array, <float*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _lda_array_
    get_resource_ptr[int64_t](_lda_array_, lda_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _ldb_array_
    get_resource_ptr[int64_t](_ldb_array_, ldb_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[float] ] _beta_array_
    get_resource_ptr[float](_beta_array_, beta_array, <float*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _ldc_array_
    get_resource_ptr[int64_t](_ldc_array_, ldc_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _group_size_
    get_resource_ptr[int64_t](_group_size_, group_size, <int64_t*>NULL)
    with nogil:
        __status__ = cublasSgemmGroupedBatched_64(<Handle>handle, <const _Operation*>(_transa_array_.data()), <const _Operation*>(_transb_array_.data()), <const int64_t*>(_m_array_.data()), <const int64_t*>(_n_array_.data()), <const int64_t*>(_k_array_.data()), <const float*>(_alpha_array_.data()), <const float* const*>aarray, <const int64_t*>(_lda_array_.data()), <const float* const*>barray, <const int64_t*>(_ldb_array_.data()), <const float*>(_beta_array_.data()), <float* const*>carray, <const int64_t*>(_ldc_array_.data()), group_count, <const int64_t*>(_group_size_.data()))
    check_status(__status__)


cpdef dgemm_grouped_batched(intptr_t handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, intptr_t aarray, lda_array, intptr_t barray, ldb_array, beta_array, intptr_t carray, ldc_array, int group_count, group_size):
    """See `cublasDgemmGroupedBatched`."""
    cdef nullable_unique_ptr[ vector[int] ] _transa_array_
    get_resource_ptr[int](_transa_array_, transa_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _transb_array_
    get_resource_ptr[int](_transb_array_, transb_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _m_array_
    get_resource_ptr[int](_m_array_, m_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _n_array_
    get_resource_ptr[int](_n_array_, n_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _k_array_
    get_resource_ptr[int](_k_array_, k_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[double] ] _alpha_array_
    get_resource_ptr[double](_alpha_array_, alpha_array, <double*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _lda_array_
    get_resource_ptr[int](_lda_array_, lda_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _ldb_array_
    get_resource_ptr[int](_ldb_array_, ldb_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[double] ] _beta_array_
    get_resource_ptr[double](_beta_array_, beta_array, <double*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _ldc_array_
    get_resource_ptr[int](_ldc_array_, ldc_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _group_size_
    get_resource_ptr[int](_group_size_, group_size, <int*>NULL)
    with nogil:
        __status__ = cublasDgemmGroupedBatched(<Handle>handle, <const _Operation*>(_transa_array_.data()), <const _Operation*>(_transb_array_.data()), <const int*>(_m_array_.data()), <const int*>(_n_array_.data()), <const int*>(_k_array_.data()), <const double*>(_alpha_array_.data()), <const double* const*>aarray, <const int*>(_lda_array_.data()), <const double* const*>barray, <const int*>(_ldb_array_.data()), <const double*>(_beta_array_.data()), <double* const*>carray, <const int*>(_ldc_array_.data()), group_count, <const int*>(_group_size_.data()))
    check_status(__status__)


cpdef dgemm_grouped_batched_64(intptr_t handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, intptr_t aarray, lda_array, intptr_t barray, ldb_array, beta_array, intptr_t carray, ldc_array, int64_t group_count, group_size):
    """See `cublasDgemmGroupedBatched_64`."""
    cdef nullable_unique_ptr[ vector[int] ] _transa_array_
    get_resource_ptr[int](_transa_array_, transa_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _transb_array_
    get_resource_ptr[int](_transb_array_, transb_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _m_array_
    get_resource_ptr[int64_t](_m_array_, m_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _n_array_
    get_resource_ptr[int64_t](_n_array_, n_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _k_array_
    get_resource_ptr[int64_t](_k_array_, k_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[double] ] _alpha_array_
    get_resource_ptr[double](_alpha_array_, alpha_array, <double*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _lda_array_
    get_resource_ptr[int64_t](_lda_array_, lda_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _ldb_array_
    get_resource_ptr[int64_t](_ldb_array_, ldb_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[double] ] _beta_array_
    get_resource_ptr[double](_beta_array_, beta_array, <double*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _ldc_array_
    get_resource_ptr[int64_t](_ldc_array_, ldc_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _group_size_
    get_resource_ptr[int64_t](_group_size_, group_size, <int64_t*>NULL)
    with nogil:
        __status__ = cublasDgemmGroupedBatched_64(<Handle>handle, <const _Operation*>(_transa_array_.data()), <const _Operation*>(_transb_array_.data()), <const int64_t*>(_m_array_.data()), <const int64_t*>(_n_array_.data()), <const int64_t*>(_k_array_.data()), <const double*>(_alpha_array_.data()), <const double* const*>aarray, <const int64_t*>(_lda_array_.data()), <const double* const*>barray, <const int64_t*>(_ldb_array_.data()), <const double*>(_beta_array_.data()), <double* const*>carray, <const int64_t*>(_ldc_array_.data()), group_count, <const int64_t*>(_group_size_.data()))
    check_status(__status__)


cpdef gemm_grouped_batched_ex(intptr_t handle, transa_array, transb_array, m_array, n_array, k_array, intptr_t alpha_array, intptr_t aarray, int atype, lda_array, intptr_t barray, int btype, ldb_array, intptr_t beta_array, intptr_t carray, int ctype, ldc_array, int group_count, group_size, int compute_type):
    """See `cublasGemmGroupedBatchedEx`."""
    cdef nullable_unique_ptr[ vector[int] ] _transa_array_
    get_resource_ptr[int](_transa_array_, transa_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _transb_array_
    get_resource_ptr[int](_transb_array_, transb_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _m_array_
    get_resource_ptr[int](_m_array_, m_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _n_array_
    get_resource_ptr[int](_n_array_, n_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _k_array_
    get_resource_ptr[int](_k_array_, k_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _lda_array_
    get_resource_ptr[int](_lda_array_, lda_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _ldb_array_
    get_resource_ptr[int](_ldb_array_, ldb_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _ldc_array_
    get_resource_ptr[int](_ldc_array_, ldc_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _group_size_
    get_resource_ptr[int](_group_size_, group_size, <int*>NULL)
    with nogil:
        __status__ = cublasGemmGroupedBatchedEx(<Handle>handle, <const _Operation*>(_transa_array_.data()), <const _Operation*>(_transb_array_.data()), <const int*>(_m_array_.data()), <const int*>(_n_array_.data()), <const int*>(_k_array_.data()), <const void*>alpha_array, <const void* const*>aarray, <DataType>atype, <const int*>(_lda_array_.data()), <const void* const*>barray, <DataType>btype, <const int*>(_ldb_array_.data()), <const void*>beta_array, <void* const*>carray, <DataType>ctype, <const int*>(_ldc_array_.data()), group_count, <const int*>(_group_size_.data()), <_ComputeType>compute_type)
    check_status(__status__)


cpdef gemm_grouped_batched_ex_64(intptr_t handle, transa_array, transb_array, m_array, n_array, k_array, intptr_t alpha_array, intptr_t aarray, int atype, lda_array, intptr_t barray, int btype, ldb_array, intptr_t beta_array, intptr_t carray, int ctype, ldc_array, int64_t group_count, group_size, int compute_type):
    """See `cublasGemmGroupedBatchedEx_64`."""
    cdef nullable_unique_ptr[ vector[int] ] _transa_array_
    get_resource_ptr[int](_transa_array_, transa_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int] ] _transb_array_
    get_resource_ptr[int](_transb_array_, transb_array, <int*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _m_array_
    get_resource_ptr[int64_t](_m_array_, m_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _n_array_
    get_resource_ptr[int64_t](_n_array_, n_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _k_array_
    get_resource_ptr[int64_t](_k_array_, k_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _lda_array_
    get_resource_ptr[int64_t](_lda_array_, lda_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _ldb_array_
    get_resource_ptr[int64_t](_ldb_array_, ldb_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _ldc_array_
    get_resource_ptr[int64_t](_ldc_array_, ldc_array, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _group_size_
    get_resource_ptr[int64_t](_group_size_, group_size, <int64_t*>NULL)
    with nogil:
        __status__ = cublasGemmGroupedBatchedEx_64(<Handle>handle, <const _Operation*>(_transa_array_.data()), <const _Operation*>(_transb_array_.data()), <const int64_t*>(_m_array_.data()), <const int64_t*>(_n_array_.data()), <const int64_t*>(_k_array_.data()), <const void*>alpha_array, <const void* const*>aarray, <DataType>atype, <const int64_t*>(_lda_array_.data()), <const void* const*>barray, <DataType>btype, <const int64_t*>(_ldb_array_.data()), <const void*>beta_array, <void* const*>carray, <DataType>ctype, <const int64_t*>(_ldc_array_.data()), group_count, <const int64_t*>(_group_size_.data()), <_ComputeType>compute_type)
    check_status(__status__)


cpdef int get_emulation_strategy(intptr_t handle) except? -1:
    """See `cublasGetEmulationStrategy`."""
    cdef _EmulationStrategy emulation_strategy
    with nogil:
        __status__ = cublasGetEmulationStrategy(<Handle>handle, &emulation_strategy)
    check_status(__status__)
    return <int>emulation_strategy


cpdef set_emulation_strategy(intptr_t handle, int emulation_strategy):
    """See `cublasSetEmulationStrategy`."""
    with nogil:
        __status__ = cublasSetEmulationStrategy(<Handle>handle, <_EmulationStrategy>emulation_strategy)
    check_status(__status__)


cpdef int get_emulation_special_values_support(intptr_t handle) except? -1:
    """See `cublasGetEmulationSpecialValuesSupport`."""
    cdef cudaEmulationSpecialValuesSupport mask
    with nogil:
        __status__ = cublasGetEmulationSpecialValuesSupport(<Handle>handle, &mask)
    check_status(__status__)
    return <int>mask


cpdef set_emulation_special_values_support(intptr_t handle, int mask):
    """See `cublasSetEmulationSpecialValuesSupport`."""
    with nogil:
        __status__ = cublasSetEmulationSpecialValuesSupport(<Handle>handle, <cudaEmulationSpecialValuesSupport>mask)
    check_status(__status__)


cpdef int get_fixed_point_emulation_mantissa_control(intptr_t handle) except? -1:
    """See `cublasGetFixedPointEmulationMantissaControl`."""
    cdef cudaEmulationMantissaControl mantissa_control
    with nogil:
        __status__ = cublasGetFixedPointEmulationMantissaControl(<Handle>handle, &mantissa_control)
    check_status(__status__)
    return <int>mantissa_control


cpdef set_fixed_point_emulation_mantissa_control(intptr_t handle, int mantissa_control):
    """See `cublasSetFixedPointEmulationMantissaControl`."""
    with nogil:
        __status__ = cublasSetFixedPointEmulationMantissaControl(<Handle>handle, <cudaEmulationMantissaControl>mantissa_control)
    check_status(__status__)


cpdef int get_fixed_point_emulation_max_mantissa_bit_count(intptr_t handle) except? -1:
    """See `cublasGetFixedPointEmulationMaxMantissaBitCount`."""
    cdef int max_mantissa_bit_count
    with nogil:
        __status__ = cublasGetFixedPointEmulationMaxMantissaBitCount(<Handle>handle, &max_mantissa_bit_count)
    check_status(__status__)
    return max_mantissa_bit_count


cpdef set_fixed_point_emulation_max_mantissa_bit_count(intptr_t handle, int max_mantissa_bit_count):
    """See `cublasSetFixedPointEmulationMaxMantissaBitCount`."""
    with nogil:
        __status__ = cublasSetFixedPointEmulationMaxMantissaBitCount(<Handle>handle, max_mantissa_bit_count)
    check_status(__status__)


cpdef int get_fixed_point_emulation_mantissa_bit_offset(intptr_t handle) except? -1:
    """See `cublasGetFixedPointEmulationMantissaBitOffset`."""
    cdef int mantissa_bit_offset
    with nogil:
        __status__ = cublasGetFixedPointEmulationMantissaBitOffset(<Handle>handle, &mantissa_bit_offset)
    check_status(__status__)
    return mantissa_bit_offset


cpdef set_fixed_point_emulation_mantissa_bit_offset(intptr_t handle, int mantissa_bit_offset):
    """See `cublasSetFixedPointEmulationMantissaBitOffset`."""
    with nogil:
        __status__ = cublasSetFixedPointEmulationMantissaBitOffset(<Handle>handle, mantissa_bit_offset)
    check_status(__status__)


cpdef intptr_t get_fixed_point_emulation_mantissa_bit_count_pointer(intptr_t handle) except? -1:
    """See `cublasGetFixedPointEmulationMantissaBitCountPointer`."""
    cdef int* mantissa_bit_count
    with nogil:
        __status__ = cublasGetFixedPointEmulationMantissaBitCountPointer(<Handle>handle, &mantissa_bit_count)
    check_status(__status__)
    return <intptr_t>mantissa_bit_count


cpdef set_fixed_point_emulation_mantissa_bit_count_pointer(intptr_t handle, intptr_t mantissa_bit_count):
    """See `cublasSetFixedPointEmulationMantissaBitCountPointer`."""
    with nogil:
        __status__ = cublasSetFixedPointEmulationMantissaBitCountPointer(<Handle>handle, <int*>mantissa_bit_count)
    check_status(__status__)
