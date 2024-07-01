# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.4.1. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int64_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum cublasStatus_t "cublasStatus_t":
    CUBLAS_STATUS_SUCCESS "CUBLAS_STATUS_SUCCESS" = 0
    CUBLAS_STATUS_NOT_INITIALIZED "CUBLAS_STATUS_NOT_INITIALIZED" = 1
    CUBLAS_STATUS_ALLOC_FAILED "CUBLAS_STATUS_ALLOC_FAILED" = 3
    CUBLAS_STATUS_INVALID_VALUE "CUBLAS_STATUS_INVALID_VALUE" = 7
    CUBLAS_STATUS_ARCH_MISMATCH "CUBLAS_STATUS_ARCH_MISMATCH" = 8
    CUBLAS_STATUS_MAPPING_ERROR "CUBLAS_STATUS_MAPPING_ERROR" = 11
    CUBLAS_STATUS_EXECUTION_FAILED "CUBLAS_STATUS_EXECUTION_FAILED" = 13
    CUBLAS_STATUS_INTERNAL_ERROR "CUBLAS_STATUS_INTERNAL_ERROR" = 14
    CUBLAS_STATUS_NOT_SUPPORTED "CUBLAS_STATUS_NOT_SUPPORTED" = 15
    CUBLAS_STATUS_LICENSE_ERROR "CUBLAS_STATUS_LICENSE_ERROR" = 16

ctypedef enum cublasFillMode_t "cublasFillMode_t":
    CUBLAS_FILL_MODE_LOWER "CUBLAS_FILL_MODE_LOWER" = 0
    CUBLAS_FILL_MODE_UPPER "CUBLAS_FILL_MODE_UPPER" = 1
    CUBLAS_FILL_MODE_FULL "CUBLAS_FILL_MODE_FULL" = 2

ctypedef enum cublasDiagType_t "cublasDiagType_t":
    CUBLAS_DIAG_NON_UNIT "CUBLAS_DIAG_NON_UNIT" = 0
    CUBLAS_DIAG_UNIT "CUBLAS_DIAG_UNIT" = 1

ctypedef enum cublasSideMode_t "cublasSideMode_t":
    CUBLAS_SIDE_LEFT "CUBLAS_SIDE_LEFT" = 0
    CUBLAS_SIDE_RIGHT "CUBLAS_SIDE_RIGHT" = 1

ctypedef enum cublasOperation_t "cublasOperation_t":
    CUBLAS_OP_N "CUBLAS_OP_N" = 0
    CUBLAS_OP_T "CUBLAS_OP_T" = 1
    CUBLAS_OP_C "CUBLAS_OP_C" = 2
    CUBLAS_OP_HERMITAN "CUBLAS_OP_HERMITAN" = 2
    CUBLAS_OP_CONJG "CUBLAS_OP_CONJG" = 3

ctypedef enum cublasPointerMode_t "cublasPointerMode_t":
    CUBLAS_POINTER_MODE_HOST "CUBLAS_POINTER_MODE_HOST" = 0
    CUBLAS_POINTER_MODE_DEVICE "CUBLAS_POINTER_MODE_DEVICE" = 1

ctypedef enum cublasAtomicsMode_t "cublasAtomicsMode_t":
    CUBLAS_ATOMICS_NOT_ALLOWED "CUBLAS_ATOMICS_NOT_ALLOWED" = 0
    CUBLAS_ATOMICS_ALLOWED "CUBLAS_ATOMICS_ALLOWED" = 1

ctypedef enum cublasGemmAlgo_t "cublasGemmAlgo_t":
    CUBLAS_GEMM_DFALT "CUBLAS_GEMM_DFALT" = -1
    CUBLAS_GEMM_DEFAULT "CUBLAS_GEMM_DEFAULT" = -1
    CUBLAS_GEMM_ALGO0 "CUBLAS_GEMM_ALGO0" = 0
    CUBLAS_GEMM_ALGO1 "CUBLAS_GEMM_ALGO1" = 1
    CUBLAS_GEMM_ALGO2 "CUBLAS_GEMM_ALGO2" = 2
    CUBLAS_GEMM_ALGO3 "CUBLAS_GEMM_ALGO3" = 3
    CUBLAS_GEMM_ALGO4 "CUBLAS_GEMM_ALGO4" = 4
    CUBLAS_GEMM_ALGO5 "CUBLAS_GEMM_ALGO5" = 5
    CUBLAS_GEMM_ALGO6 "CUBLAS_GEMM_ALGO6" = 6
    CUBLAS_GEMM_ALGO7 "CUBLAS_GEMM_ALGO7" = 7
    CUBLAS_GEMM_ALGO8 "CUBLAS_GEMM_ALGO8" = 8
    CUBLAS_GEMM_ALGO9 "CUBLAS_GEMM_ALGO9" = 9
    CUBLAS_GEMM_ALGO10 "CUBLAS_GEMM_ALGO10" = 10
    CUBLAS_GEMM_ALGO11 "CUBLAS_GEMM_ALGO11" = 11
    CUBLAS_GEMM_ALGO12 "CUBLAS_GEMM_ALGO12" = 12
    CUBLAS_GEMM_ALGO13 "CUBLAS_GEMM_ALGO13" = 13
    CUBLAS_GEMM_ALGO14 "CUBLAS_GEMM_ALGO14" = 14
    CUBLAS_GEMM_ALGO15 "CUBLAS_GEMM_ALGO15" = 15
    CUBLAS_GEMM_ALGO16 "CUBLAS_GEMM_ALGO16" = 16
    CUBLAS_GEMM_ALGO17 "CUBLAS_GEMM_ALGO17" = 17
    CUBLAS_GEMM_ALGO18 "CUBLAS_GEMM_ALGO18" = 18
    CUBLAS_GEMM_ALGO19 "CUBLAS_GEMM_ALGO19" = 19
    CUBLAS_GEMM_ALGO20 "CUBLAS_GEMM_ALGO20" = 20
    CUBLAS_GEMM_ALGO21 "CUBLAS_GEMM_ALGO21" = 21
    CUBLAS_GEMM_ALGO22 "CUBLAS_GEMM_ALGO22" = 22
    CUBLAS_GEMM_ALGO23 "CUBLAS_GEMM_ALGO23" = 23
    CUBLAS_GEMM_DEFAULT_TENSOR_OP "CUBLAS_GEMM_DEFAULT_TENSOR_OP" = 99
    CUBLAS_GEMM_DFALT_TENSOR_OP "CUBLAS_GEMM_DFALT_TENSOR_OP" = 99
    CUBLAS_GEMM_ALGO0_TENSOR_OP "CUBLAS_GEMM_ALGO0_TENSOR_OP" = 100
    CUBLAS_GEMM_ALGO1_TENSOR_OP "CUBLAS_GEMM_ALGO1_TENSOR_OP" = 101
    CUBLAS_GEMM_ALGO2_TENSOR_OP "CUBLAS_GEMM_ALGO2_TENSOR_OP" = 102
    CUBLAS_GEMM_ALGO3_TENSOR_OP "CUBLAS_GEMM_ALGO3_TENSOR_OP" = 103
    CUBLAS_GEMM_ALGO4_TENSOR_OP "CUBLAS_GEMM_ALGO4_TENSOR_OP" = 104
    CUBLAS_GEMM_ALGO5_TENSOR_OP "CUBLAS_GEMM_ALGO5_TENSOR_OP" = 105
    CUBLAS_GEMM_ALGO6_TENSOR_OP "CUBLAS_GEMM_ALGO6_TENSOR_OP" = 106
    CUBLAS_GEMM_ALGO7_TENSOR_OP "CUBLAS_GEMM_ALGO7_TENSOR_OP" = 107
    CUBLAS_GEMM_ALGO8_TENSOR_OP "CUBLAS_GEMM_ALGO8_TENSOR_OP" = 108
    CUBLAS_GEMM_ALGO9_TENSOR_OP "CUBLAS_GEMM_ALGO9_TENSOR_OP" = 109
    CUBLAS_GEMM_ALGO10_TENSOR_OP "CUBLAS_GEMM_ALGO10_TENSOR_OP" = 110
    CUBLAS_GEMM_ALGO11_TENSOR_OP "CUBLAS_GEMM_ALGO11_TENSOR_OP" = 111
    CUBLAS_GEMM_ALGO12_TENSOR_OP "CUBLAS_GEMM_ALGO12_TENSOR_OP" = 112
    CUBLAS_GEMM_ALGO13_TENSOR_OP "CUBLAS_GEMM_ALGO13_TENSOR_OP" = 113
    CUBLAS_GEMM_ALGO14_TENSOR_OP "CUBLAS_GEMM_ALGO14_TENSOR_OP" = 114
    CUBLAS_GEMM_ALGO15_TENSOR_OP "CUBLAS_GEMM_ALGO15_TENSOR_OP" = 115

ctypedef enum cublasMath_t "cublasMath_t":
    CUBLAS_DEFAULT_MATH "CUBLAS_DEFAULT_MATH" = 0
    CUBLAS_TENSOR_OP_MATH "CUBLAS_TENSOR_OP_MATH" = 1
    CUBLAS_PEDANTIC_MATH "CUBLAS_PEDANTIC_MATH" = 2
    CUBLAS_TF32_TENSOR_OP_MATH "CUBLAS_TF32_TENSOR_OP_MATH" = 3
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION" = 16

ctypedef enum cublasComputeType_t "cublasComputeType_t":
    CUBLAS_COMPUTE_16F "CUBLAS_COMPUTE_16F" = 64
    CUBLAS_COMPUTE_16F_PEDANTIC "CUBLAS_COMPUTE_16F_PEDANTIC" = 65
    CUBLAS_COMPUTE_32F "CUBLAS_COMPUTE_32F" = 68
    CUBLAS_COMPUTE_32F_PEDANTIC "CUBLAS_COMPUTE_32F_PEDANTIC" = 69
    CUBLAS_COMPUTE_32F_FAST_16F "CUBLAS_COMPUTE_32F_FAST_16F" = 74
    CUBLAS_COMPUTE_32F_FAST_16BF "CUBLAS_COMPUTE_32F_FAST_16BF" = 75
    CUBLAS_COMPUTE_32F_FAST_TF32 "CUBLAS_COMPUTE_32F_FAST_TF32" = 77
    CUBLAS_COMPUTE_64F "CUBLAS_COMPUTE_64F" = 70
    CUBLAS_COMPUTE_64F_PEDANTIC "CUBLAS_COMPUTE_64F_PEDANTIC" = 71
    CUBLAS_COMPUTE_32I "CUBLAS_COMPUTE_32I" = 72
    CUBLAS_COMPUTE_32I_PEDANTIC "CUBLAS_COMPUTE_32I_PEDANTIC" = 73


# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef int cudaDataType_t 'cudaDataType_t'
    ctypedef int cudaDataType 'cudaDataType'
    ctypedef int libraryPropertyType_t 'libraryPropertyType_t'
    ctypedef int libraryPropertyType 'libraryPropertyType'

    ctypedef struct cuComplex:
        pass
    ctypedef struct cuDoubleComplex:
        pass


ctypedef cudaDataType cublasDataType_t 'cublasDataType_t'
ctypedef void* cublasHandle_t 'cublasHandle_t'
ctypedef void (*cublasLogCallback 'cublasLogCallback')(
    const char* msg
)


###############################################################################
# Functions
###############################################################################

cdef cublasStatus_t cublasCreate(cublasHandle_t* handle) except* nogil
cdef cublasStatus_t cublasDestroy(cublasHandle_t handle) except* nogil
cdef cublasStatus_t cublasGetVersion(cublasHandle_t handle, int* version) except* nogil
cdef cublasStatus_t cublasGetProperty(libraryPropertyType type, int* value) except* nogil
cdef size_t cublasGetCudartVersion() except* nogil
cdef cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes) except* nogil
cdef cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) except* nogil
cdef cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId) except* nogil
cdef cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode) except* nogil
cdef cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) except* nogil
cdef cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode) except* nogil
cdef cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) except* nogil
cdef cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) except* nogil
cdef cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) except* nogil
cdef cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFileName) except* nogil
cdef cublasStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback) except* nogil
cdef cublasStatus_t cublasGetLoggerCallback(cublasLogCallback* userCallback) except* nogil
cdef cublasStatus_t cublasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy) except* nogil
cdef cublasStatus_t cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy) except* nogil
cdef cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) except* nogil
cdef cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) except* nogil
cdef cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream) except* nogil
cdef cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream) except* nogil
cdef cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) except* nogil
cdef cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) except* nogil
cdef cublasStatus_t cublasNrm2Ex(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil
cdef cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) except* nogil
cdef cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result) except* nogil
cdef cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) except* nogil
cdef cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) except* nogil
cdef cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil
cdef cublasStatus_t cublasDotcEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil
cdef cublasStatus_t cublasSdot(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) except* nogil
cdef cublasStatus_t cublasDdot(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) except* nogil
cdef cublasStatus_t cublasCdotu(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result) except* nogil
cdef cublasStatus_t cublasCdotc(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result) except* nogil
cdef cublasStatus_t cublasZdotu(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result) except* nogil
cdef cublasStatus_t cublasZdotc(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result) except* nogil
cdef cublasStatus_t cublasScalEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int incx, cudaDataType executionType) except* nogil
cdef cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) except* nogil
cdef cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) except* nogil
cdef cublasStatus_t cublasCscal(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasCsscal(cublasHandle_t handle, int n, const float* alpha, cuComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasZscal(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasZdscal(cublasHandle_t handle, int n, const double* alpha, cuDoubleComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasAxpyEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, cudaDataType executiontype) except* nogil
cdef cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) except* nogil
cdef cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) except* nogil
cdef cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasCopyEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy) except* nogil
cdef cublasStatus_t cublasScopy(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) except* nogil
cdef cublasStatus_t cublasDcopy(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) except* nogil
cdef cublasStatus_t cublasCcopy(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasZcopy(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy) except* nogil
cdef cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy) except* nogil
cdef cublasStatus_t cublasCswap(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasSwapEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy) except* nogil
cdef cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float* x, int incx, int* result) except* nogil
cdef cublasStatus_t cublasIdamax(cublasHandle_t handle, int n, const double* x, int incx, int* result) except* nogil
cdef cublasStatus_t cublasIcamax(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) except* nogil
cdef cublasStatus_t cublasIzamax(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) except* nogil
cdef cublasStatus_t cublasIamaxEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result) except* nogil
cdef cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float* x, int incx, int* result) except* nogil
cdef cublasStatus_t cublasIdamin(cublasHandle_t handle, int n, const double* x, int incx, int* result) except* nogil
cdef cublasStatus_t cublasIcamin(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) except* nogil
cdef cublasStatus_t cublasIzamin(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) except* nogil
cdef cublasStatus_t cublasIaminEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result) except* nogil
cdef cublasStatus_t cublasAsumEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executiontype) except* nogil
cdef cublasStatus_t cublasSasum(cublasHandle_t handle, int n, const float* x, int incx, float* result) except* nogil
cdef cublasStatus_t cublasDasum(cublasHandle_t handle, int n, const double* x, int incx, double* result) except* nogil
cdef cublasStatus_t cublasScasum(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) except* nogil
cdef cublasStatus_t cublasDzasum(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) except* nogil
cdef cublasStatus_t cublasSrot(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s) except* nogil
cdef cublasStatus_t cublasDrot(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s) except* nogil
cdef cublasStatus_t cublasCrot(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s) except* nogil
cdef cublasStatus_t cublasCsrot(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s) except* nogil
cdef cublasStatus_t cublasZrot(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s) except* nogil
cdef cublasStatus_t cublasZdrot(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s) except* nogil
cdef cublasStatus_t cublasRotEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype) except* nogil
cdef cublasStatus_t cublasSrotg(cublasHandle_t handle, float* a, float* b, float* c, float* s) except* nogil
cdef cublasStatus_t cublasDrotg(cublasHandle_t handle, double* a, double* b, double* c, double* s) except* nogil
cdef cublasStatus_t cublasCrotg(cublasHandle_t handle, cuComplex* a, cuComplex* b, float* c, cuComplex* s) except* nogil
cdef cublasStatus_t cublasZrotg(cublasHandle_t handle, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s) except* nogil
cdef cublasStatus_t cublasRotgEx(cublasHandle_t handle, void* a, void* b, cudaDataType abType, void* c, void* s, cudaDataType csType, cudaDataType executiontype) except* nogil
cdef cublasStatus_t cublasSrotm(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param) except* nogil
cdef cublasStatus_t cublasDrotm(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param) except* nogil
cdef cublasStatus_t cublasRotmEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* param, cudaDataType paramType, cudaDataType executiontype) except* nogil
cdef cublasStatus_t cublasSrotmg(cublasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param) except* nogil
cdef cublasStatus_t cublasDrotmg(cublasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param) except* nogil
cdef cublasStatus_t cublasRotmgEx(cublasHandle_t handle, void* d1, cudaDataType d1Type, void* d2, cudaDataType d2Type, void* x1, cudaDataType x1Type, const void* y1, cudaDataType y1Type, void* param, cudaDataType paramType, cudaDataType executiontype) except* nogil
cdef cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil
cdef cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil
cdef cublasStatus_t cublasCgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil
cdef cublasStatus_t cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil
cdef cublasStatus_t cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) except* nogil
cdef cublasStatus_t cublasDtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx) except* nogil
cdef cublasStatus_t cublasCtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasZtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasStbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx) except* nogil
cdef cublasStatus_t cublasDtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx) except* nogil
cdef cublasStatus_t cublasCtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasZtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasStpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx) except* nogil
cdef cublasStatus_t cublasDtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx) except* nogil
cdef cublasStatus_t cublasCtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasZtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasStrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) except* nogil
cdef cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx) except* nogil
cdef cublasStatus_t cublasCtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasZtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasStpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx) except* nogil
cdef cublasStatus_t cublasDtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx) except* nogil
cdef cublasStatus_t cublasCtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasZtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasStbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx) except* nogil
cdef cublasStatus_t cublasDtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx) except* nogil
cdef cublasStatus_t cublasCtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasZtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil
cdef cublasStatus_t cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil
cdef cublasStatus_t cublasDsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil
cdef cublasStatus_t cublasCsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasZsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil
cdef cublasStatus_t cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil
cdef cublasStatus_t cublasChbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasZhbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasSspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy) except* nogil
cdef cublasStatus_t cublasDspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy) except* nogil
cdef cublasStatus_t cublasChpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasZhpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil
cdef cublasStatus_t cublasSger(cublasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) except* nogil
cdef cublasStatus_t cublasDger(cublasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) except* nogil
cdef cublasStatus_t cublasCgeru(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasCgerc(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasZgeru(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasZgerc(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda) except* nogil
cdef cublasStatus_t cublasDsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* A, int lda) except* nogil
cdef cublasStatus_t cublasCsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasZsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasCher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasZher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasSspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP) except* nogil
cdef cublasStatus_t cublasDspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP) except* nogil
cdef cublasStatus_t cublasChpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* AP) except* nogil
cdef cublasStatus_t cublasZhpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP) except* nogil
cdef cublasStatus_t cublasSsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) except* nogil
cdef cublasStatus_t cublasDsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) except* nogil
cdef cublasStatus_t cublasCsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasZsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasSspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP) except* nogil
cdef cublasStatus_t cublasDspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP) except* nogil
cdef cublasStatus_t cublasChpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP) except* nogil
cdef cublasStatus_t cublasZhpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* AP) except* nogil
cdef cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil
cdef cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil
cdef cublasStatus_t cublasCgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil
cdef cublasStatus_t cublasZgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const float* beta, void* C, cudaDataType Ctype, int ldc) except* nogil
cdef cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil
cdef cublasStatus_t cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil
cdef cublasStatus_t cublasUint8gemmBias(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc, int m, int n, int k, const unsigned char* A, int A_bias, int lda, const unsigned char* B, int B_bias, int ldb, unsigned char* C, int C_bias, int ldc, int C_mult, int C_shift) except* nogil
cdef cublasStatus_t cublasSsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc) except* nogil
cdef cublasStatus_t cublasDsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc) except* nogil
cdef cublasStatus_t cublasCsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil
cdef cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil
cdef cublasStatus_t cublasCherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const cuComplex* A, int lda, const float* beta, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const cuDoubleComplex* A, int lda, const double* beta, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const float* beta, void* C, cudaDataType Ctype, int ldc) except* nogil
cdef cublasStatus_t cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const float* beta, void* C, cudaDataType Ctype, int ldc) except* nogil
cdef cublasStatus_t cublasSsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil
cdef cublasStatus_t cublasDsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil
cdef cublasStatus_t cublasCsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasCher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil
cdef cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil
cdef cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasSsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil
cdef cublasStatus_t cublasDsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil
cdef cublasStatus_t cublasCsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasChemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZhemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasStrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb) except* nogil
cdef cublasStatus_t cublasDtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb) except* nogil
cdef cublasStatus_t cublasCtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb) except* nogil
cdef cublasStatus_t cublasZtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb) except* nogil
cdef cublasStatus_t cublasStrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* C, int ldc) except* nogil
cdef cublasStatus_t cublasDtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, double* C, int ldc) except* nogil
cdef cublasStatus_t cublasCtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount) except* nogil
cdef cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double* beta, double* const Carray[], int ldc, int batchCount) except* nogil
cdef cublasStatus_t cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount) except* nogil
cdef cublasStatus_t cublasCgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount) except* nogil
cdef cublasStatus_t cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int ldc, int batchCount) except* nogil
cdef cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil
cdef cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil
cdef cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount) except* nogil
cdef cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount) except* nogil
cdef cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount) except* nogil
cdef cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount) except* nogil
cdef cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount) except* nogil
cdef cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) except* nogil
cdef cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) except* nogil
cdef cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, const cuComplex* B, int ldb, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float* const A[], int lda, int* P, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double* const A[], int lda, int* P, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex* const A[], int lda, int* P, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex* const A[], int lda, int* P, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasSgetriBatched(cublasHandle_t handle, int n, const float* const A[], int lda, const int* P, float* const C[], int ldc, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasDgetriBatched(cublasHandle_t handle, int n, const double* const A[], int lda, const int* P, double* const C[], int ldc, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, const int* P, cuComplex* const C[], int ldc, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, const int* P, cuDoubleComplex* const C[], int ldc, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda, const int* devIpiv, float* const Barray[], int ldb, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* const Aarray[], int lda, const int* devIpiv, double* const Barray[], int ldb, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* const Aarray[], int lda, const int* devIpiv, cuComplex* const Barray[], int ldb, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int* devIpiv, cuDoubleComplex* const Barray[], int ldb, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount) except* nogil
cdef cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount) except* nogil
cdef cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount) except* nogil
cdef cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount) except* nogil
cdef cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle, int n, const float* const A[], int lda, float* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle, int n, const double* const A[], int lda, double* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, cuComplex* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double* const Aarray[], int lda, double* const TauArray[], int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex* const Aarray[], int lda, cuComplex* const TauArray[], int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int* info, int batchSize) except* nogil
cdef cublasStatus_t cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda, float* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil
cdef cublasStatus_t cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double* const Aarray[], int lda, double* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil
cdef cublasStatus_t cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex* const Aarray[], int lda, cuComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil
cdef cublasStatus_t cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil
cdef cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc) except* nogil
cdef cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc) except* nogil
cdef cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C, int ldc) except* nogil
cdef cublasStatus_t cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* AP, float* A, int lda) except* nogil
cdef cublasStatus_t cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* AP, double* A, int lda) except* nogil
cdef cublasStatus_t cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* AP, cuComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* AP, cuDoubleComplex* A, int lda) except* nogil
cdef cublasStatus_t cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, float* AP) except* nogil
cdef cublasStatus_t cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, double* AP) except* nogil
cdef cublasStatus_t cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, cuComplex* AP) except* nogil
cdef cublasStatus_t cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* AP) except* nogil
cdef cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, int* smCountTarget) except* nogil
cdef cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget) except* nogil
cdef const char* cublasGetStatusName(cublasStatus_t status) except* nogil
cdef const char* cublasGetStatusString(cublasStatus_t status) except* nogil
cdef cublasStatus_t cublasSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* const Aarray[], int lda, const float* const xarray[], int incx, const float* beta, float* const yarray[], int incy, int batchCount) except* nogil
cdef cublasStatus_t cublasDgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* const Aarray[], int lda, const double* const xarray[], int incx, const double* beta, double* const yarray[], int incy, int batchCount) except* nogil
cdef cublasStatus_t cublasCgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const xarray[], int incx, const cuComplex* beta, cuComplex* const yarray[], int incy, int batchCount) except* nogil
cdef cublasStatus_t cublasZgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const xarray[], int incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int incy, int batchCount) except* nogil
cdef cublasStatus_t cublasSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, long long int strideA, const float* x, int incx, long long int stridex, const float* beta, float* y, int incy, long long int stridey, int batchCount) except* nogil
cdef cublasStatus_t cublasDgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, long long int strideA, const double* x, int incx, long long int stridex, const double* beta, double* y, int incy, long long int stridey, int batchCount) except* nogil
cdef cublasStatus_t cublasCgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* x, int incx, long long int stridex, const cuComplex* beta, cuComplex* y, int incy, long long int stridey, int batchCount) except* nogil
cdef cublasStatus_t cublasZgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* x, int incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy, long long int stridey, int batchCount) except* nogil
cdef cublasStatus_t cublasSetVector_64(int64_t n, int64_t elemSize, const void* x, int64_t incx, void* devicePtr, int64_t incy) except* nogil
cdef cublasStatus_t cublasGetVector_64(int64_t n, int64_t elemSize, const void* x, int64_t incx, void* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasSetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb) except* nogil
cdef cublasStatus_t cublasGetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb) except* nogil
cdef cublasStatus_t cublasSetVectorAsync_64(int64_t n, int64_t elemSize, const void* hostPtr, int64_t incx, void* devicePtr, int64_t incy, cudaStream_t stream) except* nogil
cdef cublasStatus_t cublasGetVectorAsync_64(int64_t n, int64_t elemSize, const void* devicePtr, int64_t incx, void* hostPtr, int64_t incy, cudaStream_t stream) except* nogil
cdef cublasStatus_t cublasSetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb, cudaStream_t stream) except* nogil
cdef cublasStatus_t cublasGetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb, cudaStream_t stream) except* nogil
cdef cublasStatus_t cublasNrm2Ex_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil
cdef cublasStatus_t cublasSnrm2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result) except* nogil
cdef cublasStatus_t cublasDnrm2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result) except* nogil
cdef cublasStatus_t cublasScnrm2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result) except* nogil
cdef cublasStatus_t cublasDznrm2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result) except* nogil
cdef cublasStatus_t cublasDotEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil
cdef cublasStatus_t cublasDotcEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil
cdef cublasStatus_t cublasSdot_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result) except* nogil
cdef cublasStatus_t cublasDdot_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result) except* nogil
cdef cublasStatus_t cublasCdotu_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result) except* nogil
cdef cublasStatus_t cublasCdotc_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result) except* nogil
cdef cublasStatus_t cublasZdotu_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result) except* nogil
cdef cublasStatus_t cublasZdotc_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result) except* nogil
cdef cublasStatus_t cublasScalEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int64_t incx, cudaDataType executionType) except* nogil
cdef cublasStatus_t cublasSscal_64(cublasHandle_t handle, int64_t n, const float* alpha, float* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasDscal_64(cublasHandle_t handle, int64_t n, const double* alpha, double* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasCscal_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, cuComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasCsscal_64(cublasHandle_t handle, int64_t n, const float* alpha, cuComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasZscal_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasZdscal_64(cublasHandle_t handle, int64_t n, const double* alpha, cuDoubleComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasAxpyEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, cudaDataType executiontype) except* nogil
cdef cublasStatus_t cublasSaxpy_64(cublasHandle_t handle, int64_t n, const float* alpha, const float* x, int64_t incx, float* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasDaxpy_64(cublasHandle_t handle, int64_t n, const double* alpha, const double* x, int64_t incx, double* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasCaxpy_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasZaxpy_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasCopyEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy) except* nogil
cdef cublasStatus_t cublasScopy_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasDcopy_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasCcopy_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasZcopy_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasSswap_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasDswap_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasCswap_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasZswap_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasSwapEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy) except* nogil
cdef cublasStatus_t cublasIsamax_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result) except* nogil
cdef cublasStatus_t cublasIdamax_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result) except* nogil
cdef cublasStatus_t cublasIcamax_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result) except* nogil
cdef cublasStatus_t cublasIzamax_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result) except* nogil
cdef cublasStatus_t cublasIamaxEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, int64_t* result) except* nogil
cdef cublasStatus_t cublasIsamin_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result) except* nogil
cdef cublasStatus_t cublasIdamin_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result) except* nogil
cdef cublasStatus_t cublasIcamin_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result) except* nogil
cdef cublasStatus_t cublasIzamin_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result) except* nogil
cdef cublasStatus_t cublasIaminEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, int64_t* result) except* nogil
cdef cublasStatus_t cublasAsumEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* result, cudaDataType resultType, cudaDataType executiontype) except* nogil
cdef cublasStatus_t cublasSasum_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result) except* nogil
cdef cublasStatus_t cublasDasum_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result) except* nogil
cdef cublasStatus_t cublasScasum_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result) except* nogil
cdef cublasStatus_t cublasDzasum_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result) except* nogil
cdef cublasStatus_t cublasSrot_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* c, const float* s) except* nogil
cdef cublasStatus_t cublasDrot_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* c, const double* s) except* nogil
cdef cublasStatus_t cublasCrot_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const cuComplex* s) except* nogil
cdef cublasStatus_t cublasCsrot_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const float* s) except* nogil
cdef cublasStatus_t cublasZrot_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const cuDoubleComplex* s) except* nogil
cdef cublasStatus_t cublasZdrot_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const double* s) except* nogil
cdef cublasStatus_t cublasRotEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype) except* nogil
cdef cublasStatus_t cublasSrotm_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* param) except* nogil
cdef cublasStatus_t cublasDrotm_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* param) except* nogil
cdef cublasStatus_t cublasRotmEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, const void* param, cudaDataType paramType, cudaDataType executiontype) except* nogil
cdef cublasStatus_t cublasSgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasDgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasCgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasZgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasSgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasDgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasCgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasZgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasStrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasDtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasCtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasZtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasStbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasDtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasCtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasZtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasStpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasDtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasCtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasZtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasStrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasDtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasCtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasZtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasStpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasDtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasCtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasZtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasStbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasDtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasCtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasZtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil
cdef cublasStatus_t cublasSsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasDsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasCsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasZsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasChemv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasZhemv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasSsbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasDsbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasChbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasZhbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasSspmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* AP, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasDspmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* AP, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasChpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasZhpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil
cdef cublasStatus_t cublasSger_64(cublasHandle_t handle, int64_t m, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasDger_64(cublasHandle_t handle, int64_t m, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasCgeru_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasCgerc_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasZgeru_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasZgerc_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasSsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasDsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasCsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasZsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasCher_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasZher_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasSspr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* AP) except* nogil
cdef cublasStatus_t cublasDspr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* AP) except* nogil
cdef cublasStatus_t cublasChpr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const cuComplex* x, int64_t incx, cuComplex* AP) except* nogil
cdef cublasStatus_t cublasZhpr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* AP) except* nogil
cdef cublasStatus_t cublasSsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasDsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasCsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasZsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasCher2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasZher2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil
cdef cublasStatus_t cublasSspr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* AP) except* nogil
cdef cublasStatus_t cublasDspr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* AP) except* nogil
cdef cublasStatus_t cublasChpr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* AP) except* nogil
cdef cublasStatus_t cublasZhpr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* AP) except* nogil
cdef cublasStatus_t cublasSgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* const Aarray[], int64_t lda, const float* const xarray[], int64_t incx, const float* beta, float* const yarray[], int64_t incy, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasDgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* const Aarray[], int64_t lda, const double* const xarray[], int64_t incx, const double* beta, double* const yarray[], int64_t incy, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasCgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const xarray[], int64_t incx, const cuComplex* beta, cuComplex* const yarray[], int64_t incy, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasZgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const xarray[], int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int64_t incy, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasSgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* x, int64_t incx, long long int stridex, const float* beta, float* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasDgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* x, int64_t incx, long long int stridex, const double* beta, double* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasCgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* x, int64_t incx, long long int stridex, const cuComplex* beta, cuComplex* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasZgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* x, int64_t incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasSgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasDgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCgemm3mEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasSgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil
cdef cublasStatus_t cublasGemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil
cdef cublasStatus_t cublasCgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil
cdef cublasStatus_t cublasSsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* beta, float* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasDsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* beta, double* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCsyrkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCsyrk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCherk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const cuComplex* A, int64_t lda, const float* beta, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZherk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const cuDoubleComplex* A, int64_t lda, const double* beta, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCherkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCherk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil
cdef cublasStatus_t cublasSsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasDsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCher2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const float* beta, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZher2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const double* beta, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasSsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasDsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const float* beta, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const double* beta, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasSsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasDsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasChemm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZhemm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasStrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, float* B, int64_t ldb) except* nogil
cdef cublasStatus_t cublasDtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, double* B, int64_t ldb) except* nogil
cdef cublasStatus_t cublasCtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, cuComplex* B, int64_t ldb) except* nogil
cdef cublasStatus_t cublasZtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* B, int64_t ldb) except* nogil
cdef cublasStatus_t cublasStrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasDtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, double* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasSgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* const Aarray[], int64_t lda, const float* const Barray[], int64_t ldb, const float* beta, float* const Carray[], int64_t ldc, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasDgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* const Aarray[], int64_t lda, const double* const Barray[], int64_t ldb, const double* beta, double* const Carray[], int64_t ldc, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasCgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex* beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasCgemm3mBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex* beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasZgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const Barray[], int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int64_t ldc, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasSgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* B, int64_t ldb, long long int strideB, const float* beta, float* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasDgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* B, int64_t ldb, long long int strideB, const double* beta, double* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasCgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* B, int64_t ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasCgemm3mStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* B, int64_t ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasZgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* B, int64_t ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasGemmBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int64_t lda, const void* const Barray[], cudaDataType Btype, int64_t ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int64_t ldc, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil
cdef cublasStatus_t cublasGemmStridedBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, long long int strideA, const void* B, cudaDataType Btype, int64_t ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, long long int strideC, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil
cdef cublasStatus_t cublasSgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* beta, const float* B, int64_t ldb, float* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasDgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* beta, const double* B, int64_t ldb, double* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, const cuComplex* B, int64_t ldb, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasStrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* const A[], int64_t lda, float* const B[], int64_t ldb, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasDtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* const A[], int64_t lda, double* const B[], int64_t ldb, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasCtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const A[], int64_t lda, cuComplex* const B[], int64_t ldb, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasZtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int64_t lda, cuDoubleComplex* const B[], int64_t ldb, int64_t batchCount) except* nogil
cdef cublasStatus_t cublasSdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const float* A, int64_t lda, const float* x, int64_t incx, float* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasDdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const double* A, int64_t lda, const double* x, int64_t incx, double* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasCdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, cuComplex* C, int64_t ldc) except* nogil
cdef cublasStatus_t cublasZdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* C, int64_t ldc) except* nogil
