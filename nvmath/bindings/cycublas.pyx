# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from ._internal cimport cublas as _cublas


###############################################################################
# Wrapper functions
###############################################################################

cdef cublasStatus_t cublasCreate(cublasHandle_t* handle) except* nogil:
    return _cublas._cublasCreate(handle)


cdef cublasStatus_t cublasDestroy(cublasHandle_t handle) except* nogil:
    return _cublas._cublasDestroy(handle)


cdef cublasStatus_t cublasGetVersion(cublasHandle_t handle, int* version) except* nogil:
    return _cublas._cublasGetVersion(handle, version)


cdef cublasStatus_t cublasGetProperty(libraryPropertyType type, int* value) except* nogil:
    return _cublas._cublasGetProperty(type, value)


cdef size_t cublasGetCudartVersion() except* nogil:
    return _cublas._cublasGetCudartVersion()


cdef cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes) except* nogil:
    return _cublas._cublasSetWorkspace(handle, workspace, workspaceSizeInBytes)


cdef cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) except* nogil:
    return _cublas._cublasSetStream(handle, streamId)


cdef cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId) except* nogil:
    return _cublas._cublasGetStream(handle, streamId)


cdef cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode) except* nogil:
    return _cublas._cublasGetPointerMode(handle, mode)


cdef cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) except* nogil:
    return _cublas._cublasSetPointerMode(handle, mode)


cdef cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode) except* nogil:
    return _cublas._cublasGetAtomicsMode(handle, mode)


cdef cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) except* nogil:
    return _cublas._cublasSetAtomicsMode(handle, mode)


cdef cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) except* nogil:
    return _cublas._cublasGetMathMode(handle, mode)


cdef cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) except* nogil:
    return _cublas._cublasSetMathMode(handle, mode)


cdef cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFileName) except* nogil:
    return _cublas._cublasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, logFileName)


cdef cublasStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback) except* nogil:
    return _cublas._cublasSetLoggerCallback(userCallback)


cdef cublasStatus_t cublasGetLoggerCallback(cublasLogCallback* userCallback) except* nogil:
    return _cublas._cublasGetLoggerCallback(userCallback)


cdef cublasStatus_t cublasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy) except* nogil:
    return _cublas._cublasSetVector(n, elemSize, x, incx, devicePtr, incy)


cdef cublasStatus_t cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy) except* nogil:
    return _cublas._cublasGetVector(n, elemSize, x, incx, y, incy)


cdef cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) except* nogil:
    return _cublas._cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) except* nogil:
    return _cublas._cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream) except* nogil:
    return _cublas._cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream)


cdef cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream) except* nogil:
    return _cublas._cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream)


cdef cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) except* nogil:
    return _cublas._cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) except* nogil:
    return _cublas._cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t cublasNrm2Ex(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    return _cublas._cublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType)


cdef cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) except* nogil:
    return _cublas._cublasSnrm2(handle, n, x, incx, result)


cdef cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result) except* nogil:
    return _cublas._cublasDnrm2(handle, n, x, incx, result)


cdef cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) except* nogil:
    return _cublas._cublasScnrm2(handle, n, x, incx, result)


cdef cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) except* nogil:
    return _cublas._cublasDznrm2(handle, n, x, incx, result)


cdef cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    return _cublas._cublasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t cublasDotcEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    return _cublas._cublasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t cublasSdot(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) except* nogil:
    return _cublas._cublasSdot(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasDdot(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) except* nogil:
    return _cublas._cublasDdot(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasCdotu(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result) except* nogil:
    return _cublas._cublasCdotu(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasCdotc(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result) except* nogil:
    return _cublas._cublasCdotc(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasZdotu(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result) except* nogil:
    return _cublas._cublasZdotu(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasZdotc(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result) except* nogil:
    return _cublas._cublasZdotc(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasScalEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int incx, cudaDataType executionType) except* nogil:
    return _cublas._cublasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType)


cdef cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) except* nogil:
    return _cublas._cublasSscal(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) except* nogil:
    return _cublas._cublasDscal(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasCscal(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx) except* nogil:
    return _cublas._cublasCscal(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasCsscal(cublasHandle_t handle, int n, const float* alpha, cuComplex* x, int incx) except* nogil:
    return _cublas._cublasCsscal(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasZscal(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx) except* nogil:
    return _cublas._cublasZscal(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasZdscal(cublasHandle_t handle, int n, const double* alpha, cuDoubleComplex* x, int incx) except* nogil:
    return _cublas._cublasZdscal(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasAxpyEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, cudaDataType executiontype) except* nogil:
    return _cublas._cublasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)


cdef cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) except* nogil:
    return _cublas._cublasSaxpy(handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) except* nogil:
    return _cublas._cublasDaxpy(handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy) except* nogil:
    return _cublas._cublasCaxpy(handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except* nogil:
    return _cublas._cublasZaxpy(handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t cublasCopyEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy) except* nogil:
    return _cublas._cublasCopyEx(handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t cublasScopy(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) except* nogil:
    return _cublas._cublasScopy(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasDcopy(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) except* nogil:
    return _cublas._cublasDcopy(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasCcopy(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy) except* nogil:
    return _cublas._cublasCcopy(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasZcopy(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except* nogil:
    return _cublas._cublasZcopy(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy) except* nogil:
    return _cublas._cublasSswap(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy) except* nogil:
    return _cublas._cublasDswap(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasCswap(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy) except* nogil:
    return _cublas._cublasCswap(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except* nogil:
    return _cublas._cublasZswap(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasSwapEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy) except* nogil:
    return _cublas._cublasSwapEx(handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float* x, int incx, int* result) except* nogil:
    return _cublas._cublasIsamax(handle, n, x, incx, result)


cdef cublasStatus_t cublasIdamax(cublasHandle_t handle, int n, const double* x, int incx, int* result) except* nogil:
    return _cublas._cublasIdamax(handle, n, x, incx, result)


cdef cublasStatus_t cublasIcamax(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) except* nogil:
    return _cublas._cublasIcamax(handle, n, x, incx, result)


cdef cublasStatus_t cublasIzamax(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) except* nogil:
    return _cublas._cublasIzamax(handle, n, x, incx, result)


cdef cublasStatus_t cublasIamaxEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result) except* nogil:
    return _cublas._cublasIamaxEx(handle, n, x, xType, incx, result)


cdef cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float* x, int incx, int* result) except* nogil:
    return _cublas._cublasIsamin(handle, n, x, incx, result)


cdef cublasStatus_t cublasIdamin(cublasHandle_t handle, int n, const double* x, int incx, int* result) except* nogil:
    return _cublas._cublasIdamin(handle, n, x, incx, result)


cdef cublasStatus_t cublasIcamin(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) except* nogil:
    return _cublas._cublasIcamin(handle, n, x, incx, result)


cdef cublasStatus_t cublasIzamin(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) except* nogil:
    return _cublas._cublasIzamin(handle, n, x, incx, result)


cdef cublasStatus_t cublasIaminEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result) except* nogil:
    return _cublas._cublasIaminEx(handle, n, x, xType, incx, result)


cdef cublasStatus_t cublasAsumEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executiontype) except* nogil:
    return _cublas._cublasAsumEx(handle, n, x, xType, incx, result, resultType, executiontype)


cdef cublasStatus_t cublasSasum(cublasHandle_t handle, int n, const float* x, int incx, float* result) except* nogil:
    return _cublas._cublasSasum(handle, n, x, incx, result)


cdef cublasStatus_t cublasDasum(cublasHandle_t handle, int n, const double* x, int incx, double* result) except* nogil:
    return _cublas._cublasDasum(handle, n, x, incx, result)


cdef cublasStatus_t cublasScasum(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) except* nogil:
    return _cublas._cublasScasum(handle, n, x, incx, result)


cdef cublasStatus_t cublasDzasum(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) except* nogil:
    return _cublas._cublasDzasum(handle, n, x, incx, result)


cdef cublasStatus_t cublasSrot(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s) except* nogil:
    return _cublas._cublasSrot(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasDrot(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s) except* nogil:
    return _cublas._cublasDrot(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasCrot(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s) except* nogil:
    return _cublas._cublasCrot(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasCsrot(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s) except* nogil:
    return _cublas._cublasCsrot(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasZrot(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s) except* nogil:
    return _cublas._cublasZrot(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasZdrot(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s) except* nogil:
    return _cublas._cublasZdrot(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasRotEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype) except* nogil:
    return _cublas._cublasRotEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)


cdef cublasStatus_t cublasSrotg(cublasHandle_t handle, float* a, float* b, float* c, float* s) except* nogil:
    return _cublas._cublasSrotg(handle, a, b, c, s)


cdef cublasStatus_t cublasDrotg(cublasHandle_t handle, double* a, double* b, double* c, double* s) except* nogil:
    return _cublas._cublasDrotg(handle, a, b, c, s)


cdef cublasStatus_t cublasCrotg(cublasHandle_t handle, cuComplex* a, cuComplex* b, float* c, cuComplex* s) except* nogil:
    return _cublas._cublasCrotg(handle, a, b, c, s)


cdef cublasStatus_t cublasZrotg(cublasHandle_t handle, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s) except* nogil:
    return _cublas._cublasZrotg(handle, a, b, c, s)


cdef cublasStatus_t cublasRotgEx(cublasHandle_t handle, void* a, void* b, cudaDataType abType, void* c, void* s, cudaDataType csType, cudaDataType executiontype) except* nogil:
    return _cublas._cublasRotgEx(handle, a, b, abType, c, s, csType, executiontype)


cdef cublasStatus_t cublasSrotm(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param) except* nogil:
    return _cublas._cublasSrotm(handle, n, x, incx, y, incy, param)


cdef cublasStatus_t cublasDrotm(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param) except* nogil:
    return _cublas._cublasDrotm(handle, n, x, incx, y, incy, param)


cdef cublasStatus_t cublasRotmEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* param, cudaDataType paramType, cudaDataType executiontype) except* nogil:
    return _cublas._cublasRotmEx(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype)


cdef cublasStatus_t cublasSrotmg(cublasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param) except* nogil:
    return _cublas._cublasSrotmg(handle, d1, d2, x1, y1, param)


cdef cublasStatus_t cublasDrotmg(cublasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param) except* nogil:
    return _cublas._cublasDrotmg(handle, d1, d2, x1, y1, param)


cdef cublasStatus_t cublasRotmgEx(cublasHandle_t handle, void* d1, cudaDataType d1Type, void* d2, cudaDataType d2Type, void* x1, cudaDataType x1Type, const void* y1, cudaDataType y1Type, void* param, cudaDataType paramType, cudaDataType executiontype) except* nogil:
    return _cublas._cublasRotmgEx(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype)


cdef cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil:
    return _cublas._cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil:
    return _cublas._cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasCgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    return _cublas._cublasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    return _cublas._cublasZgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil:
    return _cublas._cublasSgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil:
    return _cublas._cublasDgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    return _cublas._cublasCgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    return _cublas._cublasZgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) except* nogil:
    return _cublas._cublasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasDtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx) except* nogil:
    return _cublas._cublasDtrmv(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasCtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil:
    return _cublas._cublasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasZtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil:
    return _cublas._cublasZtrmv(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasStbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx) except* nogil:
    return _cublas._cublasStbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasDtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx) except* nogil:
    return _cublas._cublasDtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasCtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil:
    return _cublas._cublasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasZtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil:
    return _cublas._cublasZtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasStpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx) except* nogil:
    return _cublas._cublasStpmv(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasDtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx) except* nogil:
    return _cublas._cublasDtpmv(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasCtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx) except* nogil:
    return _cublas._cublasCtpmv(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasZtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) except* nogil:
    return _cublas._cublasZtpmv(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasStrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) except* nogil:
    return _cublas._cublasStrsv(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx) except* nogil:
    return _cublas._cublasDtrsv(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasCtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil:
    return _cublas._cublasCtrsv(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasZtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil:
    return _cublas._cublasZtrsv(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasStpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx) except* nogil:
    return _cublas._cublasStpsv(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasDtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx) except* nogil:
    return _cublas._cublasDtpsv(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasCtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx) except* nogil:
    return _cublas._cublasCtpsv(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasZtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) except* nogil:
    return _cublas._cublasZtpsv(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasStbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx) except* nogil:
    return _cublas._cublasStbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasDtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx) except* nogil:
    return _cublas._cublasDtbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasCtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil:
    return _cublas._cublasCtbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasZtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil:
    return _cublas._cublasZtbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil:
    return _cublas._cublasSsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasDsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil:
    return _cublas._cublasDsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasCsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    return _cublas._cublasCsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    return _cublas._cublasZsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    return _cublas._cublasChemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    return _cublas._cublasZhemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil:
    return _cublas._cublasSsbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil:
    return _cublas._cublasDsbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasChbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    return _cublas._cublasChbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZhbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    return _cublas._cublasZhbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasSspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy) except* nogil:
    return _cublas._cublasSspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t cublasDspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy) except* nogil:
    return _cublas._cublasDspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t cublasChpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    return _cublas._cublasChpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZhpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    return _cublas._cublasZhpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t cublasSger(cublasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) except* nogil:
    return _cublas._cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasDger(cublasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) except* nogil:
    return _cublas._cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasCgeru(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil:
    return _cublas._cublasCgeru(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasCgerc(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil:
    return _cublas._cublasCgerc(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasZgeru(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil:
    return _cublas._cublasZgeru(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasZgerc(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil:
    return _cublas._cublasZgerc(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda) except* nogil:
    return _cublas._cublasSsyr(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasDsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* A, int lda) except* nogil:
    return _cublas._cublasDsyr(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasCsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* A, int lda) except* nogil:
    return _cublas._cublasCsyr(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasZsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda) except* nogil:
    return _cublas._cublasZsyr(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasCher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* A, int lda) except* nogil:
    return _cublas._cublasCher(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasZher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda) except* nogil:
    return _cublas._cublasZher(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasSspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP) except* nogil:
    return _cublas._cublasSspr(handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t cublasDspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP) except* nogil:
    return _cublas._cublasDspr(handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t cublasChpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* AP) except* nogil:
    return _cublas._cublasChpr(handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t cublasZhpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP) except* nogil:
    return _cublas._cublasZhpr(handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t cublasSsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) except* nogil:
    return _cublas._cublasSsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasDsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) except* nogil:
    return _cublas._cublasDsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasCsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil:
    return _cublas._cublasCsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasZsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil:
    return _cublas._cublasZsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil:
    return _cublas._cublasCher2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil:
    return _cublas._cublasZher2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasSspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP) except* nogil:
    return _cublas._cublasSspr2(handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t cublasDspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP) except* nogil:
    return _cublas._cublasDspr2(handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t cublasChpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP) except* nogil:
    return _cublas._cublasChpr2(handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t cublasZhpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* AP) except* nogil:
    return _cublas._cublasZhpr2(handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil:
    return _cublas._cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil:
    return _cublas._cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    return _cublas._cublasCgemm3mEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasZgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const float* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    return _cublas._cublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    return _cublas._cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)


cdef cublasStatus_t cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    return _cublas._cublasCgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasUint8gemmBias(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc, int m, int n, int k, const unsigned char* A, int A_bias, int lda, const unsigned char* B, int B_bias, int ldb, unsigned char* C, int C_bias, int ldc, int C_mult, int C_shift) except* nogil:
    return _cublas._cublasUint8gemmBias(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift)


cdef cublasStatus_t cublasSsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc) except* nogil:
    return _cublas._cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasDsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc) except* nogil:
    return _cublas._cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasCsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasZsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    return _cublas._cublasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    return _cublas._cublasCsyrk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasCherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const cuComplex* A, int lda, const float* beta, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCherk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasZherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const cuDoubleComplex* A, int lda, const double* beta, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZherk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const float* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    return _cublas._cublasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const float* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    return _cublas._cublasCherk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasSsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil:
    return _cublas._cublasSsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasDsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil:
    return _cublas._cublasDsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil:
    return _cublas._cublasSsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil:
    return _cublas._cublasDsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasSsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil:
    return _cublas._cublasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasDsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil:
    return _cublas._cublasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasChemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasChemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZhemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZhemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasStrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb) except* nogil:
    return _cublas._cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t cublasDtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb) except* nogil:
    return _cublas._cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t cublasCtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb) except* nogil:
    return _cublas._cublasCtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t cublasZtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb) except* nogil:
    return _cublas._cublasZtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t cublasStrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* C, int ldc) except* nogil:
    return _cublas._cublasStrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t cublasDtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, double* C, int ldc) except* nogil:
    return _cublas._cublasDtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t cublasCtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t cublasZtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount) except* nogil:
    return _cublas._cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double* beta, double* const Carray[], int ldc, int batchCount) except* nogil:
    return _cublas._cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount) except* nogil:
    return _cublas._cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t cublasCgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount) except* nogil:
    return _cublas._cublasCgemm3mBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int ldc, int batchCount) except* nogil:
    return _cublas._cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    return _cublas._cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)


cdef cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    return _cublas._cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)


cdef cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount) except* nogil:
    return _cublas._cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount) except* nogil:
    return _cublas._cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount) except* nogil:
    return _cublas._cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount) except* nogil:
    return _cublas._cublasCgemm3mStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount) except* nogil:
    return _cublas._cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) except* nogil:
    return _cublas._cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) except* nogil:
    return _cublas._cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, const cuComplex* B, int ldb, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float* const A[], int lda, int* P, int* info, int batchSize) except* nogil:
    return _cublas._cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double* const A[], int lda, int* P, int* info, int batchSize) except* nogil:
    return _cublas._cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex* const A[], int lda, int* P, int* info, int batchSize) except* nogil:
    return _cublas._cublasCgetrfBatched(handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex* const A[], int lda, int* P, int* info, int batchSize) except* nogil:
    return _cublas._cublasZgetrfBatched(handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t cublasSgetriBatched(cublasHandle_t handle, int n, const float* const A[], int lda, const int* P, float* const C[], int ldc, int* info, int batchSize) except* nogil:
    return _cublas._cublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t cublasDgetriBatched(cublasHandle_t handle, int n, const double* const A[], int lda, const int* P, double* const C[], int ldc, int* info, int batchSize) except* nogil:
    return _cublas._cublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, const int* P, cuComplex* const C[], int ldc, int* info, int batchSize) except* nogil:
    return _cublas._cublasCgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, const int* P, cuDoubleComplex* const C[], int ldc, int* info, int batchSize) except* nogil:
    return _cublas._cublasZgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda, const int* devIpiv, float* const Barray[], int ldb, int* info, int batchSize) except* nogil:
    return _cublas._cublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* const Aarray[], int lda, const int* devIpiv, double* const Barray[], int ldb, int* info, int batchSize) except* nogil:
    return _cublas._cublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* const Aarray[], int lda, const int* devIpiv, cuComplex* const Barray[], int ldb, int* info, int batchSize) except* nogil:
    return _cublas._cublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int* devIpiv, cuDoubleComplex* const Barray[], int ldb, int* info, int batchSize) except* nogil:
    return _cublas._cublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount) except* nogil:
    return _cublas._cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount) except* nogil:
    return _cublas._cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount) except* nogil:
    return _cublas._cublasCtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount) except* nogil:
    return _cublas._cublasZtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle, int n, const float* const A[], int lda, float* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil:
    return _cublas._cublasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle, int n, const double* const A[], int lda, double* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil:
    return _cublas._cublasDmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, cuComplex* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil:
    return _cublas._cublasCmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil:
    return _cublas._cublasZmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info, int batchSize) except* nogil:
    return _cublas._cublasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double* const Aarray[], int lda, double* const TauArray[], int* info, int batchSize) except* nogil:
    return _cublas._cublasDgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex* const Aarray[], int lda, cuComplex* const TauArray[], int* info, int batchSize) except* nogil:
    return _cublas._cublasCgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int* info, int batchSize) except* nogil:
    return _cublas._cublasZgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda, float* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil:
    return _cublas._cublasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double* const Aarray[], int lda, double* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil:
    return _cublas._cublasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex* const Aarray[], int lda, cuComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil:
    return _cublas._cublasCgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil:
    return _cublas._cublasZgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc) except* nogil:
    return _cublas._cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc) except* nogil:
    return _cublas._cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex* C, int ldc) except* nogil:
    return _cublas._cublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C, int ldc) except* nogil:
    return _cublas._cublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* AP, float* A, int lda) except* nogil:
    return _cublas._cublasStpttr(handle, uplo, n, AP, A, lda)


cdef cublasStatus_t cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* AP, double* A, int lda) except* nogil:
    return _cublas._cublasDtpttr(handle, uplo, n, AP, A, lda)


cdef cublasStatus_t cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* AP, cuComplex* A, int lda) except* nogil:
    return _cublas._cublasCtpttr(handle, uplo, n, AP, A, lda)


cdef cublasStatus_t cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* AP, cuDoubleComplex* A, int lda) except* nogil:
    return _cublas._cublasZtpttr(handle, uplo, n, AP, A, lda)


cdef cublasStatus_t cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, float* AP) except* nogil:
    return _cublas._cublasStrttp(handle, uplo, n, A, lda, AP)


cdef cublasStatus_t cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, double* AP) except* nogil:
    return _cublas._cublasDtrttp(handle, uplo, n, A, lda, AP)


cdef cublasStatus_t cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, cuComplex* AP) except* nogil:
    return _cublas._cublasCtrttp(handle, uplo, n, A, lda, AP)


cdef cublasStatus_t cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* AP) except* nogil:
    return _cublas._cublasZtrttp(handle, uplo, n, A, lda, AP)


cdef cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, int* smCountTarget) except* nogil:
    return _cublas._cublasGetSmCountTarget(handle, smCountTarget)


cdef cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget) except* nogil:
    return _cublas._cublasSetSmCountTarget(handle, smCountTarget)


cdef const char* cublasGetStatusName(cublasStatus_t status) except* nogil:
    return _cublas._cublasGetStatusName(status)


cdef const char* cublasGetStatusString(cublasStatus_t status) except* nogil:
    return _cublas._cublasGetStatusString(status)


cdef cublasStatus_t cublasSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* const Aarray[], int lda, const float* const xarray[], int incx, const float* beta, float* const yarray[], int incy, int batchCount) except* nogil:
    return _cublas._cublasSgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t cublasDgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* const Aarray[], int lda, const double* const xarray[], int incx, const double* beta, double* const yarray[], int incy, int batchCount) except* nogil:
    return _cublas._cublasDgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t cublasCgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const xarray[], int incx, const cuComplex* beta, cuComplex* const yarray[], int incy, int batchCount) except* nogil:
    return _cublas._cublasCgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t cublasZgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const xarray[], int incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int incy, int batchCount) except* nogil:
    return _cublas._cublasZgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t cublasSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, long long int strideA, const float* x, int incx, long long int stridex, const float* beta, float* y, int incy, long long int stridey, int batchCount) except* nogil:
    return _cublas._cublasSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t cublasDgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, long long int strideA, const double* x, int incx, long long int stridex, const double* beta, double* y, int incy, long long int stridey, int batchCount) except* nogil:
    return _cublas._cublasDgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t cublasCgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* x, int incx, long long int stridex, const cuComplex* beta, cuComplex* y, int incy, long long int stridey, int batchCount) except* nogil:
    return _cublas._cublasCgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t cublasZgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* x, int incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy, long long int stridey, int batchCount) except* nogil:
    return _cublas._cublasZgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t cublasSetVector_64(int64_t n, int64_t elemSize, const void* x, int64_t incx, void* devicePtr, int64_t incy) except* nogil:
    return _cublas._cublasSetVector_64(n, elemSize, x, incx, devicePtr, incy)


cdef cublasStatus_t cublasGetVector_64(int64_t n, int64_t elemSize, const void* x, int64_t incx, void* y, int64_t incy) except* nogil:
    return _cublas._cublasGetVector_64(n, elemSize, x, incx, y, incy)


cdef cublasStatus_t cublasSetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb) except* nogil:
    return _cublas._cublasSetMatrix_64(rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t cublasGetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb) except* nogil:
    return _cublas._cublasGetMatrix_64(rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t cublasSetVectorAsync_64(int64_t n, int64_t elemSize, const void* hostPtr, int64_t incx, void* devicePtr, int64_t incy, cudaStream_t stream) except* nogil:
    return _cublas._cublasSetVectorAsync_64(n, elemSize, hostPtr, incx, devicePtr, incy, stream)


cdef cublasStatus_t cublasGetVectorAsync_64(int64_t n, int64_t elemSize, const void* devicePtr, int64_t incx, void* hostPtr, int64_t incy, cudaStream_t stream) except* nogil:
    return _cublas._cublasGetVectorAsync_64(n, elemSize, devicePtr, incx, hostPtr, incy, stream)


cdef cublasStatus_t cublasSetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb, cudaStream_t stream) except* nogil:
    return _cublas._cublasSetMatrixAsync_64(rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t cublasGetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb, cudaStream_t stream) except* nogil:
    return _cublas._cublasGetMatrixAsync_64(rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t cublasNrm2Ex_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    return _cublas._cublasNrm2Ex_64(handle, n, x, xType, incx, result, resultType, executionType)


cdef cublasStatus_t cublasSnrm2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result) except* nogil:
    return _cublas._cublasSnrm2_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasDnrm2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result) except* nogil:
    return _cublas._cublasDnrm2_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasScnrm2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result) except* nogil:
    return _cublas._cublasScnrm2_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasDznrm2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result) except* nogil:
    return _cublas._cublasDznrm2_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasDotEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    return _cublas._cublasDotEx_64(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t cublasDotcEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    return _cublas._cublasDotcEx_64(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t cublasSdot_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result) except* nogil:
    return _cublas._cublasSdot_64(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasDdot_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result) except* nogil:
    return _cublas._cublasDdot_64(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasCdotu_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result) except* nogil:
    return _cublas._cublasCdotu_64(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasCdotc_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result) except* nogil:
    return _cublas._cublasCdotc_64(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasZdotu_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result) except* nogil:
    return _cublas._cublasZdotu_64(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasZdotc_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result) except* nogil:
    return _cublas._cublasZdotc_64(handle, n, x, incx, y, incy, result)


cdef cublasStatus_t cublasScalEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int64_t incx, cudaDataType executionType) except* nogil:
    return _cublas._cublasScalEx_64(handle, n, alpha, alphaType, x, xType, incx, executionType)


cdef cublasStatus_t cublasSscal_64(cublasHandle_t handle, int64_t n, const float* alpha, float* x, int64_t incx) except* nogil:
    return _cublas._cublasSscal_64(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasDscal_64(cublasHandle_t handle, int64_t n, const double* alpha, double* x, int64_t incx) except* nogil:
    return _cublas._cublasDscal_64(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasCscal_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, cuComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasCscal_64(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasCsscal_64(cublasHandle_t handle, int64_t n, const float* alpha, cuComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasCsscal_64(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasZscal_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasZscal_64(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasZdscal_64(cublasHandle_t handle, int64_t n, const double* alpha, cuDoubleComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasZdscal_64(handle, n, alpha, x, incx)


cdef cublasStatus_t cublasAxpyEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, cudaDataType executiontype) except* nogil:
    return _cublas._cublasAxpyEx_64(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)


cdef cublasStatus_t cublasSaxpy_64(cublasHandle_t handle, int64_t n, const float* alpha, const float* x, int64_t incx, float* y, int64_t incy) except* nogil:
    return _cublas._cublasSaxpy_64(handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t cublasDaxpy_64(cublasHandle_t handle, int64_t n, const double* alpha, const double* x, int64_t incx, double* y, int64_t incy) except* nogil:
    return _cublas._cublasDaxpy_64(handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t cublasCaxpy_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasCaxpy_64(handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t cublasZaxpy_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasZaxpy_64(handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t cublasCopyEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy) except* nogil:
    return _cublas._cublasCopyEx_64(handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t cublasScopy_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* y, int64_t incy) except* nogil:
    return _cublas._cublasScopy_64(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasDcopy_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* y, int64_t incy) except* nogil:
    return _cublas._cublasDcopy_64(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasCcopy_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasCcopy_64(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasZcopy_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasZcopy_64(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasSswap_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy) except* nogil:
    return _cublas._cublasSswap_64(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasDswap_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy) except* nogil:
    return _cublas._cublasDswap_64(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasCswap_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasCswap_64(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasZswap_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasZswap_64(handle, n, x, incx, y, incy)


cdef cublasStatus_t cublasSwapEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy) except* nogil:
    return _cublas._cublasSwapEx_64(handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t cublasIsamax_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result) except* nogil:
    return _cublas._cublasIsamax_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasIdamax_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result) except* nogil:
    return _cublas._cublasIdamax_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasIcamax_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result) except* nogil:
    return _cublas._cublasIcamax_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasIzamax_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result) except* nogil:
    return _cublas._cublasIzamax_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasIamaxEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, int64_t* result) except* nogil:
    return _cublas._cublasIamaxEx_64(handle, n, x, xType, incx, result)


cdef cublasStatus_t cublasIsamin_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result) except* nogil:
    return _cublas._cublasIsamin_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasIdamin_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result) except* nogil:
    return _cublas._cublasIdamin_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasIcamin_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result) except* nogil:
    return _cublas._cublasIcamin_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasIzamin_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result) except* nogil:
    return _cublas._cublasIzamin_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasIaminEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, int64_t* result) except* nogil:
    return _cublas._cublasIaminEx_64(handle, n, x, xType, incx, result)


cdef cublasStatus_t cublasAsumEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* result, cudaDataType resultType, cudaDataType executiontype) except* nogil:
    return _cublas._cublasAsumEx_64(handle, n, x, xType, incx, result, resultType, executiontype)


cdef cublasStatus_t cublasSasum_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result) except* nogil:
    return _cublas._cublasSasum_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasDasum_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result) except* nogil:
    return _cublas._cublasDasum_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasScasum_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result) except* nogil:
    return _cublas._cublasScasum_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasDzasum_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result) except* nogil:
    return _cublas._cublasDzasum_64(handle, n, x, incx, result)


cdef cublasStatus_t cublasSrot_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* c, const float* s) except* nogil:
    return _cublas._cublasSrot_64(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasDrot_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* c, const double* s) except* nogil:
    return _cublas._cublasDrot_64(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasCrot_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const cuComplex* s) except* nogil:
    return _cublas._cublasCrot_64(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasCsrot_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const float* s) except* nogil:
    return _cublas._cublasCsrot_64(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasZrot_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const cuDoubleComplex* s) except* nogil:
    return _cublas._cublasZrot_64(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasZdrot_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const double* s) except* nogil:
    return _cublas._cublasZdrot_64(handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t cublasRotEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype) except* nogil:
    return _cublas._cublasRotEx_64(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)


cdef cublasStatus_t cublasSrotm_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* param) except* nogil:
    return _cublas._cublasSrotm_64(handle, n, x, incx, y, incy, param)


cdef cublasStatus_t cublasDrotm_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* param) except* nogil:
    return _cublas._cublasDrotm_64(handle, n, x, incx, y, incy, param)


cdef cublasStatus_t cublasRotmEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, const void* param, cudaDataType paramType, cudaDataType executiontype) except* nogil:
    return _cublas._cublasRotmEx_64(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype)


cdef cublasStatus_t cublasSgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil:
    return _cublas._cublasSgemv_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasDgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil:
    return _cublas._cublasDgemv_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasCgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasCgemv_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasZgemv_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasSgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil:
    return _cublas._cublasSgbmv_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasDgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil:
    return _cublas._cublasDgbmv_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasCgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasCgbmv_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasZgbmv_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasStrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx) except* nogil:
    return _cublas._cublasStrmv_64(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasDtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx) except* nogil:
    return _cublas._cublasDtrmv_64(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasCtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasCtrmv_64(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasZtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasZtrmv_64(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasStbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx) except* nogil:
    return _cublas._cublasStbmv_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasDtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx) except* nogil:
    return _cublas._cublasDtbmv_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasCtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasCtbmv_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasZtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasZtbmv_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasStpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx) except* nogil:
    return _cublas._cublasStpmv_64(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasDtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx) except* nogil:
    return _cublas._cublasDtpmv_64(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasCtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasCtpmv_64(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasZtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasZtpmv_64(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasStrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx) except* nogil:
    return _cublas._cublasStrsv_64(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasDtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx) except* nogil:
    return _cublas._cublasDtrsv_64(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasCtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasCtrsv_64(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasZtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasZtrsv_64(handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t cublasStpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx) except* nogil:
    return _cublas._cublasStpsv_64(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasDtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx) except* nogil:
    return _cublas._cublasDtpsv_64(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasCtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasCtpsv_64(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasZtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasZtpsv_64(handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t cublasStbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx) except* nogil:
    return _cublas._cublasStbsv_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasDtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx) except* nogil:
    return _cublas._cublasDtbsv_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasCtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasCtbsv_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasZtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil:
    return _cublas._cublasZtbsv_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t cublasSsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil:
    return _cublas._cublasSsymv_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasDsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil:
    return _cublas._cublasDsymv_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasCsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasCsymv_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasZsymv_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasChemv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasChemv_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZhemv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasZhemv_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasSsbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil:
    return _cublas._cublasSsbmv_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasDsbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil:
    return _cublas._cublasDsbmv_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasChbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasChbmv_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZhbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasZhbmv_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t cublasSspmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* AP, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil:
    return _cublas._cublasSspmv_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t cublasDspmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* AP, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil:
    return _cublas._cublasDspmv_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t cublasChpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasChpmv_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t cublasZhpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    return _cublas._cublasZhpmv_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t cublasSger_64(cublasHandle_t handle, int64_t m, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda) except* nogil:
    return _cublas._cublasSger_64(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasDger_64(cublasHandle_t handle, int64_t m, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda) except* nogil:
    return _cublas._cublasDger_64(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasCgeru_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasCgeru_64(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasCgerc_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasCgerc_64(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasZgeru_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasZgeru_64(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasZgerc_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasZgerc_64(handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasSsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* A, int64_t lda) except* nogil:
    return _cublas._cublasSsyr_64(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasDsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* A, int64_t lda) except* nogil:
    return _cublas._cublasDsyr_64(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasCsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasCsyr_64(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasZsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasZsyr_64(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasCher_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasCher_64(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasZher_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasZher_64(handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t cublasSspr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* AP) except* nogil:
    return _cublas._cublasSspr_64(handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t cublasDspr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* AP) except* nogil:
    return _cublas._cublasDspr_64(handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t cublasChpr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const cuComplex* x, int64_t incx, cuComplex* AP) except* nogil:
    return _cublas._cublasChpr_64(handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t cublasZhpr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* AP) except* nogil:
    return _cublas._cublasZhpr_64(handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t cublasSsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda) except* nogil:
    return _cublas._cublasSsyr2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasDsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda) except* nogil:
    return _cublas._cublasDsyr2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasCsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasCsyr2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasZsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasZsyr2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasCher2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasCher2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasZher2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil:
    return _cublas._cublasZher2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t cublasSspr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* AP) except* nogil:
    return _cublas._cublasSspr2_64(handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t cublasDspr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* AP) except* nogil:
    return _cublas._cublasDspr2_64(handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t cublasChpr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* AP) except* nogil:
    return _cublas._cublasChpr2_64(handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t cublasZhpr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* AP) except* nogil:
    return _cublas._cublasZhpr2_64(handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t cublasSgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* const Aarray[], int64_t lda, const float* const xarray[], int64_t incx, const float* beta, float* const yarray[], int64_t incy, int64_t batchCount) except* nogil:
    return _cublas._cublasSgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t cublasDgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* const Aarray[], int64_t lda, const double* const xarray[], int64_t incx, const double* beta, double* const yarray[], int64_t incy, int64_t batchCount) except* nogil:
    return _cublas._cublasDgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t cublasCgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const xarray[], int64_t incx, const cuComplex* beta, cuComplex* const yarray[], int64_t incy, int64_t batchCount) except* nogil:
    return _cublas._cublasCgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t cublasZgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const xarray[], int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int64_t incy, int64_t batchCount) except* nogil:
    return _cublas._cublasZgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t cublasSgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* x, int64_t incx, long long int stridex, const float* beta, float* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil:
    return _cublas._cublasSgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t cublasDgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* x, int64_t incx, long long int stridex, const double* beta, double* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil:
    return _cublas._cublasDgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t cublasCgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* x, int64_t incx, long long int stridex, const cuComplex* beta, cuComplex* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil:
    return _cublas._cublasCgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t cublasZgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* x, int64_t incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil:
    return _cublas._cublasZgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t cublasSgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil:
    return _cublas._cublasSgemm_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasDgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil:
    return _cublas._cublasDgemm_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCgemm_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCgemm3m_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCgemm3mEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    return _cublas._cublasCgemm3mEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasZgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZgemm_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZgemm3m_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasSgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    return _cublas._cublasSgemmEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasGemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    return _cublas._cublasGemmEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)


cdef cublasStatus_t cublasCgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    return _cublas._cublasCgemmEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasSsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* beta, float* C, int64_t ldc) except* nogil:
    return _cublas._cublasSsyrk_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasDsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* beta, double* C, int64_t ldc) except* nogil:
    return _cublas._cublasDsyrk_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasCsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCsyrk_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasZsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZsyrk_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasCsyrkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    return _cublas._cublasCsyrkEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasCsyrk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    return _cublas._cublasCsyrk3mEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasCherk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const cuComplex* A, int64_t lda, const float* beta, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCherk_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasZherk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const cuDoubleComplex* A, int64_t lda, const double* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZherk_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t cublasCherkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    return _cublas._cublasCherkEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasCherk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    return _cublas._cublasCherk3mEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t cublasSsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil:
    return _cublas._cublasSsyr2k_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasDsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil:
    return _cublas._cublasDsyr2k_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCsyr2k_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZsyr2k_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCher2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const float* beta, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCher2k_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZher2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const double* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZher2k_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasSsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil:
    return _cublas._cublasSsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasDsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil:
    return _cublas._cublasDsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const float* beta, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCherkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const double* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZherkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasSsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil:
    return _cublas._cublasSsymm_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasDsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil:
    return _cublas._cublasDsymm_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasCsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCsymm_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZsymm_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasChemm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasChemm_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasZhemm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZhemm_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t cublasStrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, float* B, int64_t ldb) except* nogil:
    return _cublas._cublasStrsm_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t cublasDtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, double* B, int64_t ldb) except* nogil:
    return _cublas._cublasDtrsm_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t cublasCtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, cuComplex* B, int64_t ldb) except* nogil:
    return _cublas._cublasCtrsm_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t cublasZtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* B, int64_t ldb) except* nogil:
    return _cublas._cublasZtrsm_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t cublasStrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc) except* nogil:
    return _cublas._cublasStrmm_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t cublasDtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, double* C, int64_t ldc) except* nogil:
    return _cublas._cublasDtrmm_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t cublasCtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCtrmm_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t cublasZtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZtrmm_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t cublasSgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* const Aarray[], int64_t lda, const float* const Barray[], int64_t ldb, const float* beta, float* const Carray[], int64_t ldc, int64_t batchCount) except* nogil:
    return _cublas._cublasSgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t cublasDgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* const Aarray[], int64_t lda, const double* const Barray[], int64_t ldb, const double* beta, double* const Carray[], int64_t ldc, int64_t batchCount) except* nogil:
    return _cublas._cublasDgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t cublasCgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex* beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount) except* nogil:
    return _cublas._cublasCgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t cublasCgemm3mBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex* beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount) except* nogil:
    return _cublas._cublasCgemm3mBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t cublasZgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const Barray[], int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int64_t ldc, int64_t batchCount) except* nogil:
    return _cublas._cublasZgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t cublasSgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* B, int64_t ldb, long long int strideB, const float* beta, float* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil:
    return _cublas._cublasSgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t cublasDgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* B, int64_t ldb, long long int strideB, const double* beta, double* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil:
    return _cublas._cublasDgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t cublasCgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* B, int64_t ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil:
    return _cublas._cublasCgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t cublasCgemm3mStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* B, int64_t ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil:
    return _cublas._cublasCgemm3mStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t cublasZgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* B, int64_t ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil:
    return _cublas._cublasZgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t cublasGemmBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int64_t lda, const void* const Barray[], cudaDataType Btype, int64_t ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int64_t ldc, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    return _cublas._cublasGemmBatchedEx_64(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)


cdef cublasStatus_t cublasGemmStridedBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, long long int strideA, const void* B, cudaDataType Btype, int64_t ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, long long int strideC, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    return _cublas._cublasGemmStridedBatchedEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)


cdef cublasStatus_t cublasSgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* beta, const float* B, int64_t ldb, float* C, int64_t ldc) except* nogil:
    return _cublas._cublasSgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t cublasDgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* beta, const double* B, int64_t ldb, double* C, int64_t ldc) except* nogil:
    return _cublas._cublasDgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t cublasCgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, const cuComplex* B, int64_t ldb, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t cublasZgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t cublasStrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* const A[], int64_t lda, float* const B[], int64_t ldb, int64_t batchCount) except* nogil:
    return _cublas._cublasStrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t cublasDtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* const A[], int64_t lda, double* const B[], int64_t ldb, int64_t batchCount) except* nogil:
    return _cublas._cublasDtrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t cublasCtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const A[], int64_t lda, cuComplex* const B[], int64_t ldb, int64_t batchCount) except* nogil:
    return _cublas._cublasCtrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t cublasZtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int64_t lda, cuDoubleComplex* const B[], int64_t ldb, int64_t batchCount) except* nogil:
    return _cublas._cublasZtrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t cublasSdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const float* A, int64_t lda, const float* x, int64_t incx, float* C, int64_t ldc) except* nogil:
    return _cublas._cublasSdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t cublasDdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const double* A, int64_t lda, const double* x, int64_t incx, double* C, int64_t ldc) except* nogil:
    return _cublas._cublasDdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t cublasCdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, cuComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasCdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t cublasZdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* C, int64_t ldc) except* nogil:
    return _cublas._cublasZdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t cublasSgemmGroupedBatched(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const float alpha_array[], const float* const Aarray[], const int lda_array[], const float* const Barray[], const int ldb_array[], const float beta_array[], float* const Carray[], const int ldc_array[], int group_count, const int group_size[]) except* nogil:
    return _cublas._cublasSgemmGroupedBatched(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)


cdef cublasStatus_t cublasSgemmGroupedBatched_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const float alpha_array[], const float* const Aarray[], const int64_t lda_array[], const float* const Barray[], const int64_t ldb_array[], const float beta_array[], float* const Carray[], const int64_t ldc_array[], int64_t group_count, const int64_t group_size[]) except* nogil:
    return _cublas._cublasSgemmGroupedBatched_64(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)


cdef cublasStatus_t cublasDgemmGroupedBatched(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const double alpha_array[], const double* const Aarray[], const int lda_array[], const double* const Barray[], const int ldb_array[], const double beta_array[], double* const Carray[], const int ldc_array[], int group_count, const int group_size[]) except* nogil:
    return _cublas._cublasDgemmGroupedBatched(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)


cdef cublasStatus_t cublasDgemmGroupedBatched_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const double alpha_array[], const double* const Aarray[], const int64_t lda_array[], const double* const Barray[], const int64_t ldb_array[], const double beta_array[], double* const Carray[], const int64_t ldc_array[], int64_t group_count, const int64_t group_size[]) except* nogil:
    return _cublas._cublasDgemmGroupedBatched_64(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)


cdef cublasStatus_t cublasGemmGroupedBatchedEx(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const void* alpha_array, const void* const Aarray[], cudaDataType_t Atype, const int lda_array[], const void* const Barray[], cudaDataType_t Btype, const int ldb_array[], const void* beta_array, void* const Carray[], cudaDataType_t Ctype, const int ldc_array[], int group_count, const int group_size[], cublasComputeType_t computeType) except* nogil:
    return _cublas._cublasGemmGroupedBatchedEx(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, Atype, lda_array, Barray, Btype, ldb_array, beta_array, Carray, Ctype, ldc_array, group_count, group_size, computeType)


cdef cublasStatus_t cublasGemmGroupedBatchedEx_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const void* alpha_array, const void* const Aarray[], cudaDataType_t Atype, const int64_t lda_array[], const void* const Barray[], cudaDataType_t Btype, const int64_t ldb_array[], const void* beta_array, void* const Carray[], cudaDataType_t Ctype, const int64_t ldc_array[], int64_t group_count, const int64_t group_size[], cublasComputeType_t computeType) except* nogil:
    return _cublas._cublasGemmGroupedBatchedEx_64(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, Atype, lda_array, Barray, Btype, ldb_array, beta_array, Carray, Ctype, ldc_array, group_count, group_size, computeType)
