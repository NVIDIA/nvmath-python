# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.6.0 to 0.7.0. Do not modify it directly.

from ._internal cimport cublasMp as _cublasMp


###############################################################################
# Wrapper functions
###############################################################################

cdef cublasMpStatus_t cublasMpCreate(cublasMpHandle_t* handle, cudaStream_t stream) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpCreate(handle, stream)


cdef cublasMpStatus_t cublasMpDestroy(cublasMpHandle_t handle) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpDestroy(handle)


cdef cublasMpStatus_t cublasMpStreamSet(cublasMpHandle_t handle, cudaStream_t stream) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpStreamSet(handle, stream)


cdef cublasMpStatus_t cublasMpStreamGet(cublasMpHandle_t handle, cudaStream_t* stream) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpStreamGet(handle, stream)


cdef cublasMpStatus_t cublasMpGetVersion(int* version) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpGetVersion(version)


cdef cublasMpStatus_t cublasMpGridCreate(int64_t nprow, int64_t npcol, cublasMpGridLayout_t layout, ncclComm_t comm, cublasMpGrid_t* grid) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpGridCreate(nprow, npcol, layout, comm, grid)


cdef cublasMpStatus_t cublasMpGridDestroy(cublasMpGrid_t grid) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpGridDestroy(grid)


cdef cublasMpStatus_t cublasMpMatrixDescriptorCreate(int64_t m, int64_t n, int64_t mb, int64_t nb, int64_t rsrc, int64_t csrc, int64_t lld, cudaDataType_t type, cublasMpGrid_t grid, cublasMpMatrixDescriptor_t* desc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpMatrixDescriptorCreate(m, n, mb, nb, rsrc, csrc, lld, type, grid, desc)


cdef cublasMpStatus_t cublasMpMatrixDescriptorDestroy(cublasMpMatrixDescriptor_t desc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpMatrixDescriptorDestroy(desc)


cdef cublasMpStatus_t cublasMpMatrixDescriptorInit(int64_t m, int64_t n, int64_t mb, int64_t nb, int64_t rsrc, int64_t csrc, int64_t lld, cudaDataType_t type, cublasMpGrid_t grid, cublasMpMatrixDescriptor_t desc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpMatrixDescriptorInit(m, n, mb, nb, rsrc, csrc, lld, type, grid, desc)


cdef cublasMpStatus_t cublasMpMatmulDescriptorCreate(cublasMpMatmulDescriptor_t* matmulDesc, cublasComputeType_t computeType) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpMatmulDescriptorCreate(matmulDesc, computeType)


cdef cublasMpStatus_t cublasMpMatmulDescriptorDestroy(cublasMpMatmulDescriptor_t matmulDesc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpMatmulDescriptorDestroy(matmulDesc)


cdef cublasMpStatus_t cublasMpMatmulDescriptorInit(cublasMpMatmulDescriptor_t matmulDesc, cublasComputeType_t computeType) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpMatmulDescriptorInit(matmulDesc, computeType)


cdef cublasMpStatus_t cublasMpMatmulDescriptorAttributeSet(cublasMpMatmulDescriptor_t matmulDesc, cublasMpMatmulDescriptorAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpMatmulDescriptorAttributeSet(matmulDesc, attr, buf, sizeInBytes)


cdef cublasMpStatus_t cublasMpMatmulDescriptorAttributeGet(cublasMpMatmulDescriptor_t matmulDesc, cublasMpMatmulDescriptorAttribute_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpMatmulDescriptorAttributeGet(matmulDesc, attr, buf, sizeInBytes, sizeWritten)


cdef cublasMpStatus_t cublasMpTrsm_bufferSize(cublasMpHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, cublasComputeType_t computeType, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpTrsm_bufferSize(handle, side, uplo, trans, diag, m, n, alpha, a, ia, ja, descA, b, ib, jb, descB, computeType, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpTrsm(cublasMpHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, cublasComputeType_t computeType, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpTrsm(handle, side, uplo, trans, diag, m, n, alpha, a, ia, ja, descA, b, ib, jb, descB, computeType, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpGemm_bufferSize(cublasMpHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpGemm_bufferSize(handle, transA, transB, m, n, k, alpha, a, ia, ja, descA, b, ib, jb, descB, beta, c, ic, jc, descC, computeType, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpGemm(cublasMpHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpGemm(handle, transA, transB, m, n, k, alpha, a, ia, ja, descA, b, ib, jb, descB, beta, c, ic, jc, descC, computeType, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpMatmul_bufferSize(cublasMpHandle_t handle, cublasMpMatmulDescriptor_t matmulDesc, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, const void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d, int64_t id, int64_t jd, cublasMpMatrixDescriptor_t descD, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpMatmul_bufferSize(handle, matmulDesc, m, n, k, alpha, a, ia, ja, descA, b, ib, jb, descB, beta, c, ic, jc, descC, d, id, jd, descD, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpMatmul(cublasMpHandle_t handle, cublasMpMatmulDescriptor_t matmulDesc, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, const void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d, int64_t id, int64_t jd, cublasMpMatrixDescriptor_t descD, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpMatmul(handle, matmulDesc, m, n, k, alpha, a, ia, ja, descA, b, ib, jb, descB, beta, c, ic, jc, descC, d, id, jd, descD, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpSyrk_bufferSize(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpSyrk_bufferSize(handle, uplo, trans, n, k, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, computeType, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpSyrk(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpSyrk(handle, uplo, trans, n, k, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, computeType, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef int64_t cublasMpNumroc(int64_t n, int64_t nb, uint32_t iproc, uint32_t isrcproc, uint32_t nprocs) except?-42 nogil:
    return _cublasMp._cublasMpNumroc(n, nb, iproc, isrcproc, nprocs)


cdef cublasMpStatus_t cublasMpGemr2D_bufferSize(cublasMpHandle_t handle, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpGemr2D_bufferSize(handle, m, n, a, ia, ja, descA, b, ib, jb, descB, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost, global_comm)


cdef cublasMpStatus_t cublasMpGemr2D(cublasMpHandle_t handle, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpGemr2D(handle, m, n, a, ia, ja, descA, b, ib, jb, descB, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost, global_comm)


cdef cublasMpStatus_t cublasMpTrmr2D_bufferSize(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpTrmr2D_bufferSize(handle, uplo, diag, m, n, a, ia, ja, descA, b, ib, jb, descB, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost, global_comm)


cdef cublasMpStatus_t cublasMpTrmr2D(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpTrmr2D(handle, uplo, diag, m, n, a, ia, ja, descA, b, ib, jb, descB, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost, global_comm)


cdef cublasMpStatus_t cublasMpGeadd_bufferSize(cublasMpHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpGeadd_bufferSize(handle, trans, m, n, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpGeadd(cublasMpHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpGeadd(handle, trans, m, n, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpTradd_bufferSize(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpTradd_bufferSize(handle, uplo, trans, m, n, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, workspaceSizeInBytesOnDevice, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpTradd(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpTradd(handle, uplo, trans, m, n, alpha, a, ia, ja, descA, beta, c, ic, jc, descC, d_work, workspaceSizeInBytesOnDevice, h_work, workspaceSizeInBytesOnHost)


cdef cublasMpStatus_t cublasMpSetEmulationStrategy(cublasMpHandle_t handle, cublasMpEmulationStrategy_t emulationStrategy) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpSetEmulationStrategy(handle, emulationStrategy)


cdef cublasMpStatus_t cublasMpGetEmulationStrategy(cublasMpHandle_t handle, cublasMpEmulationStrategy_t* emulationStrategy) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cublasMp._cublasMpGetEmulationStrategy(handle, emulationStrategy)
