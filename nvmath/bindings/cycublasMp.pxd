# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.8.0, generator version 0.3.1.dev1303+g031f1197f. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int64_t, uint32_t, uint64_t, intptr_t

from .cycublas cimport (cublasComputeType_t, cublasOperation_t, cublasFillMode_t, cublasSideMode_t,
                        cublasDiagType_t,)

###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum cublasMpStatus_t "cublasMpStatus_t":
    CUBLASMP_STATUS_SUCCESS "CUBLASMP_STATUS_SUCCESS" = 0
    CUBLASMP_STATUS_NOT_INITIALIZED "CUBLASMP_STATUS_NOT_INITIALIZED" = 1
    CUBLASMP_STATUS_ALLOCATION_FAILED "CUBLASMP_STATUS_ALLOCATION_FAILED" = 2
    CUBLASMP_STATUS_INVALID_VALUE "CUBLASMP_STATUS_INVALID_VALUE" = 3
    CUBLASMP_STATUS_ARCHITECTURE_MISMATCH "CUBLASMP_STATUS_ARCHITECTURE_MISMATCH" = 4
    CUBLASMP_STATUS_EXECUTION_FAILED "CUBLASMP_STATUS_EXECUTION_FAILED" = 5
    CUBLASMP_STATUS_INTERNAL_ERROR "CUBLASMP_STATUS_INTERNAL_ERROR" = 6
    CUBLASMP_STATUS_NOT_SUPPORTED "CUBLASMP_STATUS_NOT_SUPPORTED" = 7
    _CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR "_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cublasMpGridLayout_t "cublasMpGridLayout_t":
    CUBLASMP_GRID_LAYOUT_COL_MAJOR "CUBLASMP_GRID_LAYOUT_COL_MAJOR" = 0
    CUBLASMP_GRID_LAYOUT_ROW_MAJOR "CUBLASMP_GRID_LAYOUT_ROW_MAJOR" = 1

ctypedef enum cublasMpMatmulDescriptorAttribute_t "cublasMpMatmulDescriptorAttribute_t":
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA" = 0
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB" = 1
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMPUTE_TYPE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMPUTE_TYPE" = 2
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE" = 3
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMMUNICATION_SM_COUNT "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMMUNICATION_SM_COUNT" = 4
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE" = 5
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_POINTER "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_POINTER" = 6
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_BATCH_STRIDE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_BATCH_STRIDE" = 7
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_DATA_TYPE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_DATA_TYPE" = 8
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_POINTER "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_POINTER" = 9
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_LD "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_LD" = 10
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_BATCH_STRIDE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_BATCH_STRIDE" = 11
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_DATA_TYPE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_DATA_TYPE" = 12
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_SCALE_POINTER "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_SCALE_POINTER" = 13
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_AMAX_POINTER "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_AMAX_POINTER" = 14
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_SCALE_MODE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_SCALE_MODE" = 15
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_POINTER "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_POINTER" = 16
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_MODE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_MODE" = 17
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_POINTER "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_POINTER" = 18
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_MODE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_MODE" = 19
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_C_SCALE_POINTER "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_C_SCALE_POINTER" = 20
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_C_SCALE_MODE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_C_SCALE_MODE" = 21
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_POINTER "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_POINTER" = 22
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_MODE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_MODE" = 23
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_AMAX_D_POINTER "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_AMAX_D_POINTER" = 24
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_OUT_SCALE_POINTER "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_OUT_SCALE_POINTER" = 25
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_OUT_SCALE_MODE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_OUT_SCALE_MODE" = 26
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMMUNICATION_TYPE "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMMUNICATION_TYPE" = 27
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_RESULT_SCHEME "CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_RESULT_SCHEME" = 28

ctypedef enum cublasMpMatmulAlgoType_t "cublasMpMatmulAlgoType_t":
    CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT "CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT" = 0
    CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_P2P "CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_P2P" = 1
    CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_MULTICAST "CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_MULTICAST" = 2
    CUBLASMP_MATMUL_ALGO_TYPE_ATOMIC_P2P "CUBLASMP_MATMUL_ALGO_TYPE_ATOMIC_P2P" = 3
    CUBLASMP_MATMUL_ALGO_TYPE_ATOMIC_MULTICAST "CUBLASMP_MATMUL_ALGO_TYPE_ATOMIC_MULTICAST" = 4
    CUBLASMP_MATMUL_ALGO_TYPE_NO_OVERLAP "CUBLASMP_MATMUL_ALGO_TYPE_NO_OVERLAP" = 5

ctypedef enum cublasMpMatmulEpilogue_t "cublasMpMatmulEpilogue_t":
    CUBLASMP_MATMUL_EPILOGUE_DEFAULT "CUBLASMP_MATMUL_EPILOGUE_DEFAULT" = 0
    CUBLASMP_MATMUL_EPILOGUE_ALLREDUCE "CUBLASMP_MATMUL_EPILOGUE_ALLREDUCE" = 1
    CUBLASMP_MATMUL_EPILOGUE_RELU "CUBLASMP_MATMUL_EPILOGUE_RELU" = 2
    CUBLASMP_MATMUL_EPILOGUE_RELU_AUX "CUBLASMP_MATMUL_EPILOGUE_RELU_AUX" = (CUBLASMP_MATMUL_EPILOGUE_RELU | 128)
    CUBLASMP_MATMUL_EPILOGUE_BIAS "CUBLASMP_MATMUL_EPILOGUE_BIAS" = 4
    CUBLASMP_MATMUL_EPILOGUE_RELU_BIAS "CUBLASMP_MATMUL_EPILOGUE_RELU_BIAS" = (CUBLASMP_MATMUL_EPILOGUE_RELU | CUBLASMP_MATMUL_EPILOGUE_BIAS)
    CUBLASMP_MATMUL_EPILOGUE_RELU_AUX_BIAS "CUBLASMP_MATMUL_EPILOGUE_RELU_AUX_BIAS" = (CUBLASMP_MATMUL_EPILOGUE_RELU_AUX | CUBLASMP_MATMUL_EPILOGUE_BIAS)
    CUBLASMP_MATMUL_EPILOGUE_DRELU "CUBLASMP_MATMUL_EPILOGUE_DRELU" = (8 | 128)
    CUBLASMP_MATMUL_EPILOGUE_DRELU_BGRAD "CUBLASMP_MATMUL_EPILOGUE_DRELU_BGRAD" = (CUBLASMP_MATMUL_EPILOGUE_DRELU | 16)
    CUBLASMP_MATMUL_EPILOGUE_GELU "CUBLASMP_MATMUL_EPILOGUE_GELU" = 32
    CUBLASMP_MATMUL_EPILOGUE_GELU_AUX "CUBLASMP_MATMUL_EPILOGUE_GELU_AUX" = (CUBLASMP_MATMUL_EPILOGUE_GELU | 128)
    CUBLASMP_MATMUL_EPILOGUE_GELU_BIAS "CUBLASMP_MATMUL_EPILOGUE_GELU_BIAS" = (CUBLASMP_MATMUL_EPILOGUE_GELU | CUBLASMP_MATMUL_EPILOGUE_BIAS)
    CUBLASMP_MATMUL_EPILOGUE_GELU_AUX_BIAS "CUBLASMP_MATMUL_EPILOGUE_GELU_AUX_BIAS" = (CUBLASMP_MATMUL_EPILOGUE_GELU_AUX | CUBLASMP_MATMUL_EPILOGUE_BIAS)
    CUBLASMP_MATMUL_EPILOGUE_DGELU "CUBLASMP_MATMUL_EPILOGUE_DGELU" = (64 | 128)
    CUBLASMP_MATMUL_EPILOGUE_DGELU_BGRAD "CUBLASMP_MATMUL_EPILOGUE_DGELU_BGRAD" = (CUBLASMP_MATMUL_EPILOGUE_DGELU | 16)
    CUBLASMP_MATMUL_EPILOGUE_BGRADA "CUBLASMP_MATMUL_EPILOGUE_BGRADA" = 256
    CUBLASMP_MATMUL_EPILOGUE_BGRADB "CUBLASMP_MATMUL_EPILOGUE_BGRADB" = 512

ctypedef enum cublasMpMatmulMatrixScale_t "cublasMpMatmulMatrixScale_t":
    CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32 "CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32" = 0
    CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3 "CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3" = 1
    CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0 "CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0" = 2
    CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32 "CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32" = 3
    CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32 "CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32" = 4
    CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32 "CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32" = 5

ctypedef enum cublasMpEmulationStrategy_t "cublasMpEmulationStrategy_t":
    CUBLASMP_EMULATION_STRATEGY_DEFAULT "CUBLASMP_EMULATION_STRATEGY_DEFAULT" = 0
    CUBLASMP_EMULATION_STRATEGY_PERFORMANT "CUBLASMP_EMULATION_STRATEGY_PERFORMANT" = 1
    CUBLASMP_EMULATION_STRATEGY_EAGER "CUBLASMP_EMULATION_STRATEGY_EAGER" = 2

ctypedef enum cublasMpResultScheme_t "cublasMpResultScheme_t":
    CUBLASMP_RESULT_SCHEME_DEFAULT "CUBLASMP_RESULT_SCHEME_DEFAULT" = 0
    CUBLASMP_RESULT_SCHEME_PARTIAL "CUBLASMP_RESULT_SCHEME_PARTIAL" = 1
    CUBLASMP_RESULT_SCHEME_FULL "CUBLASMP_RESULT_SCHEME_FULL" = 2


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


ctypedef void* ncclComm_t 'ncclComm_t'
ctypedef void* cublasMpHandle_t 'cublasMpHandle_t'
ctypedef void* cublasMpGrid_t 'cublasMpGrid_t'
ctypedef void* cublasMpMatrixDescriptor_t 'cublasMpMatrixDescriptor_t'
ctypedef void* cublasMpMatmulDescriptor_t 'cublasMpMatmulDescriptor_t'
ctypedef void (*cublasMpLoggerCallback_t 'cublasMpLoggerCallback_t')(
    int logLevel,
    const char* functionName,
    const char* message
)


###############################################################################
# Functions
###############################################################################

cdef cublasMpStatus_t cublasMpCreate(cublasMpHandle_t* handle, cudaStream_t stream) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpDestroy(cublasMpHandle_t handle) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpSetStream(cublasMpHandle_t handle, cudaStream_t stream) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpGetStream(cublasMpHandle_t handle, cudaStream_t* stream) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpSetEmulationStrategy(cublasMpHandle_t handle, cublasMpEmulationStrategy_t emulationStrategy) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpGetEmulationStrategy(cublasMpHandle_t handle, cublasMpEmulationStrategy_t* emulationStrategy) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpGetVersion(int* version) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpGridCreate(int64_t nprow, int64_t npcol, cublasMpGridLayout_t layout, ncclComm_t comm, cublasMpGrid_t* grid) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpGridDestroy(cublasMpGrid_t grid) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpBufferRegister(cublasMpGrid_t grid, void* ptr, size_t size) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpBufferDeregister(cublasMpGrid_t grid, void* ptr) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMatrixDescriptorCreate(int64_t m, int64_t n, int64_t mb, int64_t nb, int64_t rsrc, int64_t csrc, int64_t lld, cudaDataType_t type, cublasMpGrid_t grid, cublasMpMatrixDescriptor_t* desc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMatrixDescriptorDestroy(cublasMpMatrixDescriptor_t desc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMatrixDescriptorInit(int64_t m, int64_t n, int64_t mb, int64_t nb, int64_t rsrc, int64_t csrc, int64_t lld, cudaDataType_t type, cublasMpGrid_t grid, cublasMpMatrixDescriptor_t desc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMatmulDescriptorCreate(cublasMpMatmulDescriptor_t* matmulDesc, cublasComputeType_t computeType) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMatmulDescriptorDestroy(cublasMpMatmulDescriptor_t matmulDesc) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMatmulDescriptorInit(cublasMpMatmulDescriptor_t matmulDesc, cublasComputeType_t computeType) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMatmulDescriptorSetAttribute(cublasMpMatmulDescriptor_t matmulDesc, cublasMpMatmulDescriptorAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMatmulDescriptorGetAttribute(cublasMpMatmulDescriptor_t matmulDesc, cublasMpMatmulDescriptorAttribute_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpTrsm_bufferSize(cublasMpHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, cublasComputeType_t computeType, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpTrsm(cublasMpHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, cublasComputeType_t computeType, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpGemm_bufferSize(cublasMpHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpGemm(cublasMpHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMatmul_bufferSize(cublasMpHandle_t handle, cublasMpMatmulDescriptor_t matmulDesc, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, const void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d, int64_t id, int64_t jd, cublasMpMatrixDescriptor_t descD, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMatmul(cublasMpHandle_t handle, cublasMpMatmulDescriptor_t matmulDesc, int64_t m, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const void* beta, const void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d, int64_t id, int64_t jd, cublasMpMatrixDescriptor_t descD, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpSyrk_bufferSize(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpSyrk(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, cublasComputeType_t computeType, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef int64_t cublasMpNumroc(int64_t n, int64_t nb, uint32_t iproc, uint32_t isrcproc, uint32_t nprocs) except?-42 nogil
cdef const char* cublasMpGetStatusString(cublasMpStatus_t status) except?NULL nogil
cdef cublasMpStatus_t cublasMpGemr2D_bufferSize(cublasMpHandle_t handle, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpGemr2D(cublasMpHandle_t handle, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpTrmr2D_bufferSize(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpTrmr2D(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasDiagType_t diag, int64_t m, int64_t n, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, void* b, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost, ncclComm_t global_comm) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpGeadd_bufferSize(cublasMpHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpGeadd(cublasMpHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpTradd_bufferSize(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, size_t* workspaceSizeInBytesOnDevice, size_t* workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpTradd(cublasMpHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t m, int64_t n, const void* alpha, const void* a, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const void* beta, void* c, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC, void* d_work, size_t workspaceSizeInBytesOnDevice, void* h_work, size_t workspaceSizeInBytesOnHost) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpMalloc(cublasMpGrid_t grid, void** ptr, size_t size) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cublasMpStatus_t cublasMpFree(cublasMpGrid_t grid, void* ptr) except?_CUBLASMPSTATUS_T_INTERNAL_LOADING_ERROR nogil
