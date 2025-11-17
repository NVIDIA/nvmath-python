# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.3.1. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.stdio cimport FILE


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum cutensorOperator_t "cutensorOperator_t":
    CUTENSOR_OP_IDENTITY "CUTENSOR_OP_IDENTITY" = 1
    CUTENSOR_OP_SQRT "CUTENSOR_OP_SQRT" = 2
    CUTENSOR_OP_RELU "CUTENSOR_OP_RELU" = 8
    CUTENSOR_OP_CONJ "CUTENSOR_OP_CONJ" = 9
    CUTENSOR_OP_RCP "CUTENSOR_OP_RCP" = 10
    CUTENSOR_OP_SIGMOID "CUTENSOR_OP_SIGMOID" = 11
    CUTENSOR_OP_TANH "CUTENSOR_OP_TANH" = 12
    CUTENSOR_OP_EXP "CUTENSOR_OP_EXP" = 22
    CUTENSOR_OP_LOG "CUTENSOR_OP_LOG" = 23
    CUTENSOR_OP_ABS "CUTENSOR_OP_ABS" = 24
    CUTENSOR_OP_NEG "CUTENSOR_OP_NEG" = 25
    CUTENSOR_OP_SIN "CUTENSOR_OP_SIN" = 26
    CUTENSOR_OP_COS "CUTENSOR_OP_COS" = 27
    CUTENSOR_OP_TAN "CUTENSOR_OP_TAN" = 28
    CUTENSOR_OP_SINH "CUTENSOR_OP_SINH" = 29
    CUTENSOR_OP_COSH "CUTENSOR_OP_COSH" = 30
    CUTENSOR_OP_ASIN "CUTENSOR_OP_ASIN" = 31
    CUTENSOR_OP_ACOS "CUTENSOR_OP_ACOS" = 32
    CUTENSOR_OP_ATAN "CUTENSOR_OP_ATAN" = 33
    CUTENSOR_OP_ASINH "CUTENSOR_OP_ASINH" = 34
    CUTENSOR_OP_ACOSH "CUTENSOR_OP_ACOSH" = 35
    CUTENSOR_OP_ATANH "CUTENSOR_OP_ATANH" = 36
    CUTENSOR_OP_CEIL "CUTENSOR_OP_CEIL" = 37
    CUTENSOR_OP_FLOOR "CUTENSOR_OP_FLOOR" = 38
    CUTENSOR_OP_MISH "CUTENSOR_OP_MISH" = 39
    CUTENSOR_OP_SWISH "CUTENSOR_OP_SWISH" = 40
    CUTENSOR_OP_SOFT_PLUS "CUTENSOR_OP_SOFT_PLUS" = 41
    CUTENSOR_OP_SOFT_SIGN "CUTENSOR_OP_SOFT_SIGN" = 42
    CUTENSOR_OP_ADD "CUTENSOR_OP_ADD" = 3
    CUTENSOR_OP_MUL "CUTENSOR_OP_MUL" = 5
    CUTENSOR_OP_MAX "CUTENSOR_OP_MAX" = 6
    CUTENSOR_OP_MIN "CUTENSOR_OP_MIN" = 7
    CUTENSOR_OP_UNKNOWN "CUTENSOR_OP_UNKNOWN" = 126

ctypedef enum cutensorStatus_t "cutensorStatus_t":
    CUTENSOR_STATUS_SUCCESS "CUTENSOR_STATUS_SUCCESS" = 0
    CUTENSOR_STATUS_NOT_INITIALIZED "CUTENSOR_STATUS_NOT_INITIALIZED" = 1
    CUTENSOR_STATUS_ALLOC_FAILED "CUTENSOR_STATUS_ALLOC_FAILED" = 3
    CUTENSOR_STATUS_INVALID_VALUE "CUTENSOR_STATUS_INVALID_VALUE" = 7
    CUTENSOR_STATUS_ARCH_MISMATCH "CUTENSOR_STATUS_ARCH_MISMATCH" = 8
    CUTENSOR_STATUS_MAPPING_ERROR "CUTENSOR_STATUS_MAPPING_ERROR" = 11
    CUTENSOR_STATUS_EXECUTION_FAILED "CUTENSOR_STATUS_EXECUTION_FAILED" = 13
    CUTENSOR_STATUS_INTERNAL_ERROR "CUTENSOR_STATUS_INTERNAL_ERROR" = 14
    CUTENSOR_STATUS_NOT_SUPPORTED "CUTENSOR_STATUS_NOT_SUPPORTED" = 15
    CUTENSOR_STATUS_LICENSE_ERROR "CUTENSOR_STATUS_LICENSE_ERROR" = 16
    CUTENSOR_STATUS_CUBLAS_ERROR "CUTENSOR_STATUS_CUBLAS_ERROR" = 17
    CUTENSOR_STATUS_CUDA_ERROR "CUTENSOR_STATUS_CUDA_ERROR" = 18
    CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE "CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE" = 19
    CUTENSOR_STATUS_INSUFFICIENT_DRIVER "CUTENSOR_STATUS_INSUFFICIENT_DRIVER" = 20
    CUTENSOR_STATUS_IO_ERROR "CUTENSOR_STATUS_IO_ERROR" = 21
    _CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR "_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cutensorAlgo_t "cutensorAlgo_t":
    CUTENSOR_ALGO_DEFAULT_PATIENT "CUTENSOR_ALGO_DEFAULT_PATIENT" = -(6)
    CUTENSOR_ALGO_GETT "CUTENSOR_ALGO_GETT" = -(4)
    CUTENSOR_ALGO_TGETT "CUTENSOR_ALGO_TGETT" = -(3)
    CUTENSOR_ALGO_TTGT "CUTENSOR_ALGO_TTGT" = -(2)
    CUTENSOR_ALGO_DEFAULT "CUTENSOR_ALGO_DEFAULT" = -(1)

ctypedef enum cutensorWorksizePreference_t "cutensorWorksizePreference_t":
    CUTENSOR_WORKSPACE_MIN "CUTENSOR_WORKSPACE_MIN" = 1
    CUTENSOR_WORKSPACE_DEFAULT "CUTENSOR_WORKSPACE_DEFAULT" = 2
    CUTENSOR_WORKSPACE_MAX "CUTENSOR_WORKSPACE_MAX" = 3

ctypedef enum cutensorOperationDescriptorAttribute_t "cutensorOperationDescriptorAttribute_t":
    CUTENSOR_OPERATION_DESCRIPTOR_TAG "CUTENSOR_OPERATION_DESCRIPTOR_TAG" = 0
    CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE "CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE" = 1
    CUTENSOR_OPERATION_DESCRIPTOR_FLOPS "CUTENSOR_OPERATION_DESCRIPTOR_FLOPS" = 2
    CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES "CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES" = 3
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT "CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT" = 4
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT "CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT" = 5
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE "CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE" = 6

ctypedef enum cutensorPlanPreferenceAttribute_t "cutensorPlanPreferenceAttribute_t":
    CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE "CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE" = 0
    CUTENSOR_PLAN_PREFERENCE_CACHE_MODE "CUTENSOR_PLAN_PREFERENCE_CACHE_MODE" = 1
    CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT "CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT" = 2
    CUTENSOR_PLAN_PREFERENCE_ALGO "CUTENSOR_PLAN_PREFERENCE_ALGO" = 3
    CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK "CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK" = 4
    CUTENSOR_PLAN_PREFERENCE_JIT "CUTENSOR_PLAN_PREFERENCE_JIT" = 5

ctypedef enum cutensorAutotuneMode_t "cutensorAutotuneMode_t":
    CUTENSOR_AUTOTUNE_MODE_NONE "CUTENSOR_AUTOTUNE_MODE_NONE" = 0
    CUTENSOR_AUTOTUNE_MODE_INCREMENTAL "CUTENSOR_AUTOTUNE_MODE_INCREMENTAL" = 1

ctypedef enum cutensorJitMode_t "cutensorJitMode_t":
    CUTENSOR_JIT_MODE_NONE "CUTENSOR_JIT_MODE_NONE" = 0
    CUTENSOR_JIT_MODE_DEFAULT "CUTENSOR_JIT_MODE_DEFAULT" = 1

ctypedef enum cutensorCacheMode_t "cutensorCacheMode_t":
    CUTENSOR_CACHE_MODE_NONE "CUTENSOR_CACHE_MODE_NONE" = 0
    CUTENSOR_CACHE_MODE_PEDANTIC "CUTENSOR_CACHE_MODE_PEDANTIC" = 1

ctypedef enum cutensorPlanAttribute_t "cutensorPlanAttribute_t":
    CUTENSOR_PLAN_REQUIRED_WORKSPACE "CUTENSOR_PLAN_REQUIRED_WORKSPACE" = 0


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


ctypedef cudaDataType_t cutensorDataType_t 'cutensorDataType_t'
ctypedef void* cutensorComputeDescriptor_t 'cutensorComputeDescriptor_t'
ctypedef void* cutensorOperationDescriptor_t 'cutensorOperationDescriptor_t'
ctypedef void* cutensorPlan_t 'cutensorPlan_t'
ctypedef void* cutensorPlanPreference_t 'cutensorPlanPreference_t'
ctypedef void* cutensorHandle_t 'cutensorHandle_t'
ctypedef void* cutensorTensorDescriptor_t 'cutensorTensorDescriptor_t'
ctypedef void* cutensorBlockSparseTensorDescriptor_t 'cutensorBlockSparseTensorDescriptor_t'
ctypedef void (*cutensorLoggerCallback_t 'cutensorLoggerCallback_t')(
    int32_t logLevel,
    const char* functionName,
    const char* message
)


###############################################################################
# Functions
###############################################################################

cdef cutensorStatus_t cutensorCreate(cutensorHandle_t* handle) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorDestroy(cutensorHandle_t handle) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorHandleResizePlanCache(cutensorHandle_t handle, const uint32_t numEntries) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorHandleWritePlanCacheToFile(const cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorHandleReadPlanCacheFromFile(cutensorHandle_t handle, const char filename[], uint32_t* numCachelinesRead) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorWriteKernelCacheToFile(const cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorReadKernelCacheFromFile(cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreateTensorDescriptor(const cutensorHandle_t handle, cutensorTensorDescriptor_t* desc, const uint32_t numModes, const int64_t extent[], const int64_t stride[], cudaDataType_t dataType, uint32_t alignmentRequirement) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorDestroyTensorDescriptor(cutensorTensorDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreateElementwiseTrinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAB, cutensorOperator_t opABC, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorElementwiseTrinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* B, const void* gamma, const void* C, void* D, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreateElementwiseBinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAC, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorElementwiseBinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* gamma, const void* C, void* D, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreatePermutation(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorPermute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, void* B, const cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreateContraction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorDestroyOperationDescriptor(cutensorOperationDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorOperationDescriptorSetAttribute(const cutensorHandle_t handle, cutensorOperationDescriptor_t desc, cutensorOperationDescriptorAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorOperationDescriptorGetAttribute(const cutensorHandle_t handle, cutensorOperationDescriptor_t desc, cutensorOperationDescriptorAttribute_t attr, void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreatePlanPreference(const cutensorHandle_t handle, cutensorPlanPreference_t* pref, cutensorAlgo_t algo, cutensorJitMode_t jitMode) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorDestroyPlanPreference(cutensorPlanPreference_t pref) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorPlanPreferenceSetAttribute(const cutensorHandle_t handle, cutensorPlanPreference_t pref, cutensorPlanPreferenceAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorPlanGetAttribute(const cutensorHandle_t handle, const cutensorPlan_t plan, cutensorPlanAttribute_t attr, void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorEstimateWorkspaceSize(const cutensorHandle_t handle, const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t planPref, const cutensorWorksizePreference_t workspacePref, uint64_t* workspaceSizeEstimate) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreatePlan(const cutensorHandle_t handle, cutensorPlan_t* plan, const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t pref, uint64_t workspaceSizeLimit) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorDestroyPlan(cutensorPlan_t plan) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* B, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreateReduction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opReduce, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorReduce(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreateContractionTrinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opD, const cutensorTensorDescriptor_t descE, const int32_t modeE[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorContractTrinary(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* B, const void* C, const void* beta, const void* D, void* E, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreateBlockSparseTensorDescriptor(cutensorHandle_t handle, cutensorBlockSparseTensorDescriptor_t* desc, const uint32_t numModes, const uint64_t numNonZeroBlocks, const uint32_t numSectionsPerMode[], const int64_t extent[], const int32_t nonZeroCoordinates[], const int64_t stride[], cudaDataType_t dataType) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorDestroyBlockSparseTensorDescriptor(cutensorBlockSparseTensorDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorCreateBlockSparseContraction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorBlockSparseTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorBlockSparseTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorBlockSparseTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorBlockSparseTensorDescriptor_t descD, const int32_t modeD[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorBlockSparseContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* const A[], const void* const B[], const void* beta, const void* const C[], void* const D[], void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef const char* cutensorGetErrorString(const cutensorStatus_t error) except?NULL nogil
cdef size_t cutensorGetVersion() except?0 nogil
cdef size_t cutensorGetCudartVersion() except?0 nogil
cdef cutensorStatus_t cutensorLoggerSetCallback(cutensorLoggerCallback_t callback) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorLoggerSetFile(FILE* file) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorLoggerOpenFile(const char* logFile) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorLoggerSetLevel(int32_t level) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorLoggerSetMask(int32_t mask) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t cutensorLoggerForceDisable() except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
