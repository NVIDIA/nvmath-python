# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.3.1. Do not modify it directly.

from ..cycutensor cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef cutensorStatus_t _cutensorCreate(cutensorHandle_t* handle) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorDestroy(cutensorHandle_t handle) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorHandleResizePlanCache(cutensorHandle_t handle, const uint32_t numEntries) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorHandleWritePlanCacheToFile(const cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorHandleReadPlanCacheFromFile(cutensorHandle_t handle, const char filename[], uint32_t* numCachelinesRead) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorWriteKernelCacheToFile(const cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorReadKernelCacheFromFile(cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreateTensorDescriptor(const cutensorHandle_t handle, cutensorTensorDescriptor_t* desc, const uint32_t numModes, const int64_t extent[], const int64_t stride[], cudaDataType_t dataType, uint32_t alignmentRequirement) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorDestroyTensorDescriptor(cutensorTensorDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreateElementwiseTrinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAB, cutensorOperator_t opABC, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorElementwiseTrinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* B, const void* gamma, const void* C, void* D, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreateElementwiseBinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAC, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorElementwiseBinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* gamma, const void* C, void* D, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreatePermutation(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorPermute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, void* B, const cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreateContraction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorDestroyOperationDescriptor(cutensorOperationDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorOperationDescriptorSetAttribute(const cutensorHandle_t handle, cutensorOperationDescriptor_t desc, cutensorOperationDescriptorAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorOperationDescriptorGetAttribute(const cutensorHandle_t handle, cutensorOperationDescriptor_t desc, cutensorOperationDescriptorAttribute_t attr, void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreatePlanPreference(const cutensorHandle_t handle, cutensorPlanPreference_t* pref, cutensorAlgo_t algo, cutensorJitMode_t jitMode) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorDestroyPlanPreference(cutensorPlanPreference_t pref) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorPlanPreferenceSetAttribute(const cutensorHandle_t handle, cutensorPlanPreference_t pref, cutensorPlanPreferenceAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorPlanGetAttribute(const cutensorHandle_t handle, const cutensorPlan_t plan, cutensorPlanAttribute_t attr, void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorEstimateWorkspaceSize(const cutensorHandle_t handle, const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t planPref, const cutensorWorksizePreference_t workspacePref, uint64_t* workspaceSizeEstimate) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreatePlan(const cutensorHandle_t handle, cutensorPlan_t* plan, const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t pref, uint64_t workspaceSizeLimit) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorDestroyPlan(cutensorPlan_t plan) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* B, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreateReduction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opReduce, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorReduce(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreateContractionTrinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opD, const cutensorTensorDescriptor_t descE, const int32_t modeE[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorContractTrinary(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* B, const void* C, const void* beta, const void* D, void* E, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreateBlockSparseTensorDescriptor(cutensorHandle_t handle, cutensorBlockSparseTensorDescriptor_t* desc, const uint32_t numModes, const uint64_t numNonZeroBlocks, const uint32_t numSectionsPerMode[], const int64_t extent[], const int32_t nonZeroCoordinates[], const int64_t stride[], cudaDataType_t dataType) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorDestroyBlockSparseTensorDescriptor(cutensorBlockSparseTensorDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorCreateBlockSparseContraction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorBlockSparseTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorBlockSparseTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorBlockSparseTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorBlockSparseTensorDescriptor_t descD, const int32_t modeD[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorBlockSparseContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* const A[], const void* const B[], const void* beta, const void* const C[], void* const D[], void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef const char* _cutensorGetErrorString(const cutensorStatus_t error) except?NULL nogil
cdef size_t _cutensorGetVersion() except?0 nogil
cdef size_t _cutensorGetCudartVersion() except?0 nogil
cdef cutensorStatus_t _cutensorLoggerSetCallback(cutensorLoggerCallback_t callback) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorLoggerSetFile(FILE* file) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorLoggerOpenFile(const char* logFile) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorLoggerSetLevel(int32_t level) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorLoggerSetMask(int32_t mask) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
cdef cutensorStatus_t _cutensorLoggerForceDisable() except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil
