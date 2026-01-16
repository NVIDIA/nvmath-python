# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.3.1. Do not modify it directly.

from ._internal cimport cutensor as _cutensor


###############################################################################
# Wrapper functions
###############################################################################

cdef cutensorStatus_t cutensorCreate(cutensorHandle_t* handle) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreate(handle)


cdef cutensorStatus_t cutensorDestroy(cutensorHandle_t handle) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorDestroy(handle)


cdef cutensorStatus_t cutensorHandleResizePlanCache(cutensorHandle_t handle, const uint32_t numEntries) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorHandleResizePlanCache(handle, numEntries)


cdef cutensorStatus_t cutensorHandleWritePlanCacheToFile(const cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorHandleWritePlanCacheToFile(handle, filename)


cdef cutensorStatus_t cutensorHandleReadPlanCacheFromFile(cutensorHandle_t handle, const char filename[], uint32_t* numCachelinesRead) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorHandleReadPlanCacheFromFile(handle, filename, numCachelinesRead)


cdef cutensorStatus_t cutensorWriteKernelCacheToFile(const cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorWriteKernelCacheToFile(handle, filename)


cdef cutensorStatus_t cutensorReadKernelCacheFromFile(cutensorHandle_t handle, const char filename[]) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorReadKernelCacheFromFile(handle, filename)


cdef cutensorStatus_t cutensorCreateTensorDescriptor(const cutensorHandle_t handle, cutensorTensorDescriptor_t* desc, const uint32_t numModes, const int64_t extent[], const int64_t stride[], cudaDataType_t dataType, uint32_t alignmentRequirement) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreateTensorDescriptor(handle, desc, numModes, extent, stride, dataType, alignmentRequirement)


cdef cutensorStatus_t cutensorDestroyTensorDescriptor(cutensorTensorDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorDestroyTensorDescriptor(desc)


cdef cutensorStatus_t cutensorCreateElementwiseTrinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAB, cutensorOperator_t opABC, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreateElementwiseTrinary(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, opAB, opABC, descCompute)


cdef cutensorStatus_t cutensorElementwiseTrinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* B, const void* gamma, const void* C, void* D, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorElementwiseTrinaryExecute(handle, plan, alpha, A, beta, B, gamma, C, D, stream)


cdef cutensorStatus_t cutensorCreateElementwiseBinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAC, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreateElementwiseBinary(handle, desc, descA, modeA, opA, descC, modeC, opC, descD, modeD, opAC, descCompute)


cdef cutensorStatus_t cutensorElementwiseBinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* gamma, const void* C, void* D, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorElementwiseBinaryExecute(handle, plan, alpha, A, gamma, C, D, stream)


cdef cutensorStatus_t cutensorCreatePermutation(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreatePermutation(handle, desc, descA, modeA, opA, descB, modeB, descCompute)


cdef cutensorStatus_t cutensorPermute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, void* B, const cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorPermute(handle, plan, alpha, A, B, stream)


cdef cutensorStatus_t cutensorCreateContraction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreateContraction(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, descCompute)


cdef cutensorStatus_t cutensorDestroyOperationDescriptor(cutensorOperationDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorDestroyOperationDescriptor(desc)


cdef cutensorStatus_t cutensorOperationDescriptorSetAttribute(const cutensorHandle_t handle, cutensorOperationDescriptor_t desc, cutensorOperationDescriptorAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorOperationDescriptorSetAttribute(handle, desc, attr, buf, sizeInBytes)


cdef cutensorStatus_t cutensorOperationDescriptorGetAttribute(const cutensorHandle_t handle, cutensorOperationDescriptor_t desc, cutensorOperationDescriptorAttribute_t attr, void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorOperationDescriptorGetAttribute(handle, desc, attr, buf, sizeInBytes)


cdef cutensorStatus_t cutensorCreatePlanPreference(const cutensorHandle_t handle, cutensorPlanPreference_t* pref, cutensorAlgo_t algo, cutensorJitMode_t jitMode) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreatePlanPreference(handle, pref, algo, jitMode)


cdef cutensorStatus_t cutensorDestroyPlanPreference(cutensorPlanPreference_t pref) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorDestroyPlanPreference(pref)


cdef cutensorStatus_t cutensorPlanPreferenceSetAttribute(const cutensorHandle_t handle, cutensorPlanPreference_t pref, cutensorPlanPreferenceAttribute_t attr, const void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorPlanPreferenceSetAttribute(handle, pref, attr, buf, sizeInBytes)


cdef cutensorStatus_t cutensorPlanGetAttribute(const cutensorHandle_t handle, const cutensorPlan_t plan, cutensorPlanAttribute_t attr, void* buf, size_t sizeInBytes) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorPlanGetAttribute(handle, plan, attr, buf, sizeInBytes)


cdef cutensorStatus_t cutensorEstimateWorkspaceSize(const cutensorHandle_t handle, const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t planPref, const cutensorWorksizePreference_t workspacePref, uint64_t* workspaceSizeEstimate) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref, workspaceSizeEstimate)


cdef cutensorStatus_t cutensorCreatePlan(const cutensorHandle_t handle, cutensorPlan_t* plan, const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t pref, uint64_t workspaceSizeLimit) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreatePlan(handle, plan, desc, pref, workspaceSizeLimit)


cdef cutensorStatus_t cutensorDestroyPlan(cutensorPlan_t plan) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorDestroyPlan(plan)


cdef cutensorStatus_t cutensorContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* B, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorContract(handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)


cdef cutensorStatus_t cutensorCreateReduction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opReduce, const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreateReduction(handle, desc, descA, modeA, opA, descC, modeC, opC, descD, modeD, opReduce, descCompute)


cdef cutensorStatus_t cutensorReduce(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorReduce(handle, plan, alpha, A, beta, C, D, workspace, workspaceSize, stream)


cdef cutensorStatus_t cutensorCreateContractionTrinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opD, const cutensorTensorDescriptor_t descE, const int32_t modeE[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreateContractionTrinary(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, opD, descE, modeE, descCompute)


cdef cutensorStatus_t cutensorContractTrinary(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* B, const void* C, const void* beta, const void* D, void* E, void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorContractTrinary(handle, plan, alpha, A, B, C, beta, D, E, workspace, workspaceSize, stream)


cdef cutensorStatus_t cutensorCreateBlockSparseTensorDescriptor(cutensorHandle_t handle, cutensorBlockSparseTensorDescriptor_t* desc, const uint32_t numModes, const uint64_t numNonZeroBlocks, const uint32_t numSectionsPerMode[], const int64_t extent[], const int32_t nonZeroCoordinates[], const int64_t stride[], cudaDataType_t dataType) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreateBlockSparseTensorDescriptor(handle, desc, numModes, numNonZeroBlocks, numSectionsPerMode, extent, nonZeroCoordinates, stride, dataType)


cdef cutensorStatus_t cutensorDestroyBlockSparseTensorDescriptor(cutensorBlockSparseTensorDescriptor_t desc) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorDestroyBlockSparseTensorDescriptor(desc)


cdef cutensorStatus_t cutensorCreateBlockSparseContraction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorBlockSparseTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorBlockSparseTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorBlockSparseTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorBlockSparseTensorDescriptor_t descD, const int32_t modeD[], const cutensorComputeDescriptor_t descCompute) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorCreateBlockSparseContraction(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, descCompute)


cdef cutensorStatus_t cutensorBlockSparseContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* const A[], const void* const B[], const void* beta, const void* const C[], void* const D[], void* workspace, uint64_t workspaceSize, cudaStream_t stream) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorBlockSparseContract(handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)


cdef const char* cutensorGetErrorString(const cutensorStatus_t error) except?NULL nogil:
    return _cutensor._cutensorGetErrorString(error)


cdef size_t cutensorGetVersion() except?0 nogil:
    return _cutensor._cutensorGetVersion()


cdef size_t cutensorGetCudartVersion() except?0 nogil:
    return _cutensor._cutensorGetCudartVersion()


cdef cutensorStatus_t cutensorLoggerSetCallback(cutensorLoggerCallback_t callback) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorLoggerSetCallback(callback)


cdef cutensorStatus_t cutensorLoggerSetFile(FILE* file) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorLoggerSetFile(file)


cdef cutensorStatus_t cutensorLoggerOpenFile(const char* logFile) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorLoggerOpenFile(logFile)


cdef cutensorStatus_t cutensorLoggerSetLevel(int32_t level) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorLoggerSetLevel(level)


cdef cutensorStatus_t cutensorLoggerSetMask(int32_t mask) except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorLoggerSetMask(mask)


cdef cutensorStatus_t cutensorLoggerForceDisable() except?_CUTENSORSTATUS_T_INTERNAL_LOADING_ERROR nogil:
    return _cutensor._cutensorLoggerForceDisable()
