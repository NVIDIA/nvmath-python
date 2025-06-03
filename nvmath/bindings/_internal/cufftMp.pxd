# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.2.6 to 11.4.0. Do not modify it directly.

from ..cycufftMp cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef cufftResult _cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMakePlanMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftGetSizeMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftEstimate1d(int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftEstimate2d(int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftEstimateMany(int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftCreate(cufftHandle* handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftGetSizeMany(cufftHandle handle, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftGetSize(cufftHandle handle, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftSetWorkArea(cufftHandle plan, void* workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftExecD2Z(cufftHandle plan, cufftDoubleReal* idata, cufftDoubleComplex* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftExecZ2D(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleReal* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftSetStream(cufftHandle plan, cudaStream_t stream) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftDestroy(cufftHandle plan) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftGetVersion(int* version) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftGetProperty(libraryPropertyType type, int* value) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftSetPlanPropertyInt64(cufftHandle plan, cufftProperty property, const long long int inputValueInt) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftGetPlanPropertyInt64(cufftHandle plan, cufftProperty property, long long int* returnPtrValue) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftResetPlanProperty(cufftHandle plan, cufftProperty property) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtSetGPUs(cufftHandle handle, int nGPUs, int* whichGPUs) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtMalloc(cufftHandle plan, cudaLibXtDesc** descriptor, cufftXtSubFormat format) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtMemcpy(cufftHandle plan, void* dstPointer, void* srcPointer, cufftXtCopyType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtFree(cudaLibXtDesc* descriptor) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtSetWorkArea(cufftHandle plan, void** workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtQueryPlan(cufftHandle plan, void* queryStruct, cufftXtQueryType queryType) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType, size_t sharedSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtMakePlanMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtGetSizeMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtExec(cufftHandle plan, void* input, void* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMpAttachComm(cufftHandle plan, cufftMpCommType comm_type, void* comm_handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtSetDistribution(cufftHandle plan, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_input, const long long int* strides_output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftXtSetSubformatDefault(cufftHandle plan, cufftXtSubFormat subformat_forward, cufftXtSubFormat subformat_inverse) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMpCreateReshape(cufftReshapeHandle* handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMpAttachReshapeComm(cufftReshapeHandle handle, cufftMpCommType comm_type, void* comm_handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMpGetReshapeSize(cufftReshapeHandle handle, size_t* workspace_size) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult ___cufftMpMakeReshape_11_2(cufftReshapeHandle handle, size_t element_size, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_input, const long long int* strides_output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMpExecReshapeAsync(cufftReshapeHandle handle, void* data_out, const void* data_in, void* workspace, cudaStream_t stream) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult _cufftMpDestroyReshape(cufftReshapeHandle handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult ___cufftMpMakeReshape_11_4(cufftReshapeHandle handle, size_t element_size, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* strides_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_output, void* comm_handle, cufftMpCommType comm_type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
