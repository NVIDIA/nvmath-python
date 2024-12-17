# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from ..cycufft cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef cufftResult _cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) except* nogil
cdef cufftResult _cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) except* nogil
cdef cufftResult _cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) except* nogil
cdef cufftResult _cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch) except* nogil
cdef cufftResult _cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t* workSize) except* nogil
cdef cufftResult _cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize) except* nogil
cdef cufftResult _cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t* workSize) except* nogil
cdef cufftResult _cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except* nogil
cdef cufftResult _cufftMakePlanMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except* nogil
cdef cufftResult _cufftGetSizeMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except* nogil
cdef cufftResult _cufftEstimate1d(int nx, cufftType type, int batch, size_t* workSize) except* nogil
cdef cufftResult _cufftEstimate2d(int nx, int ny, cufftType type, size_t* workSize) except* nogil
cdef cufftResult _cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t* workSize) except* nogil
cdef cufftResult _cufftEstimateMany(int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except* nogil
cdef cufftResult _cufftCreate(cufftHandle* handle) except* nogil
cdef cufftResult _cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t* workSize) except* nogil
cdef cufftResult _cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t* workSize) except* nogil
cdef cufftResult _cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t* workSize) except* nogil
cdef cufftResult _cufftGetSizeMany(cufftHandle handle, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workArea) except* nogil
cdef cufftResult _cufftGetSize(cufftHandle handle, size_t* workSize) except* nogil
cdef cufftResult _cufftSetWorkArea(cufftHandle plan, void* workArea) except* nogil
cdef cufftResult _cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) except* nogil
cdef cufftResult _cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) except* nogil
cdef cufftResult _cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata) except* nogil
cdef cufftResult _cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) except* nogil
cdef cufftResult _cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction) except* nogil
cdef cufftResult _cufftExecD2Z(cufftHandle plan, cufftDoubleReal* idata, cufftDoubleComplex* odata) except* nogil
cdef cufftResult _cufftExecZ2D(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleReal* odata) except* nogil
cdef cufftResult _cufftSetStream(cufftHandle plan, cudaStream_t stream) except* nogil
cdef cufftResult _cufftDestroy(cufftHandle plan) except* nogil
cdef cufftResult _cufftGetVersion(int* version) except* nogil
cdef cufftResult _cufftGetProperty(libraryPropertyType type, int* value) except* nogil
cdef cufftResult _cufftXtSetGPUs(cufftHandle handle, int nGPUs, int* whichGPUs) except* nogil
cdef cufftResult _cufftXtMalloc(cufftHandle plan, cudaLibXtDesc** descriptor, cufftXtSubFormat format) except* nogil
cdef cufftResult _cufftXtMemcpy(cufftHandle plan, void* dstPointer, void* srcPointer, cufftXtCopyType type) except* nogil
cdef cufftResult _cufftXtFree(cudaLibXtDesc* descriptor) except* nogil
cdef cufftResult _cufftXtSetWorkArea(cufftHandle plan, void** workArea) except* nogil
cdef cufftResult _cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except* nogil
cdef cufftResult _cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil
cdef cufftResult _cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil
cdef cufftResult _cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except* nogil
cdef cufftResult _cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil
cdef cufftResult _cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil
cdef cufftResult _cufftXtQueryPlan(cufftHandle plan, void* queryStruct, cufftXtQueryType queryType) except* nogil
cdef cufftResult _cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType) except* nogil
cdef cufftResult _cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType, size_t sharedSize) except* nogil
cdef cufftResult _cufftXtMakePlanMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except* nogil
cdef cufftResult _cufftXtGetSizeMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except* nogil
cdef cufftResult _cufftXtExec(cufftHandle plan, void* input, void* output, int direction) except* nogil
cdef cufftResult _cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except* nogil
cdef cufftResult _cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy, size_t* workSize) except* nogil
cdef cufftResult _cufftXtSetJITCallback(cufftHandle plan, const void* lto_callback_fatbin, size_t lto_callback_fatbin_size, cufftXtCallbackType type, void** caller_info) except* nogil
cdef cufftResult _cufftXtSetSubformatDefault(cufftHandle plan, cufftXtSubFormat subformat_forward, cufftXtSubFormat subformat_inverse) except* nogil
cdef cufftResult _cufftSetPlanPropertyInt64(cufftHandle plan, cufftProperty property, const long long int inputValueInt) except* nogil
cdef cufftResult _cufftGetPlanPropertyInt64(cufftHandle plan, cufftProperty property, long long int* returnPtrValue) except* nogil
cdef cufftResult _cufftResetPlanProperty(cufftHandle plan, cufftProperty property) except* nogil
