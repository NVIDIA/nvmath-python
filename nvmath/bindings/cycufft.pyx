# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from ._internal cimport cufft as _cufft


###############################################################################
# Wrapper functions
###############################################################################

cdef cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) except* nogil:
    return _cufft._cufftPlan1d(plan, nx, type, batch)


cdef cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) except* nogil:
    return _cufft._cufftPlan2d(plan, nx, ny, type)


cdef cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) except* nogil:
    return _cufft._cufftPlan3d(plan, nx, ny, nz, type)


cdef cufftResult cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch) except* nogil:
    return _cufft._cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)


cdef cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t* workSize) except* nogil:
    return _cufft._cufftMakePlan1d(plan, nx, type, batch, workSize)


cdef cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize) except* nogil:
    return _cufft._cufftMakePlan2d(plan, nx, ny, type, workSize)


cdef cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t* workSize) except* nogil:
    return _cufft._cufftMakePlan3d(plan, nx, ny, nz, type, workSize)


cdef cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except* nogil:
    return _cufft._cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except* nogil:
    return _cufft._cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult cufftGetSizeMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except* nogil:
    return _cufft._cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t* workSize) except* nogil:
    return _cufft._cufftEstimate1d(nx, type, batch, workSize)


cdef cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t* workSize) except* nogil:
    return _cufft._cufftEstimate2d(nx, ny, type, workSize)


cdef cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t* workSize) except* nogil:
    return _cufft._cufftEstimate3d(nx, ny, nz, type, workSize)


cdef cufftResult cufftEstimateMany(int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except* nogil:
    return _cufft._cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult cufftCreate(cufftHandle* handle) except* nogil:
    return _cufft._cufftCreate(handle)


cdef cufftResult cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t* workSize) except* nogil:
    return _cufft._cufftGetSize1d(handle, nx, type, batch, workSize)


cdef cufftResult cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t* workSize) except* nogil:
    return _cufft._cufftGetSize2d(handle, nx, ny, type, workSize)


cdef cufftResult cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t* workSize) except* nogil:
    return _cufft._cufftGetSize3d(handle, nx, ny, nz, type, workSize)


cdef cufftResult cufftGetSizeMany(cufftHandle handle, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workArea) except* nogil:
    return _cufft._cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workArea)


cdef cufftResult cufftGetSize(cufftHandle handle, size_t* workSize) except* nogil:
    return _cufft._cufftGetSize(handle, workSize)


cdef cufftResult cufftSetWorkArea(cufftHandle plan, void* workArea) except* nogil:
    return _cufft._cufftSetWorkArea(plan, workArea)


cdef cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) except* nogil:
    return _cufft._cufftSetAutoAllocation(plan, autoAllocate)


cdef cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) except* nogil:
    return _cufft._cufftExecC2C(plan, idata, odata, direction)


cdef cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata) except* nogil:
    return _cufft._cufftExecR2C(plan, idata, odata)


cdef cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) except* nogil:
    return _cufft._cufftExecC2R(plan, idata, odata)


cdef cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction) except* nogil:
    return _cufft._cufftExecZ2Z(plan, idata, odata, direction)


cdef cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal* idata, cufftDoubleComplex* odata) except* nogil:
    return _cufft._cufftExecD2Z(plan, idata, odata)


cdef cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleReal* odata) except* nogil:
    return _cufft._cufftExecZ2D(plan, idata, odata)


cdef cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) except* nogil:
    return _cufft._cufftSetStream(plan, stream)


cdef cufftResult cufftDestroy(cufftHandle plan) except* nogil:
    return _cufft._cufftDestroy(plan)


cdef cufftResult cufftGetVersion(int* version) except* nogil:
    return _cufft._cufftGetVersion(version)


cdef cufftResult cufftGetProperty(libraryPropertyType type, int* value) except* nogil:
    return _cufft._cufftGetProperty(type, value)


cdef cufftResult cufftXtSetGPUs(cufftHandle handle, int nGPUs, int* whichGPUs) except* nogil:
    return _cufft._cufftXtSetGPUs(handle, nGPUs, whichGPUs)


cdef cufftResult cufftXtMalloc(cufftHandle plan, cudaLibXtDesc** descriptor, cufftXtSubFormat format) except* nogil:
    return _cufft._cufftXtMalloc(plan, descriptor, format)


cdef cufftResult cufftXtMemcpy(cufftHandle plan, void* dstPointer, void* srcPointer, cufftXtCopyType type) except* nogil:
    return _cufft._cufftXtMemcpy(plan, dstPointer, srcPointer, type)


cdef cufftResult cufftXtFree(cudaLibXtDesc* descriptor) except* nogil:
    return _cufft._cufftXtFree(descriptor)


cdef cufftResult cufftXtSetWorkArea(cufftHandle plan, void** workArea) except* nogil:
    return _cufft._cufftXtSetWorkArea(plan, workArea)


cdef cufftResult cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except* nogil:
    return _cufft._cufftXtExecDescriptorC2C(plan, input, output, direction)


cdef cufftResult cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil:
    return _cufft._cufftXtExecDescriptorR2C(plan, input, output)


cdef cufftResult cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil:
    return _cufft._cufftXtExecDescriptorC2R(plan, input, output)


cdef cufftResult cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except* nogil:
    return _cufft._cufftXtExecDescriptorZ2Z(plan, input, output, direction)


cdef cufftResult cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil:
    return _cufft._cufftXtExecDescriptorD2Z(plan, input, output)


cdef cufftResult cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except* nogil:
    return _cufft._cufftXtExecDescriptorZ2D(plan, input, output)


cdef cufftResult cufftXtQueryPlan(cufftHandle plan, void* queryStruct, cufftXtQueryType queryType) except* nogil:
    return _cufft._cufftXtQueryPlan(plan, queryStruct, queryType)


cdef cufftResult cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType) except* nogil:
    return _cufft._cufftXtClearCallback(plan, cbType)


cdef cufftResult cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType, size_t sharedSize) except* nogil:
    return _cufft._cufftXtSetCallbackSharedSize(plan, cbType, sharedSize)


cdef cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except* nogil:
    return _cufft._cufftXtMakePlanMany(plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch, workSize, executiontype)


cdef cufftResult cufftXtGetSizeMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except* nogil:
    return _cufft._cufftXtGetSizeMany(plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch, workSize, executiontype)


cdef cufftResult cufftXtExec(cufftHandle plan, void* input, void* output, int direction) except* nogil:
    return _cufft._cufftXtExec(plan, input, output, direction)


cdef cufftResult cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except* nogil:
    return _cufft._cufftXtExecDescriptor(plan, input, output, direction)


cdef cufftResult cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy, size_t* workSize) except* nogil:
    return _cufft._cufftXtSetWorkAreaPolicy(plan, policy, workSize)


cdef cufftResult cufftXtSetJITCallback(cufftHandle plan, const void* lto_callback_fatbin, size_t lto_callback_fatbin_size, cufftXtCallbackType type, void** caller_info) except* nogil:
    return _cufft._cufftXtSetJITCallback(plan, lto_callback_fatbin, lto_callback_fatbin_size, type, caller_info)


cdef cufftResult cufftXtSetSubformatDefault(cufftHandle plan, cufftXtSubFormat subformat_forward, cufftXtSubFormat subformat_inverse) except* nogil:
    return _cufft._cufftXtSetSubformatDefault(plan, subformat_forward, subformat_inverse)


cdef cufftResult cufftSetPlanPropertyInt64(cufftHandle plan, cufftProperty property, const long long int inputValueInt) except* nogil:
    return _cufft._cufftSetPlanPropertyInt64(plan, property, inputValueInt)


cdef cufftResult cufftGetPlanPropertyInt64(cufftHandle plan, cufftProperty property, long long int* returnPtrValue) except* nogil:
    return _cufft._cufftGetPlanPropertyInt64(plan, property, returnPtrValue)


cdef cufftResult cufftResetPlanProperty(cufftHandle plan, cufftProperty property) except* nogil:
    return _cufft._cufftResetPlanProperty(plan, property)
