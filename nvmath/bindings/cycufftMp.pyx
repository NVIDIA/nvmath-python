# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.2.6 to 11.4.0. Do not modify it directly.

from ._internal cimport cufftMp as _cufftMp


###############################################################################
# Wrapper functions
###############################################################################

cdef cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftPlan1d(plan, nx, type, batch)


cdef cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftPlan2d(plan, nx, ny, type)


cdef cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftPlan3d(plan, nx, ny, nz, type)


cdef cufftResult cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)


cdef cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMakePlan1d(plan, nx, type, batch, workSize)


cdef cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMakePlan2d(plan, nx, ny, type, workSize)


cdef cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMakePlan3d(plan, nx, ny, nz, type, workSize)


cdef cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult cufftGetSizeMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftEstimate1d(nx, type, batch, workSize)


cdef cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftEstimate2d(nx, ny, type, workSize)


cdef cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftEstimate3d(nx, ny, nz, type, workSize)


cdef cufftResult cufftEstimateMany(int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize)


cdef cufftResult cufftCreate(cufftHandle* handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftCreate(handle)


cdef cufftResult cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftGetSize1d(handle, nx, type, batch, workSize)


cdef cufftResult cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftGetSize2d(handle, nx, ny, type, workSize)


cdef cufftResult cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftGetSize3d(handle, nx, ny, nz, type, workSize)


cdef cufftResult cufftGetSizeMany(cufftHandle handle, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workArea)


cdef cufftResult cufftGetSize(cufftHandle handle, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftGetSize(handle, workSize)


cdef cufftResult cufftSetWorkArea(cufftHandle plan, void* workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftSetWorkArea(plan, workArea)


cdef cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftSetAutoAllocation(plan, autoAllocate)


cdef cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftExecC2C(plan, idata, odata, direction)


cdef cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftExecR2C(plan, idata, odata)


cdef cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftExecC2R(plan, idata, odata)


cdef cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftExecZ2Z(plan, idata, odata, direction)


cdef cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal* idata, cufftDoubleComplex* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftExecD2Z(plan, idata, odata)


cdef cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleReal* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftExecZ2D(plan, idata, odata)


cdef cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftSetStream(plan, stream)


cdef cufftResult cufftDestroy(cufftHandle plan) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftDestroy(plan)


cdef cufftResult cufftGetVersion(int* version) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftGetVersion(version)


cdef cufftResult cufftGetProperty(libraryPropertyType type, int* value) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftGetProperty(type, value)


cdef cufftResult cufftSetPlanPropertyInt64(cufftHandle plan, cufftProperty property, const long long int inputValueInt) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftSetPlanPropertyInt64(plan, property, inputValueInt)


cdef cufftResult cufftGetPlanPropertyInt64(cufftHandle plan, cufftProperty property, long long int* returnPtrValue) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftGetPlanPropertyInt64(plan, property, returnPtrValue)


cdef cufftResult cufftResetPlanProperty(cufftHandle plan, cufftProperty property) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftResetPlanProperty(plan, property)


cdef cufftResult cufftXtSetGPUs(cufftHandle handle, int nGPUs, int* whichGPUs) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtSetGPUs(handle, nGPUs, whichGPUs)


cdef cufftResult cufftXtMalloc(cufftHandle plan, cudaLibXtDesc** descriptor, cufftXtSubFormat format) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtMalloc(plan, descriptor, format)


cdef cufftResult cufftXtMemcpy(cufftHandle plan, void* dstPointer, void* srcPointer, cufftXtCopyType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtMemcpy(plan, dstPointer, srcPointer, type)


cdef cufftResult cufftXtFree(cudaLibXtDesc* descriptor) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtFree(descriptor)


cdef cufftResult cufftXtSetWorkArea(cufftHandle plan, void** workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtSetWorkArea(plan, workArea)


cdef cufftResult cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtExecDescriptorC2C(plan, input, output, direction)


cdef cufftResult cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtExecDescriptorR2C(plan, input, output)


cdef cufftResult cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtExecDescriptorC2R(plan, input, output)


cdef cufftResult cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtExecDescriptorZ2Z(plan, input, output, direction)


cdef cufftResult cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtExecDescriptorD2Z(plan, input, output)


cdef cufftResult cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtExecDescriptorZ2D(plan, input, output)


cdef cufftResult cufftXtQueryPlan(cufftHandle plan, void* queryStruct, cufftXtQueryType queryType) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtQueryPlan(plan, queryStruct, queryType)


cdef cufftResult cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtClearCallback(plan, cbType)


cdef cufftResult cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType, size_t sharedSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtSetCallbackSharedSize(plan, cbType, sharedSize)


cdef cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtMakePlanMany(plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch, workSize, executiontype)


cdef cufftResult cufftXtGetSizeMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtGetSizeMany(plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch, workSize, executiontype)


cdef cufftResult cufftXtExec(cufftHandle plan, void* input, void* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtExec(plan, input, output, direction)


cdef cufftResult cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtExecDescriptor(plan, input, output, direction)


cdef cufftResult cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtSetWorkAreaPolicy(plan, policy, workSize)


cdef cufftResult cufftMpAttachComm(cufftHandle plan, cufftMpCommType comm_type, void* comm_handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMpAttachComm(plan, comm_type, comm_handle)


cdef cufftResult cufftXtSetDistribution(cufftHandle plan, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_input, const long long int* strides_output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtSetDistribution(plan, rank, lower_input, upper_input, lower_output, upper_output, strides_input, strides_output)


cdef cufftResult cufftXtSetSubformatDefault(cufftHandle plan, cufftXtSubFormat subformat_forward, cufftXtSubFormat subformat_inverse) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftXtSetSubformatDefault(plan, subformat_forward, subformat_inverse)


cdef cufftResult cufftMpCreateReshape(cufftReshapeHandle* handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMpCreateReshape(handle)


cdef cufftResult cufftMpAttachReshapeComm(cufftReshapeHandle handle, cufftMpCommType comm_type, void* comm_handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMpAttachReshapeComm(handle, comm_type, comm_handle)


cdef cufftResult cufftMpGetReshapeSize(cufftReshapeHandle handle, size_t* workspace_size) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMpGetReshapeSize(handle, workspace_size)


cdef cufftResult __cufftMpMakeReshape_11_2(cufftReshapeHandle handle, size_t element_size, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_input, const long long int* strides_output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp.___cufftMpMakeReshape_11_2(handle, element_size, rank, lower_input, upper_input, lower_output, upper_output, strides_input, strides_output)


cdef cufftResult cufftMpExecReshapeAsync(cufftReshapeHandle handle, void* data_out, const void* data_in, void* workspace, cudaStream_t stream) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMpExecReshapeAsync(handle, data_out, data_in, workspace, stream)


cdef cufftResult cufftMpDestroyReshape(cufftReshapeHandle handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp._cufftMpDestroyReshape(handle)


cdef cufftResult __cufftMpMakeReshape_11_4(cufftReshapeHandle handle, size_t element_size, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* strides_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_output, void* comm_handle, cufftMpCommType comm_type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil:
    return _cufftMp.___cufftMpMakeReshape_11_4(handle, element_size, rank, lower_input, upper_input, strides_input, lower_output, upper_output, strides_output, comm_handle, comm_type)
