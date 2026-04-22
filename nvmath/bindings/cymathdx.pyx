# This code was automatically generated across versions from 0.3.1 to 0.3.2, generator version 0.3.1.dev1418+g63712a33a.d20260318. Do not modify it directly.

from ._internal cimport mathdx as _mathdx


###############################################################################
# Wrapper functions
###############################################################################

cdef commondxStatusType commondxCreateCode(commondxCode* code) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxCreateCode(code)


cdef commondxStatusType commondxSetCodeOptionInt64(commondxCode code, commondxOption option, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxSetCodeOptionInt64(code, option, value)


cdef commondxStatusType commondxSetCodeOptionInt64s(commondxCode code, commondxOption option, size_t count, long long int* values) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxSetCodeOptionInt64s(code, option, count, values)


cdef commondxStatusType commondxSetCodeOptionStr(commondxCode code, commondxOption option, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxSetCodeOptionStr(code, option, value)


cdef commondxStatusType commondxGetCodeOptionInt64(commondxCode code, commondxOption option, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetCodeOptionInt64(code, option, value)


cdef commondxStatusType commondxGetCodeOptionsInt64s(commondxCode code, commondxOption option, size_t size, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetCodeOptionsInt64s(code, option, size, array)


cdef commondxStatusType commondxGetCodeLTOIRSize(commondxCode code, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetCodeLTOIRSize(code, size)


cdef commondxStatusType commondxGetCodeLTOIR(commondxCode code, size_t size, void* out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetCodeLTOIR(code, size, out)


cdef commondxStatusType commondxGetCodeNumLTOIRs(commondxCode code, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetCodeNumLTOIRs(code, size)


cdef commondxStatusType commondxGetCodeLTOIRSizes(commondxCode code, size_t size, size_t* out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetCodeLTOIRSizes(code, size, out)


cdef commondxStatusType commondxGetCodeLTOIRs(commondxCode code, size_t size, void** out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetCodeLTOIRs(code, size, out)


cdef commondxStatusType commondxDestroyCode(commondxCode code) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxDestroyCode(code)


cdef const char* commondxStatusToStr(commondxStatusType status) except?NULL nogil:
    return _mathdx._commondxStatusToStr(status)


cdef commondxStatusType commondxGetLastErrorStrSize(size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetLastErrorStrSize(size)


cdef commondxStatusType commondxGetLastErrorStr(commondxStatusType* code, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetLastErrorStr(code, size, value)


cdef commondxStatusType mathdxGetVersion(int* version) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._mathdxGetVersion(version)


cdef commondxStatusType mathdxGetVersionEx(int* major, int* minor, int* patch) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._mathdxGetVersionEx(major, minor, patch)


cdef commondxStatusType cublasdxCreateDescriptor(cublasdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxCreateDescriptor(handle)


cdef commondxStatusType cublasdxSetOptionStr(cublasdxDescriptor handle, commondxOption option, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxSetOptionStr(handle, option, value)


cdef commondxStatusType cublasdxSetOperatorInt64(cublasdxDescriptor handle, cublasdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxSetOperatorInt64(handle, op, value)


cdef commondxStatusType cublasdxSetOperatorInt64s(cublasdxDescriptor handle, cublasdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxSetOperatorInt64s(handle, op, count, array)


cdef commondxStatusType cublasdxCreateTensor(cublasdxDescriptor handle, cublasdxTensorType tensor_type, cublasdxTensor* tensor) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxCreateTensor(handle, tensor_type, tensor)


cdef commondxStatusType cublasdxCreateTensorStrided(cublasdxMemorySpace memory_space, commondxValueType value_type, void* ptr, long long int rank, long long int* shape, long long int* stride, cublasdxTensor* tensor) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxCreateTensorStrided(memory_space, value_type, ptr, rank, shape, stride, tensor)


cdef commondxStatusType cublasdxMakeTensorLike(cublasdxTensor input, commondxValueType value_type, cublasdxTensor* output) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxMakeTensorLike(input, value_type, output)


cdef commondxStatusType cublasdxDestroyTensor(cublasdxTensor tensor) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxDestroyTensor(tensor)


cdef commondxStatusType cublasdxDestroyPipeline(cublasdxPipeline pipeline) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxDestroyPipeline(pipeline)


cdef commondxStatusType cublasdxCreateDevicePipeline(cublasdxDescriptor handle, cublasdxDevicePipelineType device_pipeline_type, long long int pipeline_depth, cublasdxBlockSizeStrategy block_size_strategy, cublasdxTensor tensor_a, cublasdxTensor tensor_b, cublasdxPipeline* device_pipeline) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxCreateDevicePipeline(handle, device_pipeline_type, pipeline_depth, block_size_strategy, tensor_a, tensor_b, device_pipeline)


cdef commondxStatusType cublasdxCreateTilePipeline(cublasdxDescriptor handle, cublasdxTilePipelineType tile_pipeline_type, cublasdxPipeline device_pipeline, cublasdxPipeline* tile_pipeline) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxCreateTilePipeline(handle, tile_pipeline_type, device_pipeline, tile_pipeline)


cdef commondxStatusType cublasdxSetTensorOptionInt64(cublasdxTensor tensor, cublasdxTensorOption option, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxSetTensorOptionInt64(tensor, option, value)


cdef commondxStatusType cublasdxFinalizeTensors(size_t count, const cublasdxTensor* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxFinalizeTensors(count, array)


cdef commondxStatusType cublasdxFinalizePipelines(size_t count, const cublasdxPipeline* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxFinalizePipelines(count, array)


cdef commondxStatusType cublasdxFinalize(size_t countTensors, const cublasdxTensor* tensors, size_t countPipelines, const cublasdxPipeline* pipelines) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxFinalize(countTensors, tensors, countPipelines, pipelines)


cdef commondxStatusType cublasdxGetTensorTraitInt64(cublasdxTensor tensor, cublasdxTensorTrait trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetTensorTraitInt64(tensor, trait, value)


cdef commondxStatusType cublasdxGetPipelineTraitInt64(cublasdxPipeline pipeline, cublasdxPipelineTrait trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetPipelineTraitInt64(pipeline, trait, value)


cdef commondxStatusType cublasdxGetPipelineTraitInt64s(cublasdxPipeline pipeline, cublasdxPipelineTrait trait, size_t count, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetPipelineTraitInt64s(pipeline, trait, count, array)


cdef commondxStatusType cublasdxGetTensorTraitStrSize(cublasdxTensor tensor, cublasdxTensorTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetTensorTraitStrSize(tensor, trait, size)


cdef commondxStatusType cublasdxGetPipelineTraitStrSize(cublasdxPipeline pipeline, cublasdxPipelineTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetPipelineTraitStrSize(pipeline, trait, size)


cdef commondxStatusType cublasdxGetTensorTraitStr(cublasdxTensor tensor, cublasdxTensorTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetTensorTraitStr(tensor, trait, size, value)


cdef commondxStatusType cublasdxGetPipelineTraitStr(cublasdxPipeline pipeline, cublasdxPipelineTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetPipelineTraitStr(pipeline, trait, size, value)


cdef commondxStatusType cublasdxCreateDeviceFunction(cublasdxDescriptor handle, cublasdxDeviceFunctionType device_function_type, size_t count, const cublasdxTensor* array, cublasdxDeviceFunction* device_function) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxCreateDeviceFunction(handle, device_function_type, count, array, device_function)


cdef commondxStatusType cublasdxDestroyDeviceFunction(cublasdxDeviceFunction device_function) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxDestroyDeviceFunction(device_function)


cdef commondxStatusType cublasdxCreateDeviceFunctionWithPipelines(cublasdxDescriptor handle, cublasdxDeviceFunctionType device_function_type, size_t tensor_count, const cublasdxTensor* tensors, size_t pipeline_count, const cublasdxPipeline* pipelines, cublasdxDeviceFunction* device_function) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxCreateDeviceFunctionWithPipelines(handle, device_function_type, tensor_count, tensors, pipeline_count, pipelines, device_function)


cdef commondxStatusType cublasdxFinalizeDeviceFunctions(commondxCode code, size_t count, const cublasdxDeviceFunction* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxFinalizeDeviceFunctions(code, count, array)


cdef commondxStatusType cublasdxGetDeviceFunctionTraitStrSize(cublasdxDeviceFunction device_function, cublasdxDeviceFunctionTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetDeviceFunctionTraitStrSize(device_function, trait, size)


cdef commondxStatusType cublasdxGetDeviceFunctionTraitStr(cublasdxDeviceFunction device_function, cublasdxDeviceFunctionTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetDeviceFunctionTraitStr(device_function, trait, size, value)


cdef commondxStatusType cublasdxGetLTOIRSize(cublasdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetLTOIRSize(handle, lto_size)


cdef commondxStatusType cublasdxGetLTOIR(cublasdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetLTOIR(handle, size, lto)


cdef commondxStatusType cublasdxGetTraitStrSize(cublasdxDescriptor handle, cublasdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetTraitStrSize(handle, trait, size)


cdef commondxStatusType cublasdxGetTraitStr(cublasdxDescriptor handle, cublasdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetTraitStr(handle, trait, size, value)


cdef commondxStatusType cublasdxGetTraitInt64(cublasdxDescriptor handle, cublasdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetTraitInt64(handle, trait, value)


cdef commondxStatusType cublasdxGetTraitInt64s(cublasdxDescriptor handle, cublasdxTraitType trait, size_t count, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxGetTraitInt64s(handle, trait, count, array)


cdef const char* cublasdxOperatorTypeToStr(cublasdxOperatorType op) except?NULL nogil:
    return _mathdx._cublasdxOperatorTypeToStr(op)


cdef const char* cublasdxTraitTypeToStr(cublasdxTraitType trait) except?NULL nogil:
    return _mathdx._cublasdxTraitTypeToStr(trait)


cdef commondxStatusType cublasdxFinalizeCode(commondxCode code, cublasdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxFinalizeCode(code, handle)


cdef commondxStatusType cublasdxDestroyDescriptor(cublasdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cublasdxDestroyDescriptor(handle)


cdef commondxStatusType cufftdxCreateDescriptor(cufftdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxCreateDescriptor(handle)


cdef commondxStatusType cufftdxSetOptionStr(cufftdxDescriptor handle, commondxOption opt, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxSetOptionStr(handle, opt, value)


cdef commondxStatusType cufftdxGetKnobInt64Size(cufftdxDescriptor handle, size_t num_knobs, cufftdxKnobType* knobs_ptr, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxGetKnobInt64Size(handle, num_knobs, knobs_ptr, size)


cdef commondxStatusType cufftdxGetKnobInt64s(cufftdxDescriptor handle, size_t num_knobs, cufftdxKnobType* knobs_ptr, size_t size, long long int* values) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxGetKnobInt64s(handle, num_knobs, knobs_ptr, size, values)


cdef commondxStatusType cufftdxSetOperatorInt64(cufftdxDescriptor handle, cufftdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxSetOperatorInt64(handle, op, value)


cdef commondxStatusType cufftdxSetOperatorInt64s(cufftdxDescriptor handle, cufftdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxSetOperatorInt64s(handle, op, count, array)


cdef commondxStatusType cufftdxGetLTOIRSize(cufftdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxGetLTOIRSize(handle, lto_size)


cdef commondxStatusType cufftdxGetLTOIR(cufftdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxGetLTOIR(handle, size, lto)


cdef commondxStatusType cufftdxGetTraitStrSize(cufftdxDescriptor handle, cufftdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxGetTraitStrSize(handle, trait, size)


cdef commondxStatusType cufftdxGetTraitStr(cufftdxDescriptor handle, cufftdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxGetTraitStr(handle, trait, size, value)


cdef commondxStatusType cufftdxGetTraitInt64(cufftdxDescriptor handle, cufftdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxGetTraitInt64(handle, trait, value)


cdef commondxStatusType cufftdxGetTraitInt64s(cufftdxDescriptor handle, cufftdxTraitType trait, size_t count, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxGetTraitInt64s(handle, trait, count, array)


cdef commondxStatusType cufftdxGetTraitCommondxDataType(cufftdxDescriptor handle, cufftdxTraitType trait, commondxValueType* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxGetTraitCommondxDataType(handle, trait, value)


cdef commondxStatusType cufftdxFinalizeCode(commondxCode code, cufftdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxFinalizeCode(code, handle)


cdef commondxStatusType cufftdxDestroyDescriptor(cufftdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cufftdxDestroyDescriptor(handle)


cdef const char* cufftdxOperatorTypeToStr(cufftdxOperatorType op) except?NULL nogil:
    return _mathdx._cufftdxOperatorTypeToStr(op)


cdef const char* cufftdxTraitTypeToStr(cufftdxTraitType op) except?NULL nogil:
    return _mathdx._cufftdxTraitTypeToStr(op)


cdef commondxStatusType cusolverdxCreateDescriptor(cusolverdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxCreateDescriptor(handle)


cdef commondxStatusType cusolverdxSetOptionStr(cusolverdxDescriptor handle, commondxOption opt, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxSetOptionStr(handle, opt, value)


cdef commondxStatusType cusolverdxSetOperatorInt64(cusolverdxDescriptor handle, cusolverdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxSetOperatorInt64(handle, op, value)


cdef commondxStatusType cusolverdxSetOperatorInt64s(cusolverdxDescriptor handle, cusolverdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxSetOperatorInt64s(handle, op, count, array)


cdef commondxStatusType cusolverdxGetLTOIRSize(cusolverdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxGetLTOIRSize(handle, lto_size)


cdef commondxStatusType cusolverdxGetLTOIR(cusolverdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxGetLTOIR(handle, size, lto)


cdef commondxStatusType cusolverdxGetUniversalFATBINSize(cusolverdxDescriptor handle, size_t* fatbin_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxGetUniversalFATBINSize(handle, fatbin_size)


cdef commondxStatusType cusolverdxGetUniversalFATBIN(cusolverdxDescriptor handle, size_t fatbin_size, void* fatbin) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxGetUniversalFATBIN(handle, fatbin_size, fatbin)


cdef commondxStatusType cusolverdxGetTraitStrSize(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxGetTraitStrSize(handle, trait, size)


cdef commondxStatusType cusolverdxGetTraitStr(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxGetTraitStr(handle, trait, size, value)


cdef commondxStatusType cusolverdxGetTraitInt64(cusolverdxDescriptor handle, cusolverdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxGetTraitInt64(handle, trait, value)


cdef commondxStatusType cusolverdxGetTraitInt64s(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t count, long long int* values) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxGetTraitInt64s(handle, trait, count, values)


cdef commondxStatusType cusolverdxFinalizeCode(commondxCode code, cusolverdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxFinalizeCode(code, handle)


cdef commondxStatusType cusolverdxDestroyDescriptor(cusolverdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxDestroyDescriptor(handle)


cdef const char* cusolverdxOperatorTypeToStr(cusolverdxOperatorType op) except?NULL nogil:
    return _mathdx._cusolverdxOperatorTypeToStr(op)


cdef const char* cusolverdxTraitTypeToStr(cusolverdxTraitType trait) except?NULL nogil:
    return _mathdx._cusolverdxTraitTypeToStr(trait)


cdef commondxStatusType curanddxGetVersion(int* major, int* minor, int* patch) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._curanddxGetVersion(major, minor, patch)


cdef commondxStatusType curanddxCreateDescriptor(curanddxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._curanddxCreateDescriptor(handle)


cdef commondxStatusType curanddxSetOptionStr(curanddxDescriptor handle, commondxOption opt, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._curanddxSetOptionStr(handle, opt, value)


cdef commondxStatusType curanddxSetOperatorInt64(curanddxDescriptor handle, curanddxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._curanddxSetOperatorInt64(handle, op, value)


cdef commondxStatusType curanddxSetOperatorDoubles(curanddxDescriptor handle, curanddxOperatorType op, size_t count, double* values) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._curanddxSetOperatorDoubles(handle, op, count, values)


cdef commondxStatusType curanddxGetTraitStrSize(curanddxDescriptor handle, curanddxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._curanddxGetTraitStrSize(handle, trait, size)


cdef commondxStatusType curanddxGetTraitStr(curanddxDescriptor handle, curanddxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._curanddxGetTraitStr(handle, trait, size, value)


cdef commondxStatusType curanddxGetTraitInt64(curanddxDescriptor handle, curanddxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._curanddxGetTraitInt64(handle, trait, value)


cdef commondxStatusType curanddxFinalizeCode(commondxCode code, curanddxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._curanddxFinalizeCode(code, handle)


cdef commondxStatusType curanddxDestroyDescriptor(curanddxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._curanddxDestroyDescriptor(handle)


cdef const char* curanddxOperatorTypeToStr(curanddxOperatorType op) except?NULL nogil:
    return _mathdx._curanddxOperatorTypeToStr(op)


cdef const char* curanddxDistributionToStr(curanddxDistribution dist) except?NULL nogil:
    return _mathdx._curanddxDistributionToStr(dist)


cdef const char* curanddxGeneratorToStr(curanddxGenerator generator) except?NULL nogil:
    return _mathdx._curanddxGeneratorToStr(generator)


cdef const char* curanddxGenerateMethodToStr(curanddxGenerateMethod generate_method) except?NULL nogil:
    return _mathdx._curanddxGenerateMethodToStr(generate_method)


cdef const char* curanddxNormalMethodToStr(curanddxNormalMethod normal_method) except?NULL nogil:
    return _mathdx._curanddxNormalMethodToStr(normal_method)


cdef const char* curanddxTraitTypeToStr(curanddxTraitType trait) except?NULL nogil:
    return _mathdx._curanddxTraitTypeToStr(trait)


cdef commondxStatusType commondxGetCodePTXSize(commondxCode code, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetCodePTXSize(code, size)


cdef commondxStatusType commondxGetCodePTX(commondxCode code, size_t size, void* out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._commondxGetCodePTX(code, size, out)


cdef commondxStatusType cusolverdxGetTraitCommondxDataTypes(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t count, commondxValueType* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil:
    return _mathdx._cusolverdxGetTraitCommondxDataTypes(handle, trait, count, array)
