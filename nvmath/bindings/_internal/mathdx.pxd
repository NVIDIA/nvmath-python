# This code was automatically generated with version 0.2.1. Do not modify it directly.

from ..cymathdx cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef commondxStatusType _commondxCreateCode(commondxCode* code) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _commondxSetCodeOptionInt64(commondxCode code, commondxOption option, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _commondxSetCodeOptionStr(commondxCode code, commondxOption option, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _commondxGetCodeOptionInt64(commondxCode code, commondxOption option, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _commondxGetCodeOptionsInt64s(commondxCode code, commondxOption option, size_t size, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _commondxGetCodeLTOIRSize(commondxCode code, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _commondxGetCodeLTOIR(commondxCode code, size_t size, void* out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _commondxGetCodeNumLTOIRs(commondxCode code, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _commondxGetCodeLTOIRSizes(commondxCode code, size_t size, size_t* out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _commondxGetCodeLTOIRs(commondxCode code, size_t size, void** out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _commondxDestroyCode(commondxCode code) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef const char* _commondxStatusToStr(commondxStatusType status) except?NULL nogil
cdef commondxStatusType _mathdxGetVersion(int* version) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _mathdxGetVersionEx(int* major, int* minor, int* patch) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxCreateDescriptor(cublasdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxSetOptionStr(cublasdxDescriptor handle, commondxOption option, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxSetOperatorInt64(cublasdxDescriptor handle, cublasdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxSetOperatorInt64s(cublasdxDescriptor handle, cublasdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxBindTensor(cublasdxDescriptor handle, cublasdxTensorType tensor_type, cublasdxTensor* tensor) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxSetTensorOptionInt64(cublasdxTensor tensor, cublasdxTensorOption option, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxFinalizeTensors(cublasdxDescriptor handle, size_t count, const cublasdxTensor* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetTensorTraitInt64(cublasdxTensor tensor, cublasdxTensorTrait trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetTensorTraitStrSize(cublasdxTensor tensor, cublasdxTensorTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetTensorTraitStr(cublasdxTensor tensor, cublasdxTensorTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxBindDeviceFunction(cublasdxDescriptor handle, cublasdxDeviceFunctionType device_function_type, size_t count, const cublasdxTensor* array, cublasdxDeviceFunction* device_function) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxFinalizeDeviceFunctions(commondxCode code, size_t count, const cublasdxDeviceFunction* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetDeviceFunctionTraitStrSize(cublasdxDeviceFunction device_function, cublasdxDeviceFunctionTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetDeviceFunctionTraitStr(cublasdxDeviceFunction device_function, cublasdxDeviceFunctionTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetLTOIRSize(cublasdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetLTOIR(cublasdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetTraitStrSize(cublasdxDescriptor handle, cublasdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetTraitStr(cublasdxDescriptor handle, cublasdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetTraitInt64(cublasdxDescriptor handle, cublasdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxGetTraitInt64s(cublasdxDescriptor handle, cublasdxTraitType trait, size_t count, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef const char* _cublasdxOperatorTypeToStr(cublasdxOperatorType op) except?NULL nogil
cdef const char* _cublasdxTraitTypeToStr(cublasdxTraitType trait) except?NULL nogil
cdef commondxStatusType _cublasdxFinalizeCode(commondxCode code, cublasdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cublasdxDestroyDescriptor(cublasdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxCreateDescriptor(cufftdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxSetOptionStr(cufftdxDescriptor handle, commondxOption opt, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxGetKnobInt64Size(cufftdxDescriptor handle, size_t num_knobs, cufftdxKnobType* knobs_ptr, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxGetKnobInt64s(cufftdxDescriptor handle, size_t num_knobs, cufftdxKnobType* knobs_ptr, size_t size, long long int* values) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxSetOperatorInt64(cufftdxDescriptor handle, cufftdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxSetOperatorInt64s(cufftdxDescriptor handle, cufftdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxGetLTOIRSize(cufftdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxGetLTOIR(cufftdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxGetTraitStrSize(cufftdxDescriptor handle, cufftdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxGetTraitStr(cufftdxDescriptor handle, cufftdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxGetTraitInt64(cufftdxDescriptor handle, cufftdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxGetTraitInt64s(cufftdxDescriptor handle, cufftdxTraitType trait, size_t count, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxGetTraitCommondxDataType(cufftdxDescriptor handle, cufftdxTraitType trait, commondxValueType* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxFinalizeCode(commondxCode code, cufftdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cufftdxDestroyDescriptor(cufftdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef const char* _cufftdxOperatorTypeToStr(cufftdxOperatorType op) except?NULL nogil
cdef const char* _cufftdxTraitTypeToStr(cufftdxTraitType op) except?NULL nogil
cdef commondxStatusType _cusolverdxCreateDescriptor(cusolverdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxSetOptionStr(cusolverdxDescriptor handle, commondxOption opt, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxSetOperatorInt64(cusolverdxDescriptor handle, cusolverdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxSetOperatorInt64s(cusolverdxDescriptor handle, cusolverdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxGetLTOIRSize(cusolverdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxGetLTOIR(cusolverdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxGetUniversalFATBINSize(cusolverdxDescriptor handle, size_t* fatbin_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxGetUniversalFATBIN(cusolverdxDescriptor handle, size_t fatbin_size, void* fatbin) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxGetTraitStrSize(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxGetTraitStr(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxGetTraitInt64(cusolverdxDescriptor handle, cusolverdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxFinalizeCode(commondxCode code, cusolverdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType _cusolverdxDestroyDescriptor(cusolverdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef const char* _cusolverdxOperatorTypeToStr(cusolverdxOperatorType op) except?NULL nogil
cdef const char* _cusolverdxTraitTypeToStr(cusolverdxTraitType trait) except?NULL nogil
