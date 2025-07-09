# This code was automatically generated with version 0.2.1. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cymathdx cimport *


###############################################################################
# Types
###############################################################################




###############################################################################
# Enum
###############################################################################

ctypedef commondxValueType _CommondxValueType
ctypedef commondxStatusType _CommondxStatusType
ctypedef commondxPrecision _CommondxPrecision
ctypedef commondxOption _CommondxOption
ctypedef commondxExecution _CommondxExecution
ctypedef commondxCodeContainer _CommondxCodeContainer
ctypedef cublasdxApi _CublasdxApi
ctypedef cublasdxType _CublasdxType
ctypedef cublasdxTransposeMode _CublasdxTransposeMode
ctypedef cublasdxArrangement _CublasdxArrangement
ctypedef cublasdxFunction _CublasdxFunction
ctypedef cublasdxOperatorType _CublasdxOperatorType
ctypedef cublasdxTraitType _CublasdxTraitType
ctypedef cublasdxTensorType _CublasdxTensorType
ctypedef cublasdxTensorOption _CublasdxTensorOption
ctypedef cublasdxTensorTrait _CublasdxTensorTrait
ctypedef cublasdxDeviceFunctionTrait _CublasdxDeviceFunctionTrait
ctypedef cublasdxDeviceFunctionOption _CublasdxDeviceFunctionOption
ctypedef cublasdxDeviceFunctionType _CublasdxDeviceFunctionType
ctypedef cufftdxApi _CufftdxApi
ctypedef cufftdxType _CufftdxType
ctypedef cufftdxDirection _CufftdxDirection
ctypedef cufftdxComplexLayout _CufftdxComplexLayout
ctypedef cufftdxRealMode _CufftdxRealMode
ctypedef cufftdxCodeType _CufftdxCodeType
ctypedef cufftdxOperatorType _CufftdxOperatorType
ctypedef cufftdxKnobType _CufftdxKnobType
ctypedef cufftdxTraitType _CufftdxTraitType
ctypedef cusolverdxApi _CusolverdxApi
ctypedef cusolverdxType _CusolverdxType
ctypedef cusolverdxFunction _CusolverdxFunction
ctypedef cusolverdxArrangement _CusolverdxArrangement
ctypedef cusolverdxFillMode _CusolverdxFillMode
ctypedef cusolverdxSide _CusolverdxSide
ctypedef cusolverdxDiag _CusolverdxDiag
ctypedef cusolverdxOperatorType _CusolverdxOperatorType
ctypedef cusolverdxTraitType _CusolverdxTraitType


###############################################################################
# Functions
###############################################################################

cpdef long long int commondx_create_code() except? 0
cpdef commondx_set_code_option_int64(long long int code, int option, long long int value)
cpdef commondx_set_code_option_str(long long int code, int option, value)
cpdef long long int commondx_get_code_option_int64(long long int code, int option) except? 0
cpdef commondx_get_code_options_int64s(long long int code, int option, size_t size, array)
cpdef size_t commondx_get_code_ltoir_size(long long int code) except? 0
cpdef commondx_get_code_ltoir(long long int code, size_t size, out)
cpdef size_t commondx_get_code_num_ltoirs(long long int code) except? 0
cpdef commondx_get_code_ltoir_sizes(long long int code, size_t size, out)
cpdef commondx_get_code_ltoirs(long long int code, size_t size, out)
cpdef commondx_destroy_code(long long int code)
cpdef str commondx_status_to_str(int status)
cpdef int get_version() except? 0
cpdef tuple get_version_ex()
cpdef long long int cublasdx_create_descriptor() except? 0
cpdef cublasdx_set_option_str(long long int handle, int option, value)
cpdef cublasdx_set_operator_int64(long long int handle, int op, long long int value)
cpdef cublasdx_set_operator_int64s(long long int handle, int op, size_t count, array)
cpdef long long int cublasdx_bind_tensor(long long int handle, int tensor_type) except? 0
cpdef cublasdx_set_tensor_option_int64(long long int tensor, int option, long long int value)
cpdef cublasdx_finalize_tensors(long long int handle, size_t count, array)
cpdef long long int cublasdx_get_tensor_trait_int64(long long int tensor, int trait) except? 0
cpdef size_t cublasdx_get_tensor_trait_str_size(long long int tensor, int trait) except? 0
cpdef cublasdx_get_tensor_trait_str(long long int tensor, int trait, size_t size, value)
cpdef long long int cublasdx_bind_device_function(long long int handle, int device_function_type, size_t count, array) except? 0
cpdef cublasdx_finalize_device_functions(long long int code, size_t count, array)
cpdef size_t cublasdx_get_device_function_trait_str_size(long long int device_function, int trait) except? 0
cpdef cublasdx_get_device_function_trait_str(long long int device_function, int trait, size_t size, value)
cpdef size_t cublasdx_get_ltoir_size(long long int handle) except? 0
cpdef cublasdx_get_ltoir(long long int handle, size_t size, lto)
cpdef size_t cublasdx_get_trait_str_size(long long int handle, int trait) except? 0
cpdef cublasdx_get_trait_str(long long int handle, int trait, size_t size, value)
cpdef long long int cublasdx_get_trait_int64(long long int handle, int trait) except? 0
cpdef cublasdx_get_trait_int64s(long long int handle, int trait, size_t count, array)
cpdef str cublasdx_operator_type_to_str(int op)
cpdef str cublasdx_trait_type_to_str(int trait)
cpdef cublasdx_finalize_code(long long int code, long long int handle)
cpdef cublasdx_destroy_descriptor(long long int handle)
cpdef long long int cufftdx_create_descriptor() except? 0
cpdef cufftdx_set_option_str(long long int handle, int opt, value)
cpdef size_t cufftdx_get_knob_int64size(long long int handle, size_t num_knobs, knobs_ptr) except? 0
cpdef cufftdx_get_knob_int64s(long long int handle, size_t num_knobs, knobs_ptr, size_t size, intptr_t values)
cpdef cufftdx_set_operator_int64(long long int handle, int op, long long int value)
cpdef cufftdx_set_operator_int64s(long long int handle, int op, size_t count, array)
cpdef size_t cufftdx_get_ltoir_size(long long int handle) except? 0
cpdef cufftdx_get_ltoir(long long int handle, size_t size, lto)
cpdef size_t cufftdx_get_trait_str_size(long long int handle, int trait) except? 0
cpdef cufftdx_get_trait_str(long long int handle, int trait, size_t size, value)
cpdef long long int cufftdx_get_trait_int64(long long int handle, int trait) except? 0
cpdef cufftdx_get_trait_int64s(long long int handle, int trait, size_t count, array)
cpdef int cufftdx_get_trait_commondx_data_type(long long int handle, int trait) except? -1
cpdef cufftdx_finalize_code(long long int code, long long int handle)
cpdef cufftdx_destroy_descriptor(long long int handle)
cpdef str cufftdx_operator_type_to_str(int op)
cpdef str cufftdx_trait_type_to_str(int op)
cpdef long long int cusolverdx_create_descriptor() except? 0
cpdef cusolverdx_set_option_str(long long int handle, int opt, value)
cpdef cusolverdx_set_operator_int64(long long int handle, int op, long long int value)
cpdef cusolverdx_set_operator_int64s(long long int handle, int op, size_t count, array)
cpdef size_t cusolverdx_get_ltoir_size(long long int handle) except? 0
cpdef cusolverdx_get_ltoir(long long int handle, size_t size, lto)
cpdef size_t cusolverdx_get_universal_fatbin_size(long long int handle) except? 0
cpdef cusolverdx_get_universal_fatbin(long long int handle, size_t fatbin_size, fatbin)
cpdef size_t cusolverdx_get_trait_str_size(long long int handle, int trait) except? 0
cpdef cusolverdx_get_trait_str(long long int handle, int trait, size_t size, value)
cpdef long long int cusolverdx_get_trait_int64(long long int handle, int trait) except? 0
cpdef cusolverdx_finalize_code(long long int code, long long int handle)
cpdef cusolverdx_destroy_descriptor(long long int handle)
cpdef str cusolverdx_operator_type_to_str(int op)
cpdef str cusolverdx_trait_type_to_str(int trait)
