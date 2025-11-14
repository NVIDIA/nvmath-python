# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.3.1. Do not modify it directly.

cimport cython

from libc.stdint cimport intptr_t

from .cycutensor cimport *


###############################################################################
# Types
###############################################################################

ctypedef cutensorComputeDescriptor_t ComputeDescriptor
ctypedef cutensorOperationDescriptor_t OperationDescriptor
ctypedef cutensorPlan_t Plan
ctypedef cutensorPlanPreference_t PlanPreference
ctypedef cutensorHandle_t Handle
ctypedef cutensorTensorDescriptor_t TensorDescriptor
ctypedef cutensorBlockSparseTensorDescriptor_t BlockSparseTensorDescriptor
ctypedef cutensorLoggerCallback_t LoggerCallback

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef cutensorOperator_t _Operator
ctypedef cutensorStatus_t _Status
ctypedef cutensorAlgo_t _Algo
ctypedef cutensorWorksizePreference_t _WorksizePreference
ctypedef cutensorOperationDescriptorAttribute_t _OperationDescriptorAttribute
ctypedef cutensorPlanPreferenceAttribute_t _PlanPreferenceAttribute
ctypedef cutensorAutotuneMode_t _AutotuneMode
ctypedef cutensorJitMode_t _JitMode
ctypedef cutensorCacheMode_t _CacheMode
ctypedef cutensorPlanAttribute_t _PlanAttribute


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef handle_resize_plan_cache(intptr_t handle, uint32_t num_entries)
cpdef handle_write_plan_cache_to_file(intptr_t handle, filename)
cpdef uint32_t handle_read_plan_cache_from_file(intptr_t handle, filename) except? -1
cpdef write_kernel_cache_to_file(intptr_t handle, filename)
cpdef read_kernel_cache_from_file(intptr_t handle, filename)
cpdef intptr_t create_tensor_descriptor(intptr_t handle, uint32_t num_modes, extent, stride, int data_type, uint32_t alignment_requirement) except? 0
cpdef destroy_tensor_descriptor(intptr_t desc)
cpdef intptr_t create_elementwise_trinary(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_b, mode_b, int op_b, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, int op_ab, int op_abc, intptr_t desc_compute) except? 0
cpdef elementwise_trinary_execute(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t beta, intptr_t b, intptr_t gamma, intptr_t c, intptr_t d, intptr_t stream)
cpdef intptr_t create_elementwise_binary(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, int op_ac, intptr_t desc_compute) except? 0
cpdef elementwise_binary_execute(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t gamma, intptr_t c, intptr_t d, intptr_t stream)
cpdef intptr_t create_permutation(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_b, mode_b, intptr_t desc_compute) except? 0
cpdef permute(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t b, intptr_t stream)
cpdef intptr_t create_contraction(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_b, mode_b, int op_b, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, intptr_t desc_compute) except? 0
cpdef destroy_operation_descriptor(intptr_t desc)
cpdef get_operation_descriptor_attribute_dtype(int attr)
cpdef operation_descriptor_set_attribute(intptr_t handle, intptr_t desc, int attr, intptr_t buf, size_t size_in_bytes)
cpdef operation_descriptor_get_attribute(intptr_t handle, intptr_t desc, int attr, intptr_t buf, size_t size_in_bytes)
cpdef intptr_t create_plan_preference(intptr_t handle, int algo, int jit_mode) except? 0
cpdef destroy_plan_preference(intptr_t pref)
cpdef get_plan_preference_attribute_dtype(int attr)
cpdef plan_preference_set_attribute(intptr_t handle, intptr_t pref, int attr, intptr_t buf, size_t size_in_bytes)
cpdef get_plan_attribute_dtype(int attr)
cpdef plan_get_attribute(intptr_t handle, intptr_t plan, int attr, intptr_t buf, size_t size_in_bytes)
cpdef uint64_t estimate_workspace_size(intptr_t handle, intptr_t desc, intptr_t plan_pref, int workspace_pref) except? -1
cpdef intptr_t create_plan(intptr_t handle, intptr_t desc, intptr_t pref, uint64_t workspace_size_limit) except? 0
cpdef destroy_plan(intptr_t plan)
cpdef contract(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t b, intptr_t beta, intptr_t c, intptr_t d, intptr_t workspace, uint64_t workspace_size, intptr_t stream)
cpdef intptr_t create_reduction(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, int op_reduce, intptr_t desc_compute) except? 0
cpdef reduce(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t beta, intptr_t c, intptr_t d, intptr_t workspace, uint64_t workspace_size, intptr_t stream)
cpdef intptr_t create_contraction_trinary(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_b, mode_b, int op_b, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, int op_d, intptr_t desc_e, mode_e, intptr_t desc_compute) except? 0
cpdef contract_trinary(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t b, intptr_t c, intptr_t beta, intptr_t d, intptr_t e, intptr_t workspace, uint64_t workspace_size, intptr_t stream)
cpdef intptr_t create_block_sparse_tensor_descriptor(intptr_t handle, uint32_t num_modes, uint64_t num_non_zero_blocks, num_sections_per_mode, extent, non_zero_coordinates, stride, int data_type) except? 0
cpdef destroy_block_sparse_tensor_descriptor(intptr_t desc)
cpdef intptr_t create_block_sparse_contraction(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_b, mode_b, int op_b, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, intptr_t desc_compute) except? 0
cpdef block_sparse_contract(intptr_t handle, intptr_t plan, intptr_t alpha, a, b, intptr_t beta, c, d, intptr_t workspace, uint64_t workspace_size, intptr_t stream)
cpdef str get_error_string(int error)
cpdef size_t get_version() except? 0
cpdef size_t get_cudart_version() except? 0
cpdef logger_set_file(intptr_t file)
cpdef logger_open_file(log_file)
cpdef logger_set_level(int32_t level)
cpdef logger_set_mask(int32_t mask)
cpdef logger_force_disable()
