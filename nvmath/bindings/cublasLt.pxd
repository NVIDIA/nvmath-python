# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.8.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycublasLt cimport *


###############################################################################
# Types
###############################################################################

ctypedef cublasLtHandle_t Handle
ctypedef cublasLtMatrixLayout_t MatrixLayout
ctypedef cublasLtMatmulDesc_t MatmulDesc
ctypedef cublasLtMatrixTransformDesc_t MatrixTransformDesc
ctypedef cublasLtMatmulPreference_t MatmulPreference
ctypedef cublasLtLoggerCallback_t LoggerCallback

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType
ctypedef libraryPropertyType_t LibraryPropertyType


###############################################################################
# Enum
###############################################################################

ctypedef cublasLtMatmulTile_t _MatmulTile
ctypedef cublasLtMatmulStages_t _MatmulStages
ctypedef cublasLtPointerMode_t _PointerMode
ctypedef cublasLtPointerModeMask_t _PointerModeMask
ctypedef cublasLtOrder_t _Order
ctypedef cublasLtMatrixLayoutAttribute_t _MatrixLayoutAttribute
ctypedef cublasLtMatmulDescAttributes_t _MatmulDescAttribute
ctypedef cublasLtMatrixTransformDescAttributes_t _MatrixTransformDescAttribute
ctypedef cublasLtReductionScheme_t _ReductionScheme
ctypedef cublasLtEpilogue_t _Epilogue
ctypedef cublasLtMatmulSearch_t _MatmulSearch
ctypedef cublasLtMatmulPreferenceAttributes_t _MatmulPreferenceAttribute
ctypedef cublasLtMatmulAlgoCapAttributes_t _MatmulAlgoCapAttribute
ctypedef cublasLtMatmulAlgoConfigAttributes_t _MatmulAlgoConfigAttribute
ctypedef cublasLtClusterShape_t _ClusterShape
ctypedef cublasLtMatmulInnerShape_t _MatmulInnerShape
ctypedef cublasLtMatmulMatrixScale_t _MatmulMatrixScale


###############################################################################
# Functions
###############################################################################

cpdef intptr_t create() except? 0
cpdef destroy(intptr_t light_handle)
cpdef size_t get_version() except? 0
cpdef size_t get_cudart_version() except? 0
cpdef int get_property(int type) except? -1
cpdef matmul(intptr_t light_handle, intptr_t compute_desc, intptr_t alpha, intptr_t a, intptr_t adesc, intptr_t b, intptr_t bdesc, intptr_t beta, intptr_t c, intptr_t cdesc, intptr_t d, intptr_t ddesc, intptr_t algo, intptr_t workspace, size_t workspace_size_in_bytes, intptr_t stream)
cpdef matrix_transform(intptr_t light_handle, intptr_t transform_desc, intptr_t alpha, intptr_t a, intptr_t adesc, intptr_t beta, intptr_t b, intptr_t bdesc, intptr_t c, intptr_t cdesc, intptr_t stream)
cpdef intptr_t matrix_layout_create(int type, uint64_t rows, uint64_t cols, int64_t ld) except? 0
cpdef matrix_layout_destroy(intptr_t mat_layout)
cpdef get_matrix_layout_attribute_dtype(int attr)
cpdef matrix_layout_set_attribute(intptr_t mat_layout, int attr, intptr_t buf, size_t size_in_bytes)
cpdef matrix_layout_get_attribute(intptr_t mat_layout, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written)
cpdef intptr_t matmul_desc_create(int compute_type, int scale_type) except? 0
cpdef matmul_desc_destroy(intptr_t matmul_desc)
cpdef get_matmul_desc_attribute_dtype(int attr)
cpdef matmul_desc_set_attribute(intptr_t matmul_desc, int attr, intptr_t buf, size_t size_in_bytes)
cpdef matmul_desc_get_attribute(intptr_t matmul_desc, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written)
cpdef intptr_t matrix_transform_desc_create(int scale_type) except? 0
cpdef matrix_transform_desc_destroy(intptr_t transform_desc)
cpdef get_matrix_transform_desc_attribute_dtype(int attr)
cpdef matrix_transform_desc_set_attribute(intptr_t transform_desc, int attr, intptr_t buf, size_t size_in_bytes)
cpdef matrix_transform_desc_get_attribute(intptr_t transform_desc, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written)
cpdef intptr_t matmul_preference_create() except? 0
cpdef matmul_preference_destroy(intptr_t pref)
cpdef get_matmul_preference_attribute_dtype(int attr)
cpdef matmul_preference_set_attribute(intptr_t pref, int attr, intptr_t buf, size_t size_in_bytes)
cpdef matmul_preference_get_attribute(intptr_t pref, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written)
cpdef matmul_algo_get_heuristic(intptr_t light_handle, intptr_t operation_desc, intptr_t adesc, intptr_t bdesc, intptr_t cdesc, intptr_t ddesc, intptr_t preference, int requested_algo_count, intptr_t heuristic_results_array, intptr_t return_algo_count)
cpdef matmul_algo_init(intptr_t light_handle, int compute_type, int scale_type, int atype, int btype, int ctype, int dtype, int algo_id, intptr_t algo)
cpdef matmul_algo_check(intptr_t light_handle, intptr_t operation_desc, intptr_t adesc, intptr_t bdesc, intptr_t cdesc, intptr_t ddesc, intptr_t algo, intptr_t result)
cpdef get_matmul_algo_cap_attribute_dtype(int attr)
cpdef matmul_algo_cap_get_attribute(intptr_t algo, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written)
cpdef get_matmul_algo_config_attribute_dtype(int attr)
cpdef matmul_algo_config_set_attribute(intptr_t algo, int attr, intptr_t buf, size_t size_in_bytes)
cpdef matmul_algo_config_get_attribute(intptr_t algo, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written)
cpdef logger_open_file(log_file)
cpdef logger_set_level(int level)
cpdef logger_set_mask(int mask)
cpdef logger_force_disable()
cpdef str get_status_name(int status)
cpdef str get_status_string(int status)
cpdef size_t heuristics_cache_get_capacity() except? 0
cpdef heuristics_cache_set_capacity(size_t capacity)
cpdef disable_cpu_instructions_set_mask(unsigned mask)
cpdef tuple matmul_algo_get_ids(intptr_t light_handle, cublasComputeType_t compute_type, size_t scale_type, size_t atype, size_t btype, size_t ctype, size_t dtype, int requested_algo_count)
