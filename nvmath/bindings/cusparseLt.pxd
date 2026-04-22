# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.7.1 to 0.8.1, generator version 0.3.1.dev1565+g7fa82f8eb. Do not modify it directly.

cimport cython

from libc.stdint cimport intptr_t, int64_t, uint8_t, uint32_t, int32_t

from .cycusparseLt cimport *
from .cycusparseLt cimport cudaDataType, cudaStream_t, libraryPropertyType, cusparseComputeType
from .cycusparse cimport cusparseStatus_t, cusparseOrder_t, cusparseOperation_t
from .cusparse cimport LibraryPropertyType


###############################################################################
# Types
###############################################################################

ctypedef cusparseLtHandle_t Handle
ctypedef cusparseLtMatDescriptor_t MatDescriptor
ctypedef cusparseLtMatmulDescriptor_t MatmulDescriptor
ctypedef cusparseLtMatmulAlgSelection_t MatmulAlgSelection
ctypedef cusparseLtMatmulPlan_t MatmulPlan

ctypedef cudaStream_t Stream
ctypedef cudaDataType DataType


###############################################################################
# Enum
###############################################################################

ctypedef cusparseLtSparsity_t _Sparsity
ctypedef cusparseLtMatDescAttribute_t _MatDescAttribute
ctypedef cusparseComputeType _ComputeType
ctypedef cusparseLtMatmulDescAttribute_t _MatmulDescAttribute
ctypedef cusparseLtMatmulMatrixScale_t _MatmulMatrixScale
ctypedef cusparseLtMatmulAlg_t _MatmulAlg
ctypedef cusparseLtMatmulAlgAttribute_t _MatmulAlgAttribute
ctypedef cusparseLtSplitKMode_t _SplitKMode
ctypedef cusparseLtPruneAlg_t _PruneAlg


###############################################################################
# Functions
###############################################################################

cpdef str get_error_name(cusparseStatus_t status)
cpdef str get_error_string(cusparseStatus_t status)
cpdef _init(intptr_t handle)
cpdef _destroy(intptr_t handle)
cpdef int get_version(intptr_t handle) except? -1
cpdef int get_property(int property_type) except? -1
cpdef _dense_descriptor_init(intptr_t handle, intptr_t mat_descr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, int value_type, int order)
cpdef _structured_descriptor_init(intptr_t handle, intptr_t mat_descr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, int value_type, int order, int sparsity)
cpdef _mat_descriptor_destroy(intptr_t mat_descr)
cpdef get_mat_desc_attribute_dtype(int attr)
cpdef mat_desc_set_attribute(intptr_t handle, intptr_t matmul_descr, int mat_attribute, intptr_t data, size_t data_size)
cpdef mat_desc_get_attribute(intptr_t handle, intptr_t matmul_descr, int mat_attribute, intptr_t data, size_t data_size)
cpdef _matmul_descriptor_init(intptr_t handle, intptr_t matmul_descr, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, intptr_t mat_d, int compute_type)
cpdef get_matmul_desc_attribute_dtype(int attr)
cpdef matmul_desc_set_attribute(intptr_t handle, intptr_t matmul_descr, int matmul_attribute, intptr_t data, size_t data_size)
cpdef matmul_desc_get_attribute(intptr_t handle, intptr_t matmul_descr, int matmul_attribute, intptr_t data, size_t data_size)
cpdef _matmul_alg_selection_init(intptr_t handle, intptr_t alg_selection, intptr_t matmul_descr, int alg)
cpdef get_matmul_alg_attribute_dtype(int attr)
cpdef matmul_alg_set_attribute(intptr_t handle, intptr_t alg_selection, int attribute, intptr_t data, size_t data_size)
cpdef matmul_alg_get_attribute(intptr_t handle, intptr_t alg_selection, int attribute, intptr_t data, size_t data_size)
cpdef size_t matmul_get_workspace(intptr_t handle, intptr_t plan) except? -1
cpdef _matmul_plan_init(intptr_t handle, intptr_t plan, intptr_t matmul_descr, intptr_t alg_selection)
cpdef _matmul_plan_destroy(intptr_t plan)
cpdef matmul(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t d_a, intptr_t d_b, intptr_t beta, intptr_t d_c, intptr_t d_d, intptr_t workspace, intptr_t streams, int32_t num_streams)
cpdef matmul_search(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t d_a, intptr_t d_b, intptr_t beta, intptr_t d_c, intptr_t d_d, intptr_t workspace, intptr_t streams, int32_t num_streams)
cpdef sp_mma_prune(intptr_t handle, intptr_t matmul_descr, intptr_t d_in, intptr_t d_out, int prune_alg, intptr_t stream)
cpdef sp_mma_prune_check(intptr_t handle, intptr_t matmul_descr, intptr_t d_in, intptr_t valid, intptr_t stream)
cpdef sp_mma_prune2(intptr_t handle, intptr_t sparse_mat_descr, int is_sparse_a, int op, intptr_t d_in, intptr_t d_out, int prune_alg, intptr_t stream)
cpdef sp_mma_prune_check2(intptr_t handle, intptr_t sparse_mat_descr, int is_sparse_a, int op, intptr_t d_in, intptr_t d_valid, intptr_t stream)
cpdef tuple sp_mma_compressed_size(intptr_t handle, intptr_t plan)
cpdef sp_mma_compress(intptr_t handle, intptr_t plan, intptr_t d_dense, intptr_t d_compressed, intptr_t d_compressed_buffer, intptr_t stream)
cpdef tuple sp_mma_compressed_size2(intptr_t handle, intptr_t sparse_mat_descr)
cpdef sp_mma_compress2(intptr_t handle, intptr_t sparse_mat_descr, int is_sparse_a, int op, intptr_t d_dense, intptr_t d_compressed, intptr_t d_compressed_buffer, intptr_t stream)
cpdef _matmul_alg_selection_destroy(intptr_t alg_selection)

cpdef intptr_t init() except *
cpdef void destroy(intptr_t handle) except *

cpdef intptr_t dense_descriptor_init(intptr_t handle, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, int value_type, int order) except *
cpdef intptr_t structured_descriptor_init(intptr_t handle, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, int value_type, int order, int sparsity) except *
cpdef void mat_descriptor_destroy(intptr_t mat_descr) except *

cpdef intptr_t matmul_descriptor_init(intptr_t handle, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, intptr_t mat_d, int compute_type) except *
cpdef void matmul_descriptor_destroy(intptr_t matmul_descr) except *

cpdef intptr_t matmul_alg_selection_init(intptr_t handle, intptr_t matmul_descr, int alg) except *
cpdef void matmul_alg_selection_destroy(intptr_t alg_selection) except *

cpdef intptr_t matmul_plan_init(intptr_t handle, intptr_t matmul_descr, intptr_t alg_selection) except *
cpdef void matmul_plan_destroy(intptr_t plan) except *
