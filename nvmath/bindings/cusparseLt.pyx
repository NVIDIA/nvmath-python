# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.7.1 to 0.8.1, generator version 0.3.1.dev1565+g7fa82f8eb. Do not modify it directly.

cimport cython  # NOQA

from enum import IntEnum as _IntEnum

import numpy as _numpy

from libc.stdint cimport intptr_t, int64_t, uint32_t

cdef extern from *:
    """
    #ifdef _MSC_VER
    #include <malloc.h>
    static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
        return _aligned_malloc(size, alignment);
    }
    static void aligned_free_wrapper(void* ptr) {
        _aligned_free(ptr);
    }
    #else
    #include <stdlib.h>
    static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
        return aligned_alloc(alignment, size);
    }
    static void aligned_free_wrapper(void* ptr) {
        free(ptr);
    }
    #endif
    """
    void* aligned_alloc_wrapper(size_t alignment, size_t size)
    void aligned_free_wrapper(void* ptr)



###############################################################################
# Enum
###############################################################################

###############################################################################
# Types
###############################################################################



class Sparsity(_IntEnum):
    """
    See `cusparseLtSparsity_t`.
    """
    SPARSITY_50_PERCENT = CUSPARSELT_SPARSITY_50_PERCENT

class MatDescAttribute(_IntEnum):
    """
    See `cusparseLtMatDescAttribute_t`.
    """
    NUM_BATCHES = CUSPARSELT_MAT_NUM_BATCHES
    BATCH_STRIDE = CUSPARSELT_MAT_BATCH_STRIDE

class ComputeType(_IntEnum):
    """
    See `cusparseComputeType`.
    """
    COMPUTE_32I = CUSPARSE_COMPUTE_32I
    COMPUTE_16F = CUSPARSE_COMPUTE_16F
    COMPUTE_32F = CUSPARSE_COMPUTE_32F

class MatmulDescAttribute(_IntEnum):
    """
    See `cusparseLtMatmulDescAttribute_t`.
    """
    ACTIVATION_RELU = CUSPARSELT_MATMUL_ACTIVATION_RELU
    ACTIVATION_RELU_UPPERBOUND = CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND
    ACTIVATION_RELU_THRESHOLD = CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD
    ACTIVATION_GELU = CUSPARSELT_MATMUL_ACTIVATION_GELU
    ACTIVATION_GELU_SCALING = CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING
    ALPHA_VECTOR_SCALING = CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING
    BETA_VECTOR_SCALING = CUSPARSELT_MATMUL_BETA_VECTOR_SCALING
    BIAS_STRIDE = CUSPARSELT_MATMUL_BIAS_STRIDE
    BIAS_POINTER = CUSPARSELT_MATMUL_BIAS_POINTER
    SPARSE_MAT_POINTER = CUSPARSELT_MATMUL_SPARSE_MAT_POINTER
    A_SCALE_MODE = CUSPARSELT_MATMUL_A_SCALE_MODE
    B_SCALE_MODE = CUSPARSELT_MATMUL_B_SCALE_MODE
    C_SCALE_MODE = CUSPARSELT_MATMUL_C_SCALE_MODE
    D_SCALE_MODE = CUSPARSELT_MATMUL_D_SCALE_MODE
    D_OUT_SCALE_MODE = CUSPARSELT_MATMUL_D_OUT_SCALE_MODE
    A_SCALE_POINTER = CUSPARSELT_MATMUL_A_SCALE_POINTER
    B_SCALE_POINTER = CUSPARSELT_MATMUL_B_SCALE_POINTER
    C_SCALE_POINTER = CUSPARSELT_MATMUL_C_SCALE_POINTER
    D_SCALE_POINTER = CUSPARSELT_MATMUL_D_SCALE_POINTER
    D_OUT_SCALE_POINTER = CUSPARSELT_MATMUL_D_OUT_SCALE_POINTER

class MatmulMatrixScale(_IntEnum):
    """
    See `cusparseLtMatmulMatrixScale_t`.
    """
    NONE = CUSPARSELT_MATMUL_SCALE_NONE
    SCALAR_32F = CUSPARSELT_MATMUL_MATRIX_SCALE_SCALAR_32F
    VEC32_UE4M3 = CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3
    VEC64_UE8M0 = CUSPARSELT_MATMUL_MATRIX_SCALE_VEC64_UE8M0

class MatmulAlg(_IntEnum):
    """
    See `cusparseLtMatmulAlg_t`.
    """
    DEFAULT = CUSPARSELT_MATMUL_ALG_DEFAULT

class MatmulAlgAttribute(_IntEnum):
    """
    See `cusparseLtMatmulAlgAttribute_t`.
    """
    CONFIG_ID = CUSPARSELT_MATMUL_ALG_CONFIG_ID
    CONFIG_MAX_ID = CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID
    SEARCH_ITERATIONS = CUSPARSELT_MATMUL_SEARCH_ITERATIONS
    SPLIT_K = CUSPARSELT_MATMUL_SPLIT_K
    SPLIT_K_MODE = CUSPARSELT_MATMUL_SPLIT_K_MODE
    SPLIT_K_BUFFERS = CUSPARSELT_MATMUL_SPLIT_K_BUFFERS

class SplitKMode(_IntEnum):
    """
    See `cusparseLtSplitKMode_t`.
    """
    INVALID_MODE = CUSPARSELT_INVALID_MODE
    ONE_KERNEL = CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL
    TWO_KERNELS = CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS
    HEURISTIC = CUSPARSELT_HEURISTIC
    DATAPARALLEL = CUSPARSELT_DATAPARALLEL
    SPLITK = CUSPARSELT_SPLITK
    STREAMK = CUSPARSELT_STREAMK

class PruneAlg(_IntEnum):
    """
    See `cusparseLtPruneAlg_t`.
    """
    SPMMA_TILE = CUSPARSELT_PRUNE_SPMMA_TILE
    SPMMA_STRIP = CUSPARSELT_PRUNE_SPMMA_STRIP


###############################################################################
# Error handling
###############################################################################

cdef class cuSPARSELtError(Exception):

    def __init__(self, status):
        self.status = status
        cdef str err = f"cuSPARSELtError ({status})"
        super(cuSPARSELtError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuSPARSELtError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef str get_error_name(cusparseStatus_t status):
    """See `cusparseLtGetErrorName`."""
    cdef bytes _output_
    _output_ = cusparseLtGetErrorName(status)
    return _output_.decode()


cpdef str get_error_string(cusparseStatus_t status):
    """See `cusparseLtGetErrorString`."""
    cdef bytes _output_
    _output_ = cusparseLtGetErrorString(status)
    return _output_.decode()


cpdef _init(intptr_t handle):
    """See `cusparseLtInit`."""
    with nogil:
        __status__ = cusparseLtInit(<cusparseLtHandle_t*>handle)
    check_status(__status__)


cpdef _destroy(intptr_t handle):
    """See `cusparseLtDestroy`."""
    with nogil:
        __status__ = cusparseLtDestroy(<const cusparseLtHandle_t*>handle)
    check_status(__status__)


cpdef int get_version(intptr_t handle) except? -1:
    """See `cusparseLtGetVersion`."""
    cdef int version
    with nogil:
        __status__ = cusparseLtGetVersion(<const cusparseLtHandle_t*>handle, &version)
    check_status(__status__)
    return version


cpdef int get_property(int property_type) except? -1:
    """See `cusparseLtGetProperty`."""
    cdef int value
    with nogil:
        __status__ = cusparseLtGetProperty(<LibraryPropertyType>property_type, &value)
    check_status(__status__)
    return value


cpdef _dense_descriptor_init(intptr_t handle, intptr_t mat_descr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, int value_type, int order):
    """See `cusparseLtDenseDescriptorInit`."""
    with nogil:
        __status__ = cusparseLtDenseDescriptorInit(<const cusparseLtHandle_t*>handle, <cusparseLtMatDescriptor_t*>mat_descr, rows, cols, ld, alignment, <DataType>value_type, <cusparseOrder_t>order)
    check_status(__status__)


cpdef _structured_descriptor_init(intptr_t handle, intptr_t mat_descr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, int value_type, int order, int sparsity):
    """See `cusparseLtStructuredDescriptorInit`."""
    with nogil:
        __status__ = cusparseLtStructuredDescriptorInit(<const cusparseLtHandle_t*>handle, <cusparseLtMatDescriptor_t*>mat_descr, rows, cols, ld, alignment, <DataType>value_type, <cusparseOrder_t>order, <_Sparsity>sparsity)
    check_status(__status__)


cpdef _mat_descriptor_destroy(intptr_t mat_descr):
    """See `cusparseLtMatDescriptorDestroy`."""
    with nogil:
        __status__ = cusparseLtMatDescriptorDestroy(<const cusparseLtMatDescriptor_t*>mat_descr)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict mat_desc_attribute_sizes = {
    CUSPARSELT_MAT_NUM_BATCHES: _numpy.int32,
    CUSPARSELT_MAT_BATCH_STRIDE: _numpy.int64,
}

cpdef get_mat_desc_attribute_dtype(int attr):
    """Get the Python data type of the corresponding MatDescAttribute attribute.

    Args:
        attr (MatDescAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`mat_desc_get_attribute`, :func:`mat_desc_set_attribute`.
    """
    return mat_desc_attribute_sizes[attr]

###########################################################################


cpdef mat_desc_set_attribute(intptr_t handle, intptr_t matmul_descr, int mat_attribute, intptr_t data, size_t data_size):
    """See `cusparseLtMatDescSetAttribute`."""
    with nogil:
        __status__ = cusparseLtMatDescSetAttribute(<const cusparseLtHandle_t*>handle, <cusparseLtMatDescriptor_t*>matmul_descr, <_MatDescAttribute>mat_attribute, <const void*>data, data_size)
    check_status(__status__)


cpdef mat_desc_get_attribute(intptr_t handle, intptr_t matmul_descr, int mat_attribute, intptr_t data, size_t data_size):
    """See `cusparseLtMatDescGetAttribute`."""
    with nogil:
        __status__ = cusparseLtMatDescGetAttribute(<const cusparseLtHandle_t*>handle, <const cusparseLtMatDescriptor_t*>matmul_descr, <_MatDescAttribute>mat_attribute, <void*>data, data_size)
    check_status(__status__)


cpdef _matmul_descriptor_init(intptr_t handle, intptr_t matmul_descr, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, intptr_t mat_d, int compute_type):
    """See `cusparseLtMatmulDescriptorInit`."""
    with nogil:
        __status__ = cusparseLtMatmulDescriptorInit(<const cusparseLtHandle_t*>handle, <cusparseLtMatmulDescriptor_t*>matmul_descr, <cusparseOperation_t>op_a, <cusparseOperation_t>op_b, <const cusparseLtMatDescriptor_t*>mat_a, <const cusparseLtMatDescriptor_t*>mat_b, <const cusparseLtMatDescriptor_t*>mat_c, <const cusparseLtMatDescriptor_t*>mat_d, <_ComputeType>compute_type)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict matmul_desc_attribute_sizes = {
    CUSPARSELT_MATMUL_ACTIVATION_RELU: _numpy.int32,
    CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND: _numpy.float32,
    CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD: _numpy.float32,
    CUSPARSELT_MATMUL_ACTIVATION_GELU: _numpy.int32,
    CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING: _numpy.float32,
    CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING: _numpy.int32,
    CUSPARSELT_MATMUL_BETA_VECTOR_SCALING: _numpy.int32,
    CUSPARSELT_MATMUL_BIAS_STRIDE: _numpy.int64,
    CUSPARSELT_MATMUL_BIAS_POINTER: _numpy.intp,
    CUSPARSELT_MATMUL_SPARSE_MAT_POINTER: _numpy.intp,
    CUSPARSELT_MATMUL_A_SCALE_MODE: _numpy.int32,
    CUSPARSELT_MATMUL_B_SCALE_MODE: _numpy.int32,
    CUSPARSELT_MATMUL_C_SCALE_MODE: _numpy.int32,
    CUSPARSELT_MATMUL_D_SCALE_MODE: _numpy.int32,
    CUSPARSELT_MATMUL_D_OUT_SCALE_MODE: _numpy.int32,
    CUSPARSELT_MATMUL_A_SCALE_POINTER: _numpy.intp,
    CUSPARSELT_MATMUL_B_SCALE_POINTER: _numpy.intp,
    CUSPARSELT_MATMUL_C_SCALE_POINTER: _numpy.intp,
    CUSPARSELT_MATMUL_D_SCALE_POINTER: _numpy.intp,
    CUSPARSELT_MATMUL_D_OUT_SCALE_POINTER: _numpy.intp,
}

cpdef get_matmul_desc_attribute_dtype(int attr):
    """Get the Python data type of the corresponding MatmulDescAttribute attribute.

    Args:
        attr (MatmulDescAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`matmul_desc_get_attribute`, :func:`matmul_desc_set_attribute`.
    """
    return matmul_desc_attribute_sizes[attr]

###########################################################################


cpdef matmul_desc_set_attribute(intptr_t handle, intptr_t matmul_descr, int matmul_attribute, intptr_t data, size_t data_size):
    """See `cusparseLtMatmulDescSetAttribute`."""
    with nogil:
        __status__ = cusparseLtMatmulDescSetAttribute(<const cusparseLtHandle_t*>handle, <cusparseLtMatmulDescriptor_t*>matmul_descr, <_MatmulDescAttribute>matmul_attribute, <const void*>data, data_size)
    check_status(__status__)


cpdef matmul_desc_get_attribute(intptr_t handle, intptr_t matmul_descr, int matmul_attribute, intptr_t data, size_t data_size):
    """See `cusparseLtMatmulDescGetAttribute`."""
    with nogil:
        __status__ = cusparseLtMatmulDescGetAttribute(<const cusparseLtHandle_t*>handle, <const cusparseLtMatmulDescriptor_t*>matmul_descr, <_MatmulDescAttribute>matmul_attribute, <void*>data, data_size)
    check_status(__status__)


cpdef _matmul_alg_selection_init(intptr_t handle, intptr_t alg_selection, intptr_t matmul_descr, int alg):
    """See `cusparseLtMatmulAlgSelectionInit`."""
    with nogil:
        __status__ = cusparseLtMatmulAlgSelectionInit(<const cusparseLtHandle_t*>handle, <cusparseLtMatmulAlgSelection_t*>alg_selection, <const cusparseLtMatmulDescriptor_t*>matmul_descr, <_MatmulAlg>alg)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict matmul_alg_attribute_sizes = {
    CUSPARSELT_MATMUL_ALG_CONFIG_ID: _numpy.int32,
    CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID: _numpy.int32,
    CUSPARSELT_MATMUL_SEARCH_ITERATIONS: _numpy.int32,
    CUSPARSELT_MATMUL_SPLIT_K: _numpy.int32,
    CUSPARSELT_MATMUL_SPLIT_K_MODE: _numpy.int32,
    CUSPARSELT_MATMUL_SPLIT_K_BUFFERS: _numpy.int32,
}

cpdef get_matmul_alg_attribute_dtype(int attr):
    """Get the Python data type of the corresponding MatmulAlgAttribute attribute.

    Args:
        attr (MatmulAlgAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`matmul_alg_get_attribute`, :func:`matmul_alg_set_attribute`.
    """
    return matmul_alg_attribute_sizes[attr]

###########################################################################


cpdef matmul_alg_set_attribute(intptr_t handle, intptr_t alg_selection, int attribute, intptr_t data, size_t data_size):
    """See `cusparseLtMatmulAlgSetAttribute`."""
    with nogil:
        __status__ = cusparseLtMatmulAlgSetAttribute(<const cusparseLtHandle_t*>handle, <cusparseLtMatmulAlgSelection_t*>alg_selection, <_MatmulAlgAttribute>attribute, <const void*>data, data_size)
    check_status(__status__)


cpdef matmul_alg_get_attribute(intptr_t handle, intptr_t alg_selection, int attribute, intptr_t data, size_t data_size):
    """See `cusparseLtMatmulAlgGetAttribute`."""
    with nogil:
        __status__ = cusparseLtMatmulAlgGetAttribute(<const cusparseLtHandle_t*>handle, <const cusparseLtMatmulAlgSelection_t*>alg_selection, <_MatmulAlgAttribute>attribute, <void*>data, data_size)
    check_status(__status__)


cpdef size_t matmul_get_workspace(intptr_t handle, intptr_t plan) except? -1:
    """See `cusparseLtMatmulGetWorkspace`."""
    cdef size_t workspace_size
    with nogil:
        __status__ = cusparseLtMatmulGetWorkspace(<const cusparseLtHandle_t*>handle, <const cusparseLtMatmulPlan_t*>plan, &workspace_size)
    check_status(__status__)
    return workspace_size


cpdef _matmul_plan_init(intptr_t handle, intptr_t plan, intptr_t matmul_descr, intptr_t alg_selection):
    """See `cusparseLtMatmulPlanInit`."""
    with nogil:
        __status__ = cusparseLtMatmulPlanInit(<const cusparseLtHandle_t*>handle, <cusparseLtMatmulPlan_t*>plan, <const cusparseLtMatmulDescriptor_t*>matmul_descr, <const cusparseLtMatmulAlgSelection_t*>alg_selection)
    check_status(__status__)


cpdef _matmul_plan_destroy(intptr_t plan):
    """See `cusparseLtMatmulPlanDestroy`."""
    with nogil:
        __status__ = cusparseLtMatmulPlanDestroy(<const cusparseLtMatmulPlan_t*>plan)
    check_status(__status__)


cpdef matmul(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t d_a, intptr_t d_b, intptr_t beta, intptr_t d_c, intptr_t d_d, intptr_t workspace, intptr_t streams, int32_t num_streams):
    """See `cusparseLtMatmul`."""
    with nogil:
        __status__ = cusparseLtMatmul(<const cusparseLtHandle_t*>handle, <const cusparseLtMatmulPlan_t*>plan, <const void*>alpha, <const void*>d_a, <const void*>d_b, <const void*>beta, <const void*>d_c, <void*>d_d, <void*>workspace, <Stream*>streams, num_streams)
    check_status(__status__)


cpdef matmul_search(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t d_a, intptr_t d_b, intptr_t beta, intptr_t d_c, intptr_t d_d, intptr_t workspace, intptr_t streams, int32_t num_streams):
    """See `cusparseLtMatmulSearch`."""
    with nogil:
        __status__ = cusparseLtMatmulSearch(<const cusparseLtHandle_t*>handle, <cusparseLtMatmulPlan_t*>plan, <const void*>alpha, <const void*>d_a, <const void*>d_b, <const void*>beta, <const void*>d_c, <void*>d_d, <void*>workspace, <Stream*>streams, num_streams)
    check_status(__status__)


cpdef sp_mma_prune(intptr_t handle, intptr_t matmul_descr, intptr_t d_in, intptr_t d_out, int prune_alg, intptr_t stream):
    """See `cusparseLtSpMMAPrune`."""
    with nogil:
        __status__ = cusparseLtSpMMAPrune(<const cusparseLtHandle_t*>handle, <const cusparseLtMatmulDescriptor_t*>matmul_descr, <const void*>d_in, <void*>d_out, <_PruneAlg>prune_alg, <Stream>stream)
    check_status(__status__)


cpdef sp_mma_prune_check(intptr_t handle, intptr_t matmul_descr, intptr_t d_in, intptr_t valid, intptr_t stream):
    """See `cusparseLtSpMMAPruneCheck`."""
    with nogil:
        __status__ = cusparseLtSpMMAPruneCheck(<const cusparseLtHandle_t*>handle, <const cusparseLtMatmulDescriptor_t*>matmul_descr, <const void*>d_in, <int*>valid, <Stream>stream)
    check_status(__status__)


cpdef sp_mma_prune2(intptr_t handle, intptr_t sparse_mat_descr, int is_sparse_a, int op, intptr_t d_in, intptr_t d_out, int prune_alg, intptr_t stream):
    """See `cusparseLtSpMMAPrune2`."""
    with nogil:
        __status__ = cusparseLtSpMMAPrune2(<const cusparseLtHandle_t*>handle, <const cusparseLtMatDescriptor_t*>sparse_mat_descr, is_sparse_a, <cusparseOperation_t>op, <const void*>d_in, <void*>d_out, <_PruneAlg>prune_alg, <Stream>stream)
    check_status(__status__)


cpdef sp_mma_prune_check2(intptr_t handle, intptr_t sparse_mat_descr, int is_sparse_a, int op, intptr_t d_in, intptr_t d_valid, intptr_t stream):
    """See `cusparseLtSpMMAPruneCheck2`."""
    with nogil:
        __status__ = cusparseLtSpMMAPruneCheck2(<const cusparseLtHandle_t*>handle, <const cusparseLtMatDescriptor_t*>sparse_mat_descr, is_sparse_a, <cusparseOperation_t>op, <const void*>d_in, <int*>d_valid, <Stream>stream)
    check_status(__status__)


cpdef tuple sp_mma_compressed_size(intptr_t handle, intptr_t plan):
    """See `cusparseLtSpMMACompressedSize`."""
    cdef size_t compressed_size
    cdef size_t compressed_buffer_size
    with nogil:
        __status__ = cusparseLtSpMMACompressedSize(<const cusparseLtHandle_t*>handle, <const cusparseLtMatmulPlan_t*>plan, &compressed_size, &compressed_buffer_size)
    check_status(__status__)
    return (compressed_size, compressed_buffer_size)


cpdef sp_mma_compress(intptr_t handle, intptr_t plan, intptr_t d_dense, intptr_t d_compressed, intptr_t d_compressed_buffer, intptr_t stream):
    """See `cusparseLtSpMMACompress`."""
    with nogil:
        __status__ = cusparseLtSpMMACompress(<const cusparseLtHandle_t*>handle, <const cusparseLtMatmulPlan_t*>plan, <const void*>d_dense, <void*>d_compressed, <void*>d_compressed_buffer, <Stream>stream)
    check_status(__status__)


cpdef tuple sp_mma_compressed_size2(intptr_t handle, intptr_t sparse_mat_descr):
    """See `cusparseLtSpMMACompressedSize2`."""
    cdef size_t compressed_size
    cdef size_t compressed_buffer_size
    with nogil:
        __status__ = cusparseLtSpMMACompressedSize2(<const cusparseLtHandle_t*>handle, <const cusparseLtMatDescriptor_t*>sparse_mat_descr, &compressed_size, &compressed_buffer_size)
    check_status(__status__)
    return (compressed_size, compressed_buffer_size)


cpdef sp_mma_compress2(intptr_t handle, intptr_t sparse_mat_descr, int is_sparse_a, int op, intptr_t d_dense, intptr_t d_compressed, intptr_t d_compressed_buffer, intptr_t stream):
    """See `cusparseLtSpMMACompress2`."""
    with nogil:
        __status__ = cusparseLtSpMMACompress2(<const cusparseLtHandle_t*>handle, <const cusparseLtMatDescriptor_t*>sparse_mat_descr, is_sparse_a, <cusparseOperation_t>op, <const void*>d_dense, <void*>d_compressed, <void*>d_compressed_buffer, <Stream>stream)
    check_status(__status__)


cpdef _matmul_alg_selection_destroy(intptr_t alg_selection):
    """See `cusparseLtMatmulAlgSelectionDestroy`."""
    with nogil:
        __status__ = cusparseLtMatmulAlgSelectionDestroy(<const cusparseLtMatmulAlgSelection_t*>alg_selection)
    check_status(__status__)

cpdef intptr_t init() except *:
    cdef intptr_t handle = <intptr_t>aligned_alloc_wrapper(16, 1024) # Allocate max size for both 0.7.1 and 0.8.1
    if handle == 0:
        raise MemoryError()
    _init(handle)
    return handle

cpdef void destroy(intptr_t handle_ptr) except *:
    if handle_ptr != 0:
        _destroy(handle_ptr)
        aligned_free_wrapper(<void*>handle_ptr)

cpdef intptr_t dense_descriptor_init(intptr_t handle, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, int value_type, int order) except *:
    cdef intptr_t mat_descr = <intptr_t>aligned_alloc_wrapper(16, 1024) # Allocate max size for both 0.7.1 and 0.8.1
    if mat_descr == 0:
        raise MemoryError()
    _dense_descriptor_init(handle, mat_descr, rows, cols, ld, alignment, value_type, order)
    return mat_descr

cpdef intptr_t structured_descriptor_init(intptr_t handle, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, int value_type, int order, int sparsity) except *:
    cdef intptr_t mat_descr = <intptr_t>aligned_alloc_wrapper(16, 1024) # Allocate max size for both 0.7.1 and 0.8.1
    if mat_descr == 0:
        raise MemoryError()
    _structured_descriptor_init(handle, mat_descr, rows, cols, ld, alignment, value_type, order, sparsity)
    return mat_descr

cpdef void mat_descriptor_destroy(intptr_t mat_descr_ptr) except *:
    if mat_descr_ptr != 0:
        _mat_descriptor_destroy(mat_descr_ptr)
        aligned_free_wrapper(<void*>mat_descr_ptr)

cpdef intptr_t matmul_descriptor_init(intptr_t handle, int op_a, int op_b, intptr_t mat_a, intptr_t mat_b, intptr_t mat_c, intptr_t mat_d, int compute_type) except *:
    cdef intptr_t matmul_descr = <intptr_t>aligned_alloc_wrapper(16, 1024) # Allocate max size for both 0.7.1 and 0.8.1
    if matmul_descr == 0:
        raise MemoryError()
    _matmul_descriptor_init(handle, matmul_descr, op_a, op_b, mat_a, mat_b, mat_c, mat_d, compute_type)
    return matmul_descr

cpdef void matmul_descriptor_destroy(intptr_t matmul_descr_ptr) except *:
    if matmul_descr_ptr != 0:
        # There is no cusparseLtMatmulDescriptorDestroy in the C API, but we still need to free the allocated memory
        aligned_free_wrapper(<void*>matmul_descr_ptr)

cpdef intptr_t matmul_alg_selection_init(intptr_t handle, intptr_t matmul_descr, int alg) except *:
    cdef intptr_t alg_selection = <intptr_t>aligned_alloc_wrapper(16, 1024) # Allocate max size for both 0.7.1 and 0.8.1
    if alg_selection == 0:
        raise MemoryError()
    _matmul_alg_selection_init(handle, alg_selection, matmul_descr, alg)
    return alg_selection

cpdef void matmul_alg_selection_destroy(intptr_t alg_selection_ptr) except *:
    if alg_selection_ptr != 0:
        _matmul_alg_selection_destroy(alg_selection_ptr)
        aligned_free_wrapper(<void*>alg_selection_ptr)

cpdef intptr_t matmul_plan_init(intptr_t handle, intptr_t matmul_descr, intptr_t alg_selection) except *:
    cdef intptr_t plan = <intptr_t>aligned_alloc_wrapper(16, 1024) # Allocate max size for both 0.7.1 and 0.8.1
    if plan == 0:
        raise MemoryError()
    _matmul_plan_init(handle, plan, matmul_descr, alg_selection)
    return plan

cpdef void matmul_plan_destroy(intptr_t plan_ptr) except *:
    if plan_ptr != 0:
        _matmul_plan_destroy(plan_ptr)
        aligned_free_wrapper(<void*>plan_ptr)
