# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.5.0. Do not modify it directly.

cimport cython  # NOQA

from enum import IntEnum as _IntEnum

import numpy as _numpy


###############################################################################
# POD
###############################################################################




###############################################################################
# Enum
###############################################################################

class OpType(_IntEnum):
    """See `cudssOpType_t`."""
    SUM = CUDSS_SUM
    MAX = CUDSS_MAX
    MIN = CUDSS_MIN

class ConfigParam(_IntEnum):
    """See `cudssConfigParam_t`."""
    REORDERING_ALG = CUDSS_CONFIG_REORDERING_ALG
    FACTORIZATION_ALG = CUDSS_CONFIG_FACTORIZATION_ALG
    SOLVE_ALG = CUDSS_CONFIG_SOLVE_ALG
    MATCHING_TYPE = CUDSS_CONFIG_MATCHING_TYPE
    SOLVE_MODE = CUDSS_CONFIG_SOLVE_MODE
    IR_N_STEPS = CUDSS_CONFIG_IR_N_STEPS
    IR_TOL = CUDSS_CONFIG_IR_TOL
    PIVOT_TYPE = CUDSS_CONFIG_PIVOT_TYPE
    PIVOT_THRESHOLD = CUDSS_CONFIG_PIVOT_THRESHOLD
    PIVOT_EPSILON = CUDSS_CONFIG_PIVOT_EPSILON
    MAX_LU_NNZ = CUDSS_CONFIG_MAX_LU_NNZ
    HYBRID_MODE = CUDSS_CONFIG_HYBRID_MODE
    HYBRID_DEVICE_MEMORY_LIMIT = CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT
    USE_CUDA_REGISTER_MEMORY = CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY
    HOST_NTHREADS = CUDSS_CONFIG_HOST_NTHREADS
    HYBRID_EXECUTE_MODE = CUDSS_CONFIG_HYBRID_EXECUTE_MODE
    PIVOT_EPSILON_ALG = CUDSS_CONFIG_PIVOT_EPSILON_ALG

class DataParam(_IntEnum):
    """See `cudssDataParam_t`."""
    INFO = CUDSS_DATA_INFO
    LU_NNZ = CUDSS_DATA_LU_NNZ
    NPIVOTS = CUDSS_DATA_NPIVOTS
    INERTIA = CUDSS_DATA_INERTIA
    PERM_REORDER_ROW = CUDSS_DATA_PERM_REORDER_ROW
    PERM_REORDER_COL = CUDSS_DATA_PERM_REORDER_COL
    PERM_ROW = CUDSS_DATA_PERM_ROW
    PERM_COL = CUDSS_DATA_PERM_COL
    DIAG = CUDSS_DATA_DIAG
    USER_PERM = CUDSS_DATA_USER_PERM
    HYBRID_DEVICE_MEMORY_MIN = CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN
    COMM = CUDSS_DATA_COMM
    MEMORY_ESTIMATES = CUDSS_DATA_MEMORY_ESTIMATES

class Phase(_IntEnum):
    """See `cudssPhase_t`."""
    ANALYSIS = CUDSS_PHASE_ANALYSIS
    FACTORIZATION = CUDSS_PHASE_FACTORIZATION
    REFACTORIZATION = CUDSS_PHASE_REFACTORIZATION
    SOLVE = CUDSS_PHASE_SOLVE
    SOLVE_FWD = CUDSS_PHASE_SOLVE_FWD
    SOLVE_DIAG = CUDSS_PHASE_SOLVE_DIAG
    SOLVE_BWD = CUDSS_PHASE_SOLVE_BWD

class Status(_IntEnum):
    """See `cudssStatus_t`."""
    SUCCESS = CUDSS_STATUS_SUCCESS
    NOT_INITIALIZED = CUDSS_STATUS_NOT_INITIALIZED
    ALLOC_FAILED = CUDSS_STATUS_ALLOC_FAILED
    INVALID_VALUE = CUDSS_STATUS_INVALID_VALUE
    NOT_SUPPORTED = CUDSS_STATUS_NOT_SUPPORTED
    EXECUTION_FAILED = CUDSS_STATUS_EXECUTION_FAILED
    INTERNAL_ERROR = CUDSS_STATUS_INTERNAL_ERROR

class MatrixType(_IntEnum):
    """See `cudssMatrixType_t`."""
    GENERAL = CUDSS_MTYPE_GENERAL
    SYMMETRIC = CUDSS_MTYPE_SYMMETRIC
    HERMITIAN = CUDSS_MTYPE_HERMITIAN
    SPD = CUDSS_MTYPE_SPD
    HPD = CUDSS_MTYPE_HPD

class MatrixViewType(_IntEnum):
    """See `cudssMatrixViewType_t`."""
    FULL = CUDSS_MVIEW_FULL
    LOWER = CUDSS_MVIEW_LOWER
    UPPER = CUDSS_MVIEW_UPPER

class IndexBase(_IntEnum):
    """See `cudssIndexBase_t`."""
    ZERO = CUDSS_BASE_ZERO
    ONE = CUDSS_BASE_ONE

class Layout(_IntEnum):
    """See `cudssLayout_t`."""
    COL_MAJOR = CUDSS_LAYOUT_COL_MAJOR
    ROW_MAJOR = CUDSS_LAYOUT_ROW_MAJOR

class AlgType(_IntEnum):
    """See `cudssAlgType_t`."""
    ALG_DEFAULT = CUDSS_ALG_DEFAULT
    ALG_1 = CUDSS_ALG_1
    ALG_2 = CUDSS_ALG_2
    ALG_3 = CUDSS_ALG_3

class PivotType(_IntEnum):
    """See `cudssPivotType_t`."""
    PIVOT_COL = CUDSS_PIVOT_COL
    PIVOT_ROW = CUDSS_PIVOT_ROW
    PIVOT_NONE = CUDSS_PIVOT_NONE

class MatrixFormat(_IntEnum):
    """See `cudssMatrixFormat_t`."""
    DENSE = CUDSS_MFORMAT_DENSE
    CSR = CUDSS_MFORMAT_CSR
    BATCH = CUDSS_MFORMAT_BATCH

###############################################################################
# Error handling
###############################################################################

cdef class cuDSSError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value})"
        super(cuDSSError, self).__init__(err)


    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuDSSError(status)

###############################################################################
# Wrapper functions
###############################################################################

######################### Python specific utility #########################

cdef dict config_param_sizes = {
    CUDSS_CONFIG_REORDERING_ALG: _numpy.int32,
    CUDSS_CONFIG_FACTORIZATION_ALG: _numpy.int32,
    CUDSS_CONFIG_SOLVE_ALG: _numpy.int32,
    CUDSS_CONFIG_MATCHING_TYPE: _numpy.int32,
    CUDSS_CONFIG_SOLVE_MODE: _numpy.int32,
    CUDSS_CONFIG_IR_N_STEPS: _numpy.int32,
    CUDSS_CONFIG_IR_TOL: _numpy.float64,
    CUDSS_CONFIG_PIVOT_TYPE: _numpy.int32,
    CUDSS_CONFIG_PIVOT_THRESHOLD: _numpy.float64,
    CUDSS_CONFIG_PIVOT_EPSILON: _numpy.float64,
    CUDSS_CONFIG_MAX_LU_NNZ: _numpy.int64,
    CUDSS_CONFIG_HYBRID_MODE: _numpy.int32,
    CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT: _numpy.int64,
    CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY: _numpy.int32,
    CUDSS_CONFIG_HOST_NTHREADS: _numpy.int32,
    CUDSS_CONFIG_HYBRID_EXECUTE_MODE: _numpy.int32,
    CUDSS_CONFIG_PIVOT_EPSILON_ALG: _numpy.int32,
}

cpdef get_config_param_dtype(int attr):
    """Get the Python data type of the corresponding ConfigParam attribute.

    Args:
        attr (ConfigParam): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`config_get`, :func:`config_set`.
    """
    return config_param_sizes[attr]

###########################################################################


cpdef config_set(intptr_t config, int param, intptr_t value, size_t size_in_bytes):
    """See `cudssConfigSet`."""
    with nogil:
        status = cudssConfigSet(<Config>config, <_ConfigParam>param, <void*>value, size_in_bytes)
    check_status(status)


cpdef config_get(intptr_t config, int param, intptr_t value, size_t size_in_bytes, intptr_t size_written):
    """See `cudssConfigGet`."""
    with nogil:
        status = cudssConfigGet(<Config>config, <_ConfigParam>param, <void*>value, size_in_bytes, <size_t*>size_written)
    check_status(status)


######################### Python specific utility #########################

cdef dict data_param_sizes = {
    CUDSS_DATA_INFO: _numpy.int32,
    CUDSS_DATA_LU_NNZ: _numpy.int64,
    CUDSS_DATA_NPIVOTS: _numpy.int32,
    CUDSS_DATA_INERTIA: _numpy.int32,
    CUDSS_DATA_PERM_REORDER_ROW: _numpy.int32,
    CUDSS_DATA_PERM_REORDER_COL: _numpy.int32,
    CUDSS_DATA_PERM_ROW: _numpy.int32,
    CUDSS_DATA_PERM_COL: _numpy.int32,
    CUDSS_DATA_USER_PERM: _numpy.int32,
    CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN: _numpy.int64,
    CUDSS_DATA_COMM: _numpy.intp,
    CUDSS_DATA_MEMORY_ESTIMATES: _numpy.int64,
}

cpdef get_data_param_dtype(int attr):
    """Get the Python data type of the corresponding DataParam attribute.

    Args:
        attr (DataParam): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`data_get`, :func:`data_set`.
    """
    return data_param_sizes[attr]

###########################################################################


cpdef data_set(intptr_t handle, intptr_t data, int param, intptr_t value, size_t size_in_bytes):
    """See `cudssDataSet`."""
    with nogil:
        status = cudssDataSet(<Handle>handle, <Data>data, <_DataParam>param, <void*>value, size_in_bytes)
    check_status(status)


cpdef data_get(intptr_t handle, intptr_t data, int param, intptr_t value, size_t size_in_bytes, intptr_t size_written):
    """See `cudssDataGet`."""
    with nogil:
        status = cudssDataGet(<Handle>handle, <Data>data, <_DataParam>param, <void*>value, size_in_bytes, <size_t*>size_written)
    check_status(status)


cpdef execute(intptr_t handle, int phase, intptr_t solver_config, intptr_t solver_data, intptr_t input_matrix, intptr_t solution, intptr_t rhs):
    """See `cudssExecute`."""
    with nogil:
        status = cudssExecute(<Handle>handle, <_Phase>phase, <Config>solver_config, <Data>solver_data, <Matrix>input_matrix, <Matrix>solution, <Matrix>rhs)
    check_status(status)


cpdef set_stream(intptr_t handle, intptr_t stream):
    """See `cudssSetStream`."""
    with nogil:
        status = cudssSetStream(<Handle>handle, <Stream>stream)
    check_status(status)


cpdef set_comm_layer(intptr_t handle, intptr_t comm_lib_file_name):
    """See `cudssSetCommLayer`."""
    with nogil:
        status = cudssSetCommLayer(<Handle>handle, <const char*>comm_lib_file_name)
    check_status(status)


cpdef set_threading_layer(intptr_t handle, thr_lib_file_name):
    """See `cudssSetThreadingLayer`."""
    if not isinstance(thr_lib_file_name, str):
        raise TypeError("thr_lib_file_name must be a Python str")
    cdef bytes _temp_thr_lib_file_name_ = (<str>thr_lib_file_name).encode()
    cdef char* _thr_lib_file_name_ = _temp_thr_lib_file_name_
    with nogil:
        status = cudssSetThreadingLayer(<Handle>handle, <const char*>_thr_lib_file_name_)
    check_status(status)


cpdef intptr_t config_create() except? 0:
    """See `cudssConfigCreate`."""
    cdef Config solver_config
    with nogil:
        status = cudssConfigCreate(&solver_config)
    check_status(status)
    return <intptr_t>solver_config


cpdef config_destroy(intptr_t solver_config):
    """See `cudssConfigDestroy`."""
    with nogil:
        status = cudssConfigDestroy(<Config>solver_config)
    check_status(status)


cpdef intptr_t data_create(intptr_t handle) except? 0:
    """See `cudssDataCreate`."""
    cdef Data solver_data
    with nogil:
        status = cudssDataCreate(<Handle>handle, &solver_data)
    check_status(status)
    return <intptr_t>solver_data


cpdef data_destroy(intptr_t handle, intptr_t solver_data):
    """See `cudssDataDestroy`."""
    with nogil:
        status = cudssDataDestroy(<Handle>handle, <Data>solver_data)
    check_status(status)


cpdef intptr_t create() except? 0:
    """See `cudssCreate`."""
    cdef Handle handle
    with nogil:
        status = cudssCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """See `cudssDestroy`."""
    with nogil:
        status = cudssDestroy(<Handle>handle)
    check_status(status)


cpdef int get_property(int property_type) except? -1:
    """See `cudssGetProperty`."""
    cdef int value
    with nogil:
        status = cudssGetProperty(<LibraryPropertyType>property_type, &value)
    check_status(status)
    return value


cpdef intptr_t matrix_create_dn(int64_t nrows, int64_t ncols, int64_t ld, intptr_t values, int value_type, int layout) except? 0:
    """See `cudssMatrixCreateDn`."""
    cdef Matrix matrix
    with nogil:
        status = cudssMatrixCreateDn(&matrix, nrows, ncols, ld, <void*>values, <DataType>value_type, <_Layout>layout)
    check_status(status)
    return <intptr_t>matrix


cpdef intptr_t matrix_create_csr(int64_t nrows, int64_t ncols, int64_t nnz, intptr_t row_start, intptr_t row_end, intptr_t col_indices, intptr_t values, int index_type, int value_type, int mtype, int mview, int index_base) except? 0:
    """See `cudssMatrixCreateCsr`."""
    cdef Matrix matrix
    with nogil:
        status = cudssMatrixCreateCsr(&matrix, nrows, ncols, nnz, <void*>row_start, <void*>row_end, <void*>col_indices, <void*>values, <DataType>index_type, <DataType>value_type, <_MatrixType>mtype, <_MatrixViewType>mview, <_IndexBase>index_base)
    check_status(status)
    return <intptr_t>matrix


cpdef intptr_t matrix_create_batch_dn(int64_t batch_count, intptr_t nrows, intptr_t ncols, intptr_t ld, intptr_t values, int index_type, int value_type, int layout) except? 0:
    """See `cudssMatrixCreateBatchDn`."""
    cdef Matrix matrix
    with nogil:
        status = cudssMatrixCreateBatchDn(&matrix, batch_count, <void*>nrows, <void*>ncols, <void*>ld, <void**>values, <DataType>index_type, <DataType>value_type, <_Layout>layout)
    check_status(status)
    return <intptr_t>matrix


cpdef intptr_t matrix_create_batch_csr(int64_t batch_count, intptr_t nrows, intptr_t ncols, intptr_t nnz, intptr_t row_start, intptr_t row_end, intptr_t col_indices, intptr_t values, int index_type, int value_type, int mtype, int mview, int index_base) except? 0:
    """See `cudssMatrixCreateBatchCsr`."""
    cdef Matrix matrix
    with nogil:
        status = cudssMatrixCreateBatchCsr(&matrix, batch_count, <void*>nrows, <void*>ncols, <void*>nnz, <void**>row_start, <void**>row_end, <void**>col_indices, <void**>values, <DataType>index_type, <DataType>value_type, <_MatrixType>mtype, <_MatrixViewType>mview, <_IndexBase>index_base)
    check_status(status)
    return <intptr_t>matrix


cpdef matrix_destroy(intptr_t matrix):
    """See `cudssMatrixDestroy`."""
    with nogil:
        status = cudssMatrixDestroy(<Matrix>matrix)
    check_status(status)


cpdef tuple matrix_get_dn(intptr_t matrix):
    """See `cudssMatrixGetDn`."""
    cdef int64_t nrows
    cdef int64_t ncols
    cdef int64_t ld
    cdef void* values
    cdef DataType type
    cdef _Layout layout
    with nogil:
        status = cudssMatrixGetDn(<Matrix>matrix, &nrows, &ncols, &ld, &values, &type, &layout)
    check_status(status)
    return (nrows, ncols, ld, <intptr_t>values, <int>type, <int>layout)


cpdef tuple matrix_get_csr(intptr_t matrix):
    """See `cudssMatrixGetCsr`."""
    cdef int64_t nrows
    cdef int64_t ncols
    cdef int64_t nnz
    cdef void* row_start
    cdef void* row_end
    cdef void* col_indices
    cdef void* values
    cdef DataType index_type
    cdef DataType value_type
    cdef _MatrixType mtype
    cdef _MatrixViewType mview
    cdef _IndexBase index_base
    with nogil:
        status = cudssMatrixGetCsr(<Matrix>matrix, &nrows, &ncols, &nnz, &row_start, &row_end, &col_indices, &values, &index_type, &value_type, &mtype, &mview, &index_base)
    check_status(status)
    return (nrows, ncols, nnz, <intptr_t>row_start, <intptr_t>row_end, <intptr_t>col_indices, <intptr_t>values, <int>index_type, <int>value_type, <int>mtype, <int>mview, <int>index_base)


cpdef matrix_set_values(intptr_t matrix, intptr_t values):
    """See `cudssMatrixSetValues`."""
    with nogil:
        status = cudssMatrixSetValues(<Matrix>matrix, <void*>values)
    check_status(status)


cpdef matrix_set_csr_pointers(intptr_t matrix, intptr_t row_offsets, intptr_t row_end, intptr_t col_indices, intptr_t values):
    """See `cudssMatrixSetCsrPointers`."""
    with nogil:
        status = cudssMatrixSetCsrPointers(<Matrix>matrix, <void*>row_offsets, <void*>row_end, <void*>col_indices, <void*>values)
    check_status(status)


cpdef matrix_get_batch_dn(intptr_t matrix, intptr_t batch_count, intptr_t nrows, intptr_t ncols, intptr_t ld, intptr_t values, intptr_t index_type, intptr_t value_type, intptr_t layout):
    """See `cudssMatrixGetBatchDn`."""
    with nogil:
        status = cudssMatrixGetBatchDn(<Matrix>matrix, <int64_t*>batch_count, <void**>nrows, <void**>ncols, <void**>ld, <void***>values, <DataType*>index_type, <DataType*>value_type, <_Layout*>layout)
    check_status(status)


cpdef matrix_get_batch_csr(intptr_t matrix, intptr_t batch_count, intptr_t nrows, intptr_t ncols, intptr_t nnz, intptr_t row_start, intptr_t row_end, intptr_t col_indices, intptr_t values, intptr_t index_type, intptr_t value_type, intptr_t mtype, intptr_t mview, intptr_t index_base):
    """See `cudssMatrixGetBatchCsr`."""
    with nogil:
        status = cudssMatrixGetBatchCsr(<Matrix>matrix, <int64_t*>batch_count, <void**>nrows, <void**>ncols, <void**>nnz, <void***>row_start, <void***>row_end, <void***>col_indices, <void***>values, <DataType*>index_type, <DataType*>value_type, <_MatrixType*>mtype, <_MatrixViewType*>mview, <_IndexBase*>index_base)
    check_status(status)


cpdef matrix_set_batch_values(intptr_t matrix, intptr_t values):
    """See `cudssMatrixSetBatchValues`."""
    with nogil:
        status = cudssMatrixSetBatchValues(<Matrix>matrix, <void**>values)
    check_status(status)


cpdef matrix_set_batch_csr_pointers(intptr_t matrix, intptr_t row_offsets, intptr_t row_end, intptr_t col_indices, intptr_t values):
    """See `cudssMatrixSetBatchCsrPointers`."""
    with nogil:
        status = cudssMatrixSetBatchCsrPointers(<Matrix>matrix, <void**>row_offsets, <void**>row_end, <void**>col_indices, <void**>values)
    check_status(status)


cpdef int matrix_get_format(intptr_t matrix) except? -1:
    """See `cudssMatrixGetFormat`."""
    cdef int format
    with nogil:
        status = cudssMatrixGetFormat(<Matrix>matrix, &format)
    check_status(status)
    return format


cpdef get_device_mem_handler(intptr_t handle, intptr_t handler):
    """See `cudssGetDeviceMemHandler`."""
    with nogil:
        status = cudssGetDeviceMemHandler(<Handle>handle, <cudssDeviceMemHandler_t*>handler)
    check_status(status)


cpdef set_device_mem_handler(intptr_t handle, intptr_t handler):
    """See `cudssSetDeviceMemHandler`."""
    with nogil:
        status = cudssSetDeviceMemHandler(<Handle>handle, <const cudssDeviceMemHandler_t*>handler)
    check_status(status)
