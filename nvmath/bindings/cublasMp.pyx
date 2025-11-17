# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 0.5.0 to 0.6.0. Do not modify it directly.

cimport cython  # NOQA
from libc.stdint cimport int64_t
from libcpp.vector cimport vector

from enum import IntEnum as _IntEnum

import numpy as _numpy

###############################################################################
# Enum
###############################################################################

class Operation(_IntEnum):
    """See `cublasOperation_t`."""
    N = CUBLAS_OP_N
    T = CUBLAS_OP_T
    C = CUBLAS_OP_C
    HERMITAN = CUBLAS_OP_HERMITAN
    CONJG = CUBLAS_OP_CONJG

class ComputeType(_IntEnum):
    """See `cublasComputeType_t`."""
    COMPUTE_16F = CUBLAS_COMPUTE_16F
    COMPUTE_16F_PEDANTIC = CUBLAS_COMPUTE_16F_PEDANTIC
    COMPUTE_32F = CUBLAS_COMPUTE_32F
    COMPUTE_32F_PEDANTIC = CUBLAS_COMPUTE_32F_PEDANTIC
    COMPUTE_32F_FAST_16F = CUBLAS_COMPUTE_32F_FAST_16F
    COMPUTE_32F_FAST_16BF = CUBLAS_COMPUTE_32F_FAST_16BF
    COMPUTE_32F_FAST_TF32 = CUBLAS_COMPUTE_32F_FAST_TF32
    COMPUTE_64F = CUBLAS_COMPUTE_64F
    COMPUTE_64F_PEDANTIC = CUBLAS_COMPUTE_64F_PEDANTIC
    COMPUTE_32I = CUBLAS_COMPUTE_32I
    COMPUTE_32I_PEDANTIC = CUBLAS_COMPUTE_32I_PEDANTIC

class Status(_IntEnum):
    """See `cublasMpStatus_t`."""
    SUCCESS = CUBLASMP_STATUS_SUCCESS
    NOT_INITIALIZED = CUBLASMP_STATUS_NOT_INITIALIZED
    ALLOCATION_FAILED = CUBLASMP_STATUS_ALLOCATION_FAILED
    INVALID_VALUE = CUBLASMP_STATUS_INVALID_VALUE
    ARCHITECTURE_MISMATCH = CUBLASMP_STATUS_ARCHITECTURE_MISMATCH
    EXECUTION_FAILED = CUBLASMP_STATUS_EXECUTION_FAILED
    INTERNAL_ERROR = CUBLASMP_STATUS_INTERNAL_ERROR
    NOT_SUPPORTED = CUBLASMP_STATUS_NOT_SUPPORTED

class GridLayout(_IntEnum):
    """See `cublasMpGridLayout_t`."""
    COL_MAJOR = CUBLASMP_GRID_LAYOUT_COL_MAJOR
    ROW_MAJOR = CUBLASMP_GRID_LAYOUT_ROW_MAJOR

class MatmulDescriptorAttribute(_IntEnum):
    """See `cublasMpMatmulDescriptorAttribute_t`."""
    TRANSA = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA
    TRANSB = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB
    COMPUTE_TYPE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMPUTE_TYPE
    ALGO_TYPE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE
    COMMUNICATION_SM_COUNT = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMMUNICATION_SM_COUNT
    EPILOGUE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE
    BIAS_POINTER = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_POINTER
    BIAS_BATCH_STRIDE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_BATCH_STRIDE
    BIAS_DATA_TYPE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_DATA_TYPE
    EPILOGUE_AUX_POINTER = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_POINTER
    EPILOGUE_AUX_LD = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_LD
    EPILOGUE_AUX_BATCH_STRIDE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_BATCH_STRIDE
    EPILOGUE_AUX_DATA_TYPE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_DATA_TYPE
    EPILOGUE_AUX_SCALE_POINTER = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_SCALE_POINTER
    EPILOGUE_AUX_AMAX_POINTER = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_AMAX_POINTER
    EPILOGUE_AUX_SCALE_MODE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_SCALE_MODE
    A_SCALE_POINTER = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_POINTER
    A_SCALE_MODE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_MODE
    B_SCALE_POINTER = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_POINTER
    B_SCALE_MODE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_MODE
    C_SCALE_POINTER = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_C_SCALE_POINTER
    C_SCALE_MODE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_C_SCALE_MODE
    D_SCALE_POINTER = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_POINTER
    D_SCALE_MODE = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_MODE
    AMAX_D_POINTER = CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_AMAX_D_POINTER

class MatmulAlgoType(_IntEnum):
    """See `cublasMpMatmulAlgoType_t`."""
    DEFAULT = CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT
    SPLIT_P2P = CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_P2P
    SPLIT_MULTICAST = CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_MULTICAST
    ATOMIC_P2P = CUBLASMP_MATMUL_ALGO_TYPE_ATOMIC_P2P
    ATOMIC_MULTICAST = CUBLASMP_MATMUL_ALGO_TYPE_ATOMIC_MULTICAST

class MatmulEpilogue(_IntEnum):
    """See `cublasMpMatmulEpilogue_t`."""
    DEFAULT = CUBLASMP_MATMUL_EPILOGUE_DEFAULT
    ALLREDUCE = CUBLASMP_MATMUL_EPILOGUE_ALLREDUCE
    RELU = CUBLASMP_MATMUL_EPILOGUE_RELU
    RELU_AUX = CUBLASMP_MATMUL_EPILOGUE_RELU_AUX
    BIAS = CUBLASMP_MATMUL_EPILOGUE_BIAS
    RELU_BIAS = CUBLASMP_MATMUL_EPILOGUE_RELU_BIAS
    RELU_AUX_BIAS = CUBLASMP_MATMUL_EPILOGUE_RELU_AUX_BIAS
    DRELU = CUBLASMP_MATMUL_EPILOGUE_DRELU
    DRELU_BGRAD = CUBLASMP_MATMUL_EPILOGUE_DRELU_BGRAD
    GELU = CUBLASMP_MATMUL_EPILOGUE_GELU
    GELU_AUX = CUBLASMP_MATMUL_EPILOGUE_GELU_AUX
    GELU_BIAS = CUBLASMP_MATMUL_EPILOGUE_GELU_BIAS
    GELU_AUX_BIAS = CUBLASMP_MATMUL_EPILOGUE_GELU_AUX_BIAS
    DGELU = CUBLASMP_MATMUL_EPILOGUE_DGELU
    DGELU_BGRAD = CUBLASMP_MATMUL_EPILOGUE_DGELU_BGRAD
    BGRADA = CUBLASMP_MATMUL_EPILOGUE_BGRADA
    BGRADB = CUBLASMP_MATMUL_EPILOGUE_BGRADB

class MatmulMatrixScale(_IntEnum):
    """See `cublasMpMatmulMatrixScale_t`."""
    SCALAR_FP32 = CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32
    VEC16_UE4M3 = CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3
    VEC32_UE8M0 = CUBLASMP_MATMUL_MATRIX_SCALE_VEC32_UE8M0
    OUTER_VEC_FP32 = CUBLASMP_MATMUL_MATRIX_SCALE_OUTER_VEC_FP32
    VEC128_FP32 = CUBLASMP_MATMUL_MATRIX_SCALE_VEC128_FP32
    BLK128x128_FP32 = CUBLASMP_MATMUL_MATRIX_SCALE_BLK128x128_FP32


###############################################################################
# Error handling
###############################################################################

class cuBLASMpError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value})"
        err = f"{err}. You can set CUBLASMP_LOG_LEVEL=5 and CUBLASLT_LOG_LEVEL=5 environment variables to enable logging to learn more."
        super(cuBLASMpError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuBLASMpError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create(intptr_t stream) except? 0:
    """See `cublasMpCreate`."""
    cdef Handle handle
    with nogil:
        __status__ = cublasMpCreate(&handle, <Stream>stream)
    check_status(__status__)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """See `cublasMpDestroy`."""
    with nogil:
        __status__ = cublasMpDestroy(<Handle>handle)
    check_status(__status__)


cpdef stream_set(intptr_t handle, intptr_t stream):
    """See `cublasMpStreamSet`."""
    with nogil:
        __status__ = cublasMpStreamSet(<Handle>handle, <Stream>stream)
    check_status(__status__)


cpdef int get_version() except? 0:
    """See `cublasMpGetVersion`."""
    cdef int version
    with nogil:
        __status__ = cublasMpGetVersion(&version)
    check_status(__status__)
    return version


cpdef intptr_t grid_create(int64_t nprow, int64_t npcol, int layout, intptr_t comm) except? 0:
    """See `cublasMpGridCreate`."""
    cdef Grid grid
    with nogil:
        __status__ = cublasMpGridCreate(nprow, npcol, <_GridLayout>layout, <ncclComm>comm, &grid)
    check_status(__status__)
    return <intptr_t>grid


cpdef grid_destroy(intptr_t grid):
    """See `cublasMpGridDestroy`."""
    with nogil:
        __status__ = cublasMpGridDestroy(<Grid>grid)
    check_status(__status__)


cpdef intptr_t matrix_descriptor_create(int64_t m, int64_t n, int64_t mb, int64_t nb, int64_t rsrc, int64_t csrc, int64_t lld, int type, intptr_t grid) except? 0:
    """See `cublasMpMatrixDescriptorCreate`."""
    cdef MatrixDescriptor desc
    with nogil:
        __status__ = cublasMpMatrixDescriptorCreate(m, n, mb, nb, rsrc, csrc, lld, <DataType>type, <Grid>grid, &desc)
    check_status(__status__)
    return <intptr_t>desc


cpdef matrix_descriptor_destroy(intptr_t desc):
    """See `cublasMpMatrixDescriptorDestroy`."""
    with nogil:
        __status__ = cublasMpMatrixDescriptorDestroy(<MatrixDescriptor>desc)
    check_status(__status__)


cpdef intptr_t matmul_descriptor_create(int compute_type) except? 0:
    """See `cublasMpMatmulDescriptorCreate`."""
    cdef MatmulDescriptor matmul_desc
    with nogil:
        __status__ = cublasMpMatmulDescriptorCreate(&matmul_desc, <_ComputeType>compute_type)
    check_status(__status__)
    return <intptr_t>matmul_desc


cpdef matmul_descriptor_destroy(intptr_t matmul_desc):
    """See `cublasMpMatmulDescriptorDestroy`."""
    with nogil:
        __status__ = cublasMpMatmulDescriptorDestroy(<MatmulDescriptor>matmul_desc)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict matmul_descriptor_attribute_sizes = {
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMPUTE_TYPE: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMMUNICATION_SM_COUNT: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_POINTER: _numpy.intp,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_BATCH_STRIDE: _numpy.int64,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_DATA_TYPE: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_POINTER: _numpy.intp,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_LD: _numpy.int64,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_BATCH_STRIDE: _numpy.int64,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_DATA_TYPE: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_SCALE_POINTER: _numpy.intp,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_AMAX_POINTER: _numpy.intp,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_SCALE_MODE: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_POINTER: _numpy.intp,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_MODE: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_POINTER: _numpy.intp,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_MODE: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_C_SCALE_POINTER: _numpy.intp,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_C_SCALE_MODE: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_POINTER: _numpy.intp,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_MODE: _numpy.int32,
    CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_AMAX_D_POINTER: _numpy.intp,
}

cpdef get_matmul_descriptor_attribute_dtype(int attr):
    """Get the Python data type of the corresponding MatmulDescriptorAttribute attribute.

    Args:
        attr (MatmulDescriptorAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`matmul_descriptor_attribute_get`, :func:`matmul_descriptor_attribute_set`.
    """
    return matmul_descriptor_attribute_sizes[attr]

###########################################################################


cpdef matmul_descriptor_attribute_set(intptr_t matmul_desc, int attr, intptr_t buf, size_t size_in_bytes):
    """See `cublasMpMatmulDescriptorAttributeSet`."""
    with nogil:
        __status__ = cublasMpMatmulDescriptorAttributeSet(<MatmulDescriptor>matmul_desc, <_MatmulDescriptorAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(__status__)


cpdef matmul_descriptor_attribute_get(intptr_t matmul_desc, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written):
    """See `cublasMpMatmulDescriptorAttributeGet`."""
    with nogil:
        __status__ = cublasMpMatmulDescriptorAttributeGet(<MatmulDescriptor>matmul_desc, <_MatmulDescriptorAttribute>attr, <void*>buf, size_in_bytes, <size_t*>size_written)
    check_status(__status__)


cpdef tuple matmul_buffer_size(intptr_t handle, intptr_t matmul_desc, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, intptr_t d, int64_t id, int64_t jd, intptr_t desc_d):
    """See `cublasMpMatmul_bufferSize`."""
    cdef size_t workspace_size_in_bytes_on_device
    cdef size_t workspace_size_in_bytes_on_host
    with nogil:
        __status__ = cublasMpMatmul_bufferSize(<Handle>handle, <MatmulDescriptor>matmul_desc, m, n, k, <const void*>alpha, <const void*>a, ia, ja, <MatrixDescriptor>desc_a, <const void*>b, ib, jb, <MatrixDescriptor>desc_b, <const void*>beta, <const void*>c, ic, jc, <MatrixDescriptor>desc_c, <void*>d, id, jd, <MatrixDescriptor>desc_d, &workspace_size_in_bytes_on_device, &workspace_size_in_bytes_on_host)
    check_status(__status__)
    return (workspace_size_in_bytes_on_device, workspace_size_in_bytes_on_host)


cpdef matmul(intptr_t handle, intptr_t matmul_desc, int64_t m, int64_t n, int64_t k, intptr_t alpha, intptr_t a, int64_t ia, int64_t ja, intptr_t desc_a, intptr_t b, int64_t ib, int64_t jb, intptr_t desc_b, intptr_t beta, intptr_t c, int64_t ic, int64_t jc, intptr_t desc_c, intptr_t d, int64_t id, int64_t jd, intptr_t desc_d, intptr_t d_work, size_t workspace_size_in_bytes_on_device, intptr_t h_work, size_t workspace_size_in_bytes_on_host):
    """See `cublasMpMatmul`."""
    with nogil:
        __status__ = cublasMpMatmul(<Handle>handle, <MatmulDescriptor>matmul_desc, m, n, k, <const void*>alpha, <const void*>a, ia, ja, <MatrixDescriptor>desc_a, <const void*>b, ib, jb, <MatrixDescriptor>desc_b, <const void*>beta, <const void*>c, ic, jc, <MatrixDescriptor>desc_c, <void*>d, id, jd, <MatrixDescriptor>desc_d, <void*>d_work, workspace_size_in_bytes_on_device, <void*>h_work, workspace_size_in_bytes_on_host)
    check_status(__status__)


cpdef int64_t numroc(int64_t n, int64_t nb, uint32_t iproc, uint32_t isrcproc, uint32_t nprocs) except? -1:
    """See `cublasMpNumroc`."""
    return cublasMpNumroc(n, nb, iproc, isrcproc, nprocs)
