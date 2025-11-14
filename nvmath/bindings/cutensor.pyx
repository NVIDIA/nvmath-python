# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.3.1. Do not modify it directly.

cimport cython  # NOQA
cimport cpython
from libcpp.vector cimport vector

from ._internal.utils cimport get_resource_ptr, get_resource_ptrs, nullable_unique_ptr

from enum import IntEnum as _IntEnum

import ctypes
import threading
import numpy as _numpy


cdef object __symbol_lock = threading.Lock()
_COMPUTE_DESC_INIT = False
_COMPUTE_DESC_16F = None
_COMPUTE_DESC_16BF = None
_COMPUTE_DESC_TF32 = None
_COMPUTE_DESC_3XTF32 = None
_COMPUTE_DESC_32F = None
_COMPUTE_DESC_64F = None

def _load_cutensor_compute_descriptors():
    global _COMPUTE_DESC_INIT
    if _COMPUTE_DESC_INIT:
        return

    with __symbol_lock:
        try:
            lib = ctypes.CDLL("libcutensor.so.2")
            global _COMPUTE_DESC_16F, _COMPUTE_DESC_16BF, _COMPUTE_DESC_TF32, _COMPUTE_DESC_3XTF32, _COMPUTE_DESC_32F, _COMPUTE_DESC_64F
            _COMPUTE_DESC_16F = ctypes.c_void_p.in_dll(lib, "CUTENSOR_COMPUTE_DESC_16F").value
            _COMPUTE_DESC_16BF = ctypes.c_void_p.in_dll(lib, "CUTENSOR_COMPUTE_DESC_16BF").value
            _COMPUTE_DESC_TF32 = ctypes.c_void_p.in_dll(lib, "CUTENSOR_COMPUTE_DESC_TF32").value
            _COMPUTE_DESC_3XTF32 = ctypes.c_void_p.in_dll(lib, "CUTENSOR_COMPUTE_DESC_3XTF32").value
            _COMPUTE_DESC_32F = ctypes.c_void_p.in_dll(lib, "CUTENSOR_COMPUTE_DESC_32F").value
            _COMPUTE_DESC_64F = ctypes.c_void_p.in_dll(lib, "CUTENSOR_COMPUTE_DESC_64F").value
            _COMPUTE_DESC_INIT = True
        except:
            raise ImportError("Failed to load cutensor library")


class ComputeDesc:
    """See `cutensorComputeDescriptor_t`."""

    @classmethod
    def COMPUTE_16F(cls):
        _load_cutensor_compute_descriptors()
        return _COMPUTE_DESC_16F

    @classmethod
    def COMPUTE_16BF(cls):
        _load_cutensor_compute_descriptors()
        return _COMPUTE_DESC_16BF

    @classmethod
    def COMPUTE_TF32(cls):
        _load_cutensor_compute_descriptors()
        return _COMPUTE_DESC_TF32

    @classmethod
    def COMPUTE_3XTF32(cls):
        _load_cutensor_compute_descriptors()
        return _COMPUTE_DESC_3XTF32

    @classmethod
    def COMPUTE_32F(cls):
        _load_cutensor_compute_descriptors()
        return _COMPUTE_DESC_32F

    @classmethod
    def COMPUTE_64F(cls):
        _load_cutensor_compute_descriptors()
        return _COMPUTE_DESC_64F


###############################################################################
# Enum
###############################################################################

class Operator(_IntEnum):
    """See `cutensorOperator_t`."""
    OP_IDENTITY = CUTENSOR_OP_IDENTITY
    OP_SQRT = CUTENSOR_OP_SQRT
    OP_RELU = CUTENSOR_OP_RELU
    OP_CONJ = CUTENSOR_OP_CONJ
    OP_RCP = CUTENSOR_OP_RCP
    OP_SIGMOID = CUTENSOR_OP_SIGMOID
    OP_TANH = CUTENSOR_OP_TANH
    OP_EXP = CUTENSOR_OP_EXP
    OP_LOG = CUTENSOR_OP_LOG
    OP_ABS = CUTENSOR_OP_ABS
    OP_NEG = CUTENSOR_OP_NEG
    OP_SIN = CUTENSOR_OP_SIN
    OP_COS = CUTENSOR_OP_COS
    OP_TAN = CUTENSOR_OP_TAN
    OP_SINH = CUTENSOR_OP_SINH
    OP_COSH = CUTENSOR_OP_COSH
    OP_ASIN = CUTENSOR_OP_ASIN
    OP_ACOS = CUTENSOR_OP_ACOS
    OP_ATAN = CUTENSOR_OP_ATAN
    OP_ASINH = CUTENSOR_OP_ASINH
    OP_ACOSH = CUTENSOR_OP_ACOSH
    OP_ATANH = CUTENSOR_OP_ATANH
    OP_CEIL = CUTENSOR_OP_CEIL
    OP_FLOOR = CUTENSOR_OP_FLOOR
    OP_MISH = CUTENSOR_OP_MISH
    OP_SWISH = CUTENSOR_OP_SWISH
    OP_SOFT_PLUS = CUTENSOR_OP_SOFT_PLUS
    OP_SOFT_SIGN = CUTENSOR_OP_SOFT_SIGN
    OP_ADD = CUTENSOR_OP_ADD
    OP_MUL = CUTENSOR_OP_MUL
    OP_MAX = CUTENSOR_OP_MAX
    OP_MIN = CUTENSOR_OP_MIN
    OP_UNKNOWN = CUTENSOR_OP_UNKNOWN

class Status(_IntEnum):
    """See `cutensorStatus_t`."""
    SUCCESS = CUTENSOR_STATUS_SUCCESS
    NOT_INITIALIZED = CUTENSOR_STATUS_NOT_INITIALIZED
    ALLOC_FAILED = CUTENSOR_STATUS_ALLOC_FAILED
    INVALID_VALUE = CUTENSOR_STATUS_INVALID_VALUE
    ARCH_MISMATCH = CUTENSOR_STATUS_ARCH_MISMATCH
    MAPPING_ERROR = CUTENSOR_STATUS_MAPPING_ERROR
    EXECUTION_FAILED = CUTENSOR_STATUS_EXECUTION_FAILED
    INTERNAL_ERROR = CUTENSOR_STATUS_INTERNAL_ERROR
    NOT_SUPPORTED = CUTENSOR_STATUS_NOT_SUPPORTED
    LICENSE_ERROR = CUTENSOR_STATUS_LICENSE_ERROR
    CUBLAS_ERROR = CUTENSOR_STATUS_CUBLAS_ERROR
    CUDA_ERROR = CUTENSOR_STATUS_CUDA_ERROR
    INSUFFICIENT_WORKSPACE = CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE
    INSUFFICIENT_DRIVER = CUTENSOR_STATUS_INSUFFICIENT_DRIVER
    IO_ERROR = CUTENSOR_STATUS_IO_ERROR

class Algo(_IntEnum):
    """See `cutensorAlgo_t`."""
    DEFAULT_PATIENT = CUTENSOR_ALGO_DEFAULT_PATIENT
    GETT = CUTENSOR_ALGO_GETT
    TGETT = CUTENSOR_ALGO_TGETT
    TTGT = CUTENSOR_ALGO_TTGT
    DEFAULT = CUTENSOR_ALGO_DEFAULT

class WorksizePreference(_IntEnum):
    """See `cutensorWorksizePreference_t`."""
    WORKSPACE_MIN = CUTENSOR_WORKSPACE_MIN
    WORKSPACE_DEFAULT = CUTENSOR_WORKSPACE_DEFAULT
    WORKSPACE_MAX = CUTENSOR_WORKSPACE_MAX

class OperationDescriptorAttribute(_IntEnum):
    """See `cutensorOperationDescriptorAttribute_t`."""
    TAG = CUTENSOR_OPERATION_DESCRIPTOR_TAG
    SCALAR_TYPE = CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE
    FLOPS = CUTENSOR_OPERATION_DESCRIPTOR_FLOPS
    MOVED_BYTES = CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES
    PADDING_LEFT = CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT
    PADDING_RIGHT = CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT
    PADDING_VALUE = CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE

class PlanPreferenceAttribute(_IntEnum):
    """See `cutensorPlanPreferenceAttribute_t`."""
    AUTOTUNE_MODE = CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE
    CACHE_MODE = CUTENSOR_PLAN_PREFERENCE_CACHE_MODE
    INCREMENTAL_COUNT = CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT
    ALGO = CUTENSOR_PLAN_PREFERENCE_ALGO
    KERNEL_RANK = CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK
    JIT = CUTENSOR_PLAN_PREFERENCE_JIT

class AutotuneMode(_IntEnum):
    """See `cutensorAutotuneMode_t`."""
    NONE = CUTENSOR_AUTOTUNE_MODE_NONE
    INCREMENTAL = CUTENSOR_AUTOTUNE_MODE_INCREMENTAL

class JitMode(_IntEnum):
    """See `cutensorJitMode_t`."""
    NONE = CUTENSOR_JIT_MODE_NONE
    DEFAULT = CUTENSOR_JIT_MODE_DEFAULT

class CacheMode(_IntEnum):
    """See `cutensorCacheMode_t`."""
    NONE = CUTENSOR_CACHE_MODE_NONE
    PEDANTIC = CUTENSOR_CACHE_MODE_PEDANTIC

class PlanAttribute(_IntEnum):
    """See `cutensorPlanAttribute_t`."""
    REQUIRED_WORKSPACE = CUTENSOR_PLAN_REQUIRED_WORKSPACE


###############################################################################
# Error handling
###############################################################################

cdef class cuTENSORError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value})"
        super(cuTENSORError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuTENSORError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create() except? 0:
    """Initializes the cuTENSOR library and allocates the memory for the library context.

    Returns:
        intptr_t: Pointer to cutensorHandle_t.

    .. seealso:: `cutensorCreate`
    """
    cdef Handle handle
    with nogil:
        __status__ = cutensorCreate(&handle)
    check_status(__status__)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    """Frees all resources related to the provided library handle.

    Args:
        handle (intptr_t): Pointer to cutensorHandle_t.

    .. seealso:: `cutensorDestroy`
    """
    with nogil:
        __status__ = cutensorDestroy(<Handle>handle)
    check_status(__status__)


cpdef handle_resize_plan_cache(intptr_t handle, uint32_t num_entries):
    """Resizes the plan cache.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context. The cache will be attached to the handle.
        num_entries (uint32_t): Number of entries the cache will support.

    .. seealso:: `cutensorHandleResizePlanCache`
    """
    with nogil:
        __status__ = cutensorHandleResizePlanCache(<Handle>handle, <const uint32_t>num_entries)
    check_status(__status__)


cpdef handle_write_plan_cache_to_file(intptr_t handle, filename):
    """Writes the Plan-Cache (that belongs to the provided handle) to file.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        filename (str): Specifies the filename (including the absolute path) to the file that should hold all the cache information. Warning: an existing file will be overwritten.

    .. seealso:: `cutensorHandleWritePlanCacheToFile`
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a Python str")
    cdef bytes _temp_filename_ = (<str>filename).encode()
    cdef char* _filename_ = _temp_filename_
    with nogil:
        __status__ = cutensorHandleWritePlanCacheToFile(<const Handle>handle, <const char*>_filename_)
    check_status(__status__)


cpdef uint32_t handle_read_plan_cache_from_file(intptr_t handle, filename) except? -1:
    """Reads a Plan-Cache from file and overwrites the cachelines of the provided handle.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        filename (str): Specifies the filename (including the absolute path) to the file that holds all the cache information that have previously been written by ``cutensorHandleWritePlanCacheToFile``.

    Returns:
        uint32_t: On exit, this variable will hold the number of successfully-read cachelines, if CUTENSOR_STATUS_SUCCESS is returned. Otherwise, this variable will hold the number of cachelines that are required to read all cachelines associated to the cache pointed to by ``filename``; in that case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE is returned.

    .. seealso:: `cutensorHandleReadPlanCacheFromFile`
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a Python str")
    cdef bytes _temp_filename_ = (<str>filename).encode()
    cdef char* _filename_ = _temp_filename_
    cdef uint32_t num_cachelines_read
    with nogil:
        __status__ = cutensorHandleReadPlanCacheFromFile(<Handle>handle, <const char*>_filename_, &num_cachelines_read)
    check_status(__status__)
    return num_cachelines_read


cpdef write_kernel_cache_to_file(intptr_t handle, filename):
    """Writes the --per library-- kernel cache to file.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        filename (str): Specifies the filename (including the absolute path) to the file that should hold all the cache information. Warning: an existing file will be overwritten.

    .. seealso:: `cutensorWriteKernelCacheToFile`
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a Python str")
    cdef bytes _temp_filename_ = (<str>filename).encode()
    cdef char* _filename_ = _temp_filename_
    with nogil:
        __status__ = cutensorWriteKernelCacheToFile(<const Handle>handle, <const char*>_filename_)
    check_status(__status__)


cpdef read_kernel_cache_from_file(intptr_t handle, filename):
    """Reads a kernel cache from file and adds all non-existing JIT compiled kernels to the kernel cache.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        filename (str): Specifies the filename (including the absolute path) to the file that holds all the cache information that have previously been written by cutensorWriteKernelCacheToFile.

    .. seealso:: `cutensorReadKernelCacheFromFile`
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a Python str")
    cdef bytes _temp_filename_ = (<str>filename).encode()
    cdef char* _filename_ = _temp_filename_
    with nogil:
        __status__ = cutensorReadKernelCacheFromFile(<Handle>handle, <const char*>_filename_)
    check_status(__status__)


cpdef intptr_t create_tensor_descriptor(intptr_t handle, uint32_t num_modes, extent, stride, int data_type, uint32_t alignment_requirement) except? 0:
    """Creates a tensor descriptor.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        num_modes (uint32_t): Number of modes.
        extent (object): Extent of each mode (must be larger than zero). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        stride (object): stride[i] denotes the displacement (a.k.a. stride)--in elements of the base type--between two consecutive elements in the ith-mode. If stride is NULL, a packed generalized column-major memory layout is assumed (i.e., the strides increase monotonically from left to right). Each stride must be larger than zero; to be precise, a stride of zero can be achieved by omitting this mode entirely; for instance instead of writing C[a,b] = A[b,a] with strideA(a) = 0, you can write C[a,b] = A[b] directly; cuTENSOR will then automatically infer that the a-mode in A should be broadcasted. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        data_type (int): Data type of the stored entries.
        alignment_requirement (uint32_t): Alignment (in bytes) to the base pointer that will be used in conjunction with this tensor descriptor (e.g., ``cudaMalloc`` has a default alignment of 256 bytes).

    Returns:
        intptr_t: Pointer to the address where the allocated tensor descriptor object will be stored.

    .. seealso:: `cutensorCreateTensorDescriptor`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _extent_
    get_resource_ptr[int64_t](_extent_, extent, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _stride_
    get_resource_ptr[int64_t](_stride_, stride, <int64_t*>NULL)
    cdef TensorDescriptor desc
    with nogil:
        __status__ = cutensorCreateTensorDescriptor(<const Handle>handle, &desc, <const uint32_t>num_modes, <const int64_t*>(_extent_.data()), <const int64_t*>(_stride_.data()), <DataType>data_type, alignment_requirement)
    check_status(__status__)
    return <intptr_t>desc


cpdef destroy_tensor_descriptor(intptr_t desc):
    """Frees all resources related to the provided tensor descriptor.

    Args:
        desc (intptr_t): The cutensorTensorDescriptor_t object that will be deallocated.

    .. seealso:: `cutensorDestroyTensorDescriptor`
    """
    with nogil:
        __status__ = cutensorDestroyTensorDescriptor(<TensorDescriptor>desc)
    check_status(__status__)


cpdef intptr_t create_elementwise_trinary(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_b, mode_b, int op_b, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, int op_ab, int op_abc, intptr_t desc_compute) except? 0:
    """This function creates an operation descriptor that encodes an elementwise trinary operation.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc_a (intptr_t): A descriptor that holds the information about the data type, modes, and strides of A.
        mode_a (object): Array (in host memory) of size desc_a->numModes that holds the names of the modes of A (e.g., if  then mode_a = {'a','b','c'}). The mode_a[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to ``cutensorCreateTensorDescriptor``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_a (Operator): Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged.
        desc_b (intptr_t): A descriptor that holds information about the data type, modes, and strides of B.
        mode_b (object): Array (in host memory) of size desc_b->numModes that holds the names of the modes of B. mode_b[i] corresponds to extent[i] and stride[i] of the ``cutensorCreateTensorDescriptor``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_b (Operator): Unary operator that will be applied to each element of B before it is further processed. The original data of this tensor remains unchanged.
        desc_c (intptr_t): A descriptor that holds information about the data type, modes, and strides of C.
        mode_c (object): Array (in host memory) of size desc_c->numModes that holds the names of the modes of C. The mode_c[i] corresponds to extent[i] and stride[i] of the ``cutensorCreateTensorDescriptor``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_c (Operator): Unary operator that will be applied to each element of C before it is further processed. The original data of this tensor remains unchanged.
        desc_d (intptr_t): A descriptor that holds information about the data type, modes, and strides of D. Notice that we currently request desc_d and desc_c to be identical.
        mode_d (object): Array (in host memory) of size desc_d->numModes that holds the names of the modes of D. The mode_d[i] corresponds to extent[i] and stride[i] of the ``cutensorCreateTensorDescriptor``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_ab (Operator): Element-wise binary operator (see  above).
        op_abc (Operator): Element-wise binary operator (see  above).
        desc_compute (intptr_t): Determines the precision in which this operations is performed.

    Returns:
        intptr_t: This opaque struct gets allocated and filled with the information that encodes the requested elementwise operation.

    .. seealso:: `cutensorCreateElementwiseTrinary`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_a_
    get_resource_ptr[int32_t](_mode_a_, mode_a, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_b_
    get_resource_ptr[int32_t](_mode_b_, mode_b, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_c_
    get_resource_ptr[int32_t](_mode_c_, mode_c, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_d_
    get_resource_ptr[int32_t](_mode_d_, mode_d, <int32_t*>NULL)
    cdef OperationDescriptor desc
    with nogil:
        __status__ = cutensorCreateElementwiseTrinary(<const Handle>handle, &desc, <const TensorDescriptor>desc_a, <const int32_t*>(_mode_a_.data()), <_Operator>op_a, <const TensorDescriptor>desc_b, <const int32_t*>(_mode_b_.data()), <_Operator>op_b, <const TensorDescriptor>desc_c, <const int32_t*>(_mode_c_.data()), <_Operator>op_c, <const TensorDescriptor>desc_d, <const int32_t*>(_mode_d_.data()), <_Operator>op_ab, <_Operator>op_abc, <const ComputeDescriptor>desc_compute)
    check_status(__status__)
    return <intptr_t>desc


cpdef elementwise_trinary_execute(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t beta, intptr_t b, intptr_t gamma, intptr_t c, intptr_t d, intptr_t stream):
    """Performs an element-wise tensor operation for three input tensors (see ``cutensorcreateElementwiseTrinary``).

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        plan (intptr_t): Opaque handle holding all information about the desired elementwise operation (created by ``cutensorcreateElementwiseTrinary`` followed by ``cutensorcreatePlan``).
        alpha (intptr_t): Scaling factor for a (see cutensorOperationdescriptorGetattribute(desc, cUTENSOR_OPERaTION_ScaLaR_TYPE) to query the expected data type). Pointer to the host memory. If alpha is zero, a is not read and the corresponding unary operator is not applied.
        a (intptr_t): Multi-mode tensor (described by ``desca`` as part of ``cutensorcreateElementwiseTrinary``). Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to d.
        beta (intptr_t): Scaling factor for b (see cutensorOperationdescriptorGetattribute(desc, cUTENSOR_OPERaTION_ScaLaR_TYPE) to query the expected data type). Pointer to the host memory. If beta is zero, b is not read and the corresponding unary operator is not applied.
        b (intptr_t): Multi-mode tensor (described by ``descb`` as part of ``cutensorcreateElementwiseTrinary``). Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to d.
        gamma (intptr_t): Scaling factor for c (see cutensorOperationdescriptorGetattribute(desc, cUTENSOR_OPERaTION_ScaLaR_TYPE) to query the expected data type). Pointer to the host memory. If gamma is zero, c is not read and the corresponding unary operator is not applied.
        c (intptr_t): Multi-mode tensor (described by ``descc`` as part of ``cutensorcreateElementwiseTrinary``). Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to d.
        d (intptr_t): Multi-mode tensor (described by ``descd`` as part of ``cutensorcreateElementwiseTrinary``). Pointer to the GPU-accessible memory (``c`` and ``d`` may be identical, if and only if ``descc == descd``).
        stream (intptr_t): The cUda stream used to perform the operation.

    .. seealso:: `cutensorElementwiseTrinaryExecute`
    """
    with nogil:
        __status__ = cutensorElementwiseTrinaryExecute(<const Handle>handle, <const Plan>plan, <const void*>alpha, <const void*>a, <const void*>beta, <const void*>b, <const void*>gamma, <const void*>c, <void*>d, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_elementwise_binary(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, int op_ac, intptr_t desc_compute) except? 0:
    """This function creates an operation descriptor for an elementwise binary operation.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc_a (intptr_t): The descriptor that holds the information about the data type, modes, and strides of A.
        mode_a (object): Array (in host memory) of size desc_a->numModes that holds the names of the modes of A (e.g., if A_{a,b,c} => mode_a = {'a','b','c'}). The mode_a[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to ``cutensorCreateTensorDescriptor``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_a (Operator): Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged.
        desc_c (intptr_t): The descriptor that holds information about the data type, modes, and strides of C.
        mode_c (object): Array (in host memory) of size desc_c->numModes that holds the names of the modes of C. The mode_c[i] corresponds to extent[i] and stride[i] of the ``cutensorCreateTensorDescriptor``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_c (Operator): Unary operator that will be applied to each element of C before it is further processed. The original data of this tensor remains unchanged.
        desc_d (intptr_t): The descriptor that holds information about the data type, modes, and strides of D. Notice that we currently request desc_d and desc_c to be identical.
        mode_d (object): Array (in host memory) of size desc_d->numModes that holds the names of the modes of D. The mode_d[i] corresponds to extent[i] and stride[i] of the ``cutensorCreateTensorDescriptor``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_ac (Operator): Element-wise binary operator (see  above).
        desc_compute (intptr_t): Determines the precision in which this operations is performed.

    Returns:
        intptr_t: This opaque struct gets allocated and filled with the information that encodes the requested elementwise operation.

    .. seealso:: `cutensorCreateElementwiseBinary`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_a_
    get_resource_ptr[int32_t](_mode_a_, mode_a, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_c_
    get_resource_ptr[int32_t](_mode_c_, mode_c, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_d_
    get_resource_ptr[int32_t](_mode_d_, mode_d, <int32_t*>NULL)
    cdef OperationDescriptor desc
    with nogil:
        __status__ = cutensorCreateElementwiseBinary(<const Handle>handle, &desc, <const TensorDescriptor>desc_a, <const int32_t*>(_mode_a_.data()), <_Operator>op_a, <const TensorDescriptor>desc_c, <const int32_t*>(_mode_c_.data()), <_Operator>op_c, <const TensorDescriptor>desc_d, <const int32_t*>(_mode_d_.data()), <_Operator>op_ac, <const ComputeDescriptor>desc_compute)
    check_status(__status__)
    return <intptr_t>desc


cpdef elementwise_binary_execute(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t gamma, intptr_t c, intptr_t d, intptr_t stream):
    """Performs an element-wise tensor operation for two input tensors (see ``cutensorcreateElementwiseBinary``).

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        plan (intptr_t): Opaque handle holding all information about the desired elementwise operation (created by ``cutensorcreateElementwiseBinary`` followed by ``cutensorcreatePlan``).
        alpha (intptr_t): Scaling factor for a (see cutensorOperationdescriptorGetattribute(desc, cUTENSOR_OPERaTION_ScaLaR_TYPE) to query the expected data type). Pointer to the host memory. If alpha is zero, a is not read and the corresponding unary operator is not applied.
        a (intptr_t): Multi-mode tensor (described by ``desca`` as part of ``cutensorcreateElementwiseBinary``). Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to d.
        gamma (intptr_t): Scaling factor for c (see cutensorOperationdescriptorGetattribute(desc, cUTENSOR_OPERaTION_ScaLaR_TYPE) to query the expected data type). Pointer to the host memory. If gamma is zero, c is not read and the corresponding unary operator is not applied.
        c (intptr_t): Multi-mode tensor (described by ``descc`` as part of ``cutensorcreateElementwiseBinary``). Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to d.
        d (intptr_t): Multi-mode tensor (described by ``descd`` as part of ``cutensorcreateElementwiseBinary``). Pointer to the GPU-accessible memory (``c`` and ``d`` may be identical, if and only if ``descc == descd``).
        stream (intptr_t): The cUda stream used to perform the operation.

    .. seealso:: `cutensorElementwiseBinaryExecute`
    """
    with nogil:
        __status__ = cutensorElementwiseBinaryExecute(<const Handle>handle, <const Plan>plan, <const void*>alpha, <const void*>a, <const void*>gamma, <const void*>c, <void*>d, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_permutation(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_b, mode_b, intptr_t desc_compute) except? 0:
    """This function creates an operation descriptor for a tensor permutation.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc_a (intptr_t): The descriptor that holds information about the data type, modes, and strides of A.
        mode_a (object): Array of size desc_a->numModes that holds the names of the modes of A (e.g., if A_{a,b,c} => mode_a = {'a','b','c'}). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_a (Operator): Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged.
        desc_b (intptr_t): The descriptor that holds information about the data type, modes, and strides of B.
        mode_b (object): Array of size desc_b->numModes that holds the names of the modes of B. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        desc_compute (intptr_t): Determines the precision in which this operations is performed.

    Returns:
        intptr_t: This opaque struct gets allocated and filled with the information that encodes the requested permutation.

    .. seealso:: `cutensorCreatePermutation`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_a_
    get_resource_ptr[int32_t](_mode_a_, mode_a, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_b_
    get_resource_ptr[int32_t](_mode_b_, mode_b, <int32_t*>NULL)
    cdef OperationDescriptor desc
    with nogil:
        __status__ = cutensorCreatePermutation(<const Handle>handle, &desc, <const TensorDescriptor>desc_a, <const int32_t*>(_mode_a_.data()), <_Operator>op_a, <const TensorDescriptor>desc_b, <const int32_t*>(_mode_b_.data()), <const ComputeDescriptor>desc_compute)
    check_status(__status__)
    return <intptr_t>desc


cpdef permute(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t b, intptr_t stream):
    """Performs the tensor permutation that is encoded by ``plan`` (see ``cutensorCreatePermutation``).

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        plan (intptr_t): Opaque handle holding all information about the desired tensor reduction (created by ``cutensorCreatePermutation`` followed by ``cutensorCreatePlan``).
        alpha (intptr_t): Scaling factor for a (see cutensorOperationDescriptorGetattribute(desc, CUTENSOR_OPERaTION_SCaLaR_TYPE)). Pointer to the host memory. If alpha is zero, a is not read and the corresponding unary operator is not applied.
        a (intptr_t): Multi-mode tensor of type typea with nmodea modes. Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to D.
        b (intptr_t): Multi-mode tensor of type typeb with nmodeb modes. Pointer to the GPU-accessible memory.
        stream (intptr_t): The CUDa stream.

    .. seealso:: `cutensorPermute`
    """
    with nogil:
        __status__ = cutensorPermute(<const Handle>handle, <const Plan>plan, <const void*>alpha, <const void*>a, <void*>b, <const Stream>stream)
    check_status(__status__)


cpdef intptr_t create_contraction(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_b, mode_b, int op_b, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, intptr_t desc_compute) except? 0:
    """This function allocates a cutensorOperationDescriptor_t object that encodes a tensor contraction of the form .

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc_a (intptr_t): The descriptor that holds the information about the data type, modes and strides of A.
        mode_a (object): Array with 'nmode_a' entries that represent the modes of A. The mode_a[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_a (Operator): Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged.
        desc_b (intptr_t): The descriptor that holds information about the data type, modes, and strides of B.
        mode_b (object): Array with 'nmode_b' entries that represent the modes of B. The mode_b[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_b (Operator): Unary operator that will be applied to each element of B before it is further processed. The original data of this tensor remains unchanged.
        desc_c (intptr_t): Array with 'nmode_c' entries that represent the modes of C. The mode_c[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor.
        mode_c (object): The escriptor that holds information about the data type, modes, and strides of C. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_c (Operator): Unary operator that will be applied to each element of C before it is further processed. The original data of this tensor remains unchanged.
        desc_d (intptr_t): Array with 'nmode_d' entries that represent the modes of D (must be identical to mode_c for now). The mode_d[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor.
        mode_d (object): The descriptor that holds information about the data type, modes, and strides of D (must be identical to ``desc_c`` for now). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        desc_compute (intptr_t): Determines the precision in which this operations is performed.

    Returns:
        intptr_t: This opaque struct gets allocated and filled with the information that encodes the tensor contraction operation.

    .. seealso:: `cutensorCreateContraction`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_a_
    get_resource_ptr[int32_t](_mode_a_, mode_a, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_b_
    get_resource_ptr[int32_t](_mode_b_, mode_b, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_c_
    get_resource_ptr[int32_t](_mode_c_, mode_c, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_d_
    get_resource_ptr[int32_t](_mode_d_, mode_d, <int32_t*>NULL)
    cdef OperationDescriptor desc
    with nogil:
        __status__ = cutensorCreateContraction(<const Handle>handle, &desc, <const TensorDescriptor>desc_a, <const int32_t*>(_mode_a_.data()), <_Operator>op_a, <const TensorDescriptor>desc_b, <const int32_t*>(_mode_b_.data()), <_Operator>op_b, <const TensorDescriptor>desc_c, <const int32_t*>(_mode_c_.data()), <_Operator>op_c, <const TensorDescriptor>desc_d, <const int32_t*>(_mode_d_.data()), <const ComputeDescriptor>desc_compute)
    check_status(__status__)
    return <intptr_t>desc


cpdef destroy_operation_descriptor(intptr_t desc):
    """Frees all resources related to the provided descriptor.

    Args:
        desc (intptr_t): The cutensorOperationDescriptor_t object that will be deallocated.

    .. seealso:: `cutensorDestroyOperationDescriptor`
    """
    with nogil:
        __status__ = cutensorDestroyOperationDescriptor(<OperationDescriptor>desc)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict operation_descriptor_attribute_sizes = {
    CUTENSOR_OPERATION_DESCRIPTOR_TAG: _numpy.int32,
    CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE: _numpy.int32,
    CUTENSOR_OPERATION_DESCRIPTOR_FLOPS: _numpy.float32,
    CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES: _numpy.float32,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT: _numpy.uint32,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT: _numpy.uint32,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE: _numpy.uint64,
}

cpdef get_operation_descriptor_attribute_dtype(int attr):
    """Get the Python data type of the corresponding OperationDescriptorAttribute attribute.

    Args:
        attr (OperationDescriptorAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`operation_descriptor_get_attribute`, :func:`operation_descriptor_set_attribute`.
    """
    return operation_descriptor_attribute_sizes[attr]

###########################################################################


cpdef operation_descriptor_set_attribute(intptr_t handle, intptr_t desc, int attr, intptr_t buf, size_t size_in_bytes):
    """Set attribute of a cutensorOperationDescriptor_t object.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc (intptr_t): Operation descriptor that will be modified.
        attr (OperationDescriptorAttribute): Specifies the attribute that will be set.
        buf (intptr_t): This buffer (of size ``size_in_bytes``) determines the value to which ``attr`` will be set.
        size_in_bytes (size_t): Size of buf (in bytes).

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_operation_descriptor_attribute_dtype`.

    .. seealso:: `cutensorOperationDescriptorSetAttribute`
    """
    with nogil:
        __status__ = cutensorOperationDescriptorSetAttribute(<const Handle>handle, <OperationDescriptor>desc, <_OperationDescriptorAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(__status__)


cpdef operation_descriptor_get_attribute(intptr_t handle, intptr_t desc, int attr, intptr_t buf, size_t size_in_bytes):
    """This function retrieves an attribute of the provided cutensorOperationDescriptor_t object (see cutensorOperationDescriptorAttribute_t).

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc (intptr_t): The cutensorOperationDescriptor_t object whos attribute is queried.
        attr (OperationDescriptorAttribute): Specifies the attribute that will be retrieved.
        buf (intptr_t): This buffer (of size size_in_bytes) will hold the requested attribute of the provided cutensorOperationDescriptor_t object.
        size_in_bytes (size_t): Size of buf (in bytes); see cutensorOperationDescriptorAttribute_t for the exact size.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_operation_descriptor_attribute_dtype`.

    .. seealso:: `cutensorOperationDescriptorGetAttribute`
    """
    with nogil:
        __status__ = cutensorOperationDescriptorGetAttribute(<const Handle>handle, <OperationDescriptor>desc, <_OperationDescriptorAttribute>attr, <void*>buf, size_in_bytes)
    check_status(__status__)


cpdef intptr_t create_plan_preference(intptr_t handle, int algo, int jit_mode) except? 0:
    """Allocates the cutensorPlanPreference_t, enabling users to limit the applicable kernels for a given plan/operation.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        algo (Algo): Allows users to select a specific algorithm. CUTENSOR_ALGO_DEFAULT lets the heuristic choose the algorithm. Any value >= 0 selects a specific GEMM-like algorithm and deactivates the heuristic. If a specified algorithm is not supported CUTENSOR_STATUS_NOT_SUPPORTED is returned. See cutensorAlgo_t for additional choices.
        jit_mode (JitMode): Determines if cuTENSOR is allowed to use JIT-compiled kernels (leading to a longer plan-creation phase); see cutensorJitMode_t.

    Returns:
        intptr_t: Pointer to the structure holding the cutensorPlanPreference_t allocated by this function. See cutensorPlanPreference_t.

    .. seealso:: `cutensorCreatePlanPreference`
    """
    cdef PlanPreference pref
    with nogil:
        __status__ = cutensorCreatePlanPreference(<const Handle>handle, &pref, <_Algo>algo, <_JitMode>jit_mode)
    check_status(__status__)
    return <intptr_t>pref


cpdef destroy_plan_preference(intptr_t pref):
    """Frees all resources related to the provided preference.

    Args:
        pref (intptr_t): The cutensorPlanPreference_t object that will be deallocated.

    .. seealso:: `cutensorDestroyPlanPreference`
    """
    with nogil:
        __status__ = cutensorDestroyPlanPreference(<PlanPreference>pref)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict plan_preference_attribute_sizes = {
    CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE: _numpy.int32,
    CUTENSOR_PLAN_PREFERENCE_CACHE_MODE: _numpy.int32,
    CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT: _numpy.int32,
    CUTENSOR_PLAN_PREFERENCE_ALGO: _numpy.int32,
    CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK: _numpy.int32,
    CUTENSOR_PLAN_PREFERENCE_JIT: _numpy.int32,
}

cpdef get_plan_preference_attribute_dtype(int attr):
    """Get the Python data type of the corresponding PlanPreferenceAttribute attribute.

    Args:
        attr (PlanPreferenceAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`plan_preference_set_attribute`.
    """
    return plan_preference_attribute_sizes[attr]

###########################################################################


cpdef plan_preference_set_attribute(intptr_t handle, intptr_t pref, int attr, intptr_t buf, size_t size_in_bytes):
    """Set attribute of a cutensorPlanPreference_t object.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        pref (intptr_t): This opaque struct restricts the search space of viable candidates.
        attr (PlanPreferenceAttribute): Specifies the attribute that will be set.
        buf (intptr_t): This buffer (of size size_in_bytes) determines the value to which ``attr`` will be set.
        size_in_bytes (size_t): Size of buf (in bytes); see cutensorPlanPreferenceAttribute_t for the exact size.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_plan_preference_attribute_dtype`.

    .. seealso:: `cutensorPlanPreferenceSetAttribute`
    """
    with nogil:
        __status__ = cutensorPlanPreferenceSetAttribute(<const Handle>handle, <PlanPreference>pref, <_PlanPreferenceAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(__status__)


######################### Python specific utility #########################

cdef dict plan_attribute_sizes = {
    CUTENSOR_PLAN_REQUIRED_WORKSPACE: _numpy.uint64,
}

cpdef get_plan_attribute_dtype(int attr):
    """Get the Python data type of the corresponding PlanAttribute attribute.

    Args:
        attr (PlanAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`plan_get_attribute`.
    """
    return plan_attribute_sizes[attr]

###########################################################################


cpdef plan_get_attribute(intptr_t handle, intptr_t plan, int attr, intptr_t buf, size_t size_in_bytes):
    """Retrieves information about an already-created plan (see cutensorPlanAttribute_t).

    Args:
        handle (intptr_t): Denotes an already-created plan (e.g., via ``cutensorCreatePlan`` or cutensorCreatePlanAutotuned).
        plan (intptr_t): Requested attribute.
        attr (PlanAttribute): On successful exit: Holds the information of the requested attribute.
        buf (intptr_t): size of ``buf`` in bytes.
        size_in_bytes (size_t): The operation completed successfully.

    .. note:: To compute the attribute size, use the itemsize of the corresponding data
        type, which can be queried using :func:`get_plan_attribute_dtype`.

    .. seealso:: `cutensorPlanGetAttribute`
    """
    with nogil:
        __status__ = cutensorPlanGetAttribute(<const Handle>handle, <const Plan>plan, <_PlanAttribute>attr, <void*>buf, size_in_bytes)
    check_status(__status__)


cpdef uint64_t estimate_workspace_size(intptr_t handle, intptr_t desc, intptr_t plan_pref, int workspace_pref) except? -1:
    """Determines the required workspaceSize for the given operation encoded by ``desc``.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc (intptr_t): This opaque struct encodes the operation.
        plan_pref (intptr_t): This opaque struct restricts the space of viable candidates.
        workspace_pref (int): This parameter influences the size of the workspace; see cutensorWorksizePreference_t for details.

    Returns:
        uint64_t: The workspace size (in bytes) that is required for the given operation.

    .. seealso:: `cutensorEstimateWorkspaceSize`
    """
    cdef uint64_t workspace_size_estimate
    with nogil:
        __status__ = cutensorEstimateWorkspaceSize(<const Handle>handle, <const OperationDescriptor>desc, <const PlanPreference>plan_pref, <const _WorksizePreference>workspace_pref, &workspace_size_estimate)
    check_status(__status__)
    return workspace_size_estimate


cpdef intptr_t create_plan(intptr_t handle, intptr_t desc, intptr_t pref, uint64_t workspace_size_limit) except? 0:
    """This function allocates a cutensorPlan_t object, selects an appropriate kernel for a given operation (encoded by ``desc``) and prepares a plan that encodes the execution.

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc (intptr_t): This opaque struct encodes the given operation (see ``cutensorCreateContraction``, ``cutensorCreateReduction``, ``cutensorCreatePermutation``, ``cutensorCreateElementwiseBinary``, ``cutensorCreateElementwiseTrinary``, or ``cutensorCreateContractionTrinary``).
        pref (intptr_t): This opaque struct is used to restrict the space of applicable candidates/kernels (see ``cutensorCreatePlanPreference`` or cutensorPlanPreferenceAttribute_t). May be ``nullptr``, in that case default choices are assumed. Block-sparse contractions currently only support these default settings and ignore other supplied preferences.
        workspace_size_limit (uint64_t): Denotes the maximal workspace that the corresponding operation is allowed to use (see ``cutensorEstimateWorkspaceSize``).

    Returns:
        intptr_t: Pointer to the data structure created by this function that holds all information (e.g., selected kernel) necessary to perform the desired operation.

    .. seealso:: `cutensorCreatePlan`
    """
    cdef Plan plan
    with nogil:
        __status__ = cutensorCreatePlan(<const Handle>handle, &plan, <const OperationDescriptor>desc, <const PlanPreference>pref, workspace_size_limit)
    check_status(__status__)
    return <intptr_t>plan


cpdef destroy_plan(intptr_t plan):
    """Frees all resources related to the provided plan.

    Args:
        plan (intptr_t): The cutensorPlan_t object that will be deallocated.

    .. seealso:: `cutensorDestroyPlan`
    """
    with nogil:
        __status__ = cutensorDestroyPlan(<Plan>plan)
    check_status(__status__)


cpdef contract(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t b, intptr_t beta, intptr_t c, intptr_t d, intptr_t workspace, uint64_t workspace_size, intptr_t stream):
    """This routine computes the tensor contraction .

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        plan (intptr_t): Opaque handle holding the contraction execution plan (created by ``cutensorcreatecontraction`` followed by ``cutensorcreatePlan``).
        alpha (intptr_t): Scaling for a*b. Its data type is determined by 'desccompute' (see cutensorOperationdescriptorGetattribute(desc, cUTENSOR_OPERaTION_ScaLaR_TYPE)). Pointer to the host memory.
        a (intptr_t): Pointer to the data corresponding to a. Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to d.
        b (intptr_t): Pointer to the data corresponding to b. Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to d.
        beta (intptr_t): Scaling for c. Its data type is determined by 'desccompute' (see cutensorOperationdescriptorGetattribute(desc, cUTENSOR_OPERaTION_ScaLaR_TYPE)). Pointer to the host memory.
        c (intptr_t): Pointer to the data corresponding to c. Pointer to the GPU-accessible memory.
        d (intptr_t): Pointer to the data corresponding to d. Pointer to the GPU-accessible memory.
        workspace (intptr_t): Optional parameter that may be NULL. This pointer provides additional workspace, in device memory, to the library for additional optimizations; the workspace must be aligned to 256 bytes (i.e., the default alignment of cudaMalloc).
        workspace_size (uint64_t): Size of the workspace array in bytes; please refer to ``cutensorEstimateWorkspaceSize`` to query the required workspace. While ``cutensorcontract`` does not strictly require a workspace for the contraction, it is still recommended to provided some small workspace (e.g., 128 Mb).
        stream (intptr_t): The cUda stream in which all the computation is performed.

    .. seealso:: `cutensorContract`
    """
    with nogil:
        __status__ = cutensorContract(<const Handle>handle, <const Plan>plan, <const void*>alpha, <const void*>a, <const void*>b, <const void*>beta, <const void*>c, <void*>d, <void*>workspace, workspace_size, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_reduction(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, int op_reduce, intptr_t desc_compute) except? 0:
    """Creates a cutensorOperatorDescriptor_t object that encodes a tensor reduction of the form .

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc_a (intptr_t): The descriptor that holds the information about the data type, modes and strides of A.
        mode_a (object): Array with 'nmode_a' entries that represent the modes of A. mode_a[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to ``cutensorCreateTensorDescriptor``. Modes that only appear in mode_a but not in mode_c are reduced (contracted). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_a (Operator): Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged.
        desc_c (intptr_t): The descriptor that holds the information about the data type, modes and strides of C.
        mode_c (object): Array with 'nmode_c' entries that represent the modes of C. mode_c[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to ``cutensorCreateTensorDescriptor``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_c (Operator): Unary operator that will be applied to each element of C before it is further processed. The original data of this tensor remains unchanged.
        desc_d (intptr_t): Must be identical to desc_c for now.
        mode_d (object): Must be identical to mode_c for now. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_reduce (Operator): binary operator used to reduce elements of A.
        desc_compute (intptr_t): All arithmetic is performed using this data type (i.e., it affects the accuracy and performance).

    Returns:
        intptr_t: This opaque struct gets allocated and filled with the information that encodes the requested tensor reduction operation.

    .. seealso:: `cutensorCreateReduction`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_a_
    get_resource_ptr[int32_t](_mode_a_, mode_a, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_c_
    get_resource_ptr[int32_t](_mode_c_, mode_c, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_d_
    get_resource_ptr[int32_t](_mode_d_, mode_d, <int32_t*>NULL)
    cdef OperationDescriptor desc
    with nogil:
        __status__ = cutensorCreateReduction(<const Handle>handle, &desc, <const TensorDescriptor>desc_a, <const int32_t*>(_mode_a_.data()), <_Operator>op_a, <const TensorDescriptor>desc_c, <const int32_t*>(_mode_c_.data()), <_Operator>op_c, <const TensorDescriptor>desc_d, <const int32_t*>(_mode_d_.data()), <_Operator>op_reduce, <const ComputeDescriptor>desc_compute)
    check_status(__status__)
    return <intptr_t>desc


cpdef reduce(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t beta, intptr_t c, intptr_t d, intptr_t workspace, uint64_t workspace_size, intptr_t stream):
    """Performs the tensor reduction that is encoded by ``plan`` (see ``cutensorcreateReduction``).

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        plan (intptr_t): Opaque handle holding the reduction execution plan (created by ``cutensorcreateReduction`` followed by ``cutensorcreatePlan``).
        alpha (intptr_t): Scaling for a. Its data type is determined by 'desccompute' (see cutensorOperationdescriptorGetattribute(desc, cUTENSOR_OPERaTION_ScaLaR_TYPE)). Pointer to the host memory.
        a (intptr_t): Pointer to the data corresponding to a in device memory. Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to d.
        beta (intptr_t): Scaling for c. Its data type is determined by 'desccompute' (see cutensorOperationdescriptorGetattribute(desc, cUTENSOR_OPERaTION_ScaLaR_TYPE)). Pointer to the host memory.
        c (intptr_t): Pointer to the data corresponding to c in device memory. Pointer to the GPU-accessible memory.
        d (intptr_t): Pointer to the data corresponding to c in device memory. Pointer to the GPU-accessible memory.
        workspace (intptr_t): Scratchpad (device) memory of size --at least-- ``workspace_size`` bytes; the workspace must be aligned to 256 bytes (i.e., the default alignment of cudaMalloc).
        workspace_size (uint64_t): Please use :func:`estimate_workspace_size` to query the required workspace.
        stream (intptr_t): The cUda stream in which all the computation is performed.

    .. seealso:: `cutensorReduce`
    """
    with nogil:
        __status__ = cutensorReduce(<const Handle>handle, <const Plan>plan, <const void*>alpha, <const void*>a, <const void*>beta, <const void*>c, <void*>d, <void*>workspace, workspace_size, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_contraction_trinary(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_b, mode_b, int op_b, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, int op_d, intptr_t desc_e, mode_e, intptr_t desc_compute) except? 0:
    """This function allocates a cutensorOperationDescriptor_t object that encodes a tensor contraction of the form .

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc_a (intptr_t): The descriptor that holds the information about the data type, modes and strides of A.
        mode_a (object): Array with 'nmode_a' entries that represent the modes of A. The mode_a[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_a (Operator): Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged.
        desc_b (intptr_t): The descriptor that holds information about the data type, modes, and strides of B.
        mode_b (object): Array with 'nmode_b' entries that represent the modes of B. The mode_b[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_b (Operator): Unary operator that will be applied to each element of B before it is further processed. The original data of this tensor remains unchanged.
        desc_c (intptr_t): The escriptor that holds information about the data type, modes, and strides of C.
        mode_c (object): Array with 'nmode_c' entries that represent the modes of C. The mode_c[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_c (Operator): Unary operator that will be applied to each element of C before it is further processed. The original data of this tensor remains unchanged.
        desc_d (intptr_t): The escriptor that holds information about the data type, modes, and strides of D.
        mode_d (object): Array with 'nmode_d' entries that represent the modes of D. The mode_d[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_d (Operator): Unary operator that will be applied to each element of D before it is further processed. The original data of this tensor remains unchanged.
        desc_e (intptr_t): Array with 'nmode_e' entries that represent the modes of E (must be identical to mode_d for now). The mode_e[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor.
        mode_e (object): The descriptor that holds information about the data type, modes, and strides of E (must be identical to ``desc_d`` for now). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        desc_compute (intptr_t): Determines the precision in which this operations is performed.

    Returns:
        intptr_t: This opaque struct gets allocated and filled with the information that encodes the tensor contraction operation.

    .. seealso:: `cutensorCreateContractionTrinary`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_a_
    get_resource_ptr[int32_t](_mode_a_, mode_a, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_b_
    get_resource_ptr[int32_t](_mode_b_, mode_b, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_c_
    get_resource_ptr[int32_t](_mode_c_, mode_c, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_d_
    get_resource_ptr[int32_t](_mode_d_, mode_d, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_e_
    get_resource_ptr[int32_t](_mode_e_, mode_e, <int32_t*>NULL)
    cdef OperationDescriptor desc
    with nogil:
        __status__ = cutensorCreateContractionTrinary(<const Handle>handle, &desc, <const TensorDescriptor>desc_a, <const int32_t*>(_mode_a_.data()), <_Operator>op_a, <const TensorDescriptor>desc_b, <const int32_t*>(_mode_b_.data()), <_Operator>op_b, <const TensorDescriptor>desc_c, <const int32_t*>(_mode_c_.data()), <_Operator>op_c, <const TensorDescriptor>desc_d, <const int32_t*>(_mode_d_.data()), <_Operator>op_d, <const TensorDescriptor>desc_e, <const int32_t*>(_mode_e_.data()), <const ComputeDescriptor>desc_compute)
    check_status(__status__)
    return <intptr_t>desc


cpdef contract_trinary(intptr_t handle, intptr_t plan, intptr_t alpha, intptr_t a, intptr_t b, intptr_t c, intptr_t beta, intptr_t d, intptr_t e, intptr_t workspace, uint64_t workspace_size, intptr_t stream):
    """This routine computes the tensor contraction .

    Args:
        handle (intptr_t): Opaque handle holding cuTeNSOR's library context.
        plan (intptr_t): Opaque handle holding the contraction execution plan (created by ``cutensorcreatecontractionTrinary`` followed by ``cutensorcreatePlan``).
        alpha (intptr_t): Scaling for a*b*c. Its data type is determined by 'desccompute' (see cutensorOperationdescriptorGetattribute(desc, cUTeNSOR_OPeRaTION_ScaLaR_TYPe)). Pointer to the host memory.
        a (intptr_t): Pointer to the data corresponding to a. Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to e.
        b (intptr_t): Pointer to the data corresponding to b. Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to e.
        c (intptr_t): Pointer to the data corresponding to c. Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to e.
        beta (intptr_t): Scaling for d. Its data type is determined by 'desccompute' (see cutensorOperationdescriptorGetattribute(desc, cUTeNSOR_OPeRaTION_ScaLaR_TYPe)). Pointer to the host memory.
        d (intptr_t): Pointer to the data corresponding to d. Pointer to the GPU-accessible memory.
        e (intptr_t): Pointer to the data corresponding to e. Pointer to the GPU-accessible memory.
        workspace (intptr_t): Optional parameter that may be NULL. This pointer provides additional workspace, in device memory, to the library for additional optimizations; the workspace must be aligned to 256 bytes (i.e., the default alignment of cudaMalloc).
        workspace_size (uint64_t): Size of the workspace array in bytes; please refer to ``cutensorestimateWorkspaceSize`` to query the required workspace. While ``cutensorcontract`` does not strictly require a workspace for the contraction, it is still recommended to provided some small workspace (e.g., 128 Mb).
        stream (intptr_t): The cUda stream in which all the computation is performed.

    .. seealso:: `cutensorContractTrinary`
    """
    with nogil:
        __status__ = cutensorContractTrinary(<const Handle>handle, <const Plan>plan, <const void*>alpha, <const void*>a, <const void*>b, <const void*>c, <const void*>beta, <const void*>d, <void*>e, <void*>workspace, workspace_size, <Stream>stream)
    check_status(__status__)


cpdef intptr_t create_block_sparse_tensor_descriptor(intptr_t handle, uint32_t num_modes, uint64_t num_non_zero_blocks, num_sections_per_mode, extent, non_zero_coordinates, stride, int data_type) except? 0:
    """Create a block-sparse tensor descriptor.

    Args:
        handle (intptr_t): The library handle.
        num_modes (uint32_t): The number of modes. Currently, a maximum of 8 modes is supported.
        num_non_zero_blocks (uint64_t): The number of non-zero blocks in the block-sparse tensor.
        num_sections_per_mode (object): The number of sections of each mode (host array of size ``num_modes``). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``uint32_t``.

        extent (object): The extents of the sections of each mode (host array of size ``\sum_i^num_modes(num_sections_per_mode[i])``). First come the extents of the sections of the first mode, then the extents of the sections of the second mode, and so forth. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        non_zero_coordinates (object): Block-coordinates of each non-zero block (host array of size ``num_modes`` x ``num_non_zero_blocks`` Blocks can be specified in any order, however, that order must be consistent with stride and alignmentRequirement arrays. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        stride (object): The strides of each dense block (either nullptr or a host array of size ``num_modes`` x ``num_non_zero_blocks``). First the strides of the first block, then the strides of the second block, with the blocks in the same order as in non_zero_coordinates. Passing nullptr means contiguous column-major order for each block. Moreover, the strides need to be compatible in the following sense: Suppose you sort the strides of the first block, such that they are ascending; this sorting results in a permutation. If you apply this permutation to the strides of any other block, the result needs to be sorted as well. As an example:. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        data_type (int): Data type of the stored entries. We assume the same datatype for each block. Currently, the only supported values are CUDA_C_64F, CUDA_C_32F, CUDA_R_64F, and CUDA_R_32F.

    Returns:
        intptr_t: The resulting block-sparse tensor descriptor.

    .. seealso:: `cutensorCreateBlockSparseTensorDescriptor`
    """
    cdef nullable_unique_ptr[ vector[uint32_t] ] _num_sections_per_mode_
    get_resource_ptr[uint32_t](_num_sections_per_mode_, num_sections_per_mode, <uint32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _extent_
    get_resource_ptr[int64_t](_extent_, extent, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _non_zero_coordinates_
    get_resource_ptr[int32_t](_non_zero_coordinates_, non_zero_coordinates, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _stride_
    get_resource_ptr[int64_t](_stride_, stride, <int64_t*>NULL)
    cdef BlockSparseTensorDescriptor desc
    with nogil:
        __status__ = cutensorCreateBlockSparseTensorDescriptor(<Handle>handle, &desc, <const uint32_t>num_modes, <const uint64_t>num_non_zero_blocks, <const uint32_t*>(_num_sections_per_mode_.data()), <const int64_t*>(_extent_.data()), <const int32_t*>(_non_zero_coordinates_.data()), <const int64_t*>(_stride_.data()), <DataType>data_type)
    check_status(__status__)
    return <intptr_t>desc


cpdef destroy_block_sparse_tensor_descriptor(intptr_t desc):
    """Frees all resources related to the provided block-sparse tensor descriptor.

    Args:
        desc (intptr_t): The cutensorBlockSparseTensorDescrptor_t object that will be deallocated.

    .. seealso:: `cutensorDestroyBlockSparseTensorDescriptor`
    """
    with nogil:
        __status__ = cutensorDestroyBlockSparseTensorDescriptor(<BlockSparseTensorDescriptor>desc)
    check_status(__status__)


cpdef intptr_t create_block_sparse_contraction(intptr_t handle, intptr_t desc_a, mode_a, int op_a, intptr_t desc_b, mode_b, int op_b, intptr_t desc_c, mode_c, int op_c, intptr_t desc_d, mode_d, intptr_t desc_compute) except? 0:
    """This function allocates a cutensorOperationDescriptor_t object that encodes a block-sparse tensor contraction of the form .

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        desc_a (intptr_t): The descriptor that holds the information about the data type, modes, sections, section extents, strides, and non-zero blocks of A.
        mode_a (object): Array with 'nmode_a' entries that represent the modes of A. Sections, i.e., block-sizes, must match among the involved block-sparse tensors. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_a (Operator): Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged. Currently, only CUTENSOR_OP_IDENTITY is supported.
        desc_b (intptr_t): The descriptor that holds information about the the data type, modes, sections, section extents, strides, and non-zero blocks of B.
        mode_b (object): Array with 'nmode_b' entries that represent the modes of B. Sections, i.e., block-sizes, must match among the involved block-sparse tensors. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_b (Operator): Unary operator that will be applied to each element of B before it is further processed. The original data of this tensor remains unchanged. Currently, only CUTENSOR_OP_IDENTITY is supported.
        desc_c (intptr_t): Array with 'nmode_c' entries that represent the modes of C. Sections, i.e., block-sizes, must match among the involved block-sparse tensors.
        mode_c (object): The descriptor that holds information about the data type, modes, sections, section extents, strides, and non-zero blocks of C. Note that the block-sparsity pattern of C (the nonZeroCoordinates[] array used to create the decriptor) of C must be identical to that of D; and it is this block-sparsity pattern that determines which parts of the results are computed; no fill-in is allocated or computed. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        op_c (Operator): Unary operator that will be applied to each element of C before it is further processed. The original data of this tensor remains unchanged. Currently, only CUTENSOR_OP_IDENTITY is supported.
        desc_d (intptr_t): For now, this must be the same opaque pointer as desc_c, and the layouts of C and D must be identical.
        mode_d (object): Array with 'nmode_d' entries that represent the modes of D (must be identical to mode_c for now). It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int32_t``.

        desc_compute (intptr_t): Datatype of for the intermediate computation of typeCompute T = A * B.

    Returns:
        intptr_t: This opaque struct gets allocated and filled with the information that encodes the tensor contraction operation.

    .. seealso:: `cutensorCreateBlockSparseContraction`
    """
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_a_
    get_resource_ptr[int32_t](_mode_a_, mode_a, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_b_
    get_resource_ptr[int32_t](_mode_b_, mode_b, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_c_
    get_resource_ptr[int32_t](_mode_c_, mode_c, <int32_t*>NULL)
    cdef nullable_unique_ptr[ vector[int32_t] ] _mode_d_
    get_resource_ptr[int32_t](_mode_d_, mode_d, <int32_t*>NULL)
    cdef OperationDescriptor desc
    with nogil:
        __status__ = cutensorCreateBlockSparseContraction(<const Handle>handle, &desc, <const BlockSparseTensorDescriptor>desc_a, <const int32_t*>(_mode_a_.data()), <_Operator>op_a, <const BlockSparseTensorDescriptor>desc_b, <const int32_t*>(_mode_b_.data()), <_Operator>op_b, <const BlockSparseTensorDescriptor>desc_c, <const int32_t*>(_mode_c_.data()), <_Operator>op_c, <const BlockSparseTensorDescriptor>desc_d, <const int32_t*>(_mode_d_.data()), <const ComputeDescriptor>desc_compute)
    check_status(__status__)
    return <intptr_t>desc


cpdef block_sparse_contract(intptr_t handle, intptr_t plan, intptr_t alpha, a, b, intptr_t beta, c, d, intptr_t workspace, uint64_t workspace_size, intptr_t stream):
    """This routine computes the block-sparse tensor contraction .

    Args:
        handle (intptr_t): Opaque handle holding cuTENSOR's library context.
        plan (intptr_t): Opaque handle holding the contraction execution plan (created by ``cutensorcreateblockSparsecontraction`` followed by ``cutensorcreatePlan``).
        alpha (intptr_t): Scaling for a*b. Its data type is determined by 'desccompute' (see ``cutensorcreateblockSparsecontraction``). Pointer to host memory.
        a (object): Host-array of size numNonZeroblocks(a), containing pointers to GPU-accessible memory, corresponding the blocks of a. The data accessed via these pointers must not overlap with the elements written to d. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        b (object): Host-array of size numNonZeroblocks(b), containing pointers to GPU-accessible memory, corresponding the blocks of b. The data accessed via these pointers must not overlap with the elements written to d. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        beta (intptr_t): Scaling for c. Its data type is determined by 'desccompute' (see ``cutensorcreateblockSparsecontraction``). Pointer to host memory.
        c (object): Host-array of size numNonZeroblocks(c), containing pointers to GPU-accessible memory, corresponding the blocks of c. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        d (object): Host-array of size numNonZeroblocks(d), containing pointers to GPU-accessible memory, corresponding the blocks of d. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).

        workspace (intptr_t): This pointer provides the required workspace in device memory. The workspace must be aligned to 256 bytes (i.e., the default alignment of cudaMalloc).
        workspace_size (uint64_t): Size of the workspace array in bytes; please refer to ``cutensorEstimateWorkspaceSize`` to query the required workspace. For block-sparse contractions, this estimate is exact.
        stream (intptr_t): The cUda stream to which all of the computation is synchronised.

    .. seealso:: `cutensorBlockSparseContract`
    """
    cdef nullable_unique_ptr[ vector[void*] ] _a_
    get_resource_ptrs[void](_a_, a, <void*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _b_
    get_resource_ptrs[void](_b_, b, <void*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _c_
    get_resource_ptrs[void](_c_, c, <void*>NULL)
    cdef nullable_unique_ptr[ vector[void*] ] _d_
    get_resource_ptrs[void](_d_, d, <void*>NULL)
    with nogil:
        __status__ = cutensorBlockSparseContract(<const Handle>handle, <const Plan>plan, <const void*>alpha, <const void* const*>(_a_.data()), <const void* const*>(_b_.data()), <const void*>beta, <const void* const*>(_c_.data()), <void* const*>(_d_.data()), <void*>workspace, workspace_size, <Stream>stream)
    check_status(__status__)


cpdef str get_error_string(int error):
    """Returns the description string for an error code.

    Args:
        error (int): Error code to convert to string.

    .. seealso:: `cutensorGetErrorString`
    """
    cdef bytes _output_
    _output_ = cutensorGetErrorString(<const _Status>error)
    return _output_.decode()


cpdef size_t get_version() except? 0:
    """Returns Version number of the CUTENSOR library.

    .. seealso:: `cutensorGetVersion`
    """
    return cutensorGetVersion()


cpdef size_t get_cudart_version() except? 0:
    """Returns version number of the CUDA runtime that cuTENSOR was compiled against.

    .. seealso:: `cutensorGetCudartVersion`
    """
    return cutensorGetCudartVersion()


cpdef logger_set_file(intptr_t file):
    """This function sets the logging output file.

    Args:
        file (intptr_t): An open file with write permission.

    .. seealso:: `cutensorLoggerSetFile`
    """
    with nogil:
        __status__ = cutensorLoggerSetFile(<FILE*>file)
    check_status(__status__)


cpdef logger_open_file(log_file):
    """This function opens a logging output file in the given path.

    Args:
        log_file (str): Path to the logging output file.

    .. seealso:: `cutensorLoggerOpenFile`
    """
    if not isinstance(log_file, str):
        raise TypeError("log_file must be a Python str")
    cdef bytes _temp_log_file_ = (<str>log_file).encode()
    cdef char* _log_file_ = _temp_log_file_
    with nogil:
        __status__ = cutensorLoggerOpenFile(<const char*>_log_file_)
    check_status(__status__)


cpdef logger_set_level(int32_t level):
    """This function sets the value of the logging level.

    Args:
        level (int32_t): Log level, should be one of the following:.

    .. seealso:: `cutensorLoggerSetLevel`
    """
    with nogil:
        __status__ = cutensorLoggerSetLevel(level)
    check_status(__status__)


cpdef logger_set_mask(int32_t mask):
    """This function sets the value of the log mask.

    Args:
        mask (int32_t): Log mask, the bitwise OR of the following:.

    .. seealso:: `cutensorLoggerSetMask`
    """
    with nogil:
        __status__ = cutensorLoggerSetMask(mask)
    check_status(__status__)


cpdef logger_force_disable():
    """This function disables logging for the entire run.

    .. seealso:: `cutensorLoggerForceDisable`
    """
    with nogil:
        __status__ = cutensorLoggerForceDisable()
    check_status(__status__)


###############################################################################
