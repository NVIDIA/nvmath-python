# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.4.1. Do not modify it directly.

cimport cython  # NOQA
from libcpp.vector cimport vector

from enum import IntEnum as _IntEnum

import numpy as _numpy


###############################################################################
# Enum
###############################################################################

class MatmulTile(_IntEnum):
    """See `cublasLtMatmulTile_t`."""
    TILE_UNDEFINED = CUBLASLT_MATMUL_TILE_UNDEFINED
    TILE_8x8 = CUBLASLT_MATMUL_TILE_8x8
    TILE_8x16 = CUBLASLT_MATMUL_TILE_8x16
    TILE_16x8 = CUBLASLT_MATMUL_TILE_16x8
    TILE_8x32 = CUBLASLT_MATMUL_TILE_8x32
    TILE_16x16 = CUBLASLT_MATMUL_TILE_16x16
    TILE_32x8 = CUBLASLT_MATMUL_TILE_32x8
    TILE_8x64 = CUBLASLT_MATMUL_TILE_8x64
    TILE_16x32 = CUBLASLT_MATMUL_TILE_16x32
    TILE_32x16 = CUBLASLT_MATMUL_TILE_32x16
    TILE_64x8 = CUBLASLT_MATMUL_TILE_64x8
    TILE_32x32 = CUBLASLT_MATMUL_TILE_32x32
    TILE_32x64 = CUBLASLT_MATMUL_TILE_32x64
    TILE_64x32 = CUBLASLT_MATMUL_TILE_64x32
    TILE_32x128 = CUBLASLT_MATMUL_TILE_32x128
    TILE_64x64 = CUBLASLT_MATMUL_TILE_64x64
    TILE_128x32 = CUBLASLT_MATMUL_TILE_128x32
    TILE_64x128 = CUBLASLT_MATMUL_TILE_64x128
    TILE_128x64 = CUBLASLT_MATMUL_TILE_128x64
    TILE_64x256 = CUBLASLT_MATMUL_TILE_64x256
    TILE_128x128 = CUBLASLT_MATMUL_TILE_128x128
    TILE_256x64 = CUBLASLT_MATMUL_TILE_256x64
    TILE_64x512 = CUBLASLT_MATMUL_TILE_64x512
    TILE_128x256 = CUBLASLT_MATMUL_TILE_128x256
    TILE_256x128 = CUBLASLT_MATMUL_TILE_256x128
    TILE_512x64 = CUBLASLT_MATMUL_TILE_512x64
    TILE_64x96 = CUBLASLT_MATMUL_TILE_64x96
    TILE_96x64 = CUBLASLT_MATMUL_TILE_96x64
    TILE_96x128 = CUBLASLT_MATMUL_TILE_96x128
    TILE_128x160 = CUBLASLT_MATMUL_TILE_128x160
    TILE_160x128 = CUBLASLT_MATMUL_TILE_160x128
    TILE_192x128 = CUBLASLT_MATMUL_TILE_192x128
    TILE_128x192 = CUBLASLT_MATMUL_TILE_128x192
    TILE_128x96 = CUBLASLT_MATMUL_TILE_128x96
    TILE_32x256 = CUBLASLT_MATMUL_TILE_32x256
    TILE_256x32 = CUBLASLT_MATMUL_TILE_256x32

class MatmulStages(_IntEnum):
    """See `cublasLtMatmulStages_t`."""
    STAGES_UNDEFINED = CUBLASLT_MATMUL_STAGES_UNDEFINED
    STAGES_16x1 = CUBLASLT_MATMUL_STAGES_16x1
    STAGES_16x2 = CUBLASLT_MATMUL_STAGES_16x2
    STAGES_16x3 = CUBLASLT_MATMUL_STAGES_16x3
    STAGES_16x4 = CUBLASLT_MATMUL_STAGES_16x4
    STAGES_16x5 = CUBLASLT_MATMUL_STAGES_16x5
    STAGES_16x6 = CUBLASLT_MATMUL_STAGES_16x6
    STAGES_32x1 = CUBLASLT_MATMUL_STAGES_32x1
    STAGES_32x2 = CUBLASLT_MATMUL_STAGES_32x2
    STAGES_32x3 = CUBLASLT_MATMUL_STAGES_32x3
    STAGES_32x4 = CUBLASLT_MATMUL_STAGES_32x4
    STAGES_32x5 = CUBLASLT_MATMUL_STAGES_32x5
    STAGES_32x6 = CUBLASLT_MATMUL_STAGES_32x6
    STAGES_64x1 = CUBLASLT_MATMUL_STAGES_64x1
    STAGES_64x2 = CUBLASLT_MATMUL_STAGES_64x2
    STAGES_64x3 = CUBLASLT_MATMUL_STAGES_64x3
    STAGES_64x4 = CUBLASLT_MATMUL_STAGES_64x4
    STAGES_64x5 = CUBLASLT_MATMUL_STAGES_64x5
    STAGES_64x6 = CUBLASLT_MATMUL_STAGES_64x6
    STAGES_128x1 = CUBLASLT_MATMUL_STAGES_128x1
    STAGES_128x2 = CUBLASLT_MATMUL_STAGES_128x2
    STAGES_128x3 = CUBLASLT_MATMUL_STAGES_128x3
    STAGES_128x4 = CUBLASLT_MATMUL_STAGES_128x4
    STAGES_128x5 = CUBLASLT_MATMUL_STAGES_128x5
    STAGES_128x6 = CUBLASLT_MATMUL_STAGES_128x6
    STAGES_32x10 = CUBLASLT_MATMUL_STAGES_32x10
    STAGES_8x4 = CUBLASLT_MATMUL_STAGES_8x4
    STAGES_16x10 = CUBLASLT_MATMUL_STAGES_16x10
    STAGES_8x5 = CUBLASLT_MATMUL_STAGES_8x5
    STAGES_8x3 = CUBLASLT_MATMUL_STAGES_8x3
    STAGES_8xAUTO = CUBLASLT_MATMUL_STAGES_8xAUTO
    STAGES_16xAUTO = CUBLASLT_MATMUL_STAGES_16xAUTO
    STAGES_32xAUTO = CUBLASLT_MATMUL_STAGES_32xAUTO
    STAGES_64xAUTO = CUBLASLT_MATMUL_STAGES_64xAUTO
    STAGES_128xAUTO = CUBLASLT_MATMUL_STAGES_128xAUTO
    STAGES_16x80 = CUBLASLT_MATMUL_STAGES_16x80
    STAGES_64x80 = CUBLASLT_MATMUL_STAGES_64x80

class PointerMode(_IntEnum):
    """See `cublasLtPointerMode_t`."""
    HOST = CUBLASLT_POINTER_MODE_HOST
    DEVICE = CUBLASLT_POINTER_MODE_DEVICE
    DEVICE_VECTOR = CUBLASLT_POINTER_MODE_DEVICE_VECTOR
    ALPHA_DEVICE_VECTOR_BETA_ZERO = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO
    ALPHA_DEVICE_VECTOR_BETA_HOST = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST

class PointerModeMask(_IntEnum):
    """See `cublasLtPointerModeMask_t`."""
    HOST = CUBLASLT_POINTER_MODE_MASK_HOST
    DEVICE = CUBLASLT_POINTER_MODE_MASK_DEVICE
    DEVICE_VECTOR = CUBLASLT_POINTER_MODE_MASK_DEVICE_VECTOR
    ALPHA_DEVICE_VECTOR_BETA_ZERO = CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO
    ALPHA_DEVICE_VECTOR_BETA_HOST = CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST
    NO_FILTERING = CUBLASLT_POINTER_MODE_MASK_NO_FILTERING

class Order(_IntEnum):
    """See `cublasLtOrder_t`."""
    COL = CUBLASLT_ORDER_COL
    ROW = CUBLASLT_ORDER_ROW
    COL32 = CUBLASLT_ORDER_COL32
    COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C
    COL32_2R_4R4 = CUBLASLT_ORDER_COL32_2R_4R4

class MatrixLayoutAttribute(_IntEnum):
    """See `cublasLtMatrixLayoutAttribute_t`."""
    TYPE = CUBLASLT_MATRIX_LAYOUT_TYPE
    ORDER = CUBLASLT_MATRIX_LAYOUT_ORDER
    ROWS = CUBLASLT_MATRIX_LAYOUT_ROWS
    COLS = CUBLASLT_MATRIX_LAYOUT_COLS
    LD = CUBLASLT_MATRIX_LAYOUT_LD
    BATCH_COUNT = CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
    STRIDED_BATCH_OFFSET = CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
    PLANE_OFFSET = CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET

class MatmulDescAttribute(_IntEnum):
    """See `cublasLtMatmulDescAttributes_t`."""
    COMPUTE_TYPE = CUBLASLT_MATMUL_DESC_COMPUTE_TYPE
    SCALE_TYPE = CUBLASLT_MATMUL_DESC_SCALE_TYPE
    POINTER_MODE = CUBLASLT_MATMUL_DESC_POINTER_MODE
    TRANSA = CUBLASLT_MATMUL_DESC_TRANSA
    TRANSB = CUBLASLT_MATMUL_DESC_TRANSB
    TRANSC = CUBLASLT_MATMUL_DESC_TRANSC
    FILL_MODE = CUBLASLT_MATMUL_DESC_FILL_MODE
    EPILOGUE = CUBLASLT_MATMUL_DESC_EPILOGUE
    BIAS_POINTER = CUBLASLT_MATMUL_DESC_BIAS_POINTER
    BIAS_BATCH_STRIDE = CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE
    EPILOGUE_AUX_POINTER = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER
    EPILOGUE_AUX_LD = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD
    EPILOGUE_AUX_BATCH_STRIDE = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE
    ALPHA_VECTOR_BATCH_STRIDE = CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE
    SM_COUNT_TARGET = CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET
    A_SCALE_POINTER = CUBLASLT_MATMUL_DESC_A_SCALE_POINTER
    B_SCALE_POINTER = CUBLASLT_MATMUL_DESC_B_SCALE_POINTER
    C_SCALE_POINTER = CUBLASLT_MATMUL_DESC_C_SCALE_POINTER
    D_SCALE_POINTER = CUBLASLT_MATMUL_DESC_D_SCALE_POINTER
    AMAX_D_POINTER = CUBLASLT_MATMUL_DESC_AMAX_D_POINTER
    EPILOGUE_AUX_DATA_TYPE = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE
    EPILOGUE_AUX_SCALE_POINTER = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER
    EPILOGUE_AUX_AMAX_POINTER = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER
    FAST_ACCUM = CUBLASLT_MATMUL_DESC_FAST_ACCUM
    BIAS_DATA_TYPE = CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
    ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS
    ATOMIC_SYNC_NUM_CHUNKS_D_COLS = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS
    ATOMIC_SYNC_IN_COUNTERS_POINTER = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER
    ATOMIC_SYNC_OUT_COUNTERS_POINTER = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER

class MatrixTransformDescAttribute(_IntEnum):
    """See `cublasLtMatrixTransformDescAttributes_t`."""
    SCALE_TYPE = CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE
    POINTER_MODE = CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE
    TRANSA = CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA
    TRANSB = CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB

class ReductionScheme(_IntEnum):
    """See `cublasLtReductionScheme_t`."""
    NONE = CUBLASLT_REDUCTION_SCHEME_NONE
    INPLACE = CUBLASLT_REDUCTION_SCHEME_INPLACE
    COMPUTE_TYPE = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE
    OUTPUT_TYPE = CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE
    MASK = CUBLASLT_REDUCTION_SCHEME_MASK

class Epilogue(_IntEnum):
    """See `cublasLtEpilogue_t`."""
    DEFAULT = CUBLASLT_EPILOGUE_DEFAULT
    RELU = CUBLASLT_EPILOGUE_RELU
    RELU_AUX = CUBLASLT_EPILOGUE_RELU_AUX
    BIAS = CUBLASLT_EPILOGUE_BIAS
    RELU_BIAS = CUBLASLT_EPILOGUE_RELU_BIAS
    RELU_AUX_BIAS = CUBLASLT_EPILOGUE_RELU_AUX_BIAS
    DRELU = CUBLASLT_EPILOGUE_DRELU
    DRELU_BGRAD = CUBLASLT_EPILOGUE_DRELU_BGRAD
    GELU = CUBLASLT_EPILOGUE_GELU
    GELU_AUX = CUBLASLT_EPILOGUE_GELU_AUX
    GELU_BIAS = CUBLASLT_EPILOGUE_GELU_BIAS
    GELU_AUX_BIAS = CUBLASLT_EPILOGUE_GELU_AUX_BIAS
    DGELU = CUBLASLT_EPILOGUE_DGELU
    DGELU_BGRAD = CUBLASLT_EPILOGUE_DGELU_BGRAD
    BGRADA = CUBLASLT_EPILOGUE_BGRADA
    BGRADB = CUBLASLT_EPILOGUE_BGRADB

class MatmulSearch(_IntEnum):
    """See `cublasLtMatmulSearch_t`."""
    BEST_FIT = CUBLASLT_SEARCH_BEST_FIT
    LIMITED_BY_ALGO_ID = CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID
    RESERVED_02 = CUBLASLT_SEARCH_RESERVED_02
    RESERVED_03 = CUBLASLT_SEARCH_RESERVED_03
    RESERVED_04 = CUBLASLT_SEARCH_RESERVED_04
    RESERVED_05 = CUBLASLT_SEARCH_RESERVED_05

class MatmulPreferenceAttribute(_IntEnum):
    """See `cublasLtMatmulPreferenceAttributes_t`."""
    SEARCH_MODE = CUBLASLT_MATMUL_PREF_SEARCH_MODE
    MAX_WORKSPACE_BYTES = CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
    REDUCTION_SCHEME_MASK = CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK
    MIN_ALIGNMENT_A_BYTES = CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES
    MIN_ALIGNMENT_B_BYTES = CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES
    MIN_ALIGNMENT_C_BYTES = CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES
    MIN_ALIGNMENT_D_BYTES = CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES
    MAX_WAVES_COUNT = CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT
    IMPL_MASK = CUBLASLT_MATMUL_PREF_IMPL_MASK
    MATH_MODE_MASK = CUBLASLT_MATMUL_PREF_MATH_MODE_MASK
    GAUSSIAN_MODE_MASK = CUBLASLT_MATMUL_PREF_GAUSSIAN_MODE_MASK
    POINTER_MODE_MASK = CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK
    EPILOGUE_MASK = CUBLASLT_MATMUL_PREF_EPILOGUE_MASK
    SM_COUNT_TARGET = CUBLASLT_MATMUL_PREF_SM_COUNT_TARGET

class MatmulAlgoCapAttribute(_IntEnum):
    """See `cublasLtMatmulAlgoCapAttributes_t`."""
    SPLITK_SUPPORT = CUBLASLT_ALGO_CAP_SPLITK_SUPPORT
    REDUCTION_SCHEME_MASK = CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK
    CTA_SWIZZLING_SUPPORT = CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT
    STRIDED_BATCH_SUPPORT = CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT
    OUT_OF_PLACE_RESULT_SUPPORT = CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT
    UPLO_SUPPORT = CUBLASLT_ALGO_CAP_UPLO_SUPPORT
    TILE_IDS = CUBLASLT_ALGO_CAP_TILE_IDS
    CUSTOM_OPTION_MAX = CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX
    CUSTOM_MEMORY_ORDER = CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER
    POINTER_MODE_MASK = CUBLASLT_ALGO_CAP_POINTER_MODE_MASK
    EPILOGUE_MASK = CUBLASLT_ALGO_CAP_EPILOGUE_MASK
    STAGES_IDS = CUBLASLT_ALGO_CAP_STAGES_IDS
    LD_NEGATIVE = CUBLASLT_ALGO_CAP_LD_NEGATIVE
    NUMERICAL_IMPL_FLAGS = CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS
    MIN_ALIGNMENT_A_BYTES = CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES
    MIN_ALIGNMENT_B_BYTES = CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES
    MIN_ALIGNMENT_C_BYTES = CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES
    MIN_ALIGNMENT_D_BYTES = CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES
    ATOMIC_SYNC = CUBLASLT_ALGO_CAP_ATOMIC_SYNC
    MATHMODE_IMPL = CUBLASLT_ALGO_CAP_MATHMODE_IMPL
    GAUSSIAN_IMPL = CUBLASLT_ALGO_CAP_GAUSSIAN_IMPL

class MatmulAlgoConfigAttribute(_IntEnum):
    """See `cublasLtMatmulAlgoConfigAttributes_t`."""
    ID = CUBLASLT_ALGO_CONFIG_ID
    TILE_ID = CUBLASLT_ALGO_CONFIG_TILE_ID
    SPLITK_NUM = CUBLASLT_ALGO_CONFIG_SPLITK_NUM
    REDUCTION_SCHEME = CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME
    CTA_SWIZZLING = CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING
    CUSTOM_OPTION = CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
    STAGES_ID = CUBLASLT_ALGO_CONFIG_STAGES_ID
    INNER_SHAPE_ID = CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID
    CLUSTER_SHAPE_ID = CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID

class ClusterShape(_IntEnum):
    """See `cublasLtClusterShape_t`."""
    SHAPE_AUTO = CUBLASLT_CLUSTER_SHAPE_AUTO
    SHAPE_1x1x1 = CUBLASLT_CLUSTER_SHAPE_1x1x1
    SHAPE_2x1x1 = CUBLASLT_CLUSTER_SHAPE_2x1x1
    SHAPE_4x1x1 = CUBLASLT_CLUSTER_SHAPE_4x1x1
    SHAPE_1x2x1 = CUBLASLT_CLUSTER_SHAPE_1x2x1
    SHAPE_2x2x1 = CUBLASLT_CLUSTER_SHAPE_2x2x1
    SHAPE_4x2x1 = CUBLASLT_CLUSTER_SHAPE_4x2x1
    SHAPE_1x4x1 = CUBLASLT_CLUSTER_SHAPE_1x4x1
    SHAPE_2x4x1 = CUBLASLT_CLUSTER_SHAPE_2x4x1
    SHAPE_4x4x1 = CUBLASLT_CLUSTER_SHAPE_4x4x1
    SHAPE_8x1x1 = CUBLASLT_CLUSTER_SHAPE_8x1x1
    SHAPE_1x8x1 = CUBLASLT_CLUSTER_SHAPE_1x8x1
    SHAPE_8x2x1 = CUBLASLT_CLUSTER_SHAPE_8x2x1
    SHAPE_2x8x1 = CUBLASLT_CLUSTER_SHAPE_2x8x1
    SHAPE_16x1x1 = CUBLASLT_CLUSTER_SHAPE_16x1x1
    SHAPE_1x16x1 = CUBLASLT_CLUSTER_SHAPE_1x16x1
    SHAPE_3x1x1 = CUBLASLT_CLUSTER_SHAPE_3x1x1
    SHAPE_5x1x1 = CUBLASLT_CLUSTER_SHAPE_5x1x1
    SHAPE_6x1x1 = CUBLASLT_CLUSTER_SHAPE_6x1x1
    SHAPE_7x1x1 = CUBLASLT_CLUSTER_SHAPE_7x1x1
    SHAPE_9x1x1 = CUBLASLT_CLUSTER_SHAPE_9x1x1
    SHAPE_10x1x1 = CUBLASLT_CLUSTER_SHAPE_10x1x1
    SHAPE_11x1x1 = CUBLASLT_CLUSTER_SHAPE_11x1x1
    SHAPE_12x1x1 = CUBLASLT_CLUSTER_SHAPE_12x1x1
    SHAPE_13x1x1 = CUBLASLT_CLUSTER_SHAPE_13x1x1
    SHAPE_14x1x1 = CUBLASLT_CLUSTER_SHAPE_14x1x1
    SHAPE_15x1x1 = CUBLASLT_CLUSTER_SHAPE_15x1x1
    SHAPE_3x2x1 = CUBLASLT_CLUSTER_SHAPE_3x2x1
    SHAPE_5x2x1 = CUBLASLT_CLUSTER_SHAPE_5x2x1
    SHAPE_6x2x1 = CUBLASLT_CLUSTER_SHAPE_6x2x1
    SHAPE_7x2x1 = CUBLASLT_CLUSTER_SHAPE_7x2x1
    SHAPE_1x3x1 = CUBLASLT_CLUSTER_SHAPE_1x3x1
    SHAPE_2x3x1 = CUBLASLT_CLUSTER_SHAPE_2x3x1
    SHAPE_3x3x1 = CUBLASLT_CLUSTER_SHAPE_3x3x1
    SHAPE_4x3x1 = CUBLASLT_CLUSTER_SHAPE_4x3x1
    SHAPE_5x3x1 = CUBLASLT_CLUSTER_SHAPE_5x3x1
    SHAPE_3x4x1 = CUBLASLT_CLUSTER_SHAPE_3x4x1
    SHAPE_1x5x1 = CUBLASLT_CLUSTER_SHAPE_1x5x1
    SHAPE_2x5x1 = CUBLASLT_CLUSTER_SHAPE_2x5x1
    SHAPE_3x5x1 = CUBLASLT_CLUSTER_SHAPE_3x5x1
    SHAPE_1x6x1 = CUBLASLT_CLUSTER_SHAPE_1x6x1
    SHAPE_2x6x1 = CUBLASLT_CLUSTER_SHAPE_2x6x1
    SHAPE_1x7x1 = CUBLASLT_CLUSTER_SHAPE_1x7x1
    SHAPE_2x7x1 = CUBLASLT_CLUSTER_SHAPE_2x7x1
    SHAPE_1x9x1 = CUBLASLT_CLUSTER_SHAPE_1x9x1
    SHAPE_1x10x1 = CUBLASLT_CLUSTER_SHAPE_1x10x1
    SHAPE_1x11x1 = CUBLASLT_CLUSTER_SHAPE_1x11x1
    SHAPE_1x12x1 = CUBLASLT_CLUSTER_SHAPE_1x12x1
    SHAPE_1x13x1 = CUBLASLT_CLUSTER_SHAPE_1x13x1
    SHAPE_1x14x1 = CUBLASLT_CLUSTER_SHAPE_1x14x1
    SHAPE_1x15x1 = CUBLASLT_CLUSTER_SHAPE_1x15x1

class MatmulInnerShape(_IntEnum):
    """See `cublasLtMatmulInnerShape_t`."""
    UNDEFINED = CUBLASLT_MATMUL_INNER_SHAPE_UNDEFINED
    MMA884 = CUBLASLT_MATMUL_INNER_SHAPE_MMA884
    MMA1684 = CUBLASLT_MATMUL_INNER_SHAPE_MMA1684
    MMA1688 = CUBLASLT_MATMUL_INNER_SHAPE_MMA1688
    MMA16816 = CUBLASLT_MATMUL_INNER_SHAPE_MMA16816


###############################################################################
# Error handling
###############################################################################

class cuBLASLtError(Exception):

    def __init__(self, status):
        self.status = status
        from ._internal.cublasLt import _inspect_function_pointer
        self.status = status
        cdef str err
        if (_inspect_function_pointer("__cublasLtGetStatusName") != 0
                and _inspect_function_pointer("__cublasLtGetStatusString") != 0):
            err = f"{get_status_string(status)} ({get_status_name(status)})"
        else:
            from .cublas import Status
            s = Status(status)
            err = f"{s.name} ({s.value})"
        super(cuBLASLtError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise cuBLASLtError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t create() except? 0:
    """See `cublasLtCreate`."""
    cdef Handle light_handle
    with nogil:
        status = cublasLtCreate(&light_handle)
    check_status(status)
    return <intptr_t>light_handle


cpdef destroy(intptr_t light_handle):
    """See `cublasLtDestroy`."""
    with nogil:
        status = cublasLtDestroy(<Handle>light_handle)
    check_status(status)


cpdef size_t get_version():
    """See `cublasLtGetVersion`."""
    return cublasLtGetVersion()


cpdef size_t get_cudart_version():
    """See `cublasLtGetCudartVersion`."""
    return cublasLtGetCudartVersion()


cpdef int get_property(int type) except? -1:
    """See `cublasLtGetProperty`."""
    cdef int value
    with nogil:
        status = cublasLtGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


cpdef matmul(intptr_t light_handle, intptr_t compute_desc, intptr_t alpha, intptr_t a, intptr_t adesc, intptr_t b, intptr_t bdesc, intptr_t beta, intptr_t c, intptr_t cdesc, intptr_t d, intptr_t ddesc, intptr_t algo, intptr_t workspace, size_t workspace_size_in_bytes, intptr_t stream):
    """See `cublasLtMatmul`."""
    with nogil:
        status = cublasLtMatmul(<Handle>light_handle, <MatmulDesc>compute_desc, <const void*>alpha, <const void*>a, <MatrixLayout>adesc, <const void*>b, <MatrixLayout>bdesc, <const void*>beta, <const void*>c, <MatrixLayout>cdesc, <void*>d, <MatrixLayout>ddesc, <const cublasLtMatmulAlgo_t*>algo, <void*>workspace, workspace_size_in_bytes, <Stream>stream)
    check_status(status)


cpdef matrix_transform(intptr_t light_handle, intptr_t transform_desc, intptr_t alpha, intptr_t a, intptr_t adesc, intptr_t beta, intptr_t b, intptr_t bdesc, intptr_t c, intptr_t cdesc, intptr_t stream):
    """See `cublasLtMatrixTransform`."""
    with nogil:
        status = cublasLtMatrixTransform(<Handle>light_handle, <MatrixTransformDesc>transform_desc, <const void*>alpha, <const void*>a, <MatrixLayout>adesc, <const void*>beta, <const void*>b, <MatrixLayout>bdesc, <void*>c, <MatrixLayout>cdesc, <Stream>stream)
    check_status(status)


cpdef intptr_t matrix_layout_create(int type, uint64_t rows, uint64_t cols, int64_t ld) except? 0:
    """See `cublasLtMatrixLayoutCreate`."""
    cdef MatrixLayout mat_layout
    with nogil:
        status = cublasLtMatrixLayoutCreate(&mat_layout, <DataType>type, rows, cols, ld)
    check_status(status)
    return <intptr_t>mat_layout


cpdef matrix_layout_destroy(intptr_t mat_layout):
    """See `cublasLtMatrixLayoutDestroy`."""
    with nogil:
        status = cublasLtMatrixLayoutDestroy(<MatrixLayout>mat_layout)
    check_status(status)


######################### Python specific utility #########################

cdef dict matrix_layout_attribute_sizes = {
    CUBLASLT_MATRIX_LAYOUT_TYPE: _numpy.uint32,
    CUBLASLT_MATRIX_LAYOUT_ORDER: _numpy.int32,
    CUBLASLT_MATRIX_LAYOUT_ROWS: _numpy.uint64,
    CUBLASLT_MATRIX_LAYOUT_COLS: _numpy.uint64,
    CUBLASLT_MATRIX_LAYOUT_LD: _numpy.int64,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: _numpy.int32,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: _numpy.int64,
    CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET: _numpy.int64,
}

cpdef get_matrix_layout_attribute_dtype(int attr):
    """Get the Python data type of the corresponding MatrixLayoutAttribute attribute.

    Args:
        attr (MatrixLayoutAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`matrix_layout_get_attribute`, :func:`matrix_layout_set_attribute`.
    """
    return matrix_layout_attribute_sizes[attr]

###########################################################################


cpdef matrix_layout_set_attribute(intptr_t mat_layout, int attr, intptr_t buf, size_t size_in_bytes):
    """See `cublasLtMatrixLayoutSetAttribute`."""
    with nogil:
        status = cublasLtMatrixLayoutSetAttribute(<MatrixLayout>mat_layout, <_MatrixLayoutAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(status)


cpdef matrix_layout_get_attribute(intptr_t mat_layout, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written):
    """See `cublasLtMatrixLayoutGetAttribute`."""
    with nogil:
        status = cublasLtMatrixLayoutGetAttribute(<MatrixLayout>mat_layout, <_MatrixLayoutAttribute>attr, <void*>buf, size_in_bytes, <size_t*>size_written)
    check_status(status)


cpdef intptr_t matmul_desc_create(int compute_type, int scale_type) except? 0:
    """See `cublasLtMatmulDescCreate`."""
    cdef MatmulDesc matmul_desc
    with nogil:
        status = cublasLtMatmulDescCreate(&matmul_desc, <cublasComputeType_t>compute_type, <DataType>scale_type)
    check_status(status)
    return <intptr_t>matmul_desc


cpdef matmul_desc_destroy(intptr_t matmul_desc):
    """See `cublasLtMatmulDescDestroy`."""
    with nogil:
        status = cublasLtMatmulDescDestroy(<MatmulDesc>matmul_desc)
    check_status(status)


######################### Python specific utility #########################

cdef dict matmul_desc_attribute_sizes = {
    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE: _numpy.int32,
    CUBLASLT_MATMUL_DESC_SCALE_TYPE: _numpy.int32,
    CUBLASLT_MATMUL_DESC_POINTER_MODE: _numpy.int32,
    CUBLASLT_MATMUL_DESC_TRANSA: _numpy.int32,
    CUBLASLT_MATMUL_DESC_TRANSB: _numpy.int32,
    CUBLASLT_MATMUL_DESC_TRANSC: _numpy.int32,
    CUBLASLT_MATMUL_DESC_FILL_MODE: _numpy.int32,
    CUBLASLT_MATMUL_DESC_EPILOGUE: _numpy.uint32,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER: _numpy.intp,
    CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE: _numpy.int64,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER: _numpy.intp,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD: _numpy.int64,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE: _numpy.int64,
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE: _numpy.int64,
    CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET: _numpy.int32,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER: _numpy.intp,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER: _numpy.intp,
    CUBLASLT_MATMUL_DESC_C_SCALE_POINTER: _numpy.intp,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER: _numpy.intp,
    CUBLASLT_MATMUL_DESC_AMAX_D_POINTER: _numpy.intp,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE: _numpy.int32,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER: _numpy.intp,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER: _numpy.intp,
    CUBLASLT_MATMUL_DESC_FAST_ACCUM: _numpy.int8,
    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: _numpy.int32,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS: _numpy.int32,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS: _numpy.int32,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER: _numpy.int32,
    CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER: _numpy.int32,
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


cpdef matmul_desc_set_attribute(intptr_t matmul_desc, int attr, intptr_t buf, size_t size_in_bytes):
    """See `cublasLtMatmulDescSetAttribute`."""
    with nogil:
        status = cublasLtMatmulDescSetAttribute(<MatmulDesc>matmul_desc, <_MatmulDescAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(status)


cpdef matmul_desc_get_attribute(intptr_t matmul_desc, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written):
    """See `cublasLtMatmulDescGetAttribute`."""
    with nogil:
        status = cublasLtMatmulDescGetAttribute(<MatmulDesc>matmul_desc, <_MatmulDescAttribute>attr, <void*>buf, size_in_bytes, <size_t*>size_written)
    check_status(status)


cpdef intptr_t matrix_transform_desc_create(int scale_type) except? 0:
    """See `cublasLtMatrixTransformDescCreate`."""
    cdef MatrixTransformDesc transform_desc
    with nogil:
        status = cublasLtMatrixTransformDescCreate(&transform_desc, <DataType>scale_type)
    check_status(status)
    return <intptr_t>transform_desc


cpdef matrix_transform_desc_destroy(intptr_t transform_desc):
    """See `cublasLtMatrixTransformDescDestroy`."""
    with nogil:
        status = cublasLtMatrixTransformDescDestroy(<MatrixTransformDesc>transform_desc)
    check_status(status)


######################### Python specific utility #########################

cdef dict matrix_transform_desc_attribute_sizes = {
    CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE: _numpy.int32,
    CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE: _numpy.int32,
    CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA: _numpy.int32,
    CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB: _numpy.int32,
}

cpdef get_matrix_transform_desc_attribute_dtype(int attr):
    """Get the Python data type of the corresponding MatrixTransformDescAttribute attribute.

    Args:
        attr (MatrixTransformDescAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`matrix_transform_desc_get_attribute`, :func:`matrix_transform_desc_set_attribute`.
    """
    return matrix_transform_desc_attribute_sizes[attr]

###########################################################################


cpdef matrix_transform_desc_set_attribute(intptr_t transform_desc, int attr, intptr_t buf, size_t size_in_bytes):
    """See `cublasLtMatrixTransformDescSetAttribute`."""
    with nogil:
        status = cublasLtMatrixTransformDescSetAttribute(<MatrixTransformDesc>transform_desc, <_MatrixTransformDescAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(status)


cpdef matrix_transform_desc_get_attribute(intptr_t transform_desc, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written):
    """See `cublasLtMatrixTransformDescGetAttribute`."""
    with nogil:
        status = cublasLtMatrixTransformDescGetAttribute(<MatrixTransformDesc>transform_desc, <_MatrixTransformDescAttribute>attr, <void*>buf, size_in_bytes, <size_t*>size_written)
    check_status(status)


cpdef intptr_t matmul_preference_create() except? 0:
    """See `cublasLtMatmulPreferenceCreate`."""
    cdef MatmulPreference pref
    with nogil:
        status = cublasLtMatmulPreferenceCreate(&pref)
    check_status(status)
    return <intptr_t>pref


cpdef matmul_preference_destroy(intptr_t pref):
    """See `cublasLtMatmulPreferenceDestroy`."""
    with nogil:
        status = cublasLtMatmulPreferenceDestroy(<MatmulPreference>pref)
    check_status(status)


######################### Python specific utility #########################

cdef dict matmul_preference_attribute_sizes = {
    CUBLASLT_MATMUL_PREF_SEARCH_MODE: _numpy.uint32,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: _numpy.uint64,
    CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK: _numpy.uint32,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES: _numpy.uint32,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES: _numpy.uint32,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES: _numpy.uint32,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES: _numpy.uint32,
    CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT: _numpy.float32,
    CUBLASLT_MATMUL_PREF_IMPL_MASK: _numpy.uint64,
}

cpdef get_matmul_preference_attribute_dtype(int attr):
    """Get the Python data type of the corresponding MatmulPreferenceAttribute attribute.

    Args:
        attr (MatmulPreferenceAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`matmul_preference_get_attribute`, :func:`matmul_preference_set_attribute`.
    """
    return matmul_preference_attribute_sizes[attr]

###########################################################################


cpdef matmul_preference_set_attribute(intptr_t pref, int attr, intptr_t buf, size_t size_in_bytes):
    """See `cublasLtMatmulPreferenceSetAttribute`."""
    with nogil:
        status = cublasLtMatmulPreferenceSetAttribute(<MatmulPreference>pref, <_MatmulPreferenceAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(status)


cpdef matmul_preference_get_attribute(intptr_t pref, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written):
    """See `cublasLtMatmulPreferenceGetAttribute`."""
    with nogil:
        status = cublasLtMatmulPreferenceGetAttribute(<MatmulPreference>pref, <_MatmulPreferenceAttribute>attr, <void*>buf, size_in_bytes, <size_t*>size_written)
    check_status(status)


cpdef matmul_algo_get_heuristic(intptr_t light_handle, intptr_t operation_desc, intptr_t adesc, intptr_t bdesc, intptr_t cdesc, intptr_t ddesc, intptr_t preference, int requested_algo_count, intptr_t heuristic_results_array, intptr_t return_algo_count):
    """See `cublasLtMatmulAlgoGetHeuristic`."""
    with nogil:
        status = cublasLtMatmulAlgoGetHeuristic(<Handle>light_handle, <MatmulDesc>operation_desc, <MatrixLayout>adesc, <MatrixLayout>bdesc, <MatrixLayout>cdesc, <MatrixLayout>ddesc, <MatmulPreference>preference, requested_algo_count, <cublasLtMatmulHeuristicResult_t*>heuristic_results_array, <int*>return_algo_count)
    check_status(status)


cpdef matmul_algo_init(intptr_t light_handle, int compute_type, int scale_type, int atype, int btype, int ctype, int dtype, int algo_id, intptr_t algo):
    """See `cublasLtMatmulAlgoInit`."""
    with nogil:
        status = cublasLtMatmulAlgoInit(<Handle>light_handle, <cublasComputeType_t>compute_type, <DataType>scale_type, <DataType>atype, <DataType>btype, <DataType>ctype, <DataType>dtype, algo_id, <cublasLtMatmulAlgo_t*>algo)
    check_status(status)


cpdef matmul_algo_check(intptr_t light_handle, intptr_t operation_desc, intptr_t adesc, intptr_t bdesc, intptr_t cdesc, intptr_t ddesc, intptr_t algo, intptr_t result):
    """See `cublasLtMatmulAlgoCheck`."""
    with nogil:
        status = cublasLtMatmulAlgoCheck(<Handle>light_handle, <MatmulDesc>operation_desc, <MatrixLayout>adesc, <MatrixLayout>bdesc, <MatrixLayout>cdesc, <MatrixLayout>ddesc, <const cublasLtMatmulAlgo_t*>algo, <cublasLtMatmulHeuristicResult_t*>result)
    check_status(status)


######################### Python specific utility #########################

cdef dict matmul_algo_cap_attribute_sizes = {
    CUBLASLT_ALGO_CAP_SPLITK_SUPPORT: _numpy.int32,
    CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK: _numpy.uint32,
    CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT: _numpy.uint32,
    CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT: _numpy.int32,
    CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT: _numpy.int32,
    CUBLASLT_ALGO_CAP_UPLO_SUPPORT: _numpy.int32,
    CUBLASLT_ALGO_CAP_TILE_IDS: _numpy.uint32,
    CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX: _numpy.int32,
    CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER: _numpy.int32,
    CUBLASLT_ALGO_CAP_POINTER_MODE_MASK: _numpy.uint32,
    CUBLASLT_ALGO_CAP_EPILOGUE_MASK: _numpy.uint32,
    CUBLASLT_ALGO_CAP_STAGES_IDS: _numpy.uint32,
    CUBLASLT_ALGO_CAP_LD_NEGATIVE: _numpy.int32,
    CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS: _numpy.uint64,
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES: _numpy.uint32,
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES: _numpy.uint32,
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES: _numpy.uint32,
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES: _numpy.uint32,
    CUBLASLT_ALGO_CAP_ATOMIC_SYNC: _numpy.int32,
    CUBLASLT_ALGO_CAP_MATHMODE_IMPL: _numpy.int32,
    CUBLASLT_ALGO_CAP_GAUSSIAN_IMPL: _numpy.int32,
}

cpdef get_matmul_algo_cap_attribute_dtype(int attr):
    """Get the Python data type of the corresponding MatmulAlgoCapAttribute attribute.

    Args:
        attr (MatmulAlgoCapAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`matmul_algo_cap_get_attribute`.
    """
    return matmul_algo_cap_attribute_sizes[attr]

###########################################################################


cpdef matmul_algo_cap_get_attribute(intptr_t algo, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written):
    """See `cublasLtMatmulAlgoCapGetAttribute`."""
    with nogil:
        status = cublasLtMatmulAlgoCapGetAttribute(<const cublasLtMatmulAlgo_t*>algo, <_MatmulAlgoCapAttribute>attr, <void*>buf, size_in_bytes, <size_t*>size_written)
    check_status(status)


######################### Python specific utility #########################

cdef dict matmul_algo_config_attribute_sizes = {
    CUBLASLT_ALGO_CONFIG_ID: _numpy.int32,
    CUBLASLT_ALGO_CONFIG_TILE_ID: _numpy.uint32,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM: _numpy.int32,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME: _numpy.uint32,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING: _numpy.uint32,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION: _numpy.uint32,
    CUBLASLT_ALGO_CONFIG_STAGES_ID: _numpy.uint32,
    CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID: _numpy.uint16,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID: _numpy.uint16,
}

cpdef get_matmul_algo_config_attribute_dtype(int attr):
    """Get the Python data type of the corresponding MatmulAlgoConfigAttribute attribute.

    Args:
        attr (MatmulAlgoConfigAttribute): The attribute to query.

    Returns:
        The data type of the queried attribute.

    .. note:: This API has no C counterpart and is a convenient helper for
        allocating memory for :func:`matmul_algo_config_get_attribute`, :func:`matmul_algo_config_set_attribute`.
    """
    return matmul_algo_config_attribute_sizes[attr]

###########################################################################


cpdef matmul_algo_config_set_attribute(intptr_t algo, int attr, intptr_t buf, size_t size_in_bytes):
    """See `cublasLtMatmulAlgoConfigSetAttribute`."""
    with nogil:
        status = cublasLtMatmulAlgoConfigSetAttribute(<cublasLtMatmulAlgo_t*>algo, <_MatmulAlgoConfigAttribute>attr, <const void*>buf, size_in_bytes)
    check_status(status)


cpdef matmul_algo_config_get_attribute(intptr_t algo, int attr, intptr_t buf, size_t size_in_bytes, intptr_t size_written):
    """See `cublasLtMatmulAlgoConfigGetAttribute`."""
    with nogil:
        status = cublasLtMatmulAlgoConfigGetAttribute(<const cublasLtMatmulAlgo_t*>algo, <_MatmulAlgoConfigAttribute>attr, <void*>buf, size_in_bytes, <size_t*>size_written)
    check_status(status)


cpdef logger_open_file(log_file):
    """See `cublasLtLoggerOpenFile`."""
    if not isinstance(log_file, str):
        raise TypeError("log_file must be a Python str")
    cdef bytes _temp_log_file_ = (<str>log_file).encode()
    cdef char* _log_file_ = _temp_log_file_
    with nogil:
        status = cublasLtLoggerOpenFile(<const char*>_log_file_)
    check_status(status)


cpdef logger_set_level(int level):
    """See `cublasLtLoggerSetLevel`."""
    with nogil:
        status = cublasLtLoggerSetLevel(level)
    check_status(status)


cpdef logger_set_mask(int mask):
    """See `cublasLtLoggerSetMask`."""
    with nogil:
        status = cublasLtLoggerSetMask(mask)
    check_status(status)


cpdef logger_force_disable():
    """See `cublasLtLoggerForceDisable`."""
    with nogil:
        status = cublasLtLoggerForceDisable()
    check_status(status)


cpdef str get_status_name(int status):
    """See `cublasLtGetStatusName`."""
    cdef bytes _output_
    _output_ = cublasLtGetStatusName(<cublasStatus_t>status)
    return _output_.decode()


cpdef str get_status_string(int status):
    """See `cublasLtGetStatusString`."""
    cdef bytes _output_
    _output_ = cublasLtGetStatusString(<cublasStatus_t>status)
    return _output_.decode()


cpdef size_t heuristics_cache_get_capacity() except? 0:
    """See `cublasLtHeuristicsCacheGetCapacity`."""
    cdef size_t capacity
    with nogil:
        status = cublasLtHeuristicsCacheGetCapacity(&capacity)
    check_status(status)
    return capacity


cpdef heuristics_cache_set_capacity(size_t capacity):
    """See `cublasLtHeuristicsCacheSetCapacity`."""
    with nogil:
        status = cublasLtHeuristicsCacheSetCapacity(capacity)
    check_status(status)


cpdef disable_cpu_instructions_set_mask(unsigned mask):
    """See `cublasLtDisableCpuInstructionsSetMask`."""
    with nogil:
        status = cublasLtDisableCpuInstructionsSetMask(mask)
    check_status(status)


cpdef tuple matmul_algo_get_ids(intptr_t light_handle, cublasComputeType_t compute_type, size_t scale_type, size_t atype, size_t btype, size_t ctype, size_t dtype, int requested_algo_count):
    cdef vector[int] algo_ids_array
    algo_ids_array.resize(requested_algo_count)
    cdef int return_algo_count
    with nogil:
        status = cublasLtMatmulAlgoGetIds(<Handle>light_handle, compute_type, <DataType>scale_type, <DataType>atype, <DataType>btype, <DataType>ctype, <DataType>dtype, requested_algo_count, algo_ids_array.data(), &return_algo_count)
    check_status(status)
    if return_algo_count < requested_algo_count:
        algo_ids_array.resize(return_algo_count)
    return tuple(algo_ids_array)
