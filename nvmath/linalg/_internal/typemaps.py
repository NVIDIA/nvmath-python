# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Functions to link type names with CUBLAS compute types.
"""

__all__ = [
    "NAMES_TO_DEFAULT_SCALE_TYPE",
    "NAMES_TO_DEFAULT_COMPUTE_TYPE",
    "SUPPORTED_TYPES",
    "COMPUTE_TYPE_TO_DEFAULT_SCALE_TYPE",
    "SCALE_TYPE_TO_DEFAULT_COMPUTE_TYPE",
]

from nvmath.bindings import cublas  # type: ignore
from nvmath.internal.typemaps import cudaDataType


def create_default_scale_type_map():
    """
    Map the data type name to the corresponding CUDA data type that's appropriate for
    default scale.
    """

    dt = cudaDataType

    scale_type_map = {}
    scale_type_map["float8_e4m3fn", "float8_e5m2"] = dt.CUDA_R_32F
    scale_type_map["float8_e5m2", "float8_e4m3fn"] = dt.CUDA_R_32F
    scale_type_map["float8_e4m3fn", "float8_e4m3fn"] = dt.CUDA_R_32F
    scale_type_map["bfloat16", "bfloat16"] = dt.CUDA_R_32F
    scale_type_map["float16", "float16"] = dt.CUDA_R_32F
    scale_type_map["float32", "float32"] = dt.CUDA_R_32F
    scale_type_map["float64", "float64"] = dt.CUDA_R_64F
    scale_type_map["complex32", "complex32"] = dt.CUDA_C_32F
    scale_type_map["complex64", "complex64"] = dt.CUDA_C_32F
    scale_type_map["complex128", "complex128"] = dt.CUDA_C_64F

    return scale_type_map


def create_compute_type_to_scale_type_map(is_complex):
    """
    Map the compute type to the corresponding CUDA data type that's appropriate for
    default scale.
    """

    dt = cudaDataType
    ct = cublas.ComputeType

    scale_type_map = {}
    scale_type_map[ct.COMPUTE_16F] = dt.CUDA_R_16F
    scale_type_map[ct.COMPUTE_16F_PEDANTIC] = dt.CUDA_R_16F

    f32 = dt.CUDA_C_32F if is_complex else dt.CUDA_R_32F
    f64 = dt.CUDA_C_64F if is_complex else dt.CUDA_R_64F

    scale_type_map[ct.COMPUTE_32F] = f32
    scale_type_map[ct.COMPUTE_32F_PEDANTIC] = f32
    scale_type_map[ct.COMPUTE_32F_FAST_16F] = f32
    scale_type_map[ct.COMPUTE_32F_FAST_16BF] = f32
    scale_type_map[ct.COMPUTE_32F_FAST_TF32] = f32
    scale_type_map[ct.COMPUTE_64F] = f64
    scale_type_map[ct.COMPUTE_64F_PEDANTIC] = f64

    return scale_type_map


def create_scale_type_to_compute_type_map():
    """
    Map the scale type to the corresponding compute type that's an appropriate default.
    """

    dt = cudaDataType
    ct = cublas.ComputeType

    compute_type_map = {}
    compute_type_map[dt.CUDA_R_16F] = ct.COMPUTE_16F
    compute_type_map[dt.CUDA_R_16BF] = ct.COMPUTE_32F
    compute_type_map[dt.CUDA_R_32F] = ct.COMPUTE_32F
    compute_type_map[dt.CUDA_C_32F] = ct.COMPUTE_32F
    compute_type_map[dt.CUDA_R_64F] = ct.COMPUTE_64F
    compute_type_map[dt.CUDA_C_64F] = ct.COMPUTE_64F
    return compute_type_map


def create_compute_type_map():
    """
    Map the data type name to the corresponding CUDA data type that's appropriate for
    default scale.
    """

    ct = cublas.ComputeType

    compute_type_map = {}
    compute_type_map["float8_e4m3fn", "float8_e5m2"] = ct.COMPUTE_32F
    compute_type_map["float8_e5m2", "float8_e4m3fn"] = ct.COMPUTE_32F
    compute_type_map["float8_e4m3fn", "float8_e4m3fn"] = ct.COMPUTE_32F
    compute_type_map["bfloat16", "bfloat16"] = ct.COMPUTE_32F
    compute_type_map["float16", "float16"] = ct.COMPUTE_32F
    compute_type_map["float32", "float32"] = ct.COMPUTE_32F
    compute_type_map["float64", "float64"] = ct.COMPUTE_64F
    compute_type_map["complex32", "complex32"] = ct.COMPUTE_32F
    compute_type_map["complex64", "complex64"] = ct.COMPUTE_32F
    compute_type_map["complex128", "complex128"] = ct.COMPUTE_64F

    return compute_type_map


CUBLAS_COMPUTE_TYPE_TO_NAME = {
    cublas.ComputeType.COMPUTE_32F: ("float32", "complex64"),
    cublas.ComputeType.COMPUTE_64F: ("float64", "complex128"),
}

NAMES_TO_DEFAULT_SCALE_TYPE = create_default_scale_type_map()
NAMES_TO_DEFAULT_COMPUTE_TYPE = create_compute_type_map()
COMPUTE_TYPE_TO_DEFAULT_SCALE_TYPE = {
    "real": create_compute_type_to_scale_type_map(is_complex=False),
    "complex": create_compute_type_to_scale_type_map(is_complex=True),
}
SCALE_TYPE_TO_DEFAULT_COMPUTE_TYPE = create_scale_type_to_compute_type_map()

SUPPORTED_TYPES = [
    "float8_e4m3fn",
    "float8_e5m2",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "complex32",
    "complex64",
    "complex128",
]
