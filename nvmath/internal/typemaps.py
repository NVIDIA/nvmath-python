# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Functions to link type names with CUDA data and compute types.
"""

__all__ = ["COMPUTE_TYPE_TO_NAME", "DATA_TYPE_TO_NAME", "NAME_TO_DATA_TYPE", "NAME_TO_COMPUTE_TYPE", "NAME_TO_DATA_WIDTH"]

from enum import IntEnum
import re


class ComputeType(IntEnum):
    """An enumeration of CUDA compute types."""

    COMPUTE_DEFAULT = 0
    COMPUTE_16F = 1 << 0
    COMPUTE_32F = 1 << 2
    COMPUTE_64F = 1 << 4
    COMPUTE_8U = 1 << 6
    COMPUTE_8I = 1 << 8
    COMPUTE_32U = 1 << 7
    COMPUTE_32I = 1 << 9
    COMPUTE_16BF = 1 << 10
    COMPUTE_TF32 = 1 << 12


class cudaDataType(IntEnum):
    """An enumeration of `cudaDataType_t`."""

    CUDA_R_16F = 2
    CUDA_C_16F = 6
    CUDA_R_16BF = 14
    CUDA_C_16BF = 15
    CUDA_R_32F = 0
    CUDA_C_32F = 4
    CUDA_R_64F = 1
    CUDA_C_64F = 5
    CUDA_R_4I = 16
    CUDA_C_4I = 17
    CUDA_R_4U = 18
    CUDA_C_4U = 19
    CUDA_R_8I = 3
    CUDA_C_8I = 7
    CUDA_R_8U = 8
    CUDA_C_8U = 9
    CUDA_R_16I = 20
    CUDA_C_16I = 21
    CUDA_R_16U = 22
    CUDA_C_16U = 23
    CUDA_R_32I = 10
    CUDA_C_32I = 11
    CUDA_R_32U = 12
    CUDA_C_32U = 13
    CUDA_R_64I = 24
    CUDA_C_64I = 25
    CUDA_R_64U = 26
    CUDA_C_64U = 27
    CUDA_R_8F_E4M3 = 28
    CUDA_R_8F_E5M2 = 29


def create_cuda_data_type_map(cuda_data_type_enum_class):
    """
    Map the data type name to the corresponding CUDA data type.
    """
    cuda_data_type_pattern = re.compile(r"CUDA_(?P<cr>C|R)_(?P<width>\d+)(?P<type>F|I|U|BF)_?(?P<kind>(E\dM\d)?)")

    type_code_map = {"i": "int", "u": "uint", "f": "float", "bf": "bfloat"}
    # A map from (width, exponent kind) to qualifiers (finite, unsigned zero, ...) for data
    # types.
    type_qualifier_map = {(8, "e4m3"): "fn"}

    complex_types = {"float": "complex", "bfloat": "bcomplex"}

    cuda_data_type_map = {}
    data_type_width_map = {}
    for d in cuda_data_type_enum_class:
        m = cuda_data_type_pattern.match(d.name)

        is_complex = m.group("cr").lower() == "c"
        type_code = type_code_map[m.group("type").lower()]

        # We'll generalize this if and when we support Gaussian integers.
        if is_complex and type_code not in complex_types:
            continue

        width = int(m.group("width"))
        if is_complex:
            width *= 2
            type_code = complex_types[type_code]

        name = type_code + str(width)

        # Handle narrow type kinds.
        if width <= 8:
            kind = m.group("kind").lower()
            # Handle type qualifiers for narrow types.
            kind += type_qualifier_map.get((width, kind), "")
            if kind:
                name += "_" + kind

        cuda_data_type_map[name] = d
        data_type_width_map[name] = width

    return cuda_data_type_map, data_type_width_map


def create_cuda_compute_type_map(cuda_compute_type_enum_class):
    """
    Map the data type name to the corresponding CUDA compute type.
    """
    cuda_compute_type_pattern = re.compile(r"COMPUTE_(?:(?P<width>\d+)(?P<type>F|I|U|BF)|(?P<tf32>TF32))")

    type_code_map = {"i": "int", "u": "uint", "f": "float", "bf": "bfloat"}

    cuda_compute_type_map = {}
    for c in cuda_compute_type_enum_class:
        if c.name == "COMPUTE_DEFAULT":
            continue

        m = cuda_compute_type_pattern.match(c.name)

        if not m:
            raise ValueError("Internal error - unexpected enum entry")

        if m.group("tf32"):
            continue

        name = type_code_map[m.group("type").lower()] + m.group("width")
        cuda_compute_type_map[name] = c

    # Treat complex types as special case.
    cuda_compute_type_map["bcomplex32"] = cuda_compute_type_enum_class.COMPUTE_16BF
    cuda_compute_type_map["complex32"] = cuda_compute_type_enum_class.COMPUTE_16F
    cuda_compute_type_map["complex64"] = cuda_compute_type_enum_class.COMPUTE_32F
    cuda_compute_type_map["complex128"] = cuda_compute_type_enum_class.COMPUTE_64F

    return cuda_compute_type_map


NAME_TO_DATA_TYPE, NAME_TO_DATA_WIDTH = create_cuda_data_type_map(cudaDataType)
DATA_TYPE_TO_NAME = {v: k for k, v in NAME_TO_DATA_TYPE.items()}
NAME_TO_COMPUTE_TYPE = create_cuda_compute_type_map(ComputeType)
COMPUTE_TYPE_TO_NAME = {v: k for k, v in NAME_TO_COMPUTE_TYPE.items()}


FFTW_SUPPORTED_SINGLE = (cudaDataType.CUDA_R_32F, cudaDataType.CUDA_C_32F)
FFTW_SUPPORTED_DOUBLE = (cudaDataType.CUDA_R_64F, cudaDataType.CUDA_C_64F)
FFTW_SUPPORTED_FLOAT = (cudaDataType.CUDA_R_32F, cudaDataType.CUDA_R_64F)
FFTW_SUPPORTED_COMPLEX = (cudaDataType.CUDA_C_32F, cudaDataType.CUDA_C_64F)
FFTW_SUPPORTED_TYPES = sorted(FFTW_SUPPORTED_SINGLE + FFTW_SUPPORTED_DOUBLE)
