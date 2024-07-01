# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Functions to link type names with CUBLAS compute types.
"""

__all__ = ['NAME_TO_DEFAULT_SCALE_TYPE', 'NAME_TO_DEFAULT_COMPUTE_TYPE']

from nvmath.bindings import cublas
from nvmath._internal.typemaps import cudaDataType
import re

def create_default_scale_type_map():
    """
    Map the data type name to the corresponding CUDA data type that's appropriate for default scale.
    """

    dt = cudaDataType

    scale_type_map = dict()
    # scale_type_map['float8'] = dt.CUDA_R_32F # both CUDA_R_8F_E4M3 and CUDA_R_8F_E5M2 -> CUDA_R_32F
    scale_type_map['bfloat16']   = dt.CUDA_R_32F
    scale_type_map['float16']    = dt.CUDA_R_32F
    scale_type_map['float32']    = dt.CUDA_R_32F
    scale_type_map['float64']    = dt.CUDA_R_64F
    scale_type_map['complex32']  = dt.CUDA_C_32F
    scale_type_map['complex64']  = dt.CUDA_C_32F
    scale_type_map['complex128'] = dt.CUDA_C_64F

    return scale_type_map

def create_compute_type_map():
    """
    Map the data type name to the corresponding CUDA data type that's appropriate for default scale.
    """

    dt = cudaDataType
    ct = cublas.ComputeType

    compute_type_map = dict()
    # compute_type_map['float8'] = ct.COMPUTE_32F # both CUDA_R_8F_E4M3 and CUDA_R_8F_E5M2 -> CUBLAS_COMPUTE_32F
    compute_type_map['bfloat16'] = ct.COMPUTE_32F
    compute_type_map['float16'] = ct.COMPUTE_32F
    compute_type_map['float32'] = ct.COMPUTE_32F
    compute_type_map['float64'] = ct.COMPUTE_64F
    compute_type_map['complex32'] = ct.COMPUTE_32F
    compute_type_map['complex64'] = ct.COMPUTE_32F
    compute_type_map['complex128'] = ct.COMPUTE_64F

    return compute_type_map

NAME_TO_DEFAULT_SCALE_TYPE = create_default_scale_type_map()
NAME_TO_DEFAULT_COMPUTE_TYPE = create_compute_type_map()
