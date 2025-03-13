# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import numpy as np
from .types import np_float16x2, np_float16x4

TypeMap = namedtuple("TypeMap", ["np", "enum", "cpp"])

_TYPE_MAPS = [
    TypeMap(fe_type, enum, cpp)
    for (enum, (fe_type, cpp)) in enumerate(
        [
            (np.float16, "__half"),
            (np_float16x2, "__half2"),
            (np.float32, "float"),
            (np.float64, "double"),
            (np_float16x2, "commondx::detail::complex<__half>"),
            (np_float16x4, "commondx::detail::complex<__half2>"),
            (np.complex64, "commondx::detail::complex<float>"),
            (np.complex128, "commondx::detail::complex<double>"),
        ]
    )
]

NP_TYPES_TO_CPP_TYPES = {tm.np: tm.cpp for tm in _TYPE_MAPS}

_CPP_TYPES_TO_ENUM = {tm.cpp: tm.enum for tm in _TYPE_MAPS}

_ENUM_TO_NP = {tm.enum: tm.np for tm in _TYPE_MAPS}


def enum_to_np(enum):
    return _ENUM_TO_NP[enum]


def generate_type_map(name):
    struct_name = f"libmathdx_type_map_{name}"
    cpp = f"template<typename T> struct { struct_name } {{ }};"

    for cpp_type, enum_value in _CPP_TYPES_TO_ENUM.items():
        assert enum_value >= 0
        cpp = f"{cpp}\ntemplate<> struct { struct_name }<{ cpp_type }> {{ constexpr static unsigned value = { enum_value }; }};"

    return cpp, struct_name
