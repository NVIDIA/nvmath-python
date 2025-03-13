# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get algorithm configuration information.
"""

__all__ = ["AlgoConfigInterface"]

from collections import namedtuple

import numpy as np

from nvmath.bindings import cublasLt as cublaslt
from .enum_to_tuples import (
    CLUSTER_SHAPE_TO_ENUM,
    ENUM_TO_CLUSTER_SHAPE,
    MATMUL_STAGE_TO_ENUM,
    ENUM_TO_MATMUL_STAGE,
    MATMUL_TILE_TO_ENUM,
    ENUM_TO_MATMUL_TILE,
)

ConfigEnum = cublaslt.MatmulAlgoConfigAttribute


def scalar_attributes():
    return [e.name for e in ConfigEnum]


CAP_ENUM_SCALAR_ATTR = scalar_attributes()

# Special Handling.
Maps = namedtuple("Maps", ["to_enumerator", "to_value"])
SPECIAL_ATTR = {
    "CLUSTER_SHAPE_ID": Maps(CLUSTER_SHAPE_TO_ENUM, ENUM_TO_CLUSTER_SHAPE),
    "STAGES_ID": Maps(MATMUL_STAGE_TO_ENUM, ENUM_TO_MATMUL_STAGE),
    "TILE_ID": Maps(MATMUL_TILE_TO_ENUM, ENUM_TO_MATMUL_TILE),
}


class AlgoConfigInterface:
    def __init__(self, algorithm):
        """ """
        assert isinstance(algorithm, cublaslt.MatmulHeuristicResult), "Internal error."
        self.algorithm = algorithm

    def __getattr__(self, name):
        _name = name.upper()
        if _name not in CAP_ENUM_SCALAR_ATTR:
            return super().__getattr__(name)
        name = _name
        get_dtype = cublaslt.get_matmul_algo_config_attribute_dtype
        attribute_buffer = np.zeros((1,), dtype=get_dtype(ConfigEnum[name]))
        size_written = np.zeros((1,), dtype=np.uint64)
        cublaslt.matmul_algo_config_get_attribute(
            self.algorithm["algo"].ctypes.data,
            ConfigEnum[name].value,
            attribute_buffer.ctypes.data,
            attribute_buffer.itemsize,
            size_written.ctypes.data,
        )

        if name not in SPECIAL_ATTR:
            return attribute_buffer[0]

        return SPECIAL_ATTR[name].to_value[attribute_buffer[0]]

    def __setattr__(self, name, value):
        _name = name.upper()
        if _name not in CAP_ENUM_SCALAR_ATTR:
            return super().__setattr__(name, value)
        name = _name

        if name in SPECIAL_ATTR:
            value = SPECIAL_ATTR[name].to_enumerator[value]

        get_dtype = cublaslt.get_matmul_algo_config_attribute_dtype
        attribute_buffer = np.zeros((1,), dtype=get_dtype(ConfigEnum[name]))
        attribute_buffer[0] = value
        cublaslt.matmul_algo_config_set_attribute(
            self.algorithm["algo"].ctypes.data,
            ConfigEnum[name].value,
            attribute_buffer.ctypes.data,
            attribute_buffer.itemsize,
        )
