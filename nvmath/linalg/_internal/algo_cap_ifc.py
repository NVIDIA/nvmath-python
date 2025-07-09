# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get algorithm capabilities information.
"""

__all__ = ["AlgoCapInterface"]

import numpy as np

from nvmath.bindings import cublasLt as cublaslt
from .enum_to_tuples import ENUM_TO_MATMUL_STAGE, ENUM_TO_MATMUL_TILE

CapEnum = cublaslt.MatmulAlgoCapAttribute


def scalar_attributes():
    return [e.name for e in CapEnum if not e.name.endswith("_IDS")]


CAP_ENUM_SCALAR_ATTR = scalar_attributes()


class AlgoCapInterface:
    def __init__(self, algorithm):
        """ """
        assert isinstance(algorithm, cublaslt.MatmulHeuristicResult), "Internal error."
        self.algorithm = algorithm

    def __getattr__(self, name):
        _name = name.upper()
        if _name not in CAP_ENUM_SCALAR_ATTR:
            return super().__getattr__(name)
        name = _name
        get_dtype = cublaslt.get_matmul_algo_cap_attribute_dtype
        attribute_buffer = np.zeros((1,), dtype=get_dtype(CapEnum[name]))
        size_written = np.zeros((1,), dtype=np.uint64)
        cublaslt.matmul_algo_cap_get_attribute(
            self.algorithm["algo"].ctypes.data,
            CapEnum[name].value,
            attribute_buffer.ctypes.data,
            attribute_buffer.itemsize,
            size_written.ctypes.data,
        )
        return attribute_buffer[0]

    def _get_array_attribute(self, name, enum_to_value_map):
        assert name in ["TILE_IDS", "STAGES_IDS"], "Internal error."

        get_dtype = cublaslt.get_matmul_algo_cap_attribute_dtype
        dtype = get_dtype(CapEnum[name])
        size_written = np.zeros((1,), dtype=np.uint64)

        # First get the buffer size needed.
        cublaslt.matmul_algo_cap_get_attribute(
            self.algorithm["algo"].ctypes.data, CapEnum[name].value, 0, 0, size_written.ctypes.data
        )

        # Check if any data needs to be written.
        size = size_written[0]
        if size == 0:
            return ()

        # Then allocate the buffer and get the data.
        num = int(size // dtype().itemsize)
        attribute_buffer = np.zeros((num,), dtype=dtype)
        cublaslt.matmul_algo_cap_get_attribute(
            self.algorithm["algo"].ctypes.data,
            CapEnum[name].value,
            attribute_buffer.ctypes.data,
            size,
            size_written.ctypes.data,
        )

        # Convert to value.
        return tuple(enum_to_value_map[e] for e in attribute_buffer)

    @property
    def tile_ids(self):
        return self._get_array_attribute("TILE_IDS", ENUM_TO_MATMUL_TILE)

    @property
    def stages_ids(self):
        return self._get_array_attribute("STAGES_IDS", ENUM_TO_MATMUL_STAGE)
