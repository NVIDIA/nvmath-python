# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get and set matmul descriptor attributes.
"""

__all__ = ['MatmulDescInterface']

from collections.abc import Sequence
import itertools
import numbers
import operator

import numpy as np

from nvmath.bindings import cublasLt as cublaslt


DescEnum = cublaslt.MatmulDescAttribute

def scalar_attributes():
   return [e.name for e in DescEnum]

DESC_ENUM_SCALAR_ATTR = scalar_attributes()

class MatmulDescInterface:

    def __init__(self, matmul_desc):
        """
        """
        self.matmul_desc = matmul_desc

    def __getattr__(self, name):
        _name = name.upper()
        if _name not in DESC_ENUM_SCALAR_ATTR:
            return super().__getattr__(name)
        name = _name
        get_dtype = cublaslt.get_matmul_desc_attribute_dtype
        attribute_buffer = np.zeros((1,), dtype=get_dtype(DescEnum[name]))
        size_written = np.zeros((1,), dtype=np.uint64)
        cublaslt.matmul_desc_get_attribute(self.matmul_desc, DescEnum[name].value, attribute_buffer.ctypes.data, attribute_buffer.itemsize, size_written.ctypes.data)
        return attribute_buffer[0]

    def __setattr__(self, name, value):
        _name = name.upper()
        if _name not in DESC_ENUM_SCALAR_ATTR:
            return super().__setattr__(name, value)
        name = _name
        get_dtype = cublaslt.get_matmul_desc_attribute_dtype
        attribute_buffer = np.zeros((1,), dtype=get_dtype(DescEnum[name]))
        attribute_buffer[0] = value
        cublaslt.matmul_desc_set_attribute(self.matmul_desc, DescEnum[name].value, attribute_buffer.ctypes.data, attribute_buffer.itemsize)
