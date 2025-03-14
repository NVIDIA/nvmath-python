# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get and set matrix layout attributes.
"""

__all__ = ["MatrixLayoutInterface"]


import numpy as np

from nvmath.bindings import cublasLt as cublaslt


LayoutEnum = cublaslt.MatrixLayoutAttribute


def scalar_attributes():
    return [e.name for e in LayoutEnum]


LAYOUT_ENUM_SCALAR_ATTR = scalar_attributes()


class MatrixLayoutInterface:
    def __init__(self, matrix_layout):
        """ """
        self.matrix_layout = matrix_layout

    def __getattr__(self, name):
        _name = name.upper()
        if _name not in LAYOUT_ENUM_SCALAR_ATTR:
            return super().__getattr__(name)
        name = _name
        get_dtype = cublaslt.get_matrix_layout_attribute_dtype
        attribute_buffer = np.zeros((1,), dtype=get_dtype(LayoutEnum[name]))
        size_written = np.zeros((1,), dtype=np.uint64)
        cublaslt.matrix_layout_get_attribute(
            self.matrix_layout,
            LayoutEnum[name].value,
            attribute_buffer.ctypes.data,
            attribute_buffer.itemsize,
            size_written.ctypes.data,
        )
        return attribute_buffer[0]

    def __setattr__(self, name, value):
        _name = name.upper()
        if _name not in LAYOUT_ENUM_SCALAR_ATTR:
            return super().__setattr__(name, value)
        name = _name
        get_dtype = cublaslt.get_matrix_layout_attribute_dtype
        attribute_buffer = np.zeros((1,), dtype=get_dtype(LayoutEnum[name]))
        attribute_buffer[0] = value
        cublaslt.matrix_layout_set_attribute(
            self.matrix_layout, LayoutEnum[name].value, attribute_buffer.ctypes.data, attribute_buffer.itemsize
        )
