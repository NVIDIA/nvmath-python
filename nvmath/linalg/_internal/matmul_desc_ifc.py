# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get and set matmul descriptor attributes.
"""

__all__ = ["MatmulDescInterface"]

import ctypes
import logging

import numpy as np

from nvmath.bindings import cublasLt as cublaslt

logger = logging.getLogger()

DescEnum = cublaslt.MatmulDescAttribute


def scalar_attributes():
    return [e.name for e in DescEnum]


DESC_ENUM_SCALAR_ATTR = scalar_attributes()


def _get_attribute_ctype(name):
    return np.ctypeslib.as_ctypes_type(cublaslt.get_matmul_desc_attribute_dtype(DescEnum[name]))


DESC_ENUM_SCALAR_ATTR_INFO = {name: (DescEnum[name].value, _get_attribute_ctype(name)) for name in DESC_ENUM_SCALAR_ATTR}


class MatmulDescInterface:
    def __init__(self, matmul_desc):
        """ """
        self.matmul_desc = matmul_desc

    def __getattr__(self, name):
        _name = name.upper()
        logging.debug("Getting Matmul Description attribute %s.", _name)
        info = DESC_ENUM_SCALAR_ATTR_INFO.get(_name)
        if info is None:
            return super().__getattr__(name)
        enum_value, ctype = info
        name = _name
        attribute_buffer = ctype()
        size_written = ctypes.c_uint64()
        cublaslt.matmul_desc_get_attribute(
            self.matmul_desc,
            enum_value,
            ctypes.addressof(attribute_buffer),
            ctypes.sizeof(attribute_buffer),
            ctypes.addressof(size_written),
        )
        return attribute_buffer.value

    def __setattr__(self, name, value):
        _name = name.upper()
        logging.debug("Setting Matmul Description attribute %s to %s.", _name, value)
        info = DESC_ENUM_SCALAR_ATTR_INFO.get(_name)
        if info is None:
            return super().__setattr__(name, value)
        enum_value, ctype = info
        name = _name
        ctypes_value = ctype(value)
        cublaslt.matmul_desc_set_attribute(
            self.matmul_desc, enum_value, ctypes.addressof(ctypes_value), ctypes.sizeof(ctypes_value)
        )
