# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get and set matmul descriptor attributes.
"""

__all__ = ["MatmulDescInterface"]

import ctypes
import logging

import numpy as np

from nvmath.bindings import cublasMp  # type: ignore

logger = logging.getLogger()

DescEnum = cublasMp.MatmulDescriptorAttribute


def scalar_attributes():
    return [e.name for e in DescEnum]


DESC_ENUM_SCALAR_ATTR = scalar_attributes()


def _get_attribute_ctype(name):
    return np.ctypeslib.as_ctypes_type(cublasMp.get_matmul_descriptor_attribute_dtype(DescEnum[name]))


DESC_ENUM_SCALAR_ATTR_INFO = {name: (DescEnum[name].value, _get_attribute_ctype(name)) for name in DESC_ENUM_SCALAR_ATTR}  # type: ignore[valid-type]


class MatmulDescInterface:
    def __init__(self, matmul_desc):
        self.matmul_desc = matmul_desc

    def __getattr__(self, name):
        _name = name.upper()
        logging.debug("Getting Matmul Description attribute %s.", _name)
        info = DESC_ENUM_SCALAR_ATTR_INFO.get(_name)
        if info is None:
            raise AttributeError(f"No attribute named {name} in matmul descriptor")
        enum_value, ctype = info
        name = _name
        attribute_buffer = ctype()
        size_written = ctypes.c_uint64()
        cublasMp.matmul_descriptor_attribute_get(
            self.matmul_desc,
            enum_value,
            ctypes.addressof(attribute_buffer),
            ctypes.sizeof(attribute_buffer),
            ctypes.addressof(size_written),
        )
        return attribute_buffer.value

    def __setattr__(self, name, value):
        if name in ("matmul_desc"):
            # For attributes of this Python class, redirect to the original __setattr__
            return super().__setattr__(name, value)
        _name = name.upper()
        logging.debug("Setting Matmul Description attribute %s to %s.", _name, value)
        info = DESC_ENUM_SCALAR_ATTR_INFO.get(_name)
        if info is None:
            raise AttributeError(f"No attribute named {name} in matmul descriptor")
        enum_value, ctype = info
        name = _name
        ctypes_value = ctype(value)
        cublasMp.matmul_descriptor_attribute_set(
            self.matmul_desc, enum_value, ctypes.addressof(ctypes_value), ctypes.sizeof(ctypes_value)
        )
