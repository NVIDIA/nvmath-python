# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get and set matmul plan preference
attributes.
"""

__all__ = ["MatmulPreferenceInterface"]

import logging

import numpy as np

from nvmath.bindings import cublasLt as cublaslt


logger = logging.getLogger()

PreferenceEnum = cublaslt.MatmulPreferenceAttribute


def scalar_attributes():
    return [e.name for e in PreferenceEnum]


PREF_ENUM_SCALAR_ATTR = scalar_attributes()


class MatmulPreferenceInterface:
    def __init__(self, matmul_pref):
        """ """
        self.matmul_pref = matmul_pref

    def __getattr__(self, name):
        _name = name.upper()
        logging.debug("Getting Matmul Preference attribute %s.", _name)
        if _name not in PREF_ENUM_SCALAR_ATTR:
            return super().__getattr__(name)
        name = _name
        get_dtype = cublaslt.get_matmul_preference_attribute_dtype
        attribute_buffer = np.zeros((1,), dtype=get_dtype(PreferenceEnum[name]))
        size_written = np.zeros((1,), dtype=np.uint64)
        cublaslt.matmul_preference_get_attribute(
            self.matmul_pref,
            PreferenceEnum[name].value,
            attribute_buffer.ctypes.data,
            attribute_buffer.itemsize,
            size_written.ctypes.data,
        )
        return attribute_buffer[0]

    def __setattr__(self, name, value):
        _name = name.upper()
        logging.debug("Setting Matmul Preference attribute %s to %s.", _name, value)
        if _name not in PREF_ENUM_SCALAR_ATTR:
            return super().__setattr__(name, value)
        name = _name
        get_dtype = cublaslt.get_matmul_preference_attribute_dtype
        attribute_buffer = np.zeros((1,), dtype=get_dtype(PreferenceEnum[name]))
        attribute_buffer[0] = value
        cublaslt.matmul_preference_set_attribute(
            self.matmul_pref, PreferenceEnum[name].value, attribute_buffer.ctypes.data, attribute_buffer.itemsize
        )
