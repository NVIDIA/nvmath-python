# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get and set matmul plan preference
attributes.
"""

__all__ = ["MatmulPreferenceInterface"]

from nvmath._internal.attribute_ifc_factory import make_cublas_attribute_interface
from nvmath.bindings import cublasLt as cublaslt

# Create a class, MatmulPreferenceInterface, such that each enum member in
# MatmulPreferenceAttribute is exposed as a lowercase property (getter + setter).
# For example, if the class instance is stored as ``pref_ifc``,
# then the enum member ``POINTER_MODE_MASK`` becomes the property
# ``pref_ifc.pointer_mode_mask``.
MatmulPreferenceInterface = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="MatmulPreferenceInterface",
    attribute_enum=cublaslt.MatmulPreferenceAttribute,
    get_attribute_dtype_fn=cublaslt.get_matmul_preference_attribute_dtype,
    get_attribute_fn=cublaslt.matmul_preference_get_attribute,
    set_attribute_fn=cublaslt.matmul_preference_set_attribute,
)
