# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get and set matmul descriptor attributes.
"""

__all__ = ["MatmulDescInterface"]

from nvmath._internal.attribute_ifc_factory import make_cublas_attribute_interface
from nvmath.bindings import cublasLt as cublaslt

# Create a class, MatmulDescInterface, such that each enum member in
# MatmulDescAttribute is exposed as a lowercase property (getter + setter)
# and a ``set_<attr>_unchecked`` method (setter without debug logging).
# For example, if the class instance is stored as ``desc_ifc``,
# then the enum member ``POINTER_MODE`` becomes the property
# ``desc_ifc.pointer_mode``.
MatmulDescInterface = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="MatmulDescInterface",
    attribute_enum=cublaslt.MatmulDescAttribute,
    get_attribute_dtype_fn=cublaslt.get_matmul_desc_attribute_dtype,
    get_attribute_fn=cublaslt.matmul_desc_get_attribute,
    set_attribute_fn=cublaslt.matmul_desc_set_attribute,
    with_unchecked=True,  # Used by Matmul.reset_operands_unchecked hot paths.
)
