# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get and set matrix layout attributes.
"""

__all__ = ["MatrixLayoutInterface"]

from nvmath._internal.attribute_ifc_factory import make_cublas_attribute_interface
from nvmath.bindings import cublasLt as cublaslt

# Create a class, MatrixLayoutInterface, such that each enum member in
# MatrixLayoutAttribute is exposed as a lowercase property (getter + setter).
# For example, if the class instance is stored as ``layout_ifc``,
# then the enum member ``ROWS`` becomes the property
# ``layout_ifc.rows``.
MatrixLayoutInterface = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="MatrixLayoutInterface",
    attribute_enum=cublaslt.MatrixLayoutAttribute,
    get_attribute_dtype_fn=cublaslt.get_matrix_layout_attribute_dtype,
    get_attribute_fn=cublaslt.matrix_layout_get_attribute,
    set_attribute_fn=cublaslt.matrix_layout_set_attribute,
)
