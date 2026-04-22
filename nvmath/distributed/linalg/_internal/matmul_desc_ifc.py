# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get and set matmul descriptor attributes.
"""

__all__ = ["MatmulDescInterface"]

from nvmath._internal.attribute_ifc_factory import make_cublas_attribute_interface
from nvmath.bindings import cublasMp  # type: ignore

# Create a class, MatmulDescInterface, such that each enum member in
# MatmulDescriptorAttribute is exposed as a lowercase property (getter + setter)
# and a ``set_<attr>_unchecked`` method (setter without debug logging).
# For example, if the class instance is stored as ``desc_ifc``,
# then the enum member ``ALGO_TYPE`` becomes the property
# ``desc_ifc.algo_type``.
MatmulDescInterface = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="MatmulDescInterface",
    attribute_enum=cublasMp.MatmulDescriptorAttribute,
    get_attribute_dtype_fn=cublasMp.get_matmul_descriptor_attribute_dtype,
    get_attribute_fn=cublasMp.matmul_descriptor_get_attribute,
    set_attribute_fn=cublasMp.matmul_descriptor_set_attribute,
    with_unchecked=True,  # Consistent with the non-distributed MatmulDescInterface.
)
