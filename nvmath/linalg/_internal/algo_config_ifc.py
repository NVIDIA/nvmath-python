# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get and set algorithm configuration
attributes.
"""

__all__ = ["AlgoConfigInterface"]

from nvmath._internal.attribute_ifc_factory import make_cublas_attribute_interface
from nvmath.bindings import cublasLt as cublaslt

# Create a class, AlgoConfigInterface, such that each enum member in
# MatmulAlgoConfigAttribute is exposed as a lowercase property (getter + setter).
# For example, if the class instance is stored as ``config_ifc``,
# then the enum member ``CLUSTER_SHAPE_ID`` becomes the property
# ``config_ifc.cluster_shape_id``.
#
# The C bindings expect a raw pointer to the algo sub-buffer, not the
# handle dict. _get_attribute and _set_attribute bridge that gap by
# extracting `handle["algo"].ctypes.data` before forwarding the call.
_raw_get = cublaslt.matmul_algo_config_get_attribute
_raw_set = cublaslt.matmul_algo_config_set_attribute


def _get_attribute(handle, enum_value, buf_addr, buf_size, size_written_addr):
    _raw_get(handle["algo"].ctypes.data, enum_value, buf_addr, buf_size, size_written_addr)


def _set_attribute(handle, enum_value, buf_addr, buf_size):
    _raw_set(handle["algo"].ctypes.data, enum_value, buf_addr, buf_size)


AlgoConfigInterface = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="AlgoConfigInterface",
    attribute_enum=cublaslt.MatmulAlgoConfigAttribute,
    get_attribute_dtype_fn=cublaslt.get_matmul_algo_config_attribute_dtype,
    get_attribute_fn=_get_attribute,
    set_attribute_fn=_set_attribute,
)
