# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
A collection of sparse utilities.
"""

import importlib
from collections.abc import Sequence

from nvmath.internal.utils import infer_object_package
from nvmath.sparse._internal.sparse_format_helpers import PACKAGE_HELPER_MAP


def check_supported_package(native_operands):
    """
    Check if the operands belong to one of the supported packages.
    """
    package = {infer_object_package(o) for o in native_operands}
    if len(package) != 1:
        raise ValueError(f"The sparse operands do not belong to the same package: {package}.")

    package = package.pop()
    if package not in PACKAGE_HELPER_MAP:
        raise TypeError(f"The sparse operand package {package} is not supported.")

    return package


def wrap_sparse_operand(native_operand):
    """
    Wrap one "native" sparse operand so that package-agnostic API can be used.
    """

    helper = PACKAGE_HELPER_MAP[infer_object_package(native_operand)]
    sparse_format = helper.sparse_format_name(native_operand)
    attr_name_map = getattr(helper, f"{sparse_format.lower()}_attribute_names")()

    module_name = f"nvmath.sparse._internal.sparse_{sparse_format.lower()}_ifc"
    module = importlib.import_module(module_name)
    SparseHolder = getattr(module, f"{sparse_format}TensorHolder")

    return SparseHolder.create_from_tensor(native_operand, attr_name_map=attr_name_map)


def wrap_sparse_operands(native_operands):
    """
    Wrap each "native" sparse operand so that package-agnostic API can be used.
    """

    if not isinstance(native_operands, Sequence):
        return wrap_sparse_operand(native_operands)

    package = check_supported_package(native_operands)
    helper = PACKAGE_HELPER_MAP[package]

    sparse_format = {helper.sparse_format_name(o) for o in native_operands}
    if len(sparse_format) != 1:
        raise ValueError(f"The sparse operands do not have the same sparse format: {sparse_format}")
    sparse_format = sparse_format.pop()
    attr_name_map = getattr(helper, f"{sparse_format.lower()}_attribute_names")()

    module_name = f"nvmath.sparse._internal.sparse_{sparse_format.lower()}_ifc"
    module = importlib.import_module(module_name)
    SparseHolder = getattr(module, f"{sparse_format}TensorHolder")

    return [SparseHolder.create_from_tensor(o, attr_name_map=attr_name_map) for o in native_operands]


# TODO: Unify all the get_attribute functions in internal.utils
# as get(operands, attribute, attribute_desc).
def get_operands_index_type(operands):
    """
    Return the index type name of the tensors.
    """
    index_type = operands[0].index_type
    if not all(operand.index_type == index_type for operand in operands):
        index_types = {operand.index_type for operand in operands}
        raise ValueError(f"All operands must have the same index type. The index types found = {index_types}.")
    return index_type
