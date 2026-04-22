# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
A collection of sparse utilities.
"""

import importlib
from collections.abc import Sequence

from nvmath.internal.utils import infer_object_package
from nvmath.sparse._internal.sparse_format_helpers import PACKAGE_HELPER_MAP

RECOGNIZED_NAMED_FORMATS = {"BSR", "BSC", "CSR", "CSC", "COO", "DIA"}


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

    package = infer_object_package(native_operand)
    helper = PACKAGE_HELPER_MAP[package]
    sparse_format = helper.sparse_format_name(native_operand)

    # The sparse format should be a recognized named format or an UST.
    if sparse_format not in RECOGNIZED_NAMED_FORMATS:
        if package == "nvmath":
            from nvmath.sparse._internal.sparse_ust_ifc import USTensorHolder

            return USTensorHolder(native_operand)

        raise TypeError(f"The sparse format {sparse_format} is not currently supported.")

    # Recognized named format.
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

    # The sparse format should be a recognized named format or an UST.
    if sparse_format not in RECOGNIZED_NAMED_FORMATS:
        if package == "nvmath":
            from nvmath.sparse._internal.sparse_ust_ifc import USTensor

            return [USTensor(o) for o in native_operands]

        raise TypeError(f"The sparse format {sparse_format} is not currently supported.")

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


def sparse_or_dense(operand):
    """
    Determine if the operand is sparse or dense.
    """

    dense_tensor_modules = {"cupy", "numpy", "torch"}
    sparse_tensor_modules = set(PACKAGE_HELPER_MAP.keys())

    module = infer_object_package(operand)

    if module == "torch":
        import torch

        if not torch.is_tensor(operand):
            raise TypeError(f"The operand '{operand}' is not a tensor object.")

        if operand.layout == torch.strided:
            return "dense"

        return "sparse"

    # TODO: use the level spec instead of format name to infer if dense.
    if module == "nvmath":
        name = operand.tensor_format.name
        if name == "Scalar" or "Dense" in name:
            return "dense"
        return "sparse"

    # We'll check if the object is a tensor later, we'll assume here that it is
    # solely based on the package.
    if module in sparse_tensor_modules:
        return "sparse"

    if module in dense_tensor_modules:
        return "dense"

    raise TypeError(f"The package '{module}' is not supported.")
