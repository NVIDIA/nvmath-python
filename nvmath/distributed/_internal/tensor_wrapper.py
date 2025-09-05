# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Entry point to using tensors from different libraries seamlessly.
"""

__all__ = ["wrap_operand", "copy_"]

import warnings
from collections.abc import Sequence

from nvmath.internal.tensor_ifc import Tensor, TensorHolder
from nvmath.internal.tensor_wrapper import (
    infer_tensor_package as base_infer_tensor_package,
    maybe_register_package as base_maybe_register_package,
)

from .tensor_ifc import DistributedTensor
from .tensor_ifc_numpy import NumpyDistributedTensor, CudaDistributedTensor

_TENSOR_TYPES: dict[str, type[DistributedTensor]] = {"numpy": NumpyDistributedTensor, "cuda": CudaDistributedTensor}


def infer_tensor_package(tensor):
    """
    Infer the package that defines this tensor.
    """
    package = base_infer_tensor_package(tensor)
    # Use call_base=False because base_infer_tensor_package already
    # called base_maybe_register_package
    maybe_register_package(package, call_base=False)
    return package


def maybe_register_package(package, call_base=True):
    if call_base:
        base_maybe_register_package(package)
    if package == "torch":
        from .tensor_ifc_torch import TorchDistributedTensor

        _TENSOR_TYPES[package] = TorchDistributedTensor
    elif package == "cupy":
        from .tensor_ifc_cupy import CupyDistributedTensor, HostDistributedTensor

        _TENSOR_TYPES["cupy"] = CupyDistributedTensor
        _TENSOR_TYPES["cupy_host"] = HostDistributedTensor
    elif package != "numpy":
        raise AssertionError(f"Internal error: unrecognized package {package}")


def wrap_operand(native_operand: Tensor) -> DistributedTensor[Tensor]:
    """
    Wrap one "native" operand so that package-agnostic API can be used.
    """
    if isinstance(native_operand, TensorHolder):
        msg = (
            "wrap_operand() is being called unnecessarily because the input is already a TensorHolder. "
            "Only public facing APIs should call wrap_operand(). "
            "Internal APIs should assume the operands are TensorHolder already."
            "Trying to wrap a TensorHolder will become an error in the future."
        )
        warnings.warn(msg, DeprecationWarning)
        assert isinstance(native_operand, DistributedTensor)
        return native_operand
    wrapped_operand = _TENSOR_TYPES[infer_tensor_package(native_operand)](native_operand)
    return wrapped_operand


def copy_(src: Sequence[TensorHolder], dest: Sequence[TensorHolder], stream_holder):
    """
    Copy the wrapped operands in dest to the corresponding wrapped operands in src.
    """
    for s, d in zip(src, dest, strict=True):
        d.copy_(s, stream_holder=stream_holder)
