# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Entry point to using tensors from different libraries seamlessly.
"""

__all__ = ["infer_tensor_package", "wrap_operand", "wrap_operands", "to", "copy_", "TensorHolder", "AnyTensor"]

from collections.abc import Sequence
import warnings

import numpy as np

from . import formatters
from .tensor_ifc import TensorHolder, Tensor, AnyTensor
from .tensor_ifc_numpy import NumpyTensor


_TENSOR_TYPES: dict[str, type[TensorHolder]] = {"numpy": NumpyTensor}

_SUPPORTED_PACKAGES = tuple(_TENSOR_TYPES.keys())


def infer_tensor_package(tensor):
    """
    Infer the package that defines this tensor.
    """
    if issubclass(tensor.__class__, np.ndarray):
        return "numpy"
    module = tensor.__class__.__module__
    package = module.split(".")[0]
    maybe_register_package(package)
    return package


def maybe_register_package(package):
    global _SUPPORTED_PACKAGES
    if package not in _SUPPORTED_PACKAGES:
        global _TENSOR_TYPES
        from . import package_wrapper
        from .. import memory

        if package == "torch":
            from .tensor_ifc_torch import TorchTensor
            from .package_ifc_torch import TorchPackage

            _TENSOR_TYPES[package] = TorchTensor
            package_wrapper.PACKAGE[package] = TorchPackage
            memory.lazy_load_torch()
        elif package == "cupy":
            from .tensor_ifc_cupy import CupyTensor
            from .package_ifc_cupy import CupyPackage

            _TENSOR_TYPES[package] = CupyTensor
            package_wrapper.PACKAGE[package] = CupyPackage
            memory.lazy_load_cupy()
        else:
            message = f"""{package} not supported yet. Currently must be one of ['numpy', 'cupy', 'torch']"""
            raise ValueError(message)
        _SUPPORTED_PACKAGES = tuple(_TENSOR_TYPES.keys())


def wrap_operand(native_operand: Tensor) -> TensorHolder[Tensor]:
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
        return native_operand
    wrapped_operand = _TENSOR_TYPES[infer_tensor_package(native_operand)](native_operand)
    return wrapped_operand


def check_valid_package(native_operands):
    """
    Check if the operands belong to one of the supported packages.
    """
    operands_pkg = [infer_tensor_package(o) for o in native_operands]
    checks = [p in _SUPPORTED_PACKAGES for p in operands_pkg]
    if not all(checks):
        unknown = [f"{location}: {operands_pkg[location]}" for location, predicate in enumerate(checks) if predicate is False]
        unknown = formatters.array2string(unknown)
        message = f"""The operands should be ndarray-like objects from one of {_SUPPORTED_PACKAGES} packages.
The unsupported operands as a sequence of "position: package" is: \n{unknown}"""
        raise ValueError(message)

    return operands_pkg


def check_valid_operand_type(wrapped_operands):
    """
    Check if the wrapped operands are ndarray-like.
    """
    istensor = [o.istensor() for o in wrapped_operands]
    if not all(istensor):
        unknown = [
            f"{location}: {type(wrapped_operands[location].tensor)}"
            for location, predicate in enumerate(istensor)
            if predicate is False
        ]
        unknown = formatters.array2string(unknown)
        message = f"""The operands should be ndarray-like objects from one of {_SUPPORTED_PACKAGES} packages.
The unsupported operands as a sequence of "position: type" is: \n{unknown}"""
        raise ValueError(message)


def _wrapper_helper(operand: Tensor | TensorHolder, index: int, packages: list[str]) -> TensorHolder[Tensor]:
    if isinstance(operand, TensorHolder):
        msg = (
            "wrap_operands() is being called unnecessarily because at least one input is already a TensorHolder. "
            "Only public facing APIs should call wrap_operands(). "
            "Internal APIs should assume the operands are TensorHolder already."
            "Trying to wrap a TensorHolder will become an error in the future."
        )
        warnings.warn(msg, DeprecationWarning)
        return operand
    return _TENSOR_TYPES[packages[index]](operand)


def wrap_operands(native_operands: Sequence[Tensor]) -> Sequence[TensorHolder[Tensor]]:
    """
    Wrap the "native" operands so that package-agnostic API can be used.
    """

    operands_pkg = check_valid_package(native_operands)

    wrapped_operands = tuple(_wrapper_helper(o, i, operands_pkg) for i, o in enumerate(native_operands))

    check_valid_operand_type(wrapped_operands)

    return wrapped_operands


def to(operands: Sequence[TensorHolder], device_id, stream_holder) -> Sequence[TensorHolder]:
    """
    Copy the wrapped operands to the specified device_id (None or int) and return the
    wrapped operands on the device.
    """
    assert isinstance(operands, Sequence) and isinstance(operands[0], TensorHolder)
    return tuple(o.to(device_id, stream_holder) for o in operands)


def copy_(src: Sequence[TensorHolder], dest: Sequence[TensorHolder], stream_holder):
    """
    Copy the wrapped operands in dest to the corresponding wrapped operands in src.
    """
    for s, d in zip(src, dest, strict=True):
        if s.device_id == "cpu":
            # FIXME: This is probably an extra step because it's supported to copy directly
            # from cpu to device?
            s = s.to(d.device_id, stream_holder=stream_holder)
        d.copy_(s, stream_holder=stream_holder)
