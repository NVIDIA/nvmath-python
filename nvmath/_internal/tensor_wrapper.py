# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Entry point to using tensors from different libraries seamlessly.
"""

__all__ = ["infer_tensor_package", "wrap_operand", "wrap_operands", "to", "copy_"]

from collections.abc import Sequence
import functools

import numpy as np

from . import formatters
from .tensor_ifc_cupy import CupyTensor
from .tensor_ifc_numpy import NumpyTensor


_TENSOR_TYPES = {"cupy": CupyTensor, "numpy": NumpyTensor}

# Optional modules
try:
    import torch
    from .tensor_ifc_torch import TorchTensor

    _TENSOR_TYPES["torch"] = TorchTensor
    torch_asarray = functools.partial(torch.as_tensor, device="cuda")
except ImportError:
    torch = None  # type: ignore
    torch_asarray = None  # type: ignore

_SUPPORTED_PACKAGES = tuple(_TENSOR_TYPES.keys())


def infer_tensor_package(tensor):
    """
    Infer the package that defines this tensor.
    """
    if issubclass(tensor.__class__, np.ndarray):
        return "numpy"
    module = tensor.__class__.__module__
    return module.split(".")[0]


def _get_backend_asarray_func(backend):
    """
    Infer the package that defines this tensor.
    """
    if backend is torch:
        return torch_asarray
    else:
        return backend.asarray


def wrap_operand(native_operand):
    """
    Wrap one "native" operand so that package-agnostic API can be used.
    """
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


def wrap_operands(native_operands):
    """
    Wrap the "native" operands so that package-agnostic API can be used.
    """

    operands_pkg = check_valid_package(native_operands)

    wrapped_operands = tuple(_TENSOR_TYPES[operands_pkg[i]](o) for i, o in enumerate(native_operands))

    check_valid_operand_type(wrapped_operands)

    return wrapped_operands


def to(operands, device, stream_holder):
    """
    Copy the wrapped operands to the specified device ('cpu' or int) and return the
    wrapped operands on the device.
    """
    if not isinstance(operands, Sequence):
        return wrap_operand(operands.to(device, stream_holder))

    operands = tuple(o.to(device, stream_holder) for o in operands)

    return wrap_operands(operands)


def copy_(src, dest, stream_holder):
    """
    Copy the wrapped operands in dest to the corresponding wrapped operands in src.
    """
    for s, d in zip(src, dest, strict=True):
        if s.device_id is None:
            s = wrap_operand(s.to(d.device_id, stream_holder=stream_holder))
        d.copy_(s.tensor, stream_holder=stream_holder)
