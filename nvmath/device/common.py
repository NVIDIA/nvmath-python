# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from .common_cuda import MAX_SUPPORTED_CC, MIN_SUPPORTED_CC, CodeType, ComputeCapability, get_current_device_cc

__all__ = [
    "make_tensor",
    "make_fragment_like",
    "OpaqueTensor",
    "Layout",
    "axpby",
    "copy",
    "copy_fragment",
    "clear",
    "copy_wait",
]

SHARED_DEVICE_DOCSTRINGS = {
    "compiler": "A string to specify the compiler for the device code, currently supports ``None`` (default) and ``'numba'``",
    #
    "precision": """\
The computation precision specified as a numpy float dtype, currently supports ``numpy.float16``, ``numpy.float32`` and
``numpy.float64``.""".replace("\n", " "),
    #
    "code_type": "The target GPU code and compute-capability.",
    "sm": "Target mathdx compute-capability.",
    #
    "execution": "A string specifying the execution method, can be ``'Block'`` or ``'Thread'``.",
}


# TODO: maybe pre-compile regular expression
def make_binary_tempfile(content, suffix: str) -> tempfile._TemporaryFileWrapper:
    """Write `content` to a temporary file with the given `suffix`.

    A closed file object returned; it is the user's responsibility to delete the file when
    finished.

    .. seealso:: :py:func:`delete_binary_tempfiles`

    """
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
    return tmp


def delete_binary_tempfiles(filenames: list[str]):
    for name in filenames:
        if os.path.isfile(name):
            os.remove(name)


def check_in(name, value, coll, format="{name} must be in {coll_str} ; got {name} = {value}"):
    if value not in coll:
        coll_str = ", ".join(f'"{t}"' for t in coll)
        msg = format.format(name=name, value=value, coll_str=coll_str)
        raise ValueError(msg)


def check_not_in(name, value, coll, format="{name} must not be any of those value {coll_str} ; got {name} = {value}"):
    if value in coll:
        coll_str = ", ".join(f'"{t}"' for t in coll)
        msg = format.format(name=name, value=value, coll_str=coll_str)
        raise ValueError(msg)


def check_contains(set, key):
    check_in("", key, set, "{arg} must be in {set}")


def parse_sm(sm: Any) -> ComputeCapability:
    if sm is None:
        sm = get_current_device_cc()
    else:
        if not isinstance(sm, ComputeCapability) and not isinstance(sm, int):
            raise ValueError(f"sm should be a ComputeCapability or an int; got sm = {sm}")
        if isinstance(sm, int):
            sm = ComputeCapability(sm // 10, sm % 10)

    return sm


def check_sm(sm, library_name: str, var_name: str = "sm"):
    if not isinstance(sm, ComputeCapability):
        raise ValueError(f"{var_name} should be an instance of ComputeCapability ; got {var_name} = {sm}")
    if sm < MIN_SUPPORTED_CC:
        raise RuntimeError(f"Minimal compute capability {MIN_SUPPORTED_CC} is required by {library_name}, got {sm}")
    if sm > MAX_SUPPORTED_CC:
        raise RuntimeError(f"The maximum compute capability currently supported by device APIs is {MAX_SUPPORTED_CC}, got {sm}")
    if sm.minor < 0:
        raise ValueError(f"{var_name}.minor must be >= 0 ; got {var_name}.minor = {sm.minor}")


def parse_code_type(code_type: Any) -> CodeType:
    if not isinstance(code_type, Sequence) or len(code_type) != 2:
        raise ValueError(f"code_type should be an instance of CodeType or a 2-tuple ; got code_type = {code_type}")

    return CodeType(code_type[0], ComputeCapability(*code_type[1]))


def check_code_type(code_type, library_name):
    if not isinstance(code_type, CodeType):
        raise ValueError(f"code_type should be an instance of CodeType ; got code_type = {code_type}")
    check_sm(code_type.cc, library_name, "code_type.cc")
    check_in("code_type.kind", code_type.kind, ["lto"])


def pad_or_truncate(list, target_len):
    return list[:target_len] + [0] * (target_len - len(list))


class Layout:
    """
    Layout for the :py:class:`nvmath.device.OpaqueTensor`.

    .. note:: Do not create directly, use appropriate method from
        :py:func:`nvmath.device.Matmul`. Refer to
        :cublasdx_doc:`api/other_tensors.html#imported-tensor-utilities`
        for guidance on which method to use.
    """

    def __init__(self):
        raise RuntimeError("Layout should not be called directly.")

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Number of valid elements in a tensor. This is simply a product of all
        shape dimensions.

        Refer to the cuBLASDx documentation for more details on how to use this attribute:
        :cublasdx_doc:`api/other_tensors.html#imported-tensor-utilities`
        """
        pass

    @property
    @abstractmethod
    def cosize(self) -> int:
        """
        Returns a distance from last element of a tensor to its first element.
        It describes how many elements does the argument layout span.

        Refer to the cuBLASDx documentation for more details on how to use this attribute:
        :cublasdx_doc:`api/other_tensors.html#imported-tensor-utilities`
        """
        pass

    @property
    @abstractmethod
    def alignment(self) -> int:
        """
        Returns the required alignment (in bytes) for the tensor data buffer.

        Refer to the cuBLASDx documentation for more details on how to use this attribute:
        :cublasdx_doc:`api/other_tensors.html#imported-tensor-utilities`
        """
        pass


class OpaqueTensor:
    """
    Abstraction over the cuBLASDx tensor type (an alias of the CuTe tensor type).
    The CuTe tensor layout is powerful and supports layouts not provided by NumPy,
    so this a bridge to add this functionality to Python.

    .. note:: Do not create directly, use :py:func:`nvmath.device.make_tensor`.

    Refer to the cuBLASDx documentation for more details on how to use this class:
    :cublasdx_doc:`api/other_tensors.html#tensors`
    """

    buffer: np.ndarray
    layout: Layout
    leading_dimension: int | None

    def __init__(self, *args):
        raise RuntimeError("OpaqueTensor should not be called directly outside of a numba.cuda.jit(...) kernel.")


def make_tensor(array: np.ndarray, layout: Layout) -> OpaqueTensor:
    """
    make_tensor is a helper function for creating
    :py:class:`nvmath.device.OpaqueTensor` objects.

    Args:
        array (np.ndarray): The input array to be wrapped as an OpaqueTensor.
        layout (Layout): The layout of the tensor, which describes how the data is
            organized in memory.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    :cublasdx_doc:`api/other_tensors.html#create-tensor-other-label`
    """
    raise RuntimeError("make_tensor should not be called directly outside of a numba.cuda.jit(...) kernel.")


def make_fragment_like(tensor: OpaqueTensor, dtype) -> OpaqueTensor:
    """
    make_fragment_like is a helper function for creating register fragments with
    the same layout as input tensor, but different dtype.

    Args:
        tensor (OpaqueTensor): The input tensor to be used as a template for the fragment.
        dtype: The data type of the fragment to be created.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    :cublasdx_doc:`api/other_tensors.html#imported-tensor-utilities`
    """
    raise RuntimeError("make_fragment_like should not be called directly outside of a numba.cuda.jit(...) kernel.")


def axpby(alpha: float, x_tensor: OpaqueTensor, beta: float, y_tensor: OpaqueTensor) -> None:
    """
    AXPBY operation: y = alpha * x + beta * y

    Args:
        alpha (float): Scalar multiplier for x_tensor.
        x_tensor (OpaqueTensor): Input tensor x.
        beta (float): Scalar multiplier for y_tensor.
        y_tensor (OpaqueTensor): Input/output tensor y, which will be updated
            with the result.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    :cublasdx_doc:`api/other_tensors.html#imported-tensor-utilities`
    """
    raise RuntimeError("axpby should not be called directly outside of a numba.cuda.jit(...) kernel.")


def copy(src: OpaqueTensor, dst: OpaqueTensor, alignment=None):
    """
    Copies data from the source tensor to the destination tensor.

    Args:
        src (OpaqueTensor): The source tensor to copy from.
        dst (OpaqueTensor): The destination tensor to copy to.
        alignment (int, optional): The alignment (in bytes) for the copy operation.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    :cublasdx_doc:`api/other_tensors.html#cooperative-global-shared-copying`
    """
    raise RuntimeError("copy should not be called directly outside of a numba.cuda.jit(...) kernel.")


def copy_fragment(src: OpaqueTensor, dst: OpaqueTensor, alignment=None):
    """
    A bidirectional copying method to copy data between register fragments and
    global memory tensors.

    Args:
        src (OpaqueTensor): The source tensor to copy from.
        dst (OpaqueTensor): The destination tensor to copy to.
        alignment (int, optional): The alignment (in bytes) for the copy operation.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    :cublasdx_doc:`api/other_tensors.html#copying-registers-tensors`
    """
    raise RuntimeError("copy_fragment should not be called directly outside of a numba.cuda.jit(...) kernel.")


def clear(arr: OpaqueTensor):
    """
    Clears the contents of the given tensor by setting all elements to zero.

    Args:
        arr (OpaqueTensor): The tensor to be cleared.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    :cublasdx_doc:`api/other_tensors.html#imported-tensor-utilities`
    """
    raise RuntimeError("clear should not be called directly outside of a numba.cuda.jit(...) kernel.")


def copy_wait():
    """
    Creates synchronization point. It has to be called after :py:func:`nvmath.device.copy`
    to ensure that the copy operation has completed before any subsequent
    operations are executed.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    :cublasdx_doc:`api/other_tensors.html#cooperative-global-shared-copying`
    """
    raise RuntimeError("copy_wait should not be called directly outside of a numba.cuda.jit(...) kernel.")


def check_positive_integer_sequence(arg, arg_name, min_len, max_len):
    if not isinstance(arg, Sequence):
        raise ValueError(f'Parameter "{arg_name}" must be a sequence. Got: {type(arg).__name__}.')

    for i, x in enumerate(arg):
        if not isinstance(x, (int, np.integer)):
            raise ValueError(
                f'Parameter "{arg_name}" must be a sequence of positive integers. '
                + f"Element {arg_name}[{i}] is {type(x).__name__}."
            )
        if x <= 0:
            raise ValueError(f'Parameter "{arg_name}" values must be positive integers. Got {arg_name}[{i}] = {x}.')

    if min_len == max_len:
        if len(arg) != min_len:
            raise ValueError(f'Parameter "{arg_name}" must be a sequence of length {min_len}. Got length: {len(arg)}.')
    else:
        if len(arg) > max_len or len(arg) < min_len:
            raise ValueError(f'Parameter "{arg_name}" must define {min_len} to {max_len} values. Got {len(arg)} value(s).')
