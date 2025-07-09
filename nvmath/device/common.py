# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
import tempfile

import numpy as np

from .common_cuda import CodeType


__all__ = [
    "make_tensor",
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
    #
    "execution": "A string specifying the execution method, can be ``'Block'`` or ``'Thread'``.",
}


# TODO: maybe pre-compile regular expression
def make_binary_tempfile(content, suffix):
    # TODO: may need to set it False for Windows? (refer to Python API doc)
    tmp = tempfile.NamedTemporaryFile(mode="w+b", suffix=suffix, delete=True)  # noqa: SIM115
    tmp.write(content)
    tmp.flush()
    return tmp


def check_in(name, arg, set):
    if arg not in set:
        raise ValueError(f"{name} must be in {set} ; got {name} = {arg}")


def check_not_in(name, arg, set):
    if arg in set:
        raise ValueError(f"{name} must not be any of those value {set} ; got {name} = {arg}")


def check_contains(set, key):
    if key not in set:
        raise ValueError(f"{key} must be in {set}")


def check_code_type(code_type):
    if isinstance(code_type, CodeType):
        if code_type.cc.major < 7:
            raise ValueError(f"code_type.cc.major must be >= 7 ; got code_type.cc.major = {code_type.cc.major}")
        if code_type.cc.minor < 0:
            raise ValueError(f"code_type.cc.minor must be >= 0 ; got code_type.cc.minor = {code_type.cc.minor}")
        check_in("code_type.kind", code_type.kind, ["lto"])
    else:
        raise ValueError(f"code_type should be an instance of CodeType ; got code_type = {code_type}")


def pad_or_truncate(list, target_len):
    return list[:target_len] + [0] * (target_len - len(list))


class Layout:
    """Layout for the OpaqueTensor"""

    def __init__(self):
        raise RuntimeError("Layout should not be called directly.")

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Number of valid elements in a tensor. This is simply a product of all
        shape dimensions.
        """
        pass

    @property
    @abstractmethod
    def cosize(self) -> int:
        """
        Returns a distance from last element of a tensor to its first element.
        It describes how many elements does the argument layout span.
        """
        pass


class OpaqueTensor:
    buffer: np.ndarray
    layout: Layout
    leading_dimension: int | None

    def __init__(self, *args):
        raise RuntimeError("OpaqueTensor should not be called directly outside of a numba.cuda.jit(...) kernel.")


def make_tensor(array: np.ndarray, layout: Layout) -> OpaqueTensor:
    raise RuntimeError("make_tensor should not be called directly outside of a numba.cuda.jit(...) kernel.")


def axpby(alpha: float, x_tensor: OpaqueTensor, beta: float, y_tensor: OpaqueTensor) -> None:
    raise RuntimeError("axpby should not be called directly outside of a numba.cuda.jit(...) kernel.")


def copy(src: OpaqueTensor, dst: OpaqueTensor):
    raise RuntimeError("copy should not be called directly outside of a numba.cuda.jit(...) kernel.")


def copy_fragment(src: OpaqueTensor, dst: OpaqueTensor):
    raise RuntimeError("copy_fragment should not be called directly outside of a numba.cuda.jit(...) kernel.")


def clear(arr: OpaqueTensor):
    raise RuntimeError("copy_c should not be called directly outside of a numba.cuda.jit(...) kernel.")


def copy_wait():
    raise RuntimeError("copy_wait should not be called directly outside of a numba.cuda.jit(...) kernel.")
