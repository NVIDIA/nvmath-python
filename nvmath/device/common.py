# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
import os
import tempfile
from collections.abc import Sequence
from typing import Any

import numpy as np

from .common_cuda import MAX_SUPPORTED_CC, MIN_SUPPORTED_CC, CodeType, ComputeCapability, get_current_device_cc


__all__ = [
    "make_tensor",
    "OpaqueTensor",
    "Layout",
    "Partition",
    "Partitioner",
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


def check_in(name, arg, set):
    if arg not in set:
        raise ValueError(f"{name} must be in {set} ; got {name} = {arg}")


def check_not_in(name, arg, set):
    if arg in set:
        raise ValueError(f"{name} must not be any of those value {set} ; got {name} = {arg}")


def check_contains(set, key):
    if key not in set:
        raise ValueError(f"{key} must be in {set}")


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
        https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#imported-tensor-utilities
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
        https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#imported-tensor-utilities
        """
        pass

    @property
    @abstractmethod
    def cosize(self) -> int:
        """
        Returns a distance from last element of a tensor to its first element.
        It describes how many elements does the argument layout span.

        Refer to the cuBLASDx documentation for more details on how to use this attribute:
        https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#imported-tensor-utilities
        """
        pass


class OpaqueTensor:
    """
    Abstraction over the cuBLASDx tensor type (an alias of the CuTe tensor type).
    The CuTe tensor layout is powerful and supports layouts not provided by NumPy,
    so this a bridge to add this functionality to Python.

    .. note:: Do not create directly, use :py:func:`nvmath.device.make_tensor`.

    Refer to the cuBLASDx documentation for more details on how to use this class:
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#tensors
    """

    buffer: np.ndarray
    layout: Layout
    leading_dimension: int | None

    def __init__(self, *args):
        raise RuntimeError("OpaqueTensor should not be called directly outside of a numba.cuda.jit(...) kernel.")


class Partition:
    """
    Partition of a global memory tensor into a partitioned tensor. This is used
    for accessing the C matrix when working with register fragments.

    .. note:: Do not create directly, use
        :py:func:`nvmath.device.Partitioner.partition_like_C`.

    Refer to the cuBLASDx documentation for more details on how to use this class:
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#partitioner-register-tensor-other-label
    """

    def __init__(self, *args):
        raise RuntimeError("Partition should not be called directly")


class Partitioner:
    """
    Partitioner is an abstraction for partitioning a global memory tensor into a
    partitioned tensor.

    .. note:: Do not create directly, use
        :py:func:`nvmath.device.Matmul.suggest_partitioner`.

    Refer to the cuBLASDx documentation for more details on how to use this class:
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#partitioner-register-tensor-other-label
    """

    def __init__(self, *args):
        raise RuntimeError("Partitioner should not be called directly")

    @abstractmethod
    def partition_like_C(self, gmem_c: OpaqueTensor) -> Partition:
        """
        Partitions the given global memory tensor `gmem_c` into a partitioned tensor.
        The partitioned tensor is used for accessing the C matrix when working
        with register fragment.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def map_fragment_index(self, fragment_index: int) -> tuple[int, int]:
        """
        Maps the given fragment index to a global memory index.
        This is used to access the correct element in the partitioned tensor.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def is_thread_active(self) -> bool:
        """
        Checks if the current thread takes part in GEMM.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def is_predicated(self) -> bool:
        """
        Checks if the current thread is predicated.
        This is used to determine if the thread should execute the kernel.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def is_index_in_bounds(self, index: int) -> bool:
        """
        Checks if the given index is within the bounds of the partitioned tensor.
        This is used to prevent out-of-bounds access in the kernel.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")


def make_tensor(array: np.ndarray, layout: Layout) -> OpaqueTensor:
    """
    make_tensor is a helper function for creating
    :py:class:`nvmath.device.OpaqueTensor` objects.

    Args:
        array (np.ndarray): The input array to be wrapped as an OpaqueTensor.
        layout (Layout): The layout of the tensor, which describes how the data is
            organized in memory.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#create-tensor-other-label
    """
    raise RuntimeError("make_tensor should not be called directly outside of a numba.cuda.jit(...) kernel.")


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
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#imported-tensor-utilities
    """
    raise RuntimeError("axpby should not be called directly outside of a numba.cuda.jit(...) kernel.")


def copy(src: OpaqueTensor, dst: OpaqueTensor, alignment=None):
    """
    Copies data from the source tensor to the destination tensor.

    Args:
        src (OpaqueTensor): The source tensor to copy from.
        dst (OpaqueTensor): The destination tensor to copy to.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#cooperative-global-shared-copying
    """
    raise RuntimeError("copy should not be called directly outside of a numba.cuda.jit(...) kernel.")


def copy_fragment(src: OpaqueTensor, dst: OpaqueTensor):
    """
    A bidirectional copying method to copy data between register fragments and
    global memory tensors.

    Args:
        src (OpaqueTensor): The source tensor to copy from.
        dst (OpaqueTensor): The destination tensor to copy to.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#copying-registers-tensors
    """
    raise RuntimeError("copy_fragment should not be called directly outside of a numba.cuda.jit(...) kernel.")


def clear(arr: OpaqueTensor):
    """
    Clears the contents of the given tensor by setting all elements to zero.

    Args:
        arr (OpaqueTensor): The tensor to be cleared.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#imported-tensor-utilities
    """
    raise RuntimeError("clear should not be called directly outside of a numba.cuda.jit(...) kernel.")


def copy_wait():
    """
    Creates synchronization point. It has to be called after :py:func:`nvmath.device.copy`
    to ensure that the copy operation has completed before any subsequent
    operations are executed.

    Refer to the cuBLASDx documentation for more details on how to use this function:
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#cooperative-global-shared-copying
    """
    raise RuntimeError("copy_wait should not be called directly outside of a numba.cuda.jit(...) kernel.")
