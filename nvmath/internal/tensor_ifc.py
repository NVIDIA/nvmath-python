# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use tensors (or ndarray-like objects) from different libraries.
"""

from __future__ import annotations  # allows typehint of class methods to return the self class

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal, Generic, Protocol, TypeVar
from types import ModuleType

from . import typemaps
from .package_ifc import StreamHolder


class AnyTensor(Protocol):
    """Any supported external tensor object such as NumPy, CuPy, and PyTorch arrays."""

    # This empty protocol class is just a placeholder to make type checking more helpful.
    # Since the class is empty, it doesn't assume anything is true about the implementation
    # classes. Used to type hint any object that TensorHolder knows how to wrap.
    pass


"""
A generic type for the third-party Tensor which a given TensorHolder implementation wraps
around.
"""
Tensor = TypeVar("Tensor")


class TensorHolder(ABC, Generic[Tensor]):
    """
    A simple wrapper type for tensors to make the API package-agnostic.

    Methods of a TensorHolder should always return a TensorHolder instead of a Tensor (one
    of the wrapped classes) in order to prevent implementation details from the various
    Tensor implementations from leaking into nvmath-python.

    Tensors from the user should be immediately wrapped with a TensorHolder and should
    remain wrapped until just before returning to the user.
    """

    name: str
    module: ModuleType
    name_to_dtype: dict[str, Any]

    def __init__(self, tensor: Tensor):
        self.tensor: Tensor = tensor

    @property
    @abstractmethod
    def data_ptr(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> Literal["cuda"] | Literal["cpu"]:
        """The type of the device which stores the tensor."""
        raise NotImplementedError

    @property
    @abstractmethod
    def device_id(self) -> int | Literal["cpu"]:
        """The device ordinal of the device storing the tensor."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> str:
        """Name of the data type"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def empty(cls, shape: Sequence[int], device_id: int | Literal["cpu"], **context: Any) -> TensorHolder[Tensor]:
        """Create an empty TensorHolder of the specified shape and data type."""
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> Sequence[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def strides(self) -> Sequence[int]:
        raise NotImplementedError

    @abstractmethod
    def to(self, device_id: int | Literal["cpu"], stream_holder: StreamHolder | None) -> TensorHolder:
        """Copy the TensorHolder to a different device.

        No copy is performed if the TensorHolder is already on the requested device.
        """
        raise NotImplementedError

    @abstractmethod
    def copy_(self, src: TensorHolder, stream_holder: StreamHolder | None) -> None:
        """Overwrite self.tensor (in-place) with a copy of src."""
        raise NotImplementedError

    @staticmethod
    def create_name_dtype_map(conversion_function: Callable[[str], Any], exception_type: type[Exception]) -> dict[str, Any]:
        """
        Create a map between CUDA data type names and the corresponding package dtypes for
        supported data types.
        """
        names = typemaps.NAME_TO_DATA_TYPE.keys()
        name_to_dtype = {}
        for name in names:
            try:
                name_to_dtype[name] = conversion_function(name)
            except exception_type:
                pass
        return name_to_dtype

    @abstractmethod
    def istensor(self) -> bool:
        """Return whether self.tensor is the expected type."""
        raise NotImplementedError

    @abstractmethod
    def reshape(self, shape: Sequence[int], *, copy: bool | None = None) -> TensorHolder[Tensor]:
        """Reshapes tensor without changing its data.

        Args:
            shape: a new shape compatible with the original shape.

            copy (Optional[bool]): whether or not to copy the input tensor. If True, the
                function must always copy. If False, the function must never copy. If None,
                the function must avoid copying, if possible, and may copy otherwise.
        """
        raise NotImplementedError
