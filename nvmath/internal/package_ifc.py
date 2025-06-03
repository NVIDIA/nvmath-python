# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
An abstract interface to certain package-provided operations.

Even though cuda.core does not have the concept of a "current stream" or a "stream context",
we promise to honor third-party CUDA stream contexts if they match the input operand's
package. The interface class defined in this module wraps around those stream concepts to
make them uniform.

The strategy is to use cuda.core everywhere internally except for context managers which
will need to wrap back around to the external implementation.
"""

__all__ = ["Package", "StreamHolder"]

from abc import ABC, abstractmethod
from collections.abc import Hashable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import cuda.core.experimental as ccx


class AnyStream(Hashable, Protocol):
    """Any supported Stream object such as a Stream from cuda.core, CuPy, or PyTorch."""

    # This empty protocol class is just a placeholder to make type checking more helpful.
    # Since the class is empty, it doesn't assume anything is true about the implementation
    # classes.
    pass


"""
A generic type for the third-party Stream which a given Package implementation wraps
around.
"""
Stream = TypeVar("Stream")


class Package(ABC, Generic[Stream]):
    @staticmethod
    @abstractmethod
    def get_current_stream(device_id: int) -> Stream:
        """
        Obtain the current stream on the device.

        Args:
            device_id: The id (ordinal) of the device.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_stream_pointer(stream: Stream) -> int:
        """
        Obtain the stream pointer.

        Args:
            stream: The stream object.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_stream_context(stream: Stream) -> AbstractContextManager[Stream]:
        """
        Create a context manager from the stream.

        Args:
            stream: The stream object.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_external_stream(device_id: int, stream_ptr: int) -> Stream:
        """
        Wrap a stream pointer into an external stream object.

        Args:
            device_id: The id (ordinal) of the device.
            stream: The stream pointer (int) to be wrapped.
        """
        raise NotImplementedError

    @classmethod
    def create_stream(cls, external: Stream) -> ccx.Stream:
        """
        Wrap an external Stream object into a cuda.core.Stream.

        Args:
            external: The external Stream object.
        """
        return ccx.Stream.from_handle(cls.to_stream_pointer(external))


@dataclass
class StreamHolder(Generic[Stream]):
    """A data class for easing CUDA stream manipulation.

    Attributes:
        ctx: A context manager for using the specified stream.
        device_id (int): The device ID where the encapsulated stream locates.
        external: A foreign object that holds the stream alive.
        obj: The cuda.core Stream object wrapping external.
        package (str): The name of the package to which the external stream belongs.
        ptr (int): The address of the underlying ``cudaStream_t`` object.
    """

    ctx: AbstractContextManager[Stream]
    device_id: int
    external: Stream
    obj: ccx.Stream
    package: str
    ptr: int
