# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
An abstract interface to certain package-provided operations.
"""

__all__ = ["Package"]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import nullcontext, AbstractContextManager
from typing import Any


class Package(ABC):
    @staticmethod
    @abstractmethod
    def get_current_stream(device_id):
        """
        Obtain the current stream on the device.

        Args:
            device_id: The id (ordinal) of the device.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_stream_pointer(stream):
        """
        Obtain the stream pointer.

        Args:
            stream: The stream object.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_stream_context(stream):
        """
        Create a context manager from the stream.

        Args:
            stream: The stream object.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_external_stream(device_id, stream_ptr):
        """
        Wrap a stream pointer into an external stream object.

        Args:
            device_id: The id (ordinal) of the device.
            stream: The stream pointer (int) to be wrapped.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_stream(device_id):
        """
        Create a new stream on the specified device.

        Args:
            device_id: The id (ordinal) of the device.
        """
        raise NotImplementedError


@dataclass
class StreamHolder:
    """A data class for easing CUDA stream manipulation.

    Attributes:
        ctx: A context manager for using the specified stream.
        device_id (int): The device ID where the encapsulated stream locates.
        obj: A foreign object that holds the stream alive.
        package (str):
        ptr (int): The address of the underlying ``cudaStream_t`` object.
    """

    ctx: AbstractContextManager[Any] = nullcontext()
    device_id: int = -2
    obj: Any = None
    package: str = ""
    ptr: int = 0
