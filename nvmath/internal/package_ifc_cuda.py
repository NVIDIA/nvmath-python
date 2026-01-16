# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to cuda.core operations.
"""

__all__ = ["CUDAPackage"]

import contextlib

try:
    from cuda.core import Device, Stream
except ImportError:
    from cuda.core.experimental import Device, Stream

from .package_ifc import Package


class CUDAPackage(Package[Stream]):
    @staticmethod
    def get_current_stream(device_id: int):
        prev_device = Device()
        prev_device_id = prev_device.device_id
        device = Device(device_id)
        try:
            # we must ensure the context is set, otherwise
            # the stream.__hash__ can fail
            device.set_current()
            return device.default_stream
        finally:
            if prev_device_id != device_id:
                prev_device.set_current()

    @staticmethod
    def to_stream_pointer(stream: Stream) -> int:  # type: ignore[override]
        return int(stream.handle)

    @staticmethod
    def to_stream_context(stream: Stream):  # type: ignore[override]
        return contextlib.nullcontext(stream)

    @staticmethod
    def create_external_stream(device_id: int, stream_ptr: int) -> Stream:
        return Stream.from_handle(stream_ptr)

    @classmethod
    def create_stream(cls, external: Stream) -> Stream:  # type: ignore[override]
        return external
