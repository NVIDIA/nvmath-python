# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to cuda.core operations.
"""

__all__ = ["CUDAPackage"]

import contextlib

import cuda.core.experimental as ccx

from .package_ifc import Package


class CUDAPackage(Package[ccx.Stream]):
    @staticmethod
    def get_current_stream(device_id: int):
        return ccx.Device(device_id).default_stream

    @staticmethod
    def to_stream_pointer(stream: ccx.Stream) -> int:  # type: ignore[override]
        return int(stream.handle)

    @staticmethod
    def to_stream_context(stream: ccx.Stream):  # type: ignore[override]
        return contextlib.nullcontext(stream)

    @staticmethod
    def create_external_stream(device_id: int, stream_ptr: int) -> ccx.Stream:
        return ccx.Stream.from_handle(stream_ptr)

    @classmethod
    def create_stream(cls, external: ccx.Stream) -> ccx.Stream:  # type: ignore[override]
        return external
