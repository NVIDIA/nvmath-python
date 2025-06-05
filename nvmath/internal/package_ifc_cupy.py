# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to CuPy operations.
"""

__all__ = ["CupyPackage"]

import cupy as cp

from . import utils
from .package_ifc import Package


class CupyPackage(Package[cp.cuda.Stream]):
    @staticmethod
    def get_current_stream(device_id: int):
        with utils.device_ctx(device_id):
            stream = cp.cuda.get_current_stream()
        return stream

    @staticmethod
    def to_stream_pointer(stream: cp.cuda.Stream) -> int:
        return stream.ptr

    @staticmethod
    def to_stream_context(stream: cp.cuda.Stream):
        return stream

    @staticmethod
    def create_external_stream(device_id: int, stream_ptr: int) -> cp.cuda.ExternalStream:
        return cp.cuda.ExternalStream(stream_ptr)
