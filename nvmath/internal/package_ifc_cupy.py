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

# Using the functional API is faster than setting a device context
if int(cp.__version__.split(".")[0]) >= 13:
    _get_current_stream = cp.cuda.get_current_stream
else:

    def _get_current_stream(device_id: int):
        with utils.device_ctx(device_id):
            stream = cp.cuda.get_current_stream()
        return stream


# Monkey patch older versions of CuPy, so that Streams are hashable
# NOTE: We choose not to patch the BaseStream/_BaseStream class because of name change
if int(cp.__version__.split(".")[0]) < 11:
    cp.cuda.Stream.__hash__ = lambda self: hash(self.ptr)
    cp.cuda.ExternalStream.__hash__ = lambda self: hash(self.ptr)


class CupyPackage(Package[cp.cuda.Stream]):
    @staticmethod
    def get_current_stream(device_id: int):
        return _get_current_stream(device_id)

    @staticmethod
    def to_stream_pointer(stream: cp.cuda.Stream) -> int:
        return stream.ptr

    @staticmethod
    def to_stream_context(stream: cp.cuda.Stream):
        return stream

    @staticmethod
    def create_external_stream(device_id: int, stream_ptr: int) -> cp.cuda.ExternalStream:
        return cp.cuda.ExternalStream(stream_ptr)
