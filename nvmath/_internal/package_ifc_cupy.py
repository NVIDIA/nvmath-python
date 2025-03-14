# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to CuPy operations.
"""

__all__ = ["CupyPackage"]

try:
    import cupy as cp
except ImportError:
    cp = None

from . import utils
from .package_ifc import Package


class CupyPackage(Package):
    @staticmethod
    def get_current_stream(device_id):
        with utils.device_ctx(device_id):
            stream = cp.cuda.get_current_stream()
        return stream

    @staticmethod
    def to_stream_pointer(stream):
        return stream.ptr

    @staticmethod
    def to_stream_context(stream):
        return stream

    @staticmethod
    def create_external_stream(device_id, stream_ptr):
        return cp.cuda.ExternalStream(stream_ptr)

    @staticmethod
    def create_stream(device_id):
        with utils.device_ctx(device_id):
            stream = cp.cuda.Stream(null=False, non_blocking=False, ptds=False)
        return stream
