# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to Torch operations.
"""

__all__ = ["TorchPackage"]

import torch

from .package_ifc import Package


class TorchPackage(Package):
    @staticmethod
    def get_current_stream(device_id):
        return torch.cuda.current_stream(device=device_id)

    @staticmethod
    def to_stream_pointer(stream):
        return stream.cuda_stream

    @staticmethod
    def to_stream_context(stream):
        return torch.cuda.stream(stream)

    @classmethod
    def create_external_stream(device_id, stream_ptr):
        return torch.cuda.ExternalStream(stream_ptr, device=device_id)

    @staticmethod
    def create_stream(device_id):
        stream = torch.cuda.Stream(device=device_id)
        return stream
