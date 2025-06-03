# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to Torch operations.
"""

__all__ = ["TorchPackage"]

import torch

from .package_ifc import Package


class TorchPackage(Package[torch.cuda.Stream]):
    @staticmethod
    def get_current_stream(device_id: int) -> torch.cuda.Stream:
        return torch.cuda.current_stream(device=device_id)

    @staticmethod
    def to_stream_pointer(stream: torch.cuda.Stream) -> int:  # type: ignore[override]
        return stream.cuda_stream

    @staticmethod
    def to_stream_context(stream: torch.cuda.Stream) -> torch.cuda.StreamContext:  # type: ignore[override]
        return torch.cuda.stream(stream)

    @staticmethod
    def create_external_stream(device_id: int, stream_ptr: int):
        return torch.cuda.ExternalStream(stream_ptr, device=device_id)
