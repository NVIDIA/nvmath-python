# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper host APIs for random device APIs.
"""

__all__ = [
    "DirectionVectorSet",
    "get_direction_vectors32",
    "get_scramble_constants32",
    "get_direction_vectors64",
    "get_scramble_constants64",
]

from nvmath.bindings.curand import (  # type: ignore
    DirectionVectorSet,
    get_direction_vectors32,
    get_scramble_constants32,
    get_direction_vectors64,
    get_scramble_constants64,
)
