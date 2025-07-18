# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

class ComputeType(IntEnum):
    COMPUTE_16F = ...
    COMPUTE_16F_PEDANTIC = ...
    COMPUTE_32F = ...
    COMPUTE_32F_PEDANTIC = ...
    COMPUTE_32F_FAST_16F = ...
    COMPUTE_32F_FAST_16BF = ...
    COMPUTE_32F_FAST_TF32 = ...
    COMPUTE_64F = ...
    COMPUTE_64F_PEDANTIC = ...
    COMPUTE_32I = ...
    COMPUTE_32I_PEDANTIC = ...
