# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

SUPPORTED_TYPES = [
    "float32",
    "float64",
    "complex64",
    "complex128",
]


def check_dtype(dtype, operand_name):
    if dtype not in SUPPORTED_TYPES:
        raise ValueError(f"The dtype of operand {operand_name} ({dtype}) is not supported.")
