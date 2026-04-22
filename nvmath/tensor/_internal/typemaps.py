# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["get_default_compute_type_from_dtype_name", "get_supported_compute_types", "get_compute_type_name"]

import threading

from nvmath.bindings.cutensor import ComputeDesc  # type: ignore

DEFAULT_COMPUTE_TYPE: dict[str, int] = {}  # dtype_name -> default compute type
SUPPORTED_COMPUTE_TYPES: dict[str, set[int]] = {}  # dtype_name -> set of compute types supported
COMPUTE_TYPE_TO_NAME: dict[int, str] = {}  # compute type -> compute type name
_typemap_lock = threading.Lock()


def _maybe_initialize_compute_type_map():
    # https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcreatecontraction
    global DEFAULT_COMPUTE_TYPE, SUPPORTED_COMPUTE_TYPES, COMPUTE_TYPE_TO_NAME
    if not DEFAULT_COMPUTE_TYPE:  # fast path if the map is already initialized
        with _typemap_lock:
            if not DEFAULT_COMPUTE_TYPE:  # double-check to avoid race condition
                DEFAULT_COMPUTE_TYPE = {
                    "float16": ComputeDesc.COMPUTE_32F(),
                    "bfloat16": ComputeDesc.COMPUTE_32F(),
                    "float32": ComputeDesc.COMPUTE_32F(),
                    "complex64": ComputeDesc.COMPUTE_32F(),
                    "complex128": ComputeDesc.COMPUTE_64F(),
                    "float64": ComputeDesc.COMPUTE_64F(),
                }
                SUPPORTED_COMPUTE_TYPES = {
                    "float16": {ComputeDesc.COMPUTE_32F()},
                    "bfloat16": {ComputeDesc.COMPUTE_32F()},
                    "float32": {
                        ComputeDesc.COMPUTE_32F(),
                        ComputeDesc.COMPUTE_TF32(),
                        ComputeDesc.COMPUTE_3XTF32(),
                        ComputeDesc.COMPUTE_16F(),
                        ComputeDesc.COMPUTE_16BF(),
                        ComputeDesc.COMPUTE_4X16F(),
                        ComputeDesc.COMPUTE_9X16BF(),
                    },
                    "complex64": {
                        ComputeDesc.COMPUTE_32F(),
                        ComputeDesc.COMPUTE_TF32(),
                        ComputeDesc.COMPUTE_3XTF32(),
                        ComputeDesc.COMPUTE_16F(),
                        ComputeDesc.COMPUTE_16BF(),
                        ComputeDesc.COMPUTE_4X16F(),
                        ComputeDesc.COMPUTE_9X16BF(),
                    },
                    "complex128": {
                        ComputeDesc.COMPUTE_32F(),
                        ComputeDesc.COMPUTE_64F(),
                        ComputeDesc.COMPUTE_8XINT8(),
                    },
                    "float64": {
                        ComputeDesc.COMPUTE_32F(),
                        ComputeDesc.COMPUTE_64F(),
                        ComputeDesc.COMPUTE_8XINT8(),
                    },
                }
                COMPUTE_TYPE_TO_NAME = {
                    ComputeDesc.COMPUTE_32F(): "COMPUTE_32F",
                    ComputeDesc.COMPUTE_64F(): "COMPUTE_64F",
                    ComputeDesc.COMPUTE_16F(): "COMPUTE_16F",
                    ComputeDesc.COMPUTE_4X16F(): "COMPUTE_4X16F",
                    ComputeDesc.COMPUTE_16BF(): "COMPUTE_16BF",
                    ComputeDesc.COMPUTE_TF32(): "COMPUTE_TF32",
                    ComputeDesc.COMPUTE_3XTF32(): "COMPUTE_3XTF32",
                    ComputeDesc.COMPUTE_9X16BF(): "COMPUTE_9X16BF",
                    ComputeDesc.COMPUTE_8XINT8(): "COMPUTE_8XINT8",
                }
    return


def get_default_compute_type_from_dtype_name(dtype_name: str) -> int:
    _maybe_initialize_compute_type_map()
    if dtype_name not in DEFAULT_COMPUTE_TYPE:
        raise ValueError(f"Invalid data type: {dtype_name}")
    return DEFAULT_COMPUTE_TYPE[dtype_name]


def get_supported_compute_types(dtype_name: str) -> set[int]:
    # https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcreatecontraction
    _maybe_initialize_compute_type_map()
    if dtype_name not in SUPPORTED_COMPUTE_TYPES:
        raise ValueError(f"Invalid data type: {dtype_name}")
    return SUPPORTED_COMPUTE_TYPES[dtype_name]


def get_compute_type_name(compute_type: int) -> str:
    _maybe_initialize_compute_type_map()
    if compute_type not in COMPUTE_TYPE_TO_NAME:
        raise ValueError(f"Invalid compute type: {compute_type}")
    return COMPUTE_TYPE_TO_NAME[compute_type]
