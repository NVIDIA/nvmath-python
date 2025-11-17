# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["get_default_compute_type_from_dtype_name", "get_supported_compute_types"]

import threading

from nvmath.bindings.cutensor import ComputeDesc  # type: ignore

DEFAULT_COMPUTE_TYPE: dict[str, int] = {}  # dtype_name -> default compute type
SUPPORTED_COMPUTE_TYPES: dict[str, set[int]] = {}  # dtype_name -> set of compute types supported
_typemap_lock = threading.Lock()


def _maybe_initialize_compute_type_map():
    # https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcreatecontraction
    global DEFAULT_COMPUTE_TYPE, SUPPORTED_COMPUTE_TYPES
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
                    },
                    "complex64": {ComputeDesc.COMPUTE_32F(), ComputeDesc.COMPUTE_TF32(), ComputeDesc.COMPUTE_3XTF32()},
                    "complex128": {ComputeDesc.COMPUTE_32F(), ComputeDesc.COMPUTE_64F()},
                    "float64": {ComputeDesc.COMPUTE_64F(), ComputeDesc.COMPUTE_32F()},
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
