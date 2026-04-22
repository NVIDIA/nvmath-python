# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This module has internal utilities used in implementing the UST and related APIs.
"""

__all__ = []

import threading

import numpy as np

from nvmath.internal import utils


class Cache(dict):
    """
    Cache for sharing kernels and matmuls.
    """

    def __init__(self):
        self.lock = threading.Lock()

    def free(self):
        with self.lock:
            for obj in self.values():
                obj.free()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.free()


class LevelMap(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: for "subarray" like pos() and crd(), need to allocate one chunk and offset.
    # Also need to keep track of the offsets, or recompute? Add pos.base, crd.base?
    def to(self, device_id, stream_holder):
        level_map = LevelMap(self)
        for k in level_map:
            level_map[k] = self[k].to(device_id, stream_holder)
        return level_map

    def empty_like(self, stream_holder):
        level_map = LevelMap(self)
        for k in level_map:
            level_map[k] = utils.create_empty_tensor(
                self[k].__class__, self[k].shape, self[k].dtype, self[k].device_id, stream_holder, verify_strides=False
            )
        return level_map

    def copy_(self, src, stream_holder):
        for k in self:
            self[k].copy_(src[k], stream_holder)


_CTPS = {
    "complex64": "cuda::std::complex<float>",
    "complex128": "cuda::std::complex<double>",
    "float8_e4m3fn": "__nv_fp8_e4m3",
    "float8_e5m2": "__nv_fp8_e5m2",
    "bfloat16": "__nv_bfloat16",
    "float16": "__half",
    "float32": "float",
    "float64": "double",
    "int64": "long long",
    "int32": "int",
    "int16": "int",
    "int8": "int",
}

_NP_ENVELOPE = {
    "complex32": np.complex64,
    "complex64": np.complex64,
    "complex128": np.complex128,
    "float8_e4m3fn": np.float16,
    "float8_e5m2": np.float16,
    "bfloat16": np.float32,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
}

COMPLEX_PRECISION = {"complex64": "float", "complex128": "double"}


def type_str(tp):
    return _CTPS[tp]


def np_enveloping_type(tp):
    return _NP_ENVELOPE[tp]
