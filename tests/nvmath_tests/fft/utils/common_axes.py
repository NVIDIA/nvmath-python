# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

try:
    import torch
except:
    torch = None

class Framework(Enum):
    numpy = 1
    cupy = 2
    torch = 3

    @classmethod
    def enabled(cls):
        yield cls.numpy
        yield cls.cupy
        if torch is not None:
            yield cls.torch


class Backend(Enum):
    cpu = 1
    gpu = 2


class DType(Enum):
    uint8 = 1  # unsupported
    int8 = 2  # unsupported
    uint16 = 3  # unsupported
    int16 = 4  # unsupported
    uint32 = 5  # unsupported
    int32 = 6  # unsupported
    uint64 = 7  # unsupported
    int64 = 8  # unsupported

    float16 = 100
    bfloat16 = 101  # unsupported
    float32 = 102
    float64 = 103

    complex32 = 200
    complex64 = 201
    complex128 = 202


class ShapeKind(Enum):
    pow2 = 1
    pow2357 = 2
    prime = 3
    random = 4


class Direction(Enum):
    forward = "FORWARD"
    inverse = "INVERSE"


class OptFftType(Enum):
    r2c = "R2C"
    c2r = "C2R"
    c2c = "C2C"


class OptFftBlocking(Enum):
    auto = "auto"
    true = True


class OptFftLayout(Enum):
    natural = "natural"
    optimized = "optimized"
