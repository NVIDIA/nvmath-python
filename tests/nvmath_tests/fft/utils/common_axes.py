# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None


class Framework(Enum):
    numpy = 1
    cupy = 2
    torch = 3

    @classmethod
    def enabled(cls):
        yield cls.numpy
        if cp is not None:
            yield cls.cupy
        if torch is not None:
            yield cls.torch


class ExecBackend(Enum):
    cufft = 1
    fftw = 2

    @property
    def nvname(self):
        return "cuda" if self == ExecBackend.cufft else "cpu"

    @property
    def mem(self):
        return MemBackend.cuda if self == ExecBackend.cufft else MemBackend.cpu


class MemBackend(Enum):
    cuda = 1
    cpu = 2

    @property
    def exec(self):
        return ExecBackend.cufft if self == MemBackend.cuda else ExecBackend.fftw


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


class OptFftInplace(Enum):
    false = False
    true = True

    def __bool__(self):
        return self.value


class LtoCallback(Enum):
    prolog = 1
    epilog = 2
    prolog_and_epilog = 3

    def has_prolog(self):
        return self.value % 2 == 1

    def has_epilog(self):
        return self.value >= 2

    def __xor__(self, other):
        value = self.value ^ other.value
        if value == 0:
            return None
        return LtoCallback(value)


class AllowToFail(Enum):
    false = False
    true = True

    def __bool__(self):
        return self.value
