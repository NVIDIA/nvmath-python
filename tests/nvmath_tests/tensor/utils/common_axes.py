# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device

from nvmath.tensor import (
    ComputeDesc,
    ContractionAlgo,
    ContractionAutotuneMode,
    ContractionCacheMode,
    ContractionJitMode,
)


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


class MemBackend(Enum):
    cuda = 1
    cpu = 2


class DType(Enum):
    float16 = 100
    bfloat16 = 101
    float32 = 102
    float64 = 103

    complex32 = 200
    complex64 = 201
    complex128 = 202


class ComputeType(Enum):
    float16 = ComputeDesc.COMPUTE_16F()
    bfloat16 = ComputeDesc.COMPUTE_16BF()
    float32 = ComputeDesc.COMPUTE_32F()
    float64 = ComputeDesc.COMPUTE_64F()

    tf32 = ComputeDesc.COMPUTE_TF32()
    three_xtf32 = ComputeDesc.COMPUTE_3XTF32()
    eight_xint8 = ComputeDesc.COMPUTE_8XINT8()
    nine_x16bf = ComputeDesc.COMPUTE_9X16BF()
    four_x16f = ComputeDesc.COMPUTE_4X16F()


class BlockingOption(Enum):
    true = True
    auto = "auto"


class AutotuneModeOption(Enum):
    none = ContractionAutotuneMode.NONE
    incremental = ContractionAutotuneMode.INCREMENTAL


class CacheModeOption(Enum):
    none = ContractionCacheMode.NONE
    pedantic = ContractionCacheMode.PEDANTIC


class IncrementalCountOption(Enum):
    two = 2
    five = 5


class JitOption(Enum):
    off = ContractionJitMode.NONE
    on = ContractionJitMode.DEFAULT

    @classmethod
    def enabled(cls):
        yield cls.off
        # https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#_CPPv4N17cutensorJitMode_t25CUTENSOR_JIT_MODE_DEFAULTE  # noqa
        # Only supported for GPUs with compute capability >= 8.0
        device = Device()

        if device.compute_capability.major >= 8:
            yield cls.on
        del device


class AlgoOption(Enum):
    default_patient = ContractionAlgo.DEFAULT_PATIENT
    gett = ContractionAlgo.GETT
    tgett = ContractionAlgo.TGETT
    ttgt = ContractionAlgo.TTGT
    default = ContractionAlgo.DEFAULT


class KernelRankOption(Enum):
    zero = 0
    one = 1
