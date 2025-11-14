# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from functools import cache
import logging
import re
import sys
from cuda import pathfinder

logger = logging.getLogger()

#
# Note: This module should not depend on anything from the nvmath namespace!
#


# The (subset of) compute types below are shared by cuStateVec and cuTensorNet
class ComputeType(IntEnum):
    """An enumeration of CUDA compute types."""

    COMPUTE_DEFAULT = 0
    COMPUTE_16F = 1 << 0
    COMPUTE_32F = 1 << 2
    COMPUTE_64F = 1 << 4
    COMPUTE_8U = 1 << 6
    COMPUTE_8I = 1 << 8
    COMPUTE_32U = 1 << 7
    COMPUTE_32I = 1 << 9
    COMPUTE_16BF = 1 << 10
    COMPUTE_TF32 = 1 << 12


# TODO: use those exposed by CUDA Python instead, but before removing these
# duplicates, check if they are fixed to inherit IntEnum instead of Enum.
class CudaDataType(IntEnum):
    """An enumeration of `cudaDataType_t`."""

    CUDA_R_16F = 2
    CUDA_C_16F = 6
    CUDA_R_16BF = 14
    CUDA_C_16BF = 15
    CUDA_R_32F = 0
    CUDA_C_32F = 4
    CUDA_R_64F = 1
    CUDA_C_64F = 5
    CUDA_R_4I = 16
    CUDA_C_4I = 17
    CUDA_R_4U = 18
    CUDA_C_4U = 19
    CUDA_R_8I = 3
    CUDA_C_8I = 7
    CUDA_R_8U = 8
    CUDA_C_8U = 9
    CUDA_R_16I = 20
    CUDA_C_16I = 21
    CUDA_R_16U = 22
    CUDA_C_16U = 23
    CUDA_R_32I = 10
    CUDA_C_32I = 11
    CUDA_R_32U = 12
    CUDA_C_32U = 13
    CUDA_R_64I = 24
    CUDA_C_64I = 25
    CUDA_R_64U = 26
    CUDA_C_64U = 27
    CUDA_R_8F_E4M3 = 28
    CUDA_R_8F_E5M2 = 29


class LibraryPropertyType(IntEnum):
    """An enumeration of library version information."""

    MAJOR_VERSION = 0
    MINOR_VERSION = 1
    PATCH_LEVEL = 2


del IntEnum


PLATFORM_LINUX = sys.platform.startswith("linux")
PLATFORM_WIN = sys.platform.startswith("win32")


def module_init_force_cupy_lib_load():
    """
    Attempt to preload libraries at module import time. We want to do it before
    cupy, since it does not know how to properly search for libraries:
    https://github.com/cupy/cupy/issues/9127
    Fail silently if preload fails.
    """
    from nvmath.bindings import _internal

    # cutensor windows binding is not available for nvmath-python beta7.0.
    libs = (
        ("cublas", "cufft", "curand", "cusolverDn", "cusparse", "cutensor")
        if PLATFORM_LINUX
        else ("cublas", "cufft", "curand", "cusolverDn", "cusparse")
    )
    for lib in libs:
        try:
            mod = getattr(_internal, lib)
            mod._inspect_function_pointers()
        except (_internal.utils.NotSupportedError, RuntimeError):
            pass

    try:
        pathfinder.load_nvidia_dynamic_lib("nvrtc")
    except pathfinder.DynamicLibNotFoundError:
        pass


@cache
def get_nvrtc_build_id(minimal=True) -> int:
    from cuda.core.experimental import ObjectCode, Program, ProgramOptions

    code = r"""
    extern "C" __global__ void get_build_id(int* build_id) {

        *build_id = __CUDACC_VER_BUILD__;
    }
    """

    prog = Program(code, "c++", ProgramOptions(std="c++17", minimal=minimal, arch="compute_75"))
    obj = prog.compile("ptx")
    assert isinstance(obj, ObjectCode)

    pattern = re.compile(r"mov\.u32\s+%\w+,\s+(\d+)")
    m = pattern.search(obj.code.decode())
    assert m is not None

    return int(m.group(1))


@cache
def get_nvrtc_version() -> tuple[int, int, int]:
    """
    Returns the NVRTC version as a tuple of (major, minor, build).
    """
    from cuda.bindings import nvrtc

    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion error: {err}")
    # minimal support was added in CUDA 12.0
    build = get_nvrtc_build_id(minimal=major >= 12)
    return major, minor, build
