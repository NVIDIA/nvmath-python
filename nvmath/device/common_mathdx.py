# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import files, PackageNotFoundError
import os
import platform
import re
import sys
import warnings


CUDA_HOME = None
CURAND_HOME = None


PLATFORM_LINUX = sys.platform.startswith("linux")
PLATFORM_WIN = sys.platform.startswith("win32")


def conda_get_target_name():
    if PLATFORM_LINUX:
        plat = platform.processor()
        if plat == "aarch64":
            return "sbsa-linux"
        else:
            return f"{plat}-linux"
    elif PLATFORM_WIN:
        return "x64"
    else:
        raise AssertionError()


def check_cuda_home():
    # We need some CUDA headers for compiling mathDx headers.
    # We assume users properly managing their local envs (ex: no mix-n-match).
    global CUDA_HOME
    global CURAND_HOME

    # Try wheel
    try:
        # We need CUDA 12+ for device API support
        cudart = files("nvidia-cuda-runtime-cu12")
        cccl = files("nvidia-cuda-cccl-cu12")
        curand = files("nvidia-curand-cu12")
        # use cuda_fp16.h (which we need) as a proxy
        cudart = [f for f in cudart if "cuda_fp16.h" in str(f)][0]
        cudart = os.path.join(os.path.dirname(cudart.locate()), "..")
        # use cuda/std/type_traits as a proxy
        cccl = min([f for f in cccl if re.match(r".*cuda\/std\/type_traits.*", str(f))], key=lambda x: len(str(x)))
        cccl = os.path.join(os.path.dirname(cccl.locate()), "../../..")
        curand = [f for f in curand if "curand_kernel.h" in str(f)][0]
        curand = os.path.dirname(curand.locate())
    except PackageNotFoundError:
        pass
    except ValueError:
        # cccl wheel is buggy (headers missing), skip using wheels
        pass
    else:
        CUDA_HOME = (cudart, cccl)
        CURAND_HOME = curand
        return

    # Try conda
    if "CONDA_PREFIX" in os.environ:
        if PLATFORM_LINUX:
            conda_include = os.path.join(os.environ["CONDA_PREFIX"], "targets", f"{conda_get_target_name()}", "include")
        elif PLATFORM_WIN:
            conda_include = os.path.join(os.environ["CONDA_PREFIX"], "Library", "include")
        else:
            raise AssertionError()
        if os.path.isfile(os.path.join(conda_include, "cuda_fp16.h")) and os.path.isfile(
            os.path.join(conda_include, "cuda/std/type_traits")
        ):
            CUDA_HOME = (os.path.join(conda_include, ".."),)
            CURAND_HOME = os.path.join(CUDA_HOME[0], "include")
            return

    # Try local
    CUDA_PATH = os.environ.get("CUDA_PATH", None)
    CUDA_HOME = os.environ.get("CUDA_HOME", None)
    if CUDA_PATH is None and CUDA_HOME is None:
        raise RuntimeError(
            "cudart headers not found. Depending on how you install nvmath-python and other CUDA packages,\n"
            "you may need to perform one of the steps below:\n"
            "  - conda install -c conda-forge cuda-cudart-dev cuda-cccl cuda-version=12\n"
            "  - export CUDA_HOME=/path/to/CUDA/Toolkit"
        )
    elif CUDA_PATH is not None and CUDA_HOME is None:
        CUDA_HOME = CUDA_PATH
    elif CUDA_PATH is not None and CUDA_HOME is not None and CUDA_HOME != CUDA_PATH:
        warnings.warn("Both CUDA_HOME and CUDA_PATH are set but not consistent. Ignoring CUDA_PATH...")
    CUDA_HOME = (CUDA_HOME,)
    CURAND_HOME = os.path.join(CUDA_HOME[0], "include")


check_cuda_home()
