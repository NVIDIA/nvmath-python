# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from enum import IntEnum
import logging
import os
import re
import site
import sys

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


# TODO: unify all loading helpers into one
_nvrtc_obj: list[ctypes.CDLL] = []


def force_loading_nvrtc(cu_ver):
    # this logic should live in CUDA Python...
    # TODO: remove this function once NVIDIA/cuda-python#62 is resolved
    # This logic handles all cases - wheel, conda, and system installations
    global _nvrtc_obj
    if len(_nvrtc_obj) > 0:
        return

    cu_ver = cu_ver.split(".")
    major = cu_ver[0]
    if major == "11":
        # CUDA 11.2+ supports minor ver compat
        if PLATFORM_LINUX:
            cu_ver = "11.2"
        elif PLATFORM_WIN:
            cu_ver = "112"
    elif major == "12":
        if PLATFORM_LINUX:
            cu_ver = "12"
        elif PLATFORM_WIN:
            cu_ver = "120"
    else:
        raise NotImplementedError(f"CUDA {major} is not supported")

    site_paths = [site.getusersitepackages()] + site.getsitepackages() + [None]
    for sp in site_paths:
        if PLATFORM_LINUX:
            dso_dir = "lib"
            dso_path = f"libnvrtc.so.{cu_ver}"
        elif PLATFORM_WIN:
            dso_dir = "bin"
            dso_path = f"nvrtc64_{cu_ver}_0.dll"
        else:
            raise AssertionError()

        if sp is not None:
            dso_dir = os.path.join(sp, "nvidia", "cuda_nvrtc", dso_dir)
            dso_path = os.path.join(dso_dir, dso_path)
        try:
            _nvrtc_obj.append(ctypes.CDLL(dso_path, mode=ctypes.RTLD_GLOBAL))
        except OSError:
            continue
        else:
            if PLATFORM_WIN:
                import win32api

                # This absolute path will always be correct regardless of the package source
                nvrtc_path = win32api.GetModuleFileNameW(_nvrtc_obj[0]._handle)
                dso_dir = os.path.dirname(nvrtc_path)
                dso_path = os.path.join(dso_dir, [f for f in os.listdir(dso_dir) if re.match("^nvrtc-builtins.*.dll$", f)][0])
                _nvrtc_obj.append(ctypes.CDLL(dso_path))
            break
    else:
        raise RuntimeError(
            f"NVRTC from CUDA {major} not found. Depending on how you install nvmath-python and other CUDA packages,\n"
            f"you may need to perform one of the steps below:\n"
            f"  - pip install nvidia-cuda-nvrtc-cu{major}\n"
            f"  - conda install -c conda-forge cuda-nvrtc cuda-version={major}\n"
            "  - export LD_LIBRARY_PATH=/path/to/CUDA/Toolkit/lib64:$LD_LIBRARY_PATH"
        )


# TODO: unify all loading helpers into one
_libmathdx_obj: list[ctypes.CDLL] = []


def force_loading_libmathdx(cu_ver):
    # this logic should live in CUDA Python...
    # TODO: remove this function once NVIDIA/cuda-python#62 is resolved
    # This logic handles all cases - wheel, conda, and system installations
    global _libmathdx_obj
    if len(_libmathdx_obj) > 0:
        return

    cu_ver = cu_ver.split(".")
    major = cu_ver[0]
    if major != "12":
        raise NotImplementedError(f"CUDA {major} is not supported")

    site_paths = [site.getusersitepackages()] + site.getsitepackages() + [None]
    for sp in site_paths:
        if PLATFORM_LINUX:
            dso_dir = "lib"
            dso_path = "libmathdx.so.0"
        elif PLATFORM_WIN:
            dso_dir = "bin"
            dso_path = "libmathdx.dll"
        else:
            raise AssertionError("Unsupported Platform! Only Linux and Windows are supported.")

        if sp is not None:
            dso_dir = os.path.join(sp, "nvidia", f"cu{major}", dso_dir)
            dso_path = os.path.join(dso_dir, dso_path)
        try:
            _libmathdx_obj.append(ctypes.CDLL(dso_path, mode=ctypes.RTLD_GLOBAL))
        except OSError:
            continue
        else:
            if PLATFORM_WIN:
                # TODO: untested in context of libmathdx.
                import win32api

                # This absolute path will always be correct regardless of the package source
                nvrtc_path = win32api.GetModuleFileNameW(_libmathdx_obj[0]._handle)
                dso_dir = os.path.dirname(nvrtc_path)
                dso_path = os.path.join(dso_dir, [f for f in os.listdir(dso_dir) if re.match("^nvrtc-builtins.*.dll$", f)][0])
                _libmathdx_obj.append(ctypes.CDLL(dso_path))
            break
    else:
        raise RuntimeError(
            f"libmathdx not found. Depending on how you install nvmath-python and other CUDA packages,\n"
            f"you may need to perform one of the steps below:\n"
            f"  - pip install nvidia-libmathdx-cu{major}\n"
            f"  - conda install -c conda-forge libmathdx cuda-version={major}"
        )

    from nvmath.bindings import mathdx

    version = mathdx.get_version()
    if version < 201 or version >= 300:
        raise ValueError(f"libmathdx version must be >= 0.2.1 and < 0.3.0 ; got {version}")


# TODO: unify all loading helpers into one
_nvvm_obj: list[ctypes.CDLL] = []


def force_loading_nvvm():
    # this logic should live in CUDA Python...
    # This logic handles all cases - wheel, conda, and system installations
    global _nvvm_obj
    if len(_nvvm_obj) > 0:
        return

    site_paths = [site.getusersitepackages()] + site.getsitepackages() + ["conda", None]
    for sp in site_paths:
        # The SONAME is taken based on public CTK 12.x releases
        if PLATFORM_LINUX:
            dso_dir = "lib64"
            # Hack: libnvvm from Linux wheel does not have any soname (CUDAINST-3183)
            dso_path = "libnvvm.so"
            if sp == "conda" or sp is None:
                dso_path += ".4"
        elif PLATFORM_WIN:
            dso_dir = "bin"
            dso_path = "nvvm64_40_0.dll"
        else:
            raise AssertionError()

        if sp == "conda" and "CONDA_PREFIX" in os.environ:
            # nvvm is not under $CONDA_PREFIX/lib, so it's not in the default search path
            if PLATFORM_LINUX:
                dso_dir = os.path.join(os.environ["CONDA_PREFIX"], "nvvm", dso_dir)
            elif PLATFORM_WIN:
                dso_dir = os.path.join(os.environ["CONDA_PREFIX"], "Library", "nvvm", dso_dir)
            dso_path = os.path.join(dso_dir, dso_path)
        elif sp is not None:
            dso_dir = os.path.join(sp, "nvidia", "cuda_nvcc", "nvvm", dso_dir)
            dso_path = os.path.join(dso_dir, dso_path)
        try:
            _nvvm_obj.append(ctypes.CDLL(dso_path, mode=ctypes.RTLD_GLOBAL))
        except OSError:
            continue
        else:
            break
    else:
        raise RuntimeError(
            "NVVM from CUDA 12 not found. Depending on how you install nvmath-python and other CUDA packages,\n"
            "you may need to perform one of the steps below:\n"
            "  - pip install nvidia-cuda-nvcc-cu12\n"
            "  - conda install -c conda-forge cuda-nvvm cuda-version=12\n"
            "  - export LD_LIBRARY_PATH=/path/to/CUDA/Toolkit/nvvm/lib64:$LD_LIBRARY_PATH"
        )


# TODO: unify all loading helpers into one
_libcudss_obj: list[ctypes.CDLL] = []


def force_loading_cudss(cu_ver):
    # this logic should live in CUDA Python...
    global _libcudss_obj
    if len(_libcudss_obj) > 0:
        return

    cu_ver = cu_ver.split(".")
    major = cu_ver[0]
    if major != "12":
        raise NotImplementedError(f"CUDA {major} is not supported")

    site_paths = [site.getusersitepackages()] + site.getsitepackages() + [None]
    for sp in site_paths:
        if PLATFORM_LINUX:
            dso_dir = "lib"
            dso_path = "libcudss.so.0"
        elif PLATFORM_WIN:
            dso_dir = "bin"
            dso_path = "cudss64_0.dll"
        else:
            raise AssertionError("Unsupported Platform! Only Linux and Windows are supported.")

        if sp is not None:
            dso_dir = os.path.join(sp, "nvidia", f"cu{major}", dso_dir)
            dso_path = os.path.join(dso_dir, dso_path)
        try:
            _libcudss_obj.append(ctypes.CDLL(dso_path, mode=ctypes.RTLD_GLOBAL))
            logging.debug("Loaded %s", dso_path)
        except OSError as e:
            logging.debug("%s", e)
            continue
        else:
            break
    else:
        raise RuntimeError(
            f"libcudss not found. Depending on how you install nvmath-python and other CUDA packages,\n"
            f"you may need to perform one of the steps below:\n"
            f"  - pip install nvidia-cudss-cu{major}\n"
            f"  - conda install -c conda-forge libcudss cuda-version={major}"
        )


def module_init_force_cupy_lib_load():
    """
    Attempt to preload libraries at module import time.
    Fail silently if preload fails.
    """
    from nvmath.bindings import _internal

    for lib in ("cublas", "cufft", "curand", "cusolverDn", "cusparse"):
        try:
            mod = getattr(_internal, lib)
            mod._inspect_function_pointers()
        except (_internal.utils.NotSupportedError, RuntimeError):
            pass
    for cu_ver in ("12", "11"):
        try:
            force_loading_nvrtc(cu_ver)
            return
        except RuntimeError:
            pass


def get_nvrtc_build_id():
    from cuda.core.experimental import ObjectCode, Program, ProgramOptions

    code = r"""
    extern "C" __global__ void get_build_id(int* build_id) {

        *build_id = __CUDACC_VER_BUILD__;
    }
    """

    prog = Program(code, "c++", ProgramOptions(std="c++17", minimal=True, arch="compute_75"))
    obj = prog.compile("ptx")
    assert isinstance(obj, ObjectCode)

    pattern = re.compile(r"mov\.u32\s+%\w+,\s+(\d+)")
    m = pattern.search(obj.code.decode())
    assert m is not None

    return int(m.group(1))


def get_nvrtc_version():
    """
    Returns the NVRTC version as a tuple of (major, minor, build).
    """
    from cuda.bindings import nvrtc

    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion error: {err}")
    build = get_nvrtc_build_id()
    return major, minor, build
