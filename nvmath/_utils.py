# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from enum import IntEnum
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


_nvjitlink_obj: list[ctypes.CDLL] = []


def force_loading_nvjitlink():
    # this logic should live in CUDA Python...
    # This logic handles all cases - wheel, conda, and system installations
    global _nvjitlink_obj
    if len(_nvjitlink_obj) > 0:
        return
    if not PLATFORM_WIN:
        # pynvjitlink on Linux currently links to nvjitlink statically, so no
        # need to preload
        return

    site_paths = [site.getusersitepackages()] + site.getsitepackages() + [None]
    for sp in site_paths:
        # The SONAME is taken based on public CTK 12.x releases
        dso_dir = "bin"
        dso_path = "nvJitLink_120_0.dll"

        if sp is not None:
            dso_dir = os.path.join(sp, "nvidia", "nvjitlink", dso_dir)
            dso_path = os.path.join(dso_dir, dso_path)
        try:
            _nvjitlink_obj.append(ctypes.CDLL(dso_path, mode=ctypes.RTLD_GLOBAL))
        except OSError:
            continue
        else:
            break
    else:
        raise RuntimeError(
            "nvJitLink from CUDA 12 not found. Depending on how you install nvmath-python and other CUDA packages,\n"
            "you may need to perform one of the steps below:\n"
            "  - pip install nvidia-nvjitlink-cu12\n"
            "  - conda install -c conda-forge libnvjitlink cuda-version=12\n"
            "  - export LD_LIBRARY_PATH=/path/to/CUDA/Toolkit/lib64:$LD_LIBRARY_PATH"
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


_is_numba_nvvm_patched = False


def patch_numba_nvvm(nvvm):
    """Patch Numba to support wheels."""
    # this function assumes Numba is already imported
    global _is_numba_nvvm_patched
    if _is_numba_nvvm_patched:
        return

    # Patch NVVM object
    def __nvvm_new__(cls):
        with nvvm._nvvm_lock:
            # was: __INSTANCE, changed to _NVVM__INSTANCE due to name mangling...
            if cls._NVVM__INSTANCE is None:
                cls._NVVM__INSTANCE = inst = object.__new__(cls)
                try:
                    # was: inst.driver = open_cudalib('nvvm')
                    inst.driver = _nvvm_obj[0]
                except OSError as e:
                    cls._NVVM__INSTANCE = None
                    errmsg = "libNVVM cannot be found. Do `conda install " "cudatoolkit`:\n%s"
                    raise nvvm.NvvmSupportError(errmsg % e)

                # Find & populate functions
                for name, proto in inst._PROTOTYPES.items():
                    func = getattr(inst.driver, name)
                    func.restype = proto[0]
                    func.argtypes = proto[1:]
                    setattr(inst, name, func)

        return cls._NVVM__INSTANCE

    def __nvvm_init__(self):
        ir_versions = self.get_ir_version()
        self._majorIR = ir_versions[0]
        self._minorIR = ir_versions[1]
        self._majorDbg = ir_versions[2]
        self._minorDbg = ir_versions[3]
        # don't overwrite self._supported_ccs!

    force_loading_nvvm()
    nvvm.NVVM.__new__ = __nvvm_new__
    nvvm.NVVM.__init__ = __nvvm_init__
    # Pre-compute supported compute capabilities based on COMPUTE_CAPABILITIES
    # and CTK_SUPPORTED in numba/cuda/cudadrv/nvvm.py.
    n = nvvm.NVVM()  # this is a singleton
    n._supported_ccs = (
        (3, 5),
        (3, 7),
        (5, 0),
        (5, 2),
        (5, 3),
        (6, 0),
        (6, 1),
        (6, 2),
        (7, 0),
        (7, 2),
        (7, 5),
        (8, 0),
        (8, 6),
        (8, 7),
        (8, 9),
        (9, 0),
    )

    # Patch libdevice object
    if _nvvm_obj[0]._name.startswith(("libnvvm", "nvvm64")):
        # libnvvm found in sys path (ex: LD_LIBRARY_PATH), fall back to Numba's way
        # way of locating it
        from numba.cuda.cudadrv.libs import get_libdevice

        libdevice_path = get_libdevice()
        # custom CUDA path is a corner case
        if libdevice_path is None:
            raise RuntimeError(
                "cannot locate libdevice, perhaps you need to set "
                "CUDA_HOME? Please follow Numba's instruction at:\n"
                "https://numba.readthedocs.io/en/stable/cuda/overview.html#setting-cuda-installation-path"
            )
    else:
        # maybe it's pip or conda
        libdevice_path = os.path.join(os.path.dirname(_nvvm_obj[0]._name), "../libdevice/libdevice.10.bc")
    assert os.path.isfile(libdevice_path), f"{libdevice_path=}"
    with open(libdevice_path, "rb") as f:
        nvvm.LibDevice._cache_ = f.read()

    _is_numba_nvvm_patched = True
