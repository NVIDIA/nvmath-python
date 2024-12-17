# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Monkey-patching of Numba to:
#   support LTO code generation and linking
#   find libnvvm located in PYPI wheels
#

import os
import functools

import numba
from numba.cuda.cudadrv import libs
import numba.cuda.cudadrv.nvrtc as nvrtc
import numba.cuda.cudadrv.nvvm as nvvm
import pynvjitlink.patch  # type: ignore

from nvmath import _utils


#
# Numba patches
#


def __nvrtc_new__(cls):
    with nvrtc._nvrtc_lock:
        # was: __INSTANCE, changed to _NVRTC__INSTANCE due to name mangling...
        if cls._NVRTC__INSTANCE is None:
            cls._NVRTC__INSTANCE = inst = object.__new__(cls)
            try:
                # was: lib = open_cudalib('nvrtc')
                lib = _utils._nvrtc_obj[0]
            except OSError as e:
                cls._NVRTC__INSTANCE = None
                raise nvrtc.NvrtcSupportError("NVRTC cannot be loaded") from e

            # Find & populate functions
            for name, proto in inst._PROTOTYPES.items():
                func = getattr(lib, name)
                func.restype = proto[0]
                func.argtypes = proto[1:]

                @functools.wraps(func)
                def checked_call(*args, func=func, name=name):
                    error = func(*args)
                    if error == nvrtc.NvrtcResult.NVRTC_ERROR_COMPILATION:
                        raise nvrtc.NvrtcCompilationError()
                    elif error != nvrtc.NvrtcResult.NVRTC_SUCCESS:
                        try:
                            error_name = nvrtc.NvrtcResult(error).name
                        except ValueError:
                            error_name = "Unknown nvrtc_result " f"(error code: {error})"
                        msg = f"Failed to call {name}: {error_name}"
                        raise nvrtc.NvrtcError(msg)

                setattr(inst, name, checked_call)

    return cls._NVRTC__INSTANCE


#
# Monkey patching
#


def patch_codegen():
    # Check Numba version
    required_numba_ver = (0, 60)
    numba_ver = numba.version_info.short
    if numba_ver != required_numba_ver:
        raise RuntimeError(f"numba version {required_numba_ver} is required, but got {numba.__version__} (aka {numba_ver})")

    # Add new LTO-IR linker to Numba (from pynvjitlink)
    pynvjitlink.patch.patch_numba_linker(lto=True)

    # Patch Numba to support wheels
    _utils.patch_numba_nvvm(nvvm)

    # our device apis only support cuda 12+
    _utils.force_loading_nvrtc("12")
    nvrtc.NVRTC.__new__ = __nvrtc_new__
