# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Monkey-patching of Numba to:
#   support LTO code generation and linking
#   find libnvvm located in PYPI wheels
#
import functools

import numba
import numba.cuda as cuda
import numba_cuda


#
# Monkey patching
#


def patch_codegen():
    # Check Numba version
    required_numba_cuda_ver = (0, 9)
    numba_cuda_ver = tuple(map(int, numba_cuda.__version__.split(".")))[:2]
    if numba_cuda_ver < required_numba_cuda_ver:
        raise RuntimeError(
            f"numba-cuda version {required_numba_cuda_ver} is required, but got {numba_cuda.__version__} (aka {numba_cuda_ver})"
        )

    # Add new LTO-IR linker to Numba (from pynvjitlink)
    numba.config.CUDA_ENABLE_PYNVJITLINK = True
    # TODO: proper support for default lto value
    # https://github.com/NVIDIA/numba-cuda/issues/162
    cuda.jit = functools.partial(cuda.jit, lto=True)
