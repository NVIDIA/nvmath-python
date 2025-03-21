# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example on benchmarking an N-D FFT implementation as a composition of the 1D, 2D, or 3D
batched FFTs.

The reference implementation is provided in `fftn2.py` with optimizations to reduce the
number of copies compared with `fftn1.py`.
"""

import functools

import cupy as cp
from cupyx.profiler import benchmark

from caching import fft as cached_fft, FFTCache
from fftn2 import fftn

shape = 32, 8, 128, 64, 16
axes = 0, 1, 2, 3, 4

a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

nvtime = benchmark(fftn, args=(a,), kwargs={"axes": axes}, n_repeat=10)
print(f"{len(axes)}-D FFT version 2 based on nvmath-python (non-caching engine):\n{nvtime}\n")

with FFTCache() as cache:
    cached_fft = functools.partial(cached_fft, cache=cache)
    nvtime = benchmark(fftn, args=(a,), kwargs={"axes": axes, "engine": cached_fft}, n_repeat=10)
print(f"{len(axes)}-D FFT version 2 based on nvmath-python (caching engine):\n{nvtime}\n")

cptime = benchmark(cp.fft.fftn, args=(a,), kwargs={"axes": axes}, n_repeat=10)
print(f"{len(axes)}-D FFT for axes={axes} based on CuPy:\n{cptime}\n")
