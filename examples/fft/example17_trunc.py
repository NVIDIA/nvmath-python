# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example brings together truncation/padding, and caching.
"""

import functools

import cupy as cp
from cupyx.profiler import benchmark

from caching import fft as cached_fft, FFTCache
from truncation import fft as truncated_fft

shape = 512, 512, 512
axes = 0, 1
extents = 256, 256

a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

r = truncated_fft(a, axes=axes, extents=extents)

nvtime = benchmark(truncated_fft, args=(a,), kwargs={"axes": axes, "extents": extents}, n_repeat=10)
print(f"{len(axes)}-D FFT for axes={axes} and extents={extents} based on nvmath-python (non-cached version):\n{nvtime}\n")

with FFTCache() as cache:
    cached_fft = functools.partial(cached_fft, cache=cache)
    r = truncated_fft(a, axes=axes, extents=extents, engine=cached_fft)

    nvtime = benchmark(truncated_fft, args=(a,), kwargs={"axes": axes, "extents": extents, "engine": cached_fft}, n_repeat=10)
    print(f"{len(axes)}-D FFT for axes={axes} and extents={extents} based on nvmath-python (cached version):\n{nvtime}\n")

cptime = benchmark(cp.fft.fftn, args=(a,), kwargs={"axes": axes, "s": extents}, n_repeat=10)
print(f"{len(axes)}-D FFT for axes={axes} and extents={extents} based on CuPy:\n{cptime}\n")
