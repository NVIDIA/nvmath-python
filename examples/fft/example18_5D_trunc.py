# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example brings together N-D FFTs, truncation/padding, and caching.
"""

import functools

import cupy as cp
from cupyx.profiler import benchmark

from caching import fft as cached_fft, FFTCache
from fftn2 import fftn

from truncation import fft as truncated_fft

shape = 64, 128, 16, 48, 32
axes = 0, 1, 2, 3, 4
extents = tuple(s // 2 for s in shape)

a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

r = truncated_fft(a, axes=axes, extents=extents, engine=fftn)

nvtime = benchmark(truncated_fft, args=(a,), kwargs={"axes": axes, "extents": extents, "engine": fftn}, n_repeat=10)
print(
    f"{len(axes)}-D FFT for axes={axes} and extents={extents} based on nvmath-python (non-caching engine):\n{nvtime}\n"
)

# Create a cached FFTN version to use in truncated FFT to create truncated FFTN.
with FFTCache() as cache:
    cached_fft = functools.partial(cached_fft, cache=cache)
    cached_fftn = functools.wraps(fftn)(functools.partial(fftn, engine=cached_fft))

    r = truncated_fft(a, axes=axes, extents=extents, engine=cached_fftn)

    nvtime = benchmark(
        truncated_fft, args=(a,), kwargs={"axes": axes, "extents": extents, "engine": cached_fftn}, n_repeat=10
    )
    print(
        f"{len(axes)}-D FFT for axes={axes} and extents={extents} based on nvmath-python (caching engine):\n{nvtime}\n"
    )

cptime = benchmark(cp.fft.fftn, args=(a,), kwargs={"axes": axes, "s": extents}, n_repeat=10)
print(f"{len(axes)}-D FFT for axes={axes} and extents={extents} based on CuPy:\n{cptime}\n")
