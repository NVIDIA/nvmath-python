# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example showing how to reuse cached objects for repeated computation on the same problem
specification with different operands.

The cached implementation is provided in `caching.py`.
"""

import logging

import cupy as cp

from caching import fft as cached_fft, FFTCache


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

shape = 512, 256, 512
axes = 0, 1

# Explicitly managing the FFT cache
cache = FFTCache()

a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

r = cached_fft(a, axes=axes, cache=cache)

a[:] = 2 * a

# Use cached object from previous run here.
r = cached_fft(a, axes=axes, cache=cache)

# No suitable object exists in cache, so create, cache, and use a new one.
r = cached_fft(a, axes=(2,), cache=cache)

# Synchronize the default stream
cp.cuda.get_current_stream().synchronize()
print(f"Input type = {type(a)}, device = {a.device}")
print(f"FFT output type = {type(r)}, device = {r.device}")

# Release all resources in cache
cache.free()
