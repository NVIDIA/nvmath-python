# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example showing how to reuse cached FFT objects for repeated computation.

Part 1 demonstrates basic caching: the same problem specification with
different operands reuses a planned object, while a different specification
triggers a cache miss.

Part 2 demonstrates that operands whose batch dimensions are rearranged can
still produce the same cache key. The FFT plan only needs the total batch
count and the distance between consecutive FFT inputs in memory, not the
shape of the batch dimensions, so (2, 3, 64) and (3, 2, 64) have
matching keys for a 1-D FFT on the last axis.

The cached implementation is provided in `caching.py`.
"""

import logging

import cupy as cp
from caching import FFTCache
from caching import fft as cached_fft

import nvmath

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

stream = cp.cuda.get_current_stream()
cache = FFTCache()

# ---------------------------------------------------------------------------
# Part 1 — basic caching: same shape, different data
# ---------------------------------------------------------------------------

shape = 512, 256, 512
axes = 0, 1

a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

r = cached_fft(a, axes=axes, cache=cache, stream=stream)

a[:] = 2 * a

# Cache HIT — reuses planned object from the previous call.
r = cached_fft(a, axes=axes, cache=cache, stream=stream)

# Cache MISS — different axes, so a new object is created and cached.
r = cached_fft(a, axes=(2,), cache=cache, stream=stream)

stream.synchronize()
print(f"Input type = {type(a)}, device = {a.device}")
print(f"FFT output type = {type(r)}, device = {r.device}")

# ---------------------------------------------------------------------------
# Part 2 — rearranged batch dimensions but matching FFT key
# ---------------------------------------------------------------------------

axes2 = (2,)

a2 = cp.random.rand(2, 3, 64, dtype=cp.float64) + 1j * cp.random.rand(2, 3, 64, dtype=cp.float64)
b2 = cp.random.rand(3, 2, 64, dtype=cp.float64) + 1j * cp.random.rand(3, 2, 64, dtype=cp.float64)

key_a = nvmath.fft.FFT.create_key(a2, axes=axes2)
key_b = nvmath.fft.FFT.create_key(b2, axes=axes2)
assert key_a == key_b, "Expected matching FFT keys for rearranged batch dimensions"

# Cache MISS — new FFT size, creates and plans a new object.
r_a = cached_fft(a2, axes=axes2, cache=cache, stream=stream)

# Cache HIT — reuses the planned object via reset_operand.
r_b = cached_fft(b2, axes=axes2, cache=cache, stream=stream)

stream.synchronize()
print(f"Operand a2: shape={a2.shape}  ->  result shape={r_a.shape}")
print(f"Operand b2: shape={b2.shape}  ->  result shape={r_b.shape}")

ref_a = cp.fft.fftn(a2, axes=axes2)
ref_b = cp.fft.fftn(b2, axes=axes2)
assert cp.allclose(r_a, ref_a)
assert cp.allclose(r_b, ref_b)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

cache.free()
