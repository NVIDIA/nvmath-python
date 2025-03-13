# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how different data layouts are implicitly handled in nvmath-python.

The input as well as the result from the FFT operations are CuPy ndarrays.
"""

import cupy as cp

import nvmath

shape = 512, 256, 512
axes = 0, 1

a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

# Use permutation or slicing to create different layouts as a new view from `a`
for t in (a.T, a[::2, ::2, :]):
    # Forward FFT along the specified axes, batched along the complement.
    b = nvmath.fft.fft(t, axes=axes)

    # Synchronize the default stream
    cp.cuda.get_current_stream().synchronize()
    print(f"Input type = {type(t)}, device = {t.device}")
    print(f"FFT output type = {type(b)}, device = {b.device}")
