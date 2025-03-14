# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to perform inplace FFT for CuPy ndarrays using function-form
FFT APIs.
"""

import cupy as cp

import nvmath

shape = 256, 512, 512
axes = 0, 1

a = cp.ones(shape, dtype=cp.complex128)

# Forward FFT along (0,1), batched along axis=2.
b = nvmath.fft.fft(a, axes=axes, options={"inplace": True})
assert b is a  # `a` is overwritten

# Inverse FFT along (0,1), batched along axis=2.
c = nvmath.fft.ifft(b, axes=axes, options={"inplace": True})
assert c is a  # `b` is again overwritten

# Synchronize the default stream
cp.cuda.get_current_stream().synchronize()
print(f"Input type = {type(a)}, device = {a.device}")
print(f"FFT output type = {type(b)}, device = {b.device}")
print(f"IFFT output type = {type(c)}, device = {c.device}")
