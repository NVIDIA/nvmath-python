# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example showing how to perform N-dimensional complex-to-real (C2R) FFT computation with
function-form FFT APIs.
"""

import cupy as cp

import nvmath

shape = 256, 512, 512
axes = 0, 1

# Create a real type input operand and then use Real-to-complex FFT
# to get complex operand with Hermitian symmetry as input for C2R FFT
t = cp.random.rand(*shape, dtype=cp.float32)
a = nvmath.fft.rfft(t, axes=axes)

# Complex-to-real FFT along (0,1), batched along axis=2.
b = nvmath.fft.irfft(a, axes=axes)

# Synchronize the default stream
cp.cuda.get_current_stream().synchronize()
print(f"Input type = {type(t)}, device = {t.device}")
print(f"FFT output type = {type(a)}, device = {a.device}")
print(f"IFFT output type = {type(b)}, device = {b.device}")
