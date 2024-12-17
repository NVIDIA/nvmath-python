# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example showing how to perform N-dimensional real-to-complex (R2C) FFT computation with
function-form FFT APIs.
"""

import cupy as cp

import nvmath

shape = 512, 512, 512
axes = 2, 1

a = cp.random.rand(*shape, dtype=cp.float64)
a = a.transpose(0, 2, 1)

# Real-to-complex FFT along 'axes', batched along the complement.
b = nvmath.fft.rfft(a, axes=axes)

# Synchronize the default stream
cp.cuda.get_current_stream().synchronize()
print(f"Input type = {type(a)}, device = {a.device}")
print(f"FFT output type = {type(b)}, device = {b.device}")
