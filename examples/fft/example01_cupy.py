# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of function-form FFT APIs with CuPy ndarrays.

The input as well as the result from the FFT operations are CuPy ndarrays, resulting
in effortless interoperability between nvmath-python and CuPy.
"""
import cupy as cp

import nvmath

shape = 512, 256, 512
axes  = 0, 1

a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

# Forward FFT along the specified axes, batched along the complement.
b = nvmath.fft.fft(a, axes=axes)

# Inverse FFT along the specified axes, batched along the complement.
c = nvmath.fft.ifft(b, axes=axes)

# Synchronize the default stream
cp.cuda.get_current_stream().synchronize()
print(f"Input type = {type(a)}, device = {a.device}")
print(f"FFT output type = {type(b)}, device = {b.device}")
print(f"IFFT output type = {type(c)}, device = {c.device}")
