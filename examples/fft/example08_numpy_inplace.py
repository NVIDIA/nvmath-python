# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to perform inplace FFT for NumPy ndarrays using function-form
FFT APIs.

The NumPy ndarrays reside in CPU memory. There are two ways to process CPU tensors with
nvmath: either use host library to process the tensor directly or copy the tensor to GPU
memory and process it with cuFFT.

The default behaviour has changed in Beta2, the cpu arrays will default to processing with
host library: NVPL (Nvidia Performance Libraries), MKL or any other FFTW3-compatible
library. In this example, we explicitly set ``execution="cuda"``, to copy the data on GPU
for processing with cuFFT.
"""

import numpy as np

import nvmath

shape = 64, 256, 256
axes = 0, 1

a = np.ones(shape, dtype=np.complex64)
a_copy = a.copy()  # Since `a` is overwritten.

# Forward FFT along (0,1), batched along axis=2.
b = nvmath.fft.fft(a, axes=axes, options={"inplace": True}, execution="cuda")
assert b is a  # `a` is overwritten

# Inverse FFT along (0,1), batched along axis=2.
c = nvmath.fft.ifft(b, axes=axes, options={"inplace": True}, execution="cuda")
assert c is b  # `b`` is again overwritten

print(f"Input type = {type(a)}, FFT type = {type(b)}, IFFT type = {type(c)}")
