# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of function-form FFT APIs with NumPy ndarrays.

The NumPy ndarrays reside in CPU memory. There are two ways to process CPU tensors with
nvmath: either use host library to process the tensor directly or copy the tensor to GPU
memory and process it with cuFFT.

The default behaviour has changed in Beta2, the cpu arrays will default to processing with
host library: NVPL (Nvidia Performance Libraries), MKL or any other FFTW3-compatible
library. In this example, we explicitly set ``execution="cuda"``, to copy the
data on GPU for processing with cuFFT.

The input as well as the result from the FFT operations are NumPy ndarrays, resulting
in effortless interoperability between nvmath-python and NumPy.
"""

import numpy as np

import nvmath

shape = 64, 256, 128
axes = 0, 1

# NumPy ndarray, on the CPU.
a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

# Forward FFT along the specified axes, batched along the complement.
b = nvmath.fft.fft(a, axes=axes, execution="cuda")

print(f"Input type = {type(a)}, FFT output type = {type(b)}")
