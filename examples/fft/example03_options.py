# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify options to an FFT operation.

In this example, we will use a NumPy ndarray as input, and we will look at two equivalent
ways of providing options to enforce natural layout for the output.

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

shape = 64, 256, 128
axes = 0, 1

# NumPy ndarray, on the CPU.
a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

# The default result layout is set to 'optimized' (see the documentation for FFTOptions).
# In this example we'd like to enforce the same layout as the input operand.

# Alternative #1 for specifying options, using dataclass.
options = nvmath.fft.FFTOptions(result_layout="natural")
b = nvmath.fft.fft(a, axes=axes, options=options, execution="cuda")
print(f"Does the FFT result shared the same layout as the input ? {b.strides == a.strides}")
print(f"Input type = {type(a)}, FFT output type = {type(b)}")

# Alternative #2 for specifying options, using dict. The two alternatives are entirely
# equivalent.
b = nvmath.fft.fft(a, axes=axes, options={"result_layout": "natural"}, execution="cuda")
print(f"Does the FFT result shared the same layout as the input ? {b.strides == a.strides}")
print(f"Input type = {type(a)}, FFT output type = {type(b)}")
