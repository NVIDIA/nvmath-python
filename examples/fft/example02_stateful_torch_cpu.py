# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful class-form FFT APIs with Torch tensors on CPU.

There are two ways to process CPU tensors with nvmath: either use host library to process
the tensor directly or copy the tensor to GPU memory and process it with cuFFT.

The default behaviour has changed in Beta2, the cpu arrays will default to processing with
host library: NVPL (Nvidia Performance Libraries), MKL or any other FFTW3-compatible
library. In this example, we explicitly set ``execution="cuda"``, to copy the data on
GPU for processing with cuFFT.

The input as well as the result from the FFT operations are Torch tensors on CPU.
"""

import torch

import nvmath

shape = 512, 512, 512
axes = 0, 1

a = torch.ones(shape, dtype=torch.complex64)  # cpu tensor

# Create a stateful FFT object 'f'.
with nvmath.fft.FFT(a, axes=axes, execution="cuda") as f:
    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    b = f.execute()

    print(f"Input type = {type(a)}, FFT output type = {type(b)}")
