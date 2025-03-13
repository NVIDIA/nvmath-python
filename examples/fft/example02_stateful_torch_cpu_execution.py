# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful class-form FFT APIs with Torch tensors on CPU.

There are two ways to process CPU tensors with nvmath: either use host library to process
the tensor directly or copy the tensor to GPU memory and process it with cuFFT.

The default behaviour has changed in Beta2, the cpu arrays will default to processing with
host library: NVPL (Nvidia Performance Libraries), MKL or any other FFTW3-compatible
library. Passing the ``execution="cpu"`` here is optional, as the input
tensor resides in CPU memory.

The quickest way for pip users to use nvmath-python with the CPU execution space is to add
cpu to the extras:``pip install nvmath-python[cu12,cpu]`, for example.

The input as well as the result from the FFT operations are Torch tensors on CPU.
"""

import torch

import nvmath

shape = 128, 128, 128
axes = 0, 1

a = torch.ones(shape, dtype=torch.complex64)  # cpu tensor

# Create a stateful FFT object 'f'.
with nvmath.fft.FFT(a, axes=axes, execution="cpu") as f:
    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    b = f.execute()

    print(f"Input type = {type(a)}, FFT output type = {type(b)}")
