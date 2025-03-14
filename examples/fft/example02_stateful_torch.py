# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful class-form FFT APIs with Torch tensors on GPU.

The input as well as the result from the FFT operations are Torch tensors on GPU.
"""

import torch

import nvmath

shape = 512, 512, 512
axes = 0, 1

a = torch.ones(shape, dtype=torch.complex64, device="cuda")  # gpu tensor

# Create a stateful FFT object 'f'.
with nvmath.fft.FFT(a, axes=axes) as f:
    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    b = f.execute()

    # Synchronize the default stream
    torch.cuda.default_stream().synchronize()
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"FFT output type = {type(b)}, device = {b.device}")
