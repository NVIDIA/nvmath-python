# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful class-form FFT APIs with Torch tensors on CPU.

The input as well as the result from the FFT operations are Torch tensors on CPU.
"""
import torch

import nvmath

shape = 512, 512, 512
axes  = 0, 1

a = torch.ones(shape, dtype=torch.complex64)   # cpu tensor

# Create a stateful FFT object 'f'.
with nvmath.fft.FFT(a, axes=axes) as f:

    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    b = f.execute()

    print(f"Input type = {type(a)}, FFT output type = {type(b)}")