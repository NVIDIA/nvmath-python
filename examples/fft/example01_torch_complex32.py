# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example describes how to perform FFT on PyTorch tensors with low precision using function-form FFT APIs.
"""

import torch

import nvmath

shape = 16, 16, 16
axes = 0, 1

a = torch.rand(shape, dtype=torch.complex32, device=0)

# Forward FFT along (0,1), batched along axis=2.
b = nvmath.fft.fft(a, axes=axes)

# Inverse FFT along (0,1), batched along axis=2.
c = nvmath.fft.ifft(b, axes=axes)

# Synchronize the default stream
torch.cuda.default_stream().synchronize()
print(f"Input type = {type(a)}, device = {a.device}")
print(f"FFT output type = {type(b)}, device = {b.device}")
print(f"IFFT output type = {type(c)}, device = {c.device}")
