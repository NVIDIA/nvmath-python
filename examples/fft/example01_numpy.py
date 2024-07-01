# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of function-form FFT APIs with NumPy ndarrays on the CPU.

The input as well as the result from the FFT operations are NumPy ndarrays, resulting
in effortless interoperability between nvmath-python and NumPy.
"""
import numpy as np

import nvmath

shape = 512, 256, 768
axes  = 0, 1

# NumPy ndarray, on the CPU.
a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

# Forward FFT along the specified axes, batched along the complement.
b = nvmath.fft.fft(a, axes=axes)

print(f"Input type = {type(a)}, FFT output type = {type(b)}")