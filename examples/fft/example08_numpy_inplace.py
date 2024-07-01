# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to perform inplace FFT for NumPy ndarrays using function-form FFT APIs.
"""
import numpy as np

import nvmath

shape = 256, 512, 512
axes  = 0, 1

a = np.ones(shape, dtype=np.complex128)
a_copy = a.copy()    # Since `a` is overwritten.

# Forward FFT along (0,1), batched along axis=2.
b = nvmath.fft.fft(a, axes=axes, options={'inplace': True})
assert(b is a) # `a` is overwritten

# Inverse FFT along (0,1), batched along axis=2.
c = nvmath.fft.ifft(b, axes=axes, options={'inplace': True})
assert(c is b) # `b`` is again overwritten

print(f"Input type = {type(a)}, FFT type = {type(b)}, IFFT type = {type(c)}")
