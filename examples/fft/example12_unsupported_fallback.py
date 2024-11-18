# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example showing the fallback path for an unsupported layout error using the function-form FFT APIs.
"""

import cupy as cp

import nvmath

shape = 256, 256, 512
axes = 0, 2

a = cp.ones(shape, dtype=cp.complex128)

r = cp.fft.fftn(a, axes=axes)

try:
    # Forward FFT along (0,2), batched along axis=1. This is not yet supported by the cuFFT C library.
    b = nvmath.fft.fft(a, axes=axes)
except nvmath.fft.UnsupportedLayoutError as e:
    # Permute the input, and copy.
    a = a.transpose(*e.permutation).copy()
    # Perform the FFT on the permuted input and corresponding axes.
    b = nvmath.fft.fft(a, axes=e.axes)
    # Permute the result so that the axes correspond to the original input.
    permutation = list(e.permutation)
    reverse_permutation = [permutation.index(i) for i in range(a.ndim)]
    b = b.transpose(*reverse_permutation)

    # Synchronize the default stream
    cp.cuda.get_current_stream().synchronize()
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"FFT output type = {type(b)}, device = {b.device}")
