# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example showing the fallback path for an unsupported layout error using the class-form FFT APIs.
"""
import cupy as cp

import nvmath

shape = 256, 256, 512
axes  = 0, 2

a = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

r = cp.fft.fftn(a, axes=axes)

# Create a stateful FFT object 'f'. Fallback to suggested layout since the original layout-axes combination is currently not supported.
try:
    f = nvmath.fft.FFT(a, axes=axes)
    permutation = None
except nvmath.fft.UnsupportedLayoutError as e:
    # Get the permutation to use to create a supported layout-axes combination.
    permutation = e.permutation
    # Permute the input, and copy.
    a = a.transpose(*e.permutation).copy()
    # Create a stateful FFT object using the permuted input and corresponding axes.
    f = nvmath.fft.FFT(a, axes=e.axes)

# Use context manager to free resources automatically.
with f:

    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    b = f.execute(direction=nvmath.fft.FFTDirection.FORWARD)

    # Permute the result so that the axes correspond to the original (unpermuted) input 'a'.
    if permutation:
        permutation = list(permutation)
        reverse_permutation = [permutation.index(i) for i in range(a.ndim)]
        b = b.transpose(*reverse_permutation)

    # Synchronize the default stream
    cp.cuda.get_current_stream().synchronize()
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"FFT output type = {type(b)}, device = {b.device}")