# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of inplace update of input operands in stateful FFT APIs.

The input as well as the result from the FFT operations are CuPy ndarrays.

NOTE: The use of inplace updating input operands should be adopted with caution.

For the following cases, inplace updating the input operands will not affect the result
operand:
    - The input operand reside on CPU.
    - The input operand reside on GPU but the operation amounts to a C2R FFT.
"""

import cupy as cp

import nvmath

shape = 512, 512, 512
axes = 0, 1

a = cp.ones(shape, dtype=cp.complex64)

# Create a stateful FFT object 'f'.
with nvmath.fft.FFT(a, axes=axes) as f:
    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    b = f.execute(direction=nvmath.fft.FFTDirection.FORWARD)

    # Update the operand in-place with the frequency domain values.
    print("Updating 'a' in-place.")
    a[:] = b

    # Execute the new FFT.
    c = f.execute(direction=nvmath.fft.FFTDirection.INVERSE)

    # Synchronize the default stream
    cp.cuda.get_current_stream().synchronize()
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"FFT output type = {type(b)}, device = {b.device}")
    print(f"IFFT output type = {type(c)}, device = {c.device}")
