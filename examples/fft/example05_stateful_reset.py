# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to reuse the stateful API to perform FFT operations on operands
with the same layout.

In this example we will perform a forward and an inverse FFT operation to demonstrate how to
recover the original input operand.
"""

import cupy as cp

import nvmath

shape = 512, 512, 512
axes = 0, 1

a = cp.ones(shape, dtype=cp.complex64)

# Create a stateful FFT object 'f'. Note here that we need to enforce natural layout in the
# result in order to reuse the FFT object
with nvmath.fft.FFT(a, axes=axes, options={"result_layout": "natural"}) as f:
    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    b = f.execute(direction=nvmath.fft.FFTDirection.FORWARD)

    # Reset the operand to the values in frequency domain.
    f.reset_operand(b)

    # Execute the new inverse FFT.
    c = f.execute(direction=nvmath.fft.FFTDirection.INVERSE)

    # Synchronize the default stream
    cp.cuda.get_current_stream().synchronize()
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"FFT output type = {type(b)}, device = {b.device}")
    print(f"IFFT output type = {type(c)}, device = {c.device}")
