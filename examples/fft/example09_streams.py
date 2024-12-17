# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example on using multiple CUDA streams in FFT APIs.
"""

import cupy as cp

import nvmath

shape = 512, 256, 256
axes = 0, 1

a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

# Create a CUDA stream to use for instantiating, planning, and first execution of a stateful
# FFT object 'f'.
s1 = cp.cuda.Stream()

# Create a stateful FFT object 'f' on stream s1.
with nvmath.fft.FFT(a, axes=axes, options={"blocking": "auto"}, stream=s1) as f:
    # Plan the FFT on stream s1.
    f.plan(stream=s1)

    # Execute the FFT on stream s1.
    b = f.execute(stream=s1)

    # Record an event on s1 for use later.
    e1 = s1.record()

    # Create a new stream to on which the new operand c for the second execution will be
    # filled.
    s2 = cp.cuda.Stream()

    # Fill c on s2.
    with s2:
        c = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

    # In the following blocks, we will use stream s2 to perform subsequent operations. Note
    # that it's our responsibility as a user to ensure proper ordering, and we want to order
    # `reset_operand` after event e1 corresponding to the execute() call above.
    s2.wait_event(e1)

    # Alternatively, if we want to use stream s1 for subsequent operations (s2 only for
    # operand creation), we need to order `reset_operand` after the event for
    # cupy.random.rand on s2, e.g: e2 = s2.record() s1.waite_event(e2)

    # Set a new operand c on stream s2.
    f.reset_operand(c, stream=s2)

    # Execute the new FFT on stream s2.
    d = f.execute(stream=s2)

    # Synchronize s2 at the end
    s2.synchronize()
