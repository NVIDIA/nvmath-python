# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to manage memory resources used by stateful objects. This is useful when the FFT operation
needs a lot of memory and calls to execution method on a stateful object are interleaved with calls to other
operations (including another FFT) also requiring a lot of memory.

In this example, two FFT operations are performed in a loop in an interleaved manner.
We assume that the available device memory is large enough for only one FFT at a time.
"""
import logging

import cupy as cp

import nvmath


shape = 256, 512, 512
axes  = 0, 1

a = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)
b = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

# Turn on logging and set the level to DEBUG to print memory management messages.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

# Create and prepare two FFT objects.
f1 = nvmath.fft.FFT(a, axes=axes)
f1.plan()

f2 = nvmath.fft.FFT(b, axes=axes)
f2.plan()

num_iter = 3
# Use the FFT objects as context managers so that internal library resources are properly cleaned up.
with f1, f2:

    for i in range(num_iter):
        print(f"Iteration {i}")
        # Perform the first contraction, and request that the workspace be released at the end of the operation so that there is enough
        #   memory for the second one.
        r = f1.execute(release_workspace=True)

        # Update f1's operands for the next iteration.
        if i < num_iter-1:
            a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

        # Perform the second FFT, and request that the workspace be released at the end of the operation so that there is enough
        #   memory for the first FFT in the next iteration.
        r = f2.execute(release_workspace=True)

        # Update f2's operands for the next iteration.
        if i < num_iter-1:
            b[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

        # Synchronize the default stream
        cp.cuda.get_current_stream().synchronize()
        print(f"Input type = {type(a)}, device = {a.device}")
        print(f"FFT output type = {type(r)}, device = {r.device}")