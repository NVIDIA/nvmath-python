# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of inplace update of input operands in stateful matrix multiplication APIs.

The inputs as well as the result are CuPy ndarrays.
NOTE: The operands should be updated inplace only when they are in a memory space that is accessible from the execution space.
In this case, the operands reside on the GPU while the execution also happens on the GPU.
"""

import cupy as cp

import nvmath

# Prepare sample input data
m, n, k = 123, 456, 789
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)

# Turn on logging to see what's happening.
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Use the stateful object as a context manager to automatically release resources.
with nvmath.linalg.advanced.Matmul(a, b) as mm:
    # Plan the matrix multiplication. Planning returns a sequence of algorithms that can be configured as we'll see in a later example.
    mm.plan()

    # Execute the matrix multiplication.
    result = mm.execute()

    # Update the operand A in-place.
    print("Updating 'a' in-place.")
    a[:] = cp.random.rand(m, k)

    # Execute the new matrix multiplication.
    result = mm.execute()

    # Synchronize the default stream, since by default the execution is non-blocking for GPU operands.
    cp.cuda.get_current_stream().synchronize()
    print(f"Input types = {type(a), type(b)}, device = {a.device, b.device}")
    print(f"Result type = {type(result)}, device = {result.device}")
