# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to reset operands in stateful matrix multiplication APIs, and reuse the object for multiple executions. This is
needed when the memory space of the operands is not accesible from the execution space, or if it's desired to bind new (compatible) operands
to the stateful object.

The inputs as well as the result are NumPy ndarrays.
"""
import numpy as np

import nvmath

# Prepare sample input data
m, n, k = 123, 456, 789
m, n, k = 2, 3, 4
a = np.random.rand(m, k)
b = np.random.rand(k, n)

# Turn on logging to see what's happening.
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Use the stateful object as a context manager to automatically release resources.
with nvmath.linalg.advanced.Matmul(a, b) as mm:

    # Plan the matrix multiplication. Planning returns a sequence of algorithms that can be configured as we'll see in a later example.
    mm.plan()

    # Execute the matrix multiplication.
    result = mm.execute()

    # Create new operands and bind them.
    c = np.random.rand(m, k)
    d = np.random.rand(k, n)
    mm.reset_operands(c, d)

    # Execute the new matrix multiplication.
    result = mm.execute()

    # No synchronization is needed for CPU tensors, since the execution always blocks.

    print(f"Input types = {type(c), type(d)}")
    print(f"Result type = {type(result)}")
