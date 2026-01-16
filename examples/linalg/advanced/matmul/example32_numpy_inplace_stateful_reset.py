# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates

1. how to perform an in-place matrix multiplication, where the result overwrites
   operand `c`:

    c := alpha a @ b + beta c

2. how to reset operands in stateful matrix multiplication APIs, and reuse the
   object for multiple executions. This is needed when the memory space of the
   operands is not accessible from the execution space, or if it's desired to
   bind new (compatible) operands to the stateful object.

Note that operand `c` cannot be broadcast (in the batch dimensions as well as the
N dimension) when the inplace option is used, since it must be large enough to
hold the result of the computation.

The inputs as well as the result are NumPy ndarrays.
"""

import logging

import numpy as np

import nvmath

# Turn on logging to see what's happening.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Prepare sample input data.
batch, m, n, k = 3, 123, 456, 789
a1 = np.random.rand(batch, m, k)
b1 = np.random.rand(batch, k, n)
# Create operand `c` which will also hold the result.
c1_out = np.random.rand(batch, m, n)
# Take a copy of operand `c` for checking the result.
c1 = c1_out.copy()
beta = 1.0

options = {"inplace": True}
# Use the stateful object as a context manager to automatically release resources.
with nvmath.linalg.advanced.Matmul(a1, b1, c=c1_out, beta=beta, options=options) as mm:
    # Plan the matrix multiplication.
    mm.plan()

    # Execute the matrix multiplication.
    result1 = mm.execute()
    assert result1 is c1_out, "Error: the operation is not in-place."

    # Check the first result.
    reference1 = a1 @ b1 + beta * c1
    assert np.allclose(reference1, result1), "Error: the first result is incorrect."

    # Create new operands and bind them.
    a2 = np.random.rand(batch, m, k)
    b2 = np.random.rand(batch, k, n)
    c2_out = np.random.rand(batch, m, n)
    c2 = c2_out.copy()  # Take a copy for checking, since `c2_out` is overwritten.

    mm.reset_operands(a=a2, b=b2, c=c2_out)

    # Execute the second matrix multiplication.
    result2 = mm.execute()
    assert result2 is c2_out, "Error: the operation is not in-place."

    # Check the second result.
    reference2 = a2 @ b2 + beta * c2
    assert np.allclose(reference2, result2), "Error: the second result is incorrect."

    # Finally, consider the case where `c` is NOT reset and so holds its previous value.
    # The operation corresponds to `c := a3 @ b3 + (c=result2)`, since the operation is
    # in-place (`c` and `result` share the same memory).
    a3 = 2 * a1
    b3 = 2 * b1

    mm.reset_operands(a=a3, b=b3)

    result3 = mm.execute()
    assert result3 is c2_out and result3 is result2, "Error: the operation is not in-place."

    # Check the third result. Remember to use reference2 for the `c` term
    # since it is NOT reset.
    reference3 = a3 @ b3 + beta * reference2
    assert np.allclose(reference3, result3), "Error: the third result is incorrect."

    # No synchronization is needed for CPU tensors, since the call always blocks.
    print(f"Input types = {type(a1), type(b1), type(c1)}")
    print(f"Result type = {type(result1)}")
