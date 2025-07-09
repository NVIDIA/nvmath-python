# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of multiple CUDA streams with the sparse direct
solver APIs.
"""

import cupy as cp
import cupyx.scipy.sparse as sp


import nvmath

# The number of equations.
n = 8

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix.
a = sp.random(n, n, density=0.5, format="csr", dtype="float64")
a += sp.diags([2.0] * n, format="csr", dtype="float64")

# Create the RHS, which can be a matrix or vector in column-major layout.
b = cp.ones((n,))

# Create a CUDA stream to use for instantiating, planning, and first execution of a stateful
# solver object 'solver'.
s1 = cp.cuda.Stream()

# Solve a @ x = b for x.
with nvmath.sparse.advanced.DirectSolver(a, b) as solver:
    # Plan the sparse solve, where reordering to minimize fill-in and symbolic
    # factorization are done.
    plan_info = solver.plan(stream=s1)

    # Perform numerical factorization, which returns DirectSolverFactorizationInfo.
    fac_info = solver.factorize(stream=s1)

    # Solve the system.
    x = solver.solve(stream=s1)

    # Record an event on s1 for use later.
    e1 = s1.record()

    # Create a new stream to on which the new operand c for the second execution will be
    # filled.
    s2 = cp.cuda.Stream()

    # Fill c on s2.
    with s2:
        c = cp.random.rand(n)

    # In the following blocks, we will use stream s2 to perform subsequent operations. Note
    # that it's our responsibility as a user to ensure proper ordering, and we want to order
    # `reset_operands` after event e1 corresponding to the solve() call above.
    s2.wait_event(e1)

    # Alternatively, if we want to use stream s1 for subsequent operations (s2 only for
    # operand creation), we need to order `reset_operands` after the event for
    # cupy.random.rand on s2, e.g: e2 = s2.record() s1.wait_event(e2)

    # Set a new operand c on stream s2.
    solver.reset_operands(b=c, stream=s2)

    # Execute the new solve on stream s2.
    y = solver.solve(stream=s2)

    # Synchronize s2 at the end, as needed.
    s2.synchronize()

print(x)
print(y)

# Check if the result is torch tensor as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the results are of types {type(x)} and {type(y)}.")
