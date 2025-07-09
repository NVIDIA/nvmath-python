# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful sparse direct solver objects. Stateful objects
amortize the cost of preparation across multiple executions, and are especially important
for direct sparse solvers where preparatory steps like reordering can be expensive.

The inputs as well as the result are CuPy ndarrays on the GPU.
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

# Solve a @ x = b for x.
# Use the stateful object as a context manager to automatically release resources.
with nvmath.sparse.advanced.DirectSolver(a, b) as solver:
    # Each phase plan(), factorize(), solve() can be optionally configured to tailor
    # their behavior, as we shall see in later examples.

    # Plan the sparse solve, where reordering to minimize fill-in and symbolic
    # factorization are done. It returns a DirectSolverPlanInfo object, whose attributes
    # can be queried.
    plan_info = solver.plan()

    # Perform numerical factorization, which returns DirectSolverFactorizationInfo. You
    # need to refactor if the LHS values change (but the sparsity structure should remain
    # the same). Alternatively, if the LHS values change only a little, you can solve
    # solve without refactorization if the number of iterations for iterative refinement
    # is adequate and it converges.
    fac_info = solver.factorize()

    # Solve the system.
    x = solver.solve()

    # Synchronize the default stream, since by default the execution is non-blocking for GPU
    # operands.
    cp.cuda.get_current_stream().synchronize()

print(x)

# Check if the result is torch tensor as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
