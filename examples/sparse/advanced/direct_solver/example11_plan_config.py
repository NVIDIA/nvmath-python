# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to configure the various phases in a sparse direct solver
operation, such as plan(), factorize(), and solve().  It also show how to query information
returned by each phase. Stateful objects amortize the cost of preparation across multiple
executions.
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
    # Configure the reordering algorithm for the plan.
    p = solver.plan_config
    p.algorithm = nvmath.sparse.advanced.DirectSolverAlgType.ALG_1

    # Plan the operation using the specified plan configuration, which returns
    # a DirectSolverPlanInfo object.
    plan_info = solver.plan()

    # Perform numerical factorization, which returns a DirectSolverFactorizationInfo
    # object.
    fac_info = solver.factorize()
    print(f"Number of non-zeros in the factorized system = {fac_info.lu_nnz}")

    # Solve the system.
    x = solver.solve()

    # Synchronize the default stream, since by default the execution is non-blocking for GPU
    # operands.
    cp.cuda.get_current_stream().synchronize()

print(x)

# Check if the result is torch tensor as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
