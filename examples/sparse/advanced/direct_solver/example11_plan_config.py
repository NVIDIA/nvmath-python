# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to configure the various phases in a sparse direct solver
operation using both the stateless function-form API and the stateful object API. It
demonstrates how to set configurations for each phase, perform planning, factorization, and
solving. For stateful API, it also shows how to query information returned by each phase.
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

# Option 1. Use the stateless function-form API.

# Set the preferences for the plan, factorization, and solution.
plan_preferences = {
    "reordering_algorithm": nvmath.sparse.advanced.DirectSolverAlgType.ALG_1,
}
factorization_preferences = {
    "pivot_eps": 1e-12,
}
# equivalent to the dictionary version above
solution_preferences = nvmath.sparse.advanced.DirectSolverSolutionPreferences(
    ir_num_steps=50,
)
x = nvmath.sparse.advanced.direct_solver(
    a,
    b,
    plan_preferences=plan_preferences,
    factorization_preferences=factorization_preferences,
    solution_preferences=solution_preferences,
)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()

print(x)

# Check if the result is cupy ndarray as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")

# Option 2. Use the stateful object as a context manager to automatically release resources.
with nvmath.sparse.advanced.DirectSolver(a, b) as solver:
    # Configure the reordering algorithm for the plan.
    p = solver.plan_config
    p.reordering_algorithm = plan_preferences["reordering_algorithm"]

    # Plan the operation using the specified plan configuration, which returns
    # a DirectSolverPlanInfo object.
    plan_info = solver.plan()

    # Configure the factorization configuration.
    f = solver.factorization_config
    f.pivot_eps = factorization_preferences["pivot_eps"]
    # Perform numerical factorization, which returns a DirectSolverFactorizationInfo
    # object.
    fac_info = solver.factorize()
    print(f"Number of non-zeros in the factorized system = {fac_info.lu_nnz}")

    # Configure the solution configuration.
    s = solver.solution_config
    s.ir_num_steps = solution_preferences.ir_num_steps
    # Solve the system.
    x = solver.solve()

    # Synchronize the default stream, since by default the execution is non-blocking for GPU
    # operands.
    cp.cuda.get_current_stream().synchronize()

print(x)

# Check if the result is cupy ndarray as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
