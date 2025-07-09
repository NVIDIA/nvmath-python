# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of resetting the operands in stateful sparse direct
solver objects. Stateful objects amortize the cost of preparation across multiple
executions.

The LHS is a SciPy CSR matrix while the RHS and solution are NumPy ndarrays on the CPU.
"""

import numpy as np
import scipy.sparse as sp


import nvmath

# The number of equations.
n = 8

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix.
a = sp.random_array((n, n), density=0.5, format="csr", dtype="float64")
a += sp.diags_array([2.0] * n, format="csr", dtype="float64")

# Create the RHS, which can be a matrix or vector in column-major layout.
b = np.ones((n,))

# Solve a @ x = b for x.
# Use the stateful object as a context manager to automatically release resources.
with nvmath.sparse.advanced.DirectSolver(a, b) as solver:
    # Plan the sparse solve.
    plan_info = solver.plan()

    # Perform numerical factorization, which returns factorization_info. You
    fac_info = solver.factorize()

    # Solve the system.
    x = solver.solve()
    print(x)

    # Modify the rhs in-place and solve.
    b[:] *= 10
    x = solver.solve()
    print(x)

    # For out-of-place modifications, the operand has to be reset.
    b = b / 10
    solver.reset_operands(b=b)
    x = solver.solve()
    print(x)

    # Now let's modify the LHS. For small changes, the LHS can be modified
    # and iterative refinement used in the solve() to obtain the result
    # without re-factorization.
    solution_config = solver.solution_config
    solution_config.ir_num_steps = 30

    # Update A in place.
    a.data *= 1.1
    x = solver.solve()
    print(x)

    # For larger changes to A, it's better to refactorize.
    solution_config.ir_num_steps = 0

    a.data *= 10
    solver.factorize()
    x = solver.solve()
    print(x)

    # 'A' can also be updated out-of-place and operands reset.
    a = a / 10
    solver.reset_operands(a=a)
    solver.factorize()
    x = solver.solve()
    print(x)

# Check if the result is numpy ndarray as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
