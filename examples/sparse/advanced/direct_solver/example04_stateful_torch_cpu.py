# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful sparse direct solver objects. Stateful objects
amortize the cost of preparation across multiple executions, and are especially important
for direct sparse solvers where preparatory steps like reordering can be expensive.

The inputs as well as the result are PyTorch tensors on the CPU.
"""

import torch

import nvmath

# The number of equations.
n = 8

# Prepare sample input data.
# Create a diagonally dominant random CSR matrix.
a = torch.rand(n, n) + torch.diag(torch.tensor([10] * n))
a = a.to_sparse_csr()
# Note that torch uses int64 for index buffers, whereas cuDSS currently requires int32.
a = torch.sparse_csr_tensor(
    a.crow_indices().to(dtype=torch.int32), a.col_indices().to(dtype=torch.int32), a.values(), size=a.size()
)

# Create the RHS, which can be a matrix or vector in column-major layout.
b = torch.ones(2, n).T

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

# No synchronization is needed for CPU tensors, since the execution always blocks.
print(x)

# Check if the result is torch tensor as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
print(f"Inputs were located on devices {a.device} and {b.device} and the result is on {x.device}")
