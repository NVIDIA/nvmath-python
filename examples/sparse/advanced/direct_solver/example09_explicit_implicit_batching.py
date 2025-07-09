# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to use batched operands in a sparse direct solver operation.
We distinguish batching into explicit, where the samples in a batch are provided as
a sequence of matrices (or vectors for the RHS), and implicit, where the samples are
inferred from 3D or higher-dimensional tensors for the LHS and RHS.

This example shows that the batching specification for the LHS and RHS can be independent
- the LHS can be batched explicitly while the RHS can be batched implicitly and vice-versa.
We use CuPy operands in this example on the GPU.
"""

import cupy as cp
import cupyx.scipy.sparse as sp

import nvmath

# The number of equations.
n = 8
batch = 2

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix.
a = sp.random(n, n, density=0.5, format="csr", dtype="float64")
a += sp.diags([2.0] * n, format="csr", dtype="float64")

# Create an explicit batch for the LHS as a sequence of sparse systems of equations.
# In this case, all systems have the same number of equations but this is not required
# as we'll see in a later example.
a = [a, 10 * a]

# Create an implicitly-batched RHS as a 3D ndarray, where each matrix sample is col-major.
b = cp.ones((batch, 2, n)).transpose(0, 2, 1)

# Solve a @ x = b for x.
x = nvmath.sparse.advanced.direct_solver(a, b)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()

print(x)

# Check if the result is cupy array as well.
print(
    f"Inputs were sequences of types {type(a[0])} and {type(b[0])} and the result \
sequence is of type {type(x[0])}."
)
