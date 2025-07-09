# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to use batched operands in a sparse direct solver operation.
We distinguish batching into explicit, where the samples in a batch are provided as
a sequence of matrices (or vectors for the RHS), and implicit, where the samples are
inferred from 3D or higher-dimensional tensors for the LHS and RHS. The batching for
the LHS and RHS can be independent - the LHS can be batched explicitly while the RHS
can be batched implicitly and vice-versa.

This example shows that each sample in an explicitly batched system can be of different
size (number of equations), resulting in a flexible user interface. We use CuPy operands
on the GPU.
"""

import cupy as cp
import cupyx.scipy.sparse as sp

import nvmath

# The number of equations in each sample in the batch of size 3.
n = 4, 8, 12

# Prepare sample input data.
# Create an explicit batch for the LHS as a sequence of sparse systems of equations, where
# each sample in the batch has a different number of equations.
a = []
for size in n:
    o = sp.random(size, size, density=0.5, format="csr", dtype="float64")
    o += sp.diags([2.0] * size, format="csr", dtype="float64")
    a.append(o)

# Create the RHS batch, as a sequence of matrices or vectors of size corresponding to the
# number of equations in each sample.
b = []
for size in n:
    b.append(cp.ones((size, 2), order="F"))

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
