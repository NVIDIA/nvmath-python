# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful binary tensor contraction objects.
Stateful objects amortize the cost of preparation across multiple executions.

The inputs as well as the result are CuPy ndarrays.
"""

import cupy as cp

import nvmath

a = cp.random.rand(4, 4, 12, 12)
b = cp.random.rand(12, 12, 8, 8)

c = cp.random.rand(4, 4, 8, 8)

alpha, beta = 0.3, 0.9

# result[i,j,m,n] = \sum_{k,l} alpha * a[i,j,k,l] * b[k,l,m,n] + beta * c[i,j,m,n]

# Create a stateful BinaryContraction object 'contraction'.
with nvmath.tensor.BinaryContraction("ijkl,klmn->ijmn", a, b, c=c) as contraction:
    # Plan the Contraction.
    contraction.plan()

    # Execute the Contraction.
    result = contraction.execute(alpha=alpha, beta=beta)

    # Synchronize the default stream
    cp.cuda.get_current_stream().synchronize()
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"Contraction output type = {type(result)}, device = {result.device}")
