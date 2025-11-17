# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful ternary tensor contraction objects.
Stateful objects amortize the cost of preparation across multiple executions.

The inputs as well as the result are CuPy ndarrays.
"""

import cupy as cp

import nvmath

a = cp.random.rand(4, 6, 8)
b = cp.random.rand(6, 8, 3)
c = cp.random.rand(3, 9)


# result[i,j,m,n] = \sum_{k,l} a[i,j,k] * b[j,k,l] * c[l,n]

# Create a stateful TernaryContraction object 'contraction'.
with nvmath.tensor.TernaryContraction("ijk,jkl,ln->in", a, b, c) as contraction:
    # Plan the Contraction.
    contraction.plan()

    # Execute the Contraction.
    result = contraction.execute()

    # Synchronize the default stream
    cp.cuda.get_current_stream().synchronize()
    assert cp.allclose(result, cp.einsum("ijk,jkl,ln->in", a, b, c))
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"Contraction output type = {type(result)}, device = {result.device}")
