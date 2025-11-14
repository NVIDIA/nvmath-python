# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful ternary tensor contraction
objects with scalars.

Stateful objects allow users to reuse the same contraction object for multiple executions
with different set of scalar parameters. The inputs as well as the result are CuPy ndarrays.
"""

import cupy as cp

import nvmath

a = cp.random.rand(8, 8, 8, 8)
b = cp.random.rand(8, 8, 8, 8)
c = cp.random.rand(8, 8, 8, 8)

# d may not be needed for the contraction depending on the specific contraction,
# but here we pass it to demonstrate how to reuse the same contraction object.
d = cp.random.rand(8, 8, 8, 8)

with nvmath.tensor.TernaryContraction("ijkl,klmn,mnpq->ijpq", a, b, c, d=d) as contraction:
    # Plan the contraction.
    contraction.plan()

    # Execute the contraction.
    # NOTE: when d is specified for ternary contraction, beta must be set

    # Case 1: alpha = 1.0 (default) and beta = 1.0 (must be set)
    # result[i,j,p,q] = \sum_{k,l,m,n} a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q] + d[i,j,p,q]
    result = contraction.execute(beta=1.0)
    assert cp.allclose(result, cp.einsum("ijkl,klmn,mnpq->ijpq", a, b, c) + d)

    # Case 2: alpha = 2 and beta = 0 (equivalent to d=0)
    # result[i,j,p,q] = \sum_{k,l,m,n} alpha * a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q]
    alpha = 2
    beta = 0
    result = contraction.execute(alpha=alpha, beta=beta)
    # NOTE: If d is not provided during the initialization of the contraction object,
    #       beta does not need to be provided.
    assert cp.allclose(result, cp.einsum("ijkl,klmn,mnpq->ijpq", a, b, c) * alpha)

    # Case 3: alpha = 1.0 (default) and beta = 0.2
    # result[i,j,p,q] =
    # \sum_{k,l,m,n} a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q] + beta * d[i,j,p,q]
    beta = 0.2
    result = contraction.execute(beta=beta)
    assert cp.allclose(result, cp.einsum("ijkl,klmn,mnpq->ijpq", a, b, c) + d * beta)

    # Case 4: alpha = 1.4 and beta = 0.5
    # result[i,j,p,q] =
    # \sum_{k,l,m,n} alpha * a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q] + beta * d[i,j,p,q]
    alpha = 1.4
    beta = 0.5
    result = contraction.execute(alpha=alpha, beta=beta)
    assert cp.allclose(result, cp.einsum("ijkl,klmn,mnpq->ijpq", a, b, c) * alpha + d * beta)
