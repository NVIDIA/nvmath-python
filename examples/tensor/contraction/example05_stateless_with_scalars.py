# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateless ternary tensor contraction
function-form APIs with scalars.

Stateless function-form APIs allow users to perform a single contraction with
a given set of scalar parameters. The inputs as well as the result are NumPy ndarrays.
"""

import numpy as np

import nvmath

a = np.random.rand(8, 8, 8, 8)
b = np.random.rand(8, 8, 8, 8)
c = np.random.rand(8, 8, 8, 8)
d = np.random.rand(8, 8, 8, 8)


# Case 1: alpha = 1.0 (default) and d = None (default and beta does not need to be set)
# result[i,j,p,q] = \sum_{k,l,m,n} a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q]
result = nvmath.tensor.ternary_contraction("ijkl,klmn,mnpq->ijpq", a, b, c)
assert np.allclose(result, np.einsum("ijkl,klmn,mnpq->ijpq", a, b, c))

# Case 2: alpha = 2 and d = None (default and beta does not need to be set)
# result[i,j,p,q] = \sum_{k,l,m,n} alpha * a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q]
alpha = 2
result = nvmath.tensor.ternary_contraction("ijkl,klmn,mnpq->ijpq", a, b, c, alpha=alpha)
assert np.allclose(result, np.einsum("ijkl,klmn,mnpq->ijpq", a, b, c) * alpha)

# Case 3: alpha = 1.0 (default) and beta = 0.2 with a non-zero d
# result[i,j,p,q] = \sum_{k,l,m,n} a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q] + beta * d[i,j,p,q]
beta = 0.2
result = nvmath.tensor.ternary_contraction("ijkl,klmn,mnpq->ijpq", a, b, c, d=d, beta=beta)
assert np.allclose(result, np.einsum("ijkl,klmn,mnpq->ijpq", a, b, c) + d * beta)


# Case 4: alpha = 1.4 and beta = 0.5 with a non-zero d
# result[i,j,p,q] = \sum_{k,l,m,n} alpha * a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q]
alpha = 1.4
beta = 0.5
result = nvmath.tensor.ternary_contraction("ijkl,klmn,mnpq->ijpq", a, b, c, d=d, alpha=alpha, beta=beta)
assert np.allclose(result, np.einsum("ijkl,klmn,mnpq->ijpq", a, b, c) * alpha + d * beta)
