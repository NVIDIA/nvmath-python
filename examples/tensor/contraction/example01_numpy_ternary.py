# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic ternary tensor contraction using NumPy arrays.

nvmath-python supports multiple frameworks. The result of each operation is a tensor of the
same framework that was used to pass the inputs. It is also located on the same device as
the inputs.
"""

import numpy as np

import nvmath

a = np.random.rand(8, 8, 8, 8)
b = np.random.rand(8, 8, 8, 8)
c = np.random.rand(8, 8, 8, 8)

# result[i,j,p,q] = \sum_{k,l,m,n} a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q]
# when d is specified for ternary contraction, beta must be set
result = nvmath.tensor.ternary_contraction("ijkl,klmn,mnpq->ijpq", a, b, c)

assert np.allclose(result, np.einsum("ijkl,klmn,mnpq->ijpq", a, b, c))

print(f"Input type = {type(a), type(b), type(c)}, contraction result type = {type(result)}")

# Optionally, users may scale the contraction result with a scale factor alpha,
# and/or add an additional operand d to the contraction result with a scale factor beta

alpha, beta = 1.3, 0.7
d = np.random.rand(8, 8, 8, 8)


# result[i,j,p,q] = \sum_{k,l,m,n} alpha * a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q]
#                   + beta * d[i,j,p,q]
# when d is specified for ternary contraction, beta must be set
result = nvmath.tensor.ternary_contraction("ijkl,klmn,mnpq->ijpq", a, b, c, d=d, alpha=alpha, beta=beta)

assert np.allclose(result, alpha * np.einsum("ijkl,klmn,mnpq->ijpq", a, b, c) + beta * d)

print(f"Input type = {type(a), type(b), type(c)}, contraction result type = {type(result)}")
