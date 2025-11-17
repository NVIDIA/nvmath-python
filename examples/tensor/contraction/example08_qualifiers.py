# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of qualifiers to specify the operators for the operands
in a ternary tensor contraction operation. As of cuTensor 2.3.1, only the conjugate operator
is supported in the contraction APIs when the operands are complex.

The inputs as well as the result are CuPy ndarrays.
"""

import cupy as cp
import numpy as np

import nvmath

a = cp.random.rand(8, 8, 8, 8) + 1j * cp.random.rand(8, 8, 8, 8)
b = cp.random.rand(8, 8, 8, 8) + 1j * cp.random.rand(8, 8, 8, 8)
c = cp.random.rand(8, 8, 8, 8) + 1j * cp.random.rand(8, 8, 8, 8)
d = cp.random.rand(8, 8, 8, 8) + 1j * cp.random.rand(8, 8, 8, 8)

# create an array of qualifiers (of length # of operands) with the default identity operator
qualifiers = np.full(4, nvmath.tensor.Operator.OP_IDENTITY, dtype=nvmath.tensor.tensor_qualifiers_dtype)
# set the qualifier for operand b to conjugate
qualifiers[1] = nvmath.tensor.Operator.OP_CONJ

# result[i,j,p,q] = \sum_{k,l,m,n} a[i,j,k,l] * b[k,l,m,n].conj() * c[m,n,p,q] + d[i,j,p,q]
result = nvmath.tensor.ternary_contraction("ijkl,klmn,mnpq->ijpq", a, b, c, d=d, qualifiers=qualifiers, beta=1)
reference = cp.einsum("ijkl,klmn,mnpq->ijpq", a, b.conj(), c) + d
assert cp.allclose(result, reference)
