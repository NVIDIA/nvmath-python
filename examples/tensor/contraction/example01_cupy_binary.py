# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic binary tensor contraction using CuPy arrays.

nvmath-python supports multiple frameworks. The result of each operation is a tensor of the
same framework that was used to pass the inputs. It is also located on the same device as
the inputs.
"""

import cupy as cp

import nvmath

a = cp.random.rand(4, 4, 12, 12)
b = cp.random.rand(12, 12, 8, 8)

# result[i,j,m,n] = \sum_{k,l} a[i,j,k,l] * b[k,l,m,n]
result = nvmath.tensor.binary_contraction("ijkl,klmn->ijmn", a, b)

assert cp.allclose(result, cp.einsum("ijkl,klmn->ijmn", a, b))

print(f"Input type = {type(a), type(b)}, contraction result type = {type(result)}")

# Optionally, users may scale the contraction result with a scale factor alpha,
# and/or add an additional operand c to the contraction result with a scale factor beta

alpha, beta = 1.3, 0.7
c = cp.random.rand(4, 4, 8, 8)

# result[i,j,m,n] = \sum_{k,l} alpha * a[i,j,k,l] * b[k,l,m,n] + beta * c[i,j,m,n]
# when c is specified for binary contraction, beta must be set
result = nvmath.tensor.binary_contraction("ijkl,klmn->ijmn", a, b, c=c, alpha=alpha, beta=beta)

assert cp.allclose(result, alpha * cp.einsum("ijkl,klmn->ijmn", a, b) + beta * c)

print(f"Input type = {type(a), type(b), type(c)}, contraction result type = {type(result)}")
