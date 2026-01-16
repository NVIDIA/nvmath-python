# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic ternary tensor contraction using Torch tensors.

nvmath-python supports multiple frameworks. The result of each operation is a tensor of the
same framework that was used to pass the inputs. It is also located on the same device as
the inputs.
"""

import torch

import nvmath

a = torch.rand(8, 8, 8, 8, device="cuda")
b = torch.rand(8, 8, 8, 8, device="cuda")
c = torch.rand(8, 8, 8, 8, device="cuda")


# result[i,j,p,q] = \sum_{k,l,m,n} a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q]
result = nvmath.tensor.ternary_contraction("ijkl,klmn,mnpq->ijpq", a, b, c)

assert torch.allclose(result, torch.einsum("ijkl,klmn,mnpq->ijpq", a, b, c))

print(f"Input type = {type(a), type(b), type(c)}, contraction result type = {type(result)}")

# Optionally, users may scale the contraction result with a scale factor alpha,
# and/or add an additional operand d to the contraction result with a scale factor beta

alpha, beta = 1.3, 0.7
d = torch.rand(8, 8, 8, 8, device="cuda")

# result[i,j,p,q] = \sum_{k,l,m,n} alpha * a[i,j,k,l] * b[k,l,m,n] * c[m,n,p,q]
#                   + beta * d[i,j,p,q]
# when d is specified for ternary contraction, beta must be set
result = nvmath.tensor.ternary_contraction("ijkl,klmn,mnpq->ijpq", a, b, c, d=d, alpha=alpha, beta=beta)

assert torch.allclose(result, alpha * torch.einsum("ijkl,klmn,mnpq->ijpq", a, b, c) + beta * d)

print(f"Input type = {type(a), type(b), type(c)}, contraction result type = {type(result)}")
