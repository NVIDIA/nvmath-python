# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic binary tensor contraction using Torch tensors.


nvmath-python supports multiple frameworks. The result of each operation is a tensor of the
same framework that was used to pass the inputs. It is also located on the same device as
the inputs.
"""

import torch

import nvmath

a = torch.rand(4, 4, 12, 12, device="cuda")
b = torch.rand(12, 12, 8, 8, device="cuda")

# result[i,j,m,n] = \sum_{k,l} a[i,j,k,l] * b[k,l,m,n]
result = nvmath.tensor.binary_contraction("ijkl,klmn->ijmn", a, b)

assert torch.allclose(result, torch.einsum("ijkl,klmn->ijmn", a, b))

print(f"Input type = {type(a), type(b)}, contraction result type = {type(result)}")

# Optionally, users may scale the contraction result with a scale factor alpha,
# and/or add an additional operand c to the contraction result with a scale factor beta

alpha, beta = 1.3, 0.7
c = torch.rand(4, 4, 8, 8, device="cuda")

# result[i,j,m,n] = \sum_{k,l} alpha * a[i,j,k,l] * b[k,l,m,n] + beta * c[i,j,m,n]
# when c is specified for binary contraction, beta must be set
result = nvmath.tensor.binary_contraction("ijkl,klmn->ijmn", a, b, c=c, alpha=alpha, beta=beta)

assert torch.allclose(result, alpha * torch.einsum("ijkl,klmn->ijmn", a, b) + beta * c)

print(f"Input type = {type(a), type(b), type(c)}, contraction result type = {type(result)}")
