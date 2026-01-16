# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful ternary tensor contraction objects.
Stateful objects amortize the cost of preparation across multiple executions.

The inputs as well as the result are Torch tensors.
"""

import torch

import nvmath

a = torch.rand(4, 6, 8, device="cuda")
b = torch.rand(6, 8, 3, device="cuda")
c = torch.rand(3, 9, device="cuda")


# result[i,j,m,n] = \sum_{k,l} a[i,j,k] * b[j,k,l] * c[l,n]

# Create a stateful TernaryContraction object 'contraction'.
with nvmath.tensor.TernaryContraction("ijk,jkl,ln->in", a, b, c) as contraction:
    # Plan the Contraction.
    contraction.plan()

    # Execute the Contraction.
    result = contraction.execute()

    # Synchronize the default stream
    torch.cuda.default_stream().synchronize()
    assert torch.allclose(result, torch.einsum("ijk,jkl,ln->in", a, b, c))
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"Contraction output type = {type(result)}, device = {result.device}")
