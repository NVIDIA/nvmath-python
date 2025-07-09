# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
TODO
"""

import torch

import nvmath

# Prepare sample input data.
n = 8
batch = 2
device_id = 0

# Create a diagonally dominant random CSR matrix.
a = torch.rand(n, n) + torch.diag(torch.tensor([10] * n))
a = torch.stack([a] * batch, dim=0)
a = a.to_sparse_csr()
# Note that torch uses int64 for index buffers, whereas cuDSS currently requires int32.
a = torch.sparse_csr_tensor(
    a.crow_indices().to(dtype=torch.int32), a.col_indices().to(dtype=torch.int32), a.values(), size=a.size(), device=device_id
)

# Create the RHS, which is a sequence of vectors of shape (n, 1). Note that the vector is
# provided as a column-matrix to disambiguate between the batched vector vs a matrix case,
# both of which are logically 2D.
b = torch.ones(batch, 1, n, device=device_id).permute(0, 2, 1)

# Solve a @ x = b for x.
x = nvmath.sparse.advanced.direct_solver(a, b)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
torch.cuda.default_stream().synchronize()

print(x)

# Check if the result is torch tensor as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
print(f"Inputs were located on devices {a.device} and {b.device} and the result is on {x.device}")
