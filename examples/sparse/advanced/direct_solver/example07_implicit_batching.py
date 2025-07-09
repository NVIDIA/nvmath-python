# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to use batched operands in a sparse direct solver operation.
We distinguish batching into explicit, where the samples in a batch are provided as
a sequence of matrices (or vectors for the RHS), and implicit, where the samples are
inferred from 3D or higher-dimensional tensors for the LHS and RHS. The batching for
the LHS and RHS can be independent - the LHS can be batched explicitly while the RHS
can be batched implicitly and vice-versa.

This example illustrates implicit batching of both the LHS and RHS using PyTorch operands,
since it's currently the only package that supports CSR sparse tensors (>= 3D). In this
example, the operands are on the GPU.
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

# Create the RHS, which can be a matrix or vector in column-major layout.
b = torch.ones(batch, 3, n, device=device_id).permute(0, 2, 1)

# Solve a @ x = b for x.
x = nvmath.sparse.advanced.direct_solver(a, b)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
torch.cuda.default_stream().synchronize()

print(x)

# Check if the result is torch tensor as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
print(f"Inputs were located on devices {a.device} and {b.device} and the result is on {x.device}")
