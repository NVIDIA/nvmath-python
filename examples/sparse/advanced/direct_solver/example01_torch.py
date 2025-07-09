# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic direct solution of a sparse linear system using PyTorch
operands with float64 dtype on the GPU.

nvmath-python supports multiple frameworks, memory spaces, and execution spaces. For the
LHS, CSR format from SciPy, CuPy, and PyTorch is supported, while the RHS is a dense
ndarray or tensor from the corresponding package NumPy, CuPy, or PyTorch. The result
is of the same type and in the same memory space as the RHS, ensuring effortless
interoperability.
"""

import torch

import nvmath

# The number of equations.
n = 8
device_id = 0

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix.
a = torch.rand(n, n) + torch.diag(torch.tensor([10] * n))
a = a.to_sparse_csr()
# Note that torch uses int64 for index buffers, whereas cuDSS currently requires int32.
a = torch.sparse_csr_tensor(
    a.crow_indices().to(dtype=torch.int32), a.col_indices().to(dtype=torch.int32), a.values(), size=a.size(), device=device_id
)

# Create the RHS, which can be a matrix or vector in column-major layout.
b = torch.ones(2, n, device=device_id).T

# Solve a @ x = b for x.
x = nvmath.sparse.advanced.direct_solver(a, b)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
torch.cuda.default_stream().synchronize()

print(x)

# Check if the result is torch tensor as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
print(f"Inputs were located on devices {a.device} and {b.device} and the result is on {x.device}")
