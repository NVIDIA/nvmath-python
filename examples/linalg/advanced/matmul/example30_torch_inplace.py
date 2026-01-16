# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to perform an in-place matrix multiplication, where
the result overwrites operand `c`:

    c := alpha a @ b + beta c

Note that operand `c` cannot be broadcast (in the batch dimensions as well as the
N dimension) when the inplace option is used, since it must be large enough to
hold the result of the computation.
"""

import torch

import nvmath

# Prepare sample input data.
batch, m, n, k = 3, 123, 456, 789
a = torch.rand(batch, m, k, device="cuda")
b = torch.rand(batch, k, n, device="cuda")
c = torch.rand(batch, m, n, device="cuda")
beta = 1.0

# Specify that the operation should be performed in-place.
options = {"inplace": True}
result = nvmath.linalg.advanced.matmul(a, b, c=c, beta=beta, options=options)

assert result is c, "Error: the operation is not in-place."

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
torch.cuda.default_stream().synchronize()

# Check if the result is torch tensor as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(result)}.")
print(f"Inputs were of data types {a.dtype} and {b.dtype} and the result is of data type {result.dtype}.")
