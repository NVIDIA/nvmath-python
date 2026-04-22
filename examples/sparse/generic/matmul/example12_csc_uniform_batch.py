# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates batching, where batch dimensions are specified for
all of the operands.

The operands are UST on the GPU backed by torch dense tensors.
"""

import torch

import nvmath

# The problem parameters.
device_id = 0
dtype = torch.float64
m, n, k = 4, 2, 6

# The batch dimensions - multiple batch dimensions are supported.
batch = 5, 7

# Create a batched torch CSC tensor.
a = torch.rand(*batch, m, k, dtype=dtype, device=device_id)
a = a.to_sparse_csc()
a = nvmath.sparse.ust.Tensor.from_package(a)
print(f"The shape of a = {a.shape}.")

# Dense 'b' and 'c'.
b = torch.ones(*batch, k, n, dtype=dtype, device=device_id)
b = nvmath.sparse.ust.Tensor.from_package(b)
print(f"The shape of b = {b.shape}.")
c = torch.ones(*batch, m, n, dtype=dtype, device=device_id)
c = nvmath.sparse.ust.Tensor.from_package(c)
print(f"The shape of c = {c.shape}.")

# Create a stateful object that specifies the operation c := a @ b + c
# Use the codegen option since cuSPARSE doesn't currently support
# batched CSC matrices.
with nvmath.sparse.Matmul(a, b, c, options={"codegen": True}) as mm:
    # Plan the SpMM operation. As we will see in later examples, planning
    # can be customized by providing prologs, epilog, and semiring operations.
    # The planning needs to be done only once for each problem specification.
    mm.plan()

    # Execute the operation.
    r = mm.execute()

    # The result `r` is `c` since the operation is in-place.
    assert r is c, "Error: the operation is not in-place."
