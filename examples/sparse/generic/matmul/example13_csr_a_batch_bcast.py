# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates batching, where batch dimensions are specified for
`b` and `c` while `a` is broadcast in the batch dimensions.

The operands are UST on the GPU backed by torch dense tensors.
"""

import torch

import nvmath

# The problem parameters.
device_id = 0
dtype = torch.float64
m, n, k = 4, 2, 6

# The batch dimensions - multiple batch dimensions are supported.
batch = 7, 6

# Create a torch CSR tensor.
a = torch.rand(m, k, dtype=dtype, device=device_id)
a = a.to_sparse_csr()
a = nvmath.sparse.ust.Tensor.from_package(a)
print(f"The shape of a = {a.shape}.")

# Dense 'b' and 'c'.
b = torch.ones(*batch, k, n, dtype=dtype, device=device_id)
b = nvmath.sparse.ust.Tensor.from_package(b)
print(f"The shape of b = {b.shape}.")
# Note that since the SpMM is in-place (`c` is overwritten), `c` cannot be
# broadcast.
c = torch.ones(*batch, m, n, dtype=dtype, device=device_id)
c = nvmath.sparse.ust.Tensor.from_package(c)
print(f"The shape of c = {c.shape}.")

# Create a stateful object that specifies the operation c := alpha a @ b + beta c
alpha, beta = 1.2, 2.4
with nvmath.sparse.Matmul(a, b, c, alpha=alpha, beta=beta) as mm:
    # Plan the SpMM operation. As we will see in later examples, planning
    # can be customized by providing prologs, epilog, and semiring operations.
    # The planning needs to be done only once for each problem specification.
    mm.plan()

    # Execute the operation.
    r = mm.execute()

    # The result `r` is `c` since the operation is in-place.
    assert r is c, "Error: the operation is not in-place."
