# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates the use of the stateful SpMM APIs. Stateful objects
amortize the cost of preparation across multiple executions, and are especially
important for sparse operations where preparatory steps can be expensive.

The operands are UST on the CPU backed by dense torch tensors.
"""

import logging

import torch

import nvmath

# The problem parameters.
device_id = "cpu"
dtype = torch.float64
m, n, k = 4, 2, 6

# Create a torch CSR tensor. We'll mask a dense random tensor to target 50%
# NNZ density and convert it to sparse CSR.
a = torch.rand(m, k, dtype=dtype, device=device_id)
a = a * (a > 0.5)
a = a.to_sparse_csr()

# View the sparse operand as UST.
a = nvmath.sparse.ust.Tensor.from_package(a)
print(f"a = \n{a}")

# Dense 'b' and 'c' UST backed by torch.
b = torch.ones(k, n, dtype=dtype, device=device_id)
b = nvmath.sparse.ust.Tensor.from_package(b)
print(f"b = \n{b}")

c = torch.ones(m, n, dtype=dtype, device=device_id)
c = nvmath.sparse.ust.Tensor.from_package(c)

# Turn on logging to see what's happening under the hood.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

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

print(f"c := {alpha} a @ b + {beta} c =\n{r}")
print(r.to_package())
