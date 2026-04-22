# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to implement a custom semiring to replace the customary `add`
and `multiply` operations in the dot product that comprises each element of the
result in a matrix multiplication: `c[i,j] = c[i,j] + a[i,k] * b[k,j]`.

Here we use the `max+` semiring (also called tropical semiring).
"""

import logging

import numba
import numba.cuda
import torch
from numba.cuda.np.numpy_support import farray

import nvmath

dtype = torch.float64

indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
values = torch.tensor([2.0, 4.0], dtype=dtype)
shape = 2, 2
a = torch.sparse_coo_tensor(indices, values, shape, device=0).coalesce()
a = a.to_sparse_csr()

# Create a UST from a torch CSR matrix.
a = nvmath.sparse.ust.Tensor.from_package(a)
print(a)

# Dense 'b' and 'c'.
b = torch.ones(*shape, dtype=dtype, device=0)
b = nvmath.sparse.ust.Tensor.from_package(b)
print(b)
c = torch.zeros(*shape, dtype=dtype, device=0) + 3.0
c = nvmath.sparse.ust.Tensor.from_package(c)


# Define the "tropical" semiring.
def plus(a, b):
    """The `mul` semiring operation."""
    return a + b


def maximum(a, b):
    """The `add` semiring operation."""
    return max(a, b)


# The codegen path also needs the atomic version of the `add` operation.
def atomic_maximum(a, b):
    """The atomic version of the `add` semiring operation."""
    a_array = farray(a, (1,), dtype=numba.float64)
    return numba.cuda.atomic.max(a_array, 0, b)


# Compile the semiring operations to LTO-IR.
mul = nvmath.sparse.compile_matmul_mul(plus, dtype="float64")
add = nvmath.sparse.compile_matmul_add(maximum, dtype="float64")
atomic_add = nvmath.sparse.compile_matmul_atomic_add(atomic_maximum, dtype="float64")

# Turn on logging.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# c := a @ b + c
with nvmath.sparse.Matmul(a, b, c, beta=1.0, options={"codegen": False}) as mm:
    # Plan the SpMM operation with the semiring operations.
    mm.plan(semiring={"mul": mul, "add": add, "atomic_add": atomic_add})

    # Execute the SpMM using the custom add and multiply.
    r = mm.execute()

print(f"max+ result (ust view) = \n{r}")
print(f"max+ result (torch view) = \n{r.to_package()}")
