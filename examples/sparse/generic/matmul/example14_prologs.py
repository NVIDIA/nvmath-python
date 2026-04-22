# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to use prologs for operands that will be fused into the SpMM.
"""

import math

import cupy as cp
import cupyx.scipy.sparse as sp

import nvmath

# The problem data.
n = 4

dtype = cp.float64

# Create a sparse operand in DIA format and view it as UST.
values = cp.array([[0.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [-1.0, -2.0, -3.0, 0.0]], dtype=dtype)
offsets = cp.array([1, 0, -1], dtype=cp.int32)
a = sp.dia_matrix((values, offsets), shape=(n, n))
a = nvmath.sparse.ust.Tensor.from_package(a)
print(f"a = \n{a}")
print(f"In dense form, a = \n{a.to_package().toarray()}")

# Dense 'b' and 'c', also viewed as UST.
b = cp.ones((n, n), dtype=dtype)
b = nvmath.sparse.ust.Tensor.from_package(b)
print(f"b = \n{b}")

c = cp.zeros((n, n), dtype=dtype)
c = nvmath.sparse.ust.Tensor.from_package(c)


# The prolog for `a`.
def transform_a(a):
    return 3.14 * math.sin(a)


# The prolog for `b`.
def transform_b(b):
    return 6.28 * math.cos(b)


# Compile the prologs to LTO-IR using the helpers (or use your own compiler).
prolog_a = nvmath.sparse.compile_matmul_prolog(transform_a, operand_label="a", dtype="float64")
prolog_b = nvmath.sparse.compile_matmul_prolog(transform_b, operand_label="b", dtype="float64")

# c := 3.14 sin(a) @  6.28 cos(b)
with nvmath.sparse.Matmul(a, b, c, beta=1.0) as mm:
    # Plan the SpMM operation with the prologs.
    mm.plan(prologs={"a": prolog_a, "b": prolog_b})

    # Execute the SpMM.
    r = mm.execute()

print(f"c := 3.14 sin(a) @  6.28 cos(b) = {r}")
