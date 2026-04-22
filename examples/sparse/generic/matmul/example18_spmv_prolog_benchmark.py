# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the performance of UST and SpMM with code generation for
various sparse formats (COO, CSR, DIA) for c := 2 * a @ 3 * b.

We'll use a vector for operand `b` in the SpMM.
"""

import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.profiler import benchmark

import nvmath

# The problem details.
dtype = cp.float32
n = 1024 * 32

# Create the sparse operand: tridiagonal matrix in DIA format along with UST view.
values = cp.array([[0.0] + [-1.0] * (n - 1), [4.0] * n, [1.0] * (n - 1) + [0.0]], dtype=dtype)
offsets = cp.array([-1, 0, 1], dtype=cp.int32)
a_dia = sp.dia_matrix((values, offsets), shape=(n, n))
u_dia = nvmath.sparse.ust.Tensor.from_package(a_dia)

# Convert to COO and UST view.
a_coo = sp.coo_matrix(a_dia)

# Recall that the COO matrix needs to be coalesced.
a_coo.sum_duplicates()

u_coo = nvmath.sparse.ust.Tensor.from_package(a_coo)

# Convert to CSR and UST view.
a_csr = sp.csr_matrix(a_dia)
u_csr = nvmath.sparse.ust.Tensor.from_package(a_csr)

# Create the dense operands and their corresponding UST views.
b = cp.ones((n,), dtype=dtype)
b_u = nvmath.sparse.ust.Tensor.from_package(b)

c = cp.zeros((n,), dtype=dtype)
c_u = nvmath.sparse.ust.Tensor.from_package(c)


# Prolog functions.
def double(a):
    return 2 * a


def triple(a):
    return 3 * a


# Compile the prolog to LTO-IR using the helpers (or your own compiler).
prolog_a = nvmath.sparse.compile_matmul_prolog(double, operand_label="a", dtype="float32")
prolog_b = nvmath.sparse.compile_matmul_prolog(triple, operand_label="b", dtype="float32")


# Execution operations.
def matmul_cp(a, b):
    """Implement 2 * a @ 3 * b using CuPy."""
    return double(a) @ triple(b)


def matmul_ust(mm):
    """Implement 2 * a @ 3 * b using UST and SpMM."""
    return mm.execute()


# Create the SpMM objects encapsulating the computation for each of the 3 formats and plan
# the operation.

# Set the `codegen` option to True to always generate a kernel for UST operands, even if
# dispatch is supported.

mm_coo = nvmath.sparse.Matmul(u_coo, b_u, c_u, options={"codegen": True})
mm_coo.plan(prologs={"a": prolog_a, "b": prolog_b})

mm_csr = nvmath.sparse.Matmul(u_csr, b_u, c_u, options={"codegen": True})
mm_csr.plan(prologs={"a": prolog_a, "b": prolog_b})

mm_dia = nvmath.sparse.Matmul(u_dia, b_u, c_u, options={"codegen": True})
mm_dia.plan(prologs={"a": prolog_a, "b": prolog_b})

# Benchmark the SpMM using CuPy.
p = benchmark(matmul_cp, (a_coo, b), n_repeat=10)
print(f"CuPy COO >>> {p}")
p = benchmark(matmul_cp, (a_csr, b), n_repeat=10)
print(f"CuPy CSR >>> {p}")
p = benchmark(matmul_cp, (a_dia, b), n_repeat=10)
print(f"CuPy DIA >>> {p}")

# Benchmark the fused SpMM using UST.
p = benchmark(matmul_ust, (mm_coo,), n_repeat=10)
print(f"UST COO  >>> {p}")
p = benchmark(matmul_ust, (mm_csr,), n_repeat=10)
print(f"UST CSR  >>> {p}")
p = benchmark(matmul_ust, (mm_dia,), n_repeat=10)
print(f"UST DIA  >>> {p}")

# Free stateful objects.
mm_coo.free()
mm_csr.free()
mm_dia.free()
