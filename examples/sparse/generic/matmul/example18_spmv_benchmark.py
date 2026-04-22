# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the performance of UST and SpMM with code generation for
various sparse formats (COO, CSR, DIA).

We'll use a vector for operand `b` in the SpMM.
"""

import cupy as cp
import cupyx.scipy.sparse as sp
import torch
from cupyx.profiler import benchmark

import nvmath

# The problem details.
dtype_cp, dtype_torch = cp.float32, torch.float32
n = 1024 * 32

# Create the sparse operand: tridiagonal matrix in DIA format along with UST view.
values = cp.array([[0.0] + [-1.0] * (n - 1), [4.0] * n, [1.0] * (n - 1) + [0.0]], dtype=dtype_cp)
offsets = cp.array([-1, 0, 1], dtype=cp.int32)
a_cp_dia = sp.dia_matrix((values, offsets), shape=(n, n))
u_dia = nvmath.sparse.ust.Tensor.from_package(a_cp_dia)

# Convert to cupy and torch COO, and UST view.
a_cp_coo = sp.coo_matrix(a_cp_dia)

# Recall that the COO matrix needs to be coalesced.
a_cp_coo.sum_duplicates()

u_coo = nvmath.sparse.ust.Tensor.from_package(a_cp_coo)

# Create torch COO.
indices = torch.from_dlpack(cp.vstack((a_cp_coo.row, a_cp_coo.col)))
values = torch.from_dlpack(a_cp_coo.data)
a_torch_coo = torch.sparse_coo_tensor(indices, values, a_cp_coo.shape).coalesce()

# Convert to CSR and UST view.
a_cp_csr = sp.csr_matrix(a_cp_dia)
a_torch_csr = a_torch_coo.to_sparse_csr()

u_csr = nvmath.sparse.ust.Tensor.from_package(a_cp_csr)

# Create the cupy and torch dense operands and their corresponding UST views.
b_cp = cp.ones((n,), dtype=dtype_cp)
b_torch = torch.ones(n, dtype=dtype_torch, device=0)
b_u = nvmath.sparse.ust.Tensor.from_package(b_cp)

c_cp = cp.zeros((n,), dtype=dtype_cp)
c_torch = torch.zeros(n, dtype=dtype_torch, device=0)
c_u = nvmath.sparse.ust.Tensor.from_package(c_cp)


# Execution operations.
def matmul_cp(a, b):
    """Implement SpMM using CuPy."""
    return a @ b


def matmul_torch(a, b):
    """Implement SpMM using torch."""
    return torch.mv(a, b)


def matmul_ust(mm):
    """Implement SpMM using UST."""
    return mm.execute()


# Create the SpMM objects encapsulating the computation for each of the 3 formats and plan
# the operation.

# Set the `codegen` option to True to always generate a kernel for UST operands, even if
# dispatch is supported.

mm_coo = nvmath.sparse.Matmul(u_coo, b_u, c_u, options={"codegen": True})
mm_coo.plan()

mm_csr = nvmath.sparse.Matmul(u_csr, b_u, c_u, options={"codegen": True})
mm_csr.plan()

mm_dia = nvmath.sparse.Matmul(u_dia, b_u, c_u, options={"codegen": True})
mm_dia.plan()

# Benchmark the SpMM using CuPy.
p = benchmark(matmul_cp, (a_cp_coo, b_cp), n_repeat=10)
print(f"CuPy COO    >>> {p}")
p = benchmark(matmul_cp, (a_cp_csr, b_cp), n_repeat=10)
print(f"CuPy CSR    >>> {p}")
p = benchmark(matmul_cp, (a_cp_dia, b_cp), n_repeat=10)
print(f"CuPy DIA    >>> {p}")

# Benchmark the SpMM using PyTorch.
p = benchmark(matmul_torch, (a_torch_coo, b_torch), n_repeat=10)
print(f"PyTorch COO >>> {p}")
p = benchmark(matmul_torch, (a_torch_csr, b_torch), n_repeat=10)
print(f"PyTorch CSR >>> {p}")

# Benchmark the fused SpMM using UST.
p = benchmark(matmul_ust, (mm_coo,), n_repeat=10)
print(f"UST COO     >>> {p}")
p = benchmark(matmul_ust, (mm_csr,), n_repeat=10)
print(f"UST CSR     >>> {p}")
p = benchmark(matmul_ust, (mm_dia,), n_repeat=10)
print(f"UST DIA     >>> {p}")

# Free stateful objects.
mm_coo.free()
mm_csr.free()
mm_dia.free()
