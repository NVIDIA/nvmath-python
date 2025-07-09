# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic direct solution of a sparse linear system using CuPy
operands with complex64 dtype on the GPU.

nvmath-python supports multiple frameworks, memory spaces, and execution spaces. For the
LHS, CSR format from SciPy, CuPy, and PyTorch is supported, while the RHS is a dense
ndarray or tensor from the corresponding package NumPy, CuPy, or PyTorch. The result
is of the same type and in the same memory space as the RHS, ensuring effortless
interoperability.
"""

import cupy as cp
import cupyx.scipy.sparse as sp

import nvmath

# The number of equations.
n = 8

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix of complex64 values.
a = sp.random(n, n, density=0.5, format="csr", dtype="float32").astype("complex64")
a += sp.diags([2.0] * n, format="csr", dtype="float32")

# Create the RHS, which can be a matrix or vector in column-major layout.
b = cp.ones((2, n), dtype=cp.float32) + 1j
b = b.T

# Solve a @ x = b for x.
x = nvmath.sparse.advanced.direct_solver(a, b)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()

print(x)

# Check if the result is cupy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
assert isinstance(x, cp.ndarray)
