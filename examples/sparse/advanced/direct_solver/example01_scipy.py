# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic direct solution of a sparse linear system.

nvmath-python supports multiple frameworks, memory spaces, and execution spaces. For the
LHS, CSR format from SciPy, CuPy, and PyTorch is supported, while the RHS is a dense
ndarray or tensor from the corresponding package NumPy, CuPy, or PyTorch. The result
is of the same type and in the same memory space as the RHS, ensuring effortless
interoperability.
"""

import numpy as np
import scipy.sparse as sp

import nvmath

# The number of equations.
n = 8

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix.
a = sp.random_array((n, n), density=0.5, format="csr", dtype="float64")
a += sp.diags_array([2.0] * n, format="csr", dtype="float64")

# Create the RHS, which can be a matrix or vector in column-major layout.
b = np.ones((n, 2), order="F")

# Solve a @ x = b for x.
x = nvmath.sparse.advanced.direct_solver(a, b)
print(x)

# No synchronization is needed for CPU tensors, since the execution always blocks.

# Check if the result is numpy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
assert isinstance(x, np.ndarray)
