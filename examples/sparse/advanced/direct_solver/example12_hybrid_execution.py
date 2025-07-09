# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to use the hybrid (CPU-GPU) execution mode, which can be
beneficial to speedup phases of the execution that are not amenable to parallelization, or
for factorizing and solving small matrices.
"""

import nvmath

import numpy as np
import scipy.sparse as sp

# The number of equations.
n = 8

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix.
a = sp.random_array((n, n), density=0.5, format="csr", dtype="float64")
a += sp.diags_array([2.0] * n, format="csr", dtype="float64")

# Create the RHS, which can be a matrix or vector in column-major layout.
b = np.ones((n,))

# Specify hybrid execution mode using the execution option. The operands
# can be in the CPU or GPU memory space.
x = nvmath.sparse.advanced.direct_solver(a, b, execution="hybrid")

# Alternatively, hybrid execution can be specified as an ExecutionHybrid object.
e = nvmath.sparse.advanced.ExecutionHybrid(num_threads=1)
x = nvmath.sparse.advanced.direct_solver(a, b, execution=e)

print(x)

# No synchronization is needed for CPU tensors, since the execution always blocks.

# Check if the result is numpy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
assert isinstance(x, np.ndarray)
