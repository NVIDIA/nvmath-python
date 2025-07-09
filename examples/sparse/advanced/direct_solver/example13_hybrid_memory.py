# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to use the hybrid (CPU-GPU) memory mode, which can be
used when the L and U factors don't fit into device memory, as in the case of large
systems of equations.
"""

import nvmath

import cupy as cp
import cupyx.scipy.sparse as sp

# The number of equations.
n = 8

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix.
a = sp.random(n, n, density=0.5, format="csr", dtype="float64")
a += sp.diags([2.0] * n, format="csr", dtype="float64")

# Create the RHS, which can be a matrix or vector in column-major layout.
b = cp.ones((n, 2), order="F")

# Specify hybrid memory mode using the `hybrid_memory_mode_options` for CUDA execution.
x = nvmath.sparse.advanced.direct_solver(
    a, b, execution={"name": "cuda", "hybrid_memory_mode_options": {"hybrid_memory_mode": True}}
)

# Alternatively, hybrid memory mode can be specified in the ExecutionCUDA object.
hmmo = nvmath.sparse.advanced.HybridMemoryModeOptions(hybrid_memory_mode=True)
e = nvmath.sparse.advanced.ExecutionCUDA(hybrid_memory_mode_options=hmmo)
x = nvmath.sparse.advanced.direct_solver(a, b, execution=e)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()

print(x)

# Check if the result is numpy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
assert isinstance(x, cp.ndarray)
