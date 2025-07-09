# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify the threading library, which can be used to
speed up reordering in plan() as well as factorize() and solve() steps for hybrid
execution. It also shows how you can provide your own cuDSS library handle.

You can provide your own threading library implementation that follows the API
specification described in
https://docs.nvidia.com/cuda/cudss/advanced_features.html#threading-layer-api-in-cudss
or use the prebuilt library provided for Linux based on GNU OpenMP libcudss_mtlayer_gomp.so.
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
b = cp.ones((n,))

# Create a cuDSS library handle, in case you want to share it across multiple sparse
# direct solver operations.
h = nvmath.bindings.cudss.create()
o = {"handle": h}

# Specify the threading library. Here we use the prebuilt library libcudss_mtlayer_gomp.so
# whose location depends on the installation method. For example, if nvmath-python was
# installed using `pip`, the location can be obtained using `pip show nvidia-cudss-cu12`
# and appending `nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0` to the location shown.
# multithreading_lib = "/path/to/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0"
# o = nvmath.sparse.advanced.DirectSolverOptions(
#        multithreading_lib=multithreading_lib, handle=h
# )
x = nvmath.sparse.advanced.direct_solver(a, b, options=o, execution="hybrid")

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()

print(x)

# Destroy the cudss handle.
nvmath.bindings.cudss.destroy(h)

# Check if the result is numpy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
assert isinstance(x, cp.ndarray)
