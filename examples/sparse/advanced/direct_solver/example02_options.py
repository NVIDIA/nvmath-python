# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify options to a sparse direct solver operation.

In this example, we will use SciPy sparse CSR and NumPy ndarrays as input, and we will
look at two equivalent ways to specify that the system is symmetric postive-definite.
"""

import numpy as np
import scipy.sparse as sp

import nvmath

# The number of equations.
n = 8

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix.
a = sp.random_array((n, n), density=0.25, format="csr", dtype="float64")
a += a.T + sp.diags_array([2.0] * n, format="csr", dtype="float64")

# Create the RHS, which can be a matrix or vector in column-major layout.
b = np.ones((n, 2), order="F")

# Here we'd like to specify that the LHS is symmetric and positive-definite, and we show two
# alternatives for doing so. Tip: use help(nvmath.sparse.advanced.DirectSolverMatrixType) to
# the available LHS types.
sparse_system_type = nvmath.sparse.advanced.DirectSolverMatrixType.SPD

# Alternative #1 for specifying options, using a dataclass.
# Tip: use help(nvmath.sparse.advanced.DirectSolverOptions) to see available options.
options = nvmath.sparse.advanced.DirectSolverOptions(sparse_system_type=sparse_system_type)
x = nvmath.sparse.advanced.direct_solver(a, b, options=options)

# Alternative #2 for specifying options, using dict. The two alternatives are entirely
# equivalent.
x = nvmath.sparse.advanced.direct_solver(a, b, options={"sparse_system_type": sparse_system_type})

# No synchronization is needed for CPU tensors, since the execution always blocks.

print(x)

# Check if the result is numpy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
assert isinstance(x, np.ndarray)
