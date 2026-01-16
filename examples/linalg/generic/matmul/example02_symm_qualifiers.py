# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify the operand structure (triangular, symmetric, ...)
to a generic matrix multiplication operation.

In this example, we will multiply a symmetric CuPy ndarray with a general one. The result
is also a general CuPy ndarray.
"""

import cupy as cp
import numpy as np

import nvmath

# Prepare sample input data.
n, k = 123, 789
a = cp.random.rand(k, k).astype(cp.float32)
b = cp.random.rand(k, n).astype(cp.float32)

# We can use structured matrices as inputs by providing the corresponding qualifier which
# describes the matrix. By default, all inputs are assumed to be general matrices.
# MatrixQualifiers are provided as a NumPy ndarray of custom NumPy dtype,
# nvmath.linalg.matrix_qualifiers_dtype.
qualifiers = np.full((2,), nvmath.linalg.GeneralMatrixQualifier.create(), dtype=nvmath.linalg.matrix_qualifiers_dtype)
qualifiers[0] = nvmath.linalg.SymmetricMatrixQualifier.create(uplo=nvmath.linalg.FillMode.LOWER)

result = nvmath.linalg.matmul(a, b, qualifiers=qualifiers)

# No synchronization is needed for CPU tensors, since the execution always blocks.

# Create the symmetric matrix from the lower-triangular part of `a`.
s = cp.tril(a, k=-1)
s += s.T + cp.diag(cp.diag(a))
print(cp.allclose(s @ b, result))
