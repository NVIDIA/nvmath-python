# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify the operand structure (triangular, symmetric, ...)
to a generic matrix multiplication operation.

In this example, we will multiply a general NumPy ndarray with a diagonal one. The
result is also a general NumPy ndarray.
"""

import numpy as np

import nvmath

# Prepare sample input data.
m, k = 123, 789
# Transpose and conjugate operations are not supported for the diagonal matmul API, so we
# must provide a column-order matrix.
a = np.random.rand(m, k).astype(np.float32, order="F")
# The matmul function accepts diagonal matrices as a vector. To extract the main diagonal
# from an existing NumPy array, see np.diag() or np.diagonal().
b = np.random.rand(k).astype(np.float32)

# We can use structured matrices as inputs by providing the corresponding qualifier which
# describes the matrix. By default, all inputs are assumed to be general matrices.
# MatrixQualifiers are provided as a NumPy ndarray of custom NumPy dtype,
# nvmath.linalg.matrix_qualifiers_dtype.
qualifiers = np.full((2,), nvmath.linalg.GeneralMatrixQualifier.create(), dtype=nvmath.linalg.matrix_qualifiers_dtype)
qualifiers[1] = nvmath.linalg.DiagonalMatrixQualifier.create()

result = nvmath.linalg.matmul(a, b, execution="cuda", qualifiers=qualifiers)

# No synchronization is needed for CPU tensors, since the execution always blocks.
print(np.allclose(a @ np.diag(b), result))
