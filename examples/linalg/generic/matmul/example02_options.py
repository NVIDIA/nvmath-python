# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify options to a matrix multiplication operation.

In this example, we will use NumPy ndarrays as input, and we will look at two equivalent
ways to specify the compute type.
"""

import numpy as np

import nvmath

# Prepare sample input data.
m, k = 123, 789
a = np.random.rand(m, k).astype(np.float32)
b = np.tril(np.random.rand(k, k).astype(np.float32))

# We can choose the execution space for the matrix multiplication using ExecutionCUDA or
# ExecutionCPU. By default, the execution space matches the operands, so in order to execute
# a matrix multiplication on NumPy arrays using CUDA we need to specify ExecutionCUDA.
# Tip: use help(nvmath.linalg.generic.ExecutionCUDA) to see available options.
execution = nvmath.linalg.ExecutionCUDA()

# We can use structured matrices as inputs by providing the corresponding qualifier which
# describes the matrix. By default, all inputs are assumed to be general matrices.
# MatrixQualifiers are provided as an array of custom NumPy dtype,
# nvmath.linalg.matrix_qualifiers_dtype.
qualifiers = np.full((2,), nvmath.linalg.GeneralMatrixQualifier.create(), dtype=nvmath.linalg.matrix_qualifiers_dtype)
qualifiers[1] = nvmath.linalg.TriangularMatrixQualifier.create(uplo=nvmath.linalg.FillMode.LOWER)

result = nvmath.linalg.matmul(a, b, execution=execution, qualifiers=qualifiers)

# No synchronization is needed for CPU tensors, since the execution always blocks.

# Check if the result is numpy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(result)}.")
print(f"Inputs were of data types {a.dtype} and {b.dtype} and the result is of data type {result.dtype}.")
assert isinstance(result, np.ndarray)
