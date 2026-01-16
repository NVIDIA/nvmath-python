# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to perform an in-place matrix multiplication, where
the result overwrites operand `c`:

    c := alpha a @ b + beta c

Note that operand `c` cannot be broadcast (in the batch dimensions as well as the
N dimension) when the inplace option is used, since it must be large enough to
hold the result of the computation.
"""

import numpy as np

import nvmath

# Prepare sample input data.
batch, m, n, k = 3, 123, 456, 789
a = np.random.rand(batch, m, k).astype(np.float32)
b = np.random.rand(batch, k, n).astype(np.float32)
c = np.random.rand(batch, m, n).astype(np.float32)
beta = 1.0

# Specify that the operation should be performed in-place.
options = {"inplace": True}
result = nvmath.linalg.advanced.matmul(a, b, c=c, beta=beta, options=options)

assert result is c, "Error: the operation is not in-place."

# No synchronization is needed for CPU tensors, since the execution always blocks.

# Check if the result is NumPy ndarray as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(result)}.")
print(f"Inputs were of data types {a.dtype} and {b.dtype} and the result is of data type {result.dtype}.")
assert isinstance(result, np.ndarray)
