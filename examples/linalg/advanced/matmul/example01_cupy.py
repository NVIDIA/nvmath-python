# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of CuPy arrays.

nvmath-python supports multiple frameworks. The result of each operation is a tensor of the same
framework that was used to pass the inputs. It is also located on the same device as the inputs.
"""

import cupy as cp

import nvmath

# Prepare sample input data.
n, m, k = 123, 456, 789
a = cp.random.rand(n, k)
b = cp.random.rand(k, m)

# Perform the multiplication.
result = nvmath.linalg.advanced.matmul(a, b)

# Synchronize the default stream, since by default the execution is non-blocking for GPU operands.
cp.cuda.get_current_stream().synchronize()

# Check if the result is cupy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(result)}.")
assert isinstance(result, cp.ndarray)
