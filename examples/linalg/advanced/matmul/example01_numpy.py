# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of numpy arrays.

nvmath-python supports multiple frameworks. The result of each operation is a tensor of the
same framework that was used to pass the inputs. It is also located on the CPU like the
inputs.
"""

import numpy as np

import nvmath

# Prepare sample input data.
m, n, k = 123, 456, 789
a = np.random.rand(m, k)
b = np.random.rand(k, n)

# Perform the multiplication.
result = nvmath.linalg.advanced.matmul(a, b)

# No synchronization is needed for CPU tensors, since the execution always blocks.

# Check if the result is numpy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(result)}.")
assert isinstance(result, np.ndarray)
