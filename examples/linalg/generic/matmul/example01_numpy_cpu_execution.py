# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of NumPy arrays using the generic API.

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

# The execution happens on the CPU by default since the operands are on the CPU.

# Perform the multiplication.
result = nvmath.linalg.matmul(a, b)

print(np.allclose(a @ b, result))
