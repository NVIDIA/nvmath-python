# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of PyTorch GPU arrays using the
generic API.

nvmath-python supports multiple frameworks. The result of each operation is a tensor of the
same framework that was used to pass the inputs. It is also located on the GPU like the
inputs.
"""

import torch

import nvmath

# Prepare sample input data.
m, n, k = 123, 456, 789
device_id = 0
a = torch.rand(m, k, device=device_id)
b = torch.rand(k, n, device=device_id)

# The execution happens on the GPU by default since the operands are on the GPU.

# Perform the multiplication.
result = nvmath.linalg.matmul(a, b)

print(torch.allclose(a @ b, result))
