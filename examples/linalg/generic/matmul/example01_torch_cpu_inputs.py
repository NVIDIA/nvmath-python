# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of PyTorch CPU arrays using the
generic API.

nvmath-python supports multiple frameworks. The result of each operation is a tensor of the
same framework that was used to pass the inputs. It is also located on the CPU like the
inputs.
"""

import torch

import nvmath

# Prepare sample input data.
m, n, k = 123, 456, 789
a = torch.rand(m, k)
b = torch.rand(k, n)

# We can choose the execution space for the matrix multiplication using ExecutionCUDA or
# ExecutionCPU. By default, the execution space matches the operands, so in order to execute
# a matrix multiplication on NumPy arrays using CUDA we need to specify ExecutionCUDA.
# Tip: use help(nvmath.linalg.ExecutionCUDA) to see available options.
execution = nvmath.linalg.ExecutionCUDA()

# Perform the multiplication.
result = nvmath.linalg.matmul(a, b, execution=execution)

# Alternatively, the execution space can be specified as a string "cuda", which is
# identical to providing a default-constructed ExecutionCUDA() object.
result = nvmath.linalg.matmul(a, b, execution="cuda")

print(torch.allclose(a @ b, result))
