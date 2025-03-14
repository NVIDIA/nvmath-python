# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful matrix multiplication objects. Stateful objects
amortize the cost of preparation across multiple executions.

The inputs as well as the result are PyTorch tensors on the CPU.
"""

import torch

import nvmath

# Prepare sample input data
m, n, k = 123, 456, 789
a = torch.rand(m, k)
b = torch.rand(k, n)

# Use the stateful object as a context manager to automatically release resources.
with nvmath.linalg.advanced.Matmul(a, b) as mm:
    # Plan the matrix multiplication. Planning returns a sequence of algorithms that can be
    # configured, as we'll see in a later example.
    mm.plan()

    # Execute the matrix multiplication.
    result = mm.execute()

    # No synchronization is needed for CPU tensors, since the execution always blocks.

    print(f"Input types = {type(a), type(b)}, device = {a.device, b.device}")
    print(f"Result type = {type(result)}, device = {result.device}")
