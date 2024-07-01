# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of torch tensors.

nvmath-python supports multiple frameworks. The result of each operation is a tensor of the same
framework that was used to pass the inputs. It is also located on the same device as the inputs.
"""
import torch

import nvmath

# Prepare sample input data.
m, n, k = 123, 456, 789
a = torch.rand(m, k)
b = torch.rand(k, n)

# Perform the multiplication.
print("Running the multiplication on CPU tensors...")
result = nvmath.linalg.advanced.matmul(a, b)

# No synchronization is needed for CPU tensors, since the execution always blocks.

# Check if the result is numpy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(result)}.")
print(f"Inputs were located on devices {a.device} and {b.device} and the result is on {result.device}")

# Now, move the tensors to the GPU and verify that the result is on the GPU as well.
a_gpu = a.cuda()
b_gpu = b.cuda()
print("\nRunning the multiplication on GPU tensors...")
result = nvmath.linalg.advanced.matmul(a_gpu, b_gpu)

# Synchronize the default stream, since by default the execution is non-blocking for GPU operands.
torch.cuda.default_stream().synchronize()

print(f"Inputs were of types {type(a_gpu)} and {type(b_gpu)} and the result is of type {type(result)}.")
print(f"Inputs were located on devices {a_gpu.device} and {b_gpu.device} and the result is on {result.device}")
