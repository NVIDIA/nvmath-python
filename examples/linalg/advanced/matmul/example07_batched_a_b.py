# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates nvmath-python's capability to execute batched multiplications.

Executing multiple multiplications together (in a batch) yields better performance than executing
them separately.

In this example we will multiply each of our `a` matrices with the corresponding `b` matrix.
"""
import cupy as cp

import nvmath

# Enable logging.
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Prepare sample input data.
batch_size = 32
m = n = k = 2000
a_batch = cp.random.rand(batch_size, m, k)
b_batch = cp.random.rand(batch_size, k, n)
print(f"a shape is: {a_batch.shape}, b shape is: {b_batch.shape}")

# Execute the multiplication.
result = nvmath.linalg.advanced.matmul(a_batch, b_batch)

# Synchronize the default stream, since by default the execution is non-blocking for GPU operands.
cp.cuda.get_current_stream().synchronize()
print(f"Input types = {type(a_batch), type(b_batch)}, device = {a_batch.device, b_batch.device}")
print(f"Result type = {type(result)}, device = {result.device}")
