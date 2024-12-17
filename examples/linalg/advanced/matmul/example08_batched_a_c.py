# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates nvmath's capability to execute batched multiplications.

Executing multiple multiplications together (in a batch) yields better performance than
executing them separately. nvmath supports broadcasting, so if one of the inputs is batched
and the other one is not, it will be broadcasted to match the batch size.

In this example we will multiply each of our `a` matrices with the same `b` matrix, and add
each of the `c` (m, 1) matrices to the result. Since each `c` is a column matrix, it's also
broadcast across the columns of each result in the batch.
"""

import cupy as cp

import nvmath

# Prepare sample input data.
batch_size = 32
m = n = k = 2000
a_batch = cp.random.rand(batch_size, m, k)
b = cp.random.rand(k, n)
c = cp.random.rand(batch_size, m, 1)
beta = 1.2
print(f"a shape is: {a_batch.shape}, b shape is: {b.shape} and c shape is: {c.shape}")

# Execute the multiplication.
result = nvmath.linalg.advanced.matmul(a_batch, b, c, beta=beta)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()
print(f"Input types = {type(a_batch), type(b)}, device = {a_batch.device, b.device}")
print(f"Result type = {type(result)}, device = {result.device}")
