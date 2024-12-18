# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates autotuning capability of nvmath-python.

Autotuning will benchmark the algorithms proposed in the planning stage and reorder them
according to their performance.
"""

import cupy as cp

import nvmath

# Enable logging.
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Prepare sample input data.
m, n, k = 2048, 4096, 1024
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)
bias = cp.random.rand(m, 1)

# Use the stateful object as a context manager to automatically release resources.
with nvmath.linalg.advanced.Matmul(a, b) as mm:
    epilog = nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS
    mm.plan(epilog=epilog, epilog_inputs={"bias": bias})

    # Run the autotuning. It will benchmark the algorithms found during planning and reorder
    # them according to their actual performance. See the logs section "autotuning phase" to
    # see what happens under the hood.
    mm.autotune(iterations=5)

    # Execute the multiplication.
    result = mm.execute()

    # Synchronize the default stream, since by default the execution is non-blocking for GPU
    # operands.
    cp.cuda.get_current_stream().synchronize()
    print(f"Input types = {type(a), type(b)}, device = {a.device, b.device}")
    print(f"Result type = {type(result)}, device = {result.device}")
