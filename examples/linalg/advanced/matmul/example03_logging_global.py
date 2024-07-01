# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to turn on logging using the global logger.
"""
import cupy as cp

import nvmath

# Turn on logging. Here we use the global logger, set the level to "debug", and use a custom format for the log.
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Prepare sample input data.
m, n, k = 64, 128, 256
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)
alpha = 0.45

# Perform the GEMM.
result = nvmath.linalg.advanced.matmul(a, b, alpha=alpha)

# Synchronize the default stream, since by default the execution is non-blocking for GPU operands.
cp.cuda.get_current_stream().synchronize()
