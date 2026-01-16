# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to perform an in-place matrix multiplication with
the BIAS epilog, where the result overwrites operand `c`:

    c := alpha a @ b + beta c + bias

Note that operand `c` cannot be broadcast (in the batch dimensions as well as the
N dimension) when the inplace option is used, since it must be large enough to
hold the result of the computation.

Also recall that not all operand layouts are supported with epilogs. The user must
choose a supported layout for the selected epilog to avoid an "unsupported" exception.
Refer to the cuBLASLt documentation: https://docs.nvidia.com/cuda/cublas/#cublasltmatmul.
"""

import logging

import cupy as cp

import nvmath

# Turn on logging to see what's happening.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Prepare sample input data.
batch, m, n, k = 3, 123, 456, 789
a = cp.random.rand(batch, m, k)
b = cp.random.rand(batch, k, n)

# The result (also `c` since the operation is in-place) layout for each matrix in the
# batch must be COL when used with the BIAS epilog.
c = cp.random.rand(batch, n, m).transpose(0, 2, 1)

bias = cp.random.rand(m, 1)

options = {"inplace": True}
# Use the stateful object as a context manager to automatically release resources.
with nvmath.linalg.advanced.Matmul(a, b, c=c, beta=1.0, options=options) as mm:
    # Plan the matrix multiplication for the BIAS epilog.
    epilog = nvmath.linalg.advanced.MatmulEpilog.BIAS
    mm.plan(epilog=epilog, epilog_inputs={"bias": bias})

    # Execute the matrix multiplication.
    result = mm.execute()
    assert result is c, "Error: the operation is not in-place."

    # Synchronize the default stream, since by default the execution is non-blocking for GPU
    # operands.
    cp.cuda.get_current_stream().synchronize()
