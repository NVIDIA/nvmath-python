# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates usage of epilogs.

Epilogs allow you to execute extra computations after the matrix multiplication in a single fused kernel.
In this example we'll use the BGRADB epilog, which generates an extra output "bgradb" corresponding to the
reduction of the B matrix.
"""

import cupy as cp

import nvmath

# Prepare sample input data.
m, n, k = 64, 128, 256
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)

# Perform the multiplication with BGRADB epilog.
# The auxiliary output "auxiliary" is a dict containing the bias gradient with the key "bgradb".
epilog = nvmath.linalg.advanced.MatmulEpilog.BGRADB
result, auxiliary = nvmath.linalg.advanced.matmul(a, b, epilog=epilog)

# Synchronize the default stream, since by default the execution is non-blocking for GPU operands.
cp.cuda.get_current_stream().synchronize()
print(
    f"Inputs were of types {type(a)} and {type(b)}, and the result type is {type(result)}, and the auxiliary output is of type {type(auxiliary)}."
)
