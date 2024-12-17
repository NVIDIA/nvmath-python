# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates usage of epilogs.

Epilogs allow you to execute extra computations after the matrix multiplication in a single
fused kernel. In this example we'll use the BGRADA epilog, which generates an extra output
"bgrada" corresponding to the reduction of the A matrix.
"""

import cupy as cp

import nvmath

# Prepare sample input data.
m, n, k = 64, 128, 256
a = cp.random.rand(k, m).T  # Currently, it's required that 'a' is in column-major layout.
b = cp.random.rand(k, n)

# Perform the multiplication with BGRADA epilog. The auxiliary output "auxiliary" is a dict
# containing the bias gradient with the key "bgrada".
epilog = nvmath.linalg.advanced.MatmulEpilog.BGRADA
result, auxiliary = nvmath.linalg.advanced.matmul(a, b, epilog=epilog)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()
print(
    f"Inputs were of types {type(a)} and {type(b)}, and the result type is {type(result)}, "
    f"and the auxiliary output is of type {type(auxiliary)}."
)
