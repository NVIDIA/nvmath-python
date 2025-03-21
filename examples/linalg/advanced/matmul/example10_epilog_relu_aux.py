# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates usage of epilogs that generate auxiliary output.

Epilogs allow you to execute extra computations after the matrix multiplication in a single
fused kernel. In this example we'll use the RELU_AUX epilog, which generates an extra output
"relu_aux". We will see in a later example how to use the auxiliary output as input to other
epilogs like DRELU.
"""

import cupy as cp

import nvmath

# Prepare sample input data.
m, n, k = 64, 128, 256
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)

# Perform the multiplication with RELU_AUX epilog.
epilog = nvmath.linalg.advanced.MatmulEpilog.RELU_AUX
result, auxiliary = nvmath.linalg.advanced.matmul(a, b, epilog=epilog)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()
print(
    f"Inputs were of types {type(a)} and {type(b)}, and the result type is {type(result)}, "
    f"and the auxiliary output is of type {type(auxiliary)}."
)
