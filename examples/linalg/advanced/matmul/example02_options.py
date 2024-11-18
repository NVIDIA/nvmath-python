# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify options to a matrix multiplication operation.

In this example, we will use NumPy ndarrays as input, and we will look at two equivalent ways to specify the compute type.
"""

import numpy as np

import nvmath

# Prepare sample input data.
m, n, k = 123, 456, 789
a = np.random.rand(m, k).astype(np.float32)
b = np.random.rand(k, n).astype(np.float32)

# Here we'd like to use COMPUTE_32F_FAST_TF32 for the compute type, and we show two alternatives for doing so.
# Tip: use help(nvmath.linalg.advanced.MatmulComputeType) to see available compute types.
compute_type = nvmath.linalg.advanced.MatmulComputeType.COMPUTE_32F_FAST_TF32

# Alternative #1 for specifying options, using a dataclass.
# Tip: use help(nvmath.linalg.advanced.MatmulOptions) to see available options.
options = nvmath.linalg.advanced.MatmulOptions(compute_type=compute_type)
result = nvmath.linalg.advanced.matmul(a, b, options=options)

# Alternative #2 for specifying options, using dict. The two alternatives are entirely equivalent.
result = nvmath.linalg.advanced.matmul(a, b, options={"compute_type": compute_type})

# No synchronization is needed for CPU tensors, since the execution always blocks.

# Check if the result is numpy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(result)}.")
print(f"Inputs were of data types {a.dtype} and {b.dtype} and the result is of data type {result.dtype}.")
assert isinstance(result, np.ndarray)
