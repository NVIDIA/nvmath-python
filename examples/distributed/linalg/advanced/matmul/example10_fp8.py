# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic distributed matrix multiplication of FP8 tensors.

In narrow-precision operations, quantization scales must be provided for each tensor. These
scales are used to dequantize input operands and quantize the result. Without proper
scaling, the results of FP8 operations will likely exceed the type's range.

FP8 is only supported on devices with compute capability 8.9 or higher.

$ mpiexec -n 4 python example10_fp8.py
"""

import numpy as np
import torch
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import Slab
from nvmath.distributed.linalg.advanced import matrix_qualifiers_dtype

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % torch.cuda.device_count()
# cuBLASMp requires NCCL communication backend.
nvmath.distributed.initialize(device_id, comm, backends=["nccl"])

# Prepare sample input data. Note that N, M and K (for local GEMMs) must be divisible
# by 16 for FP8.
m, n, k = 128, 64, 96

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# FP8 operations require TN input layout.

with torch.cuda.device(device_id):
    a = (torch.rand(*row_wise_distribution.shape(rank, (m, k)), device="cuda") * 10).type(torch.float8_e4m3fn)
    b = (torch.rand(*row_wise_distribution.shape(rank, (n, k)), device="cuda") * 10).type(torch.float8_e4m3fn)

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with col_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [col_wise_distribution, col_wise_distribution, row_wise_distribution]

# Prepare quantization scales. The scales must allow the result to fit within the dynamic
# range of the data type used. Scales can be provided either as a dictionary or as a
# MatmulQuantizationScales object. Note that scales are only allowed for FP8 operands.
scales = {"a": 1, "b": 1, "d": 0.1}

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True

# Perform the multiplication. The result of the multiplication will be:
# (scales.a * A) @ (scales.b * B) * scales.d
result = nvmath.distributed.linalg.advanced.matmul(
    a, b, distributions=distributions, quantization_scales=scales, qualifiers=qualifiers
)

# Check how scaling helped to fit into the dynamic range of float8_e4m3fn type.
result_without_scaling = nvmath.distributed.linalg.advanced.matmul(
    a, b, distributions=distributions, quantization_scales={"a": 1, "b": 1, "d": 1}, qualifiers=qualifiers
)

if rank == 0:
    print("Without scaling, most of the elements were clamped to the maximum value of float8_e4m3fn type (448):")
    # Printing the tensor synchronizes on the default CUDA stream.
    print(result_without_scaling)
    print(f"\nWith D scale set to {scales['d']}, they were scaled down to fit into the dynamic range of float8_e4m3fn:")
    print(result)
