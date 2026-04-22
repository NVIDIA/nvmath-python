# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates using RELU epilog with FP8 matrix multiplication.

In FP8 operations, quantization scales must be provided for each tensor. These scales are
used to dequantize input operands and quantize the result. The RELU epilog is applied
after scaling but before final quantization.

FP8 is only supported with compute capability 8.9 or higher.

$ mpiexec -n 4 python example16_fp8_epilog.py
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

# Prepare sample input data with some negative values
m, n, k = 128, 64, 96

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# FP8 operations require TN input layout.

with torch.cuda.device(device_id):
    a = (torch.rand(*row_wise_distribution.shape(rank, (m, k)), device="cuda") * 20 - 10).type(torch.float8_e4m3fn)
    b = (torch.rand(*row_wise_distribution.shape(rank, (n, k)), device="cuda") * 20 - 10).type(torch.float8_e4m3fn)

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with col_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [col_wise_distribution, col_wise_distribution, row_wise_distribution]

# Set quantization scales to keep values in range
scales = {"a": 1, "b": 1, "d": 0.1}

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True

# First perform multiplication without RELU
result_no_relu = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    quantization_scales=scales,
)

# Now perform multiplication with RELU epilog
result_with_relu = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    epilog=nvmath.distributed.linalg.advanced.MatmulEpilog.RELU,
    quantization_scales=scales,
)

if rank == 0:
    print("Result without RELU (notice negative values):")
    # Printing the tensor synchronizes on the default CUDA stream.
    print(result_no_relu)
    print("\nResult with RELU (all values >= 0):")
    print(result_with_relu)

# Verify that all values in the RELU result are non-negative
assert torch.all(result_with_relu.type(torch.float32) >= 0), "RELU result contains negative values!"
