# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how the reset_operands method of a Matmul object can be used to
change both the operands and their quantization scales.

FP8 is only supported compute capability 8.9 or higher.

$ mpiexec -n 4 python example12_fp8_reset.py
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

# Prepare sample input data
m, n, k = 128, 256, 16

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# FP8 operations require TN input layout.

with torch.cuda.device(device_id):
    a = torch.ones(*row_wise_distribution.shape(rank, (m, k)), device="cuda", dtype=torch.float8_e5m2)
    b = torch.ones(*row_wise_distribution.shape(rank, (n, k)), device="cuda", dtype=torch.float8_e4m3fn)

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with col_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [col_wise_distribution, col_wise_distribution, row_wise_distribution]

if rank == 0:
    print(f"A = \n{a}")
    print(f"\nB = \n{b}")

scales = {"a": 3, "b": 2, "d": 1}

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True

with nvmath.distributed.linalg.advanced.Matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    quantization_scales=scales,
    options={"result_type": nvmath.CudaDataType.CUDA_R_8F_E5M2},
) as mm:
    # Plan the multiplication
    mm.plan()

    # Execute the multiplication and print the result
    result = mm.execute()
    if rank == 0:
        # Printing the tensor synchronizes on the default CUDA stream.
        print(f"\nA (A scale: {scales['a']}) @ B (B scale: {scales['b']}) = (D scale: {scales['d']}) \n{result}")

    # Replace A with a matrix filled with 128 and adjust A and D scales.
    # Note that since no new scale for B is specified, it will remain unchanged.
    with torch.cuda.device(device_id):
        new_a = torch.full(row_wise_distribution.shape(rank, (m, k)), 128, device="cuda").type(torch.float8_e5m2).T
    if rank == 0:
        print(f"\nnew A = \n{new_a}")
    new_a_scale = 1
    new_d_scale = 0.01
    mm.reset_operands(a=new_a, quantization_scales={"a": new_a_scale, "d": new_d_scale})

    # Execute the multiplication again and print the new result
    result2 = mm.execute()
    if rank == 0:
        # Printing the tensor synchronizes on the default CUDA stream.
        print(f"\nA (A scale: {new_a_scale}) @ B (B scale: {scales['b']}) = (D scale: {new_d_scale}) \n{result2}")
