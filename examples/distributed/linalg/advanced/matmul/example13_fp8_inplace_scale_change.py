# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how quantization scales passed as GPU tensors can be modified
in-place without needing to call reset_operands().

FP8 is only supported with compute capability 8.9 or higher.

$ mpiexec -n 4 python example13_fp8_inplace_scale_change.py
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
# cuBLASMp requires NCCL communication backends
nvmath.distributed.initialize(device_id, comm, backends=["nccl"])

# Prepare sample input data
m, n, k = 128, 256, 16

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# FP8 operations require TN input layout.

with torch.cuda.device(device_id):
    a = torch.ones(*row_wise_distribution.shape(rank, (m, k)), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.ones(*row_wise_distribution.shape(rank, (n, k)), device="cuda", dtype=torch.float8_e5m2)

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with col_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [col_wise_distribution, col_wise_distribution, row_wise_distribution]

if rank == 0:
    print(f"A = \n{a}")
    print(f"\nB = \n{b}")

# Create 1D single-element float32 GPU tensors to hold the quantization scales.
# These will be modified in-place later.
scales = {
    "a": torch.full((1,), 3, dtype=torch.float32, device=f"cuda:{device_id}"),
    "b": torch.full((1,), 2, dtype=torch.float32, device=f"cuda:{device_id}"),
    "d": torch.full((1,), 1, dtype=torch.float32, device=f"cuda:{device_id}"),
}

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
    torch.cuda.default_stream().synchronize()
    if rank == 0:
        print(
            f"\nA (A scale: {scales['a'].item()}) @ B (B scale: {scales['b'].item()}) = "
            f"(D scale: {scales['d'].item()}) \n{result}"
        )

    # Modify the quantization scales for A and D in-place
    scales["a"][:] = 2
    scales["d"][:] = 0.25

    # Execute the multiplication again with the new quantization scales and print the result
    result2 = mm.execute()
    torch.cuda.default_stream().synchronize()
    if rank == 0:
        print(
            f"\nA (A scale: {scales['a'].item()}) @ B (B scale: {scales['b'].item()}) = "
            f"(D scale: {scales['d'].item()}) \n{result2}"
        )
