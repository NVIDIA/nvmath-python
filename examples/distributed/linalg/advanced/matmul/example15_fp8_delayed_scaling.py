# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to implement a simple delayed scaling algorithm. We use the
amax value from the previous iteration to set the scale for the next iteration. In a more
advanced setup, an average amax from N previous iterations could be used as well. In each
iteration, we multiply two normally-distributed matrices A and B and add matrix C to the
result.

FP8 is only supported with compute capability 8.9 or higher.

$ mpiexec -n 4 python example15_fp8_delayed_scaling.py
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

m, n, k = 256, 256, 256

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# FP8 operations require TN input layout.

with torch.cuda.device(device_id):
    a = torch.zeros(*row_wise_distribution.shape(rank, (m, k)), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros(*row_wise_distribution.shape(rank, (n, k)), device="cuda", dtype=torch.float8_e4m3fn)
    c = torch.zeros(*col_wise_distribution.shape(rank, (n, m)), device="cuda", dtype=torch.float16)

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a, b, and c (see example01 for more information).
a = a.T  # a is now (k, m) with col_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution
c = c.T  # c is now (m, n) with row_wise_distribution

# Distributions for A, B, and C.
distributions = [col_wise_distribution, col_wise_distribution, row_wise_distribution]


def regenerate_inputs():
    with torch.cuda.device(device_id):
        a[:] = torch.randn(a.shape, device="cuda") * 10
        b[:] = torch.randn(b.shape, device="cuda") * 10
        c[:] = torch.randn(c.shape, device="cuda") * 10
        return a, b, c


# Keep D scale in a GPU tensor instead of a Python float to allow in-place changes
dscale = torch.ones((1,), dtype=torch.float32, device=f"cuda:{device_id}")
scales = {"a": 1, "b": 1, "d": dscale}

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True

# Request FP8 output and AMAX calculation
options = {"result_type": nvmath.CudaDataType.CUDA_R_8F_E4M3, "result_amax": True}

with nvmath.distributed.linalg.advanced.Matmul(
    a, b, c=c, distributions=distributions, qualifiers=qualifiers, beta=1, quantization_scales=scales, options=options
) as mm:
    mm.plan()

    for iteration in range(10):
        # Populate a, b, and c with fresh random data
        regenerate_inputs()

        # Execute the matrix multiplication
        result, aux = mm.execute()
        # Get the global amax (cuBLASMp returns the local amax)
        amax = comm.allreduce(aux["result_amax"].item(), op=MPI.MAX)

        # Calculate the percentage of clamped values on this process.
        max_representable_value = 448
        clamped_percent = (
            100 * ((result == max_representable_value) | (result == -max_representable_value)).sum().item() / result.nelement()
        )

        # Print a report. Note that the percentage of clamped values will rapidly decrease
        if rank == 0:
            print(
                f"Iteration {iteration} with dscale={dscale.item():05f}: "
                f"amax={amax:.2f}, {clamped_percent:.02f}% of values were clamped to the max value."
            )

        torch.cuda.default_stream().synchronize()

        # Update D scale for the next iteration
        dscale[:] = max_representable_value / amax
