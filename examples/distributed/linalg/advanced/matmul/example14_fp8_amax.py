# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to obtain the maximum absolute value (amax) in the result,
computed before quantization.

In previous examples, quantization scales were set manually to appropriate values. Amax can
be used to automatically set proper scales in FP8 operations, as it indicates how much the
result needs to be scaled to fit into the dynamic range of the result type. In this example,
we first compute the result without scaling, then use amax to compute the correct scale, and
repeat the multiplication. While this approach is inefficient, it demonstrates the concept.
For a more practical example, see example15_fp8_delayed_scaling.py, which uses amax from
previous iterations to choose scales for subsequent multiplications.

FP8 is only supported with compute capability 8.9 or higher.

$ mpiexec -n 4 python example14_fp8_amax.py
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

torch.manual_seed(13)

# Fill the input tensors with random numbers from (0, 30).
m, n, k = 128, 128, 16

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# FP8 operations require TN input layout.

with torch.cuda.device(device_id):
    a = (torch.rand(*row_wise_distribution.shape(rank, (m, k)), device="cuda") * 30).type(torch.float8_e4m3fn)
    b = (torch.rand(*row_wise_distribution.shape(rank, (n, k)), device="cuda") * 30).type(torch.float8_e4m3fn)

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with col_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [col_wise_distribution, col_wise_distribution, row_wise_distribution]

# To request amax, set `result_amax` option to True.
options = {"result_amax": True}

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True

# When result_amax is set, a tuple containing the actual result and the auxiliary outputs
# will be returned instead of just the result.
result, aux = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    quantization_scales={"a": 1, "b": 1, "d": 1},
    options=options,
)

# With all quantization scales set to 1, most of the elements are clamped to the maximum
# value:
if rank == 0:
    print("Result is:")
    # Printing the tensor synchronizes on the default CUDA stream.
    print(result)

    # Local Amax will be present in the auxiliary outputs dictionary as "result_amax".
    print(f"Matmul returned the result and the auxiliary outputs of type {type(aux)}: {aux}")

# Get the global amax (cuBLASMp returns the local amax)
amax = comm.allreduce(aux["result_amax"].item(), op=MPI.MAX)

if rank == 0:
    print("The global amax is", amax)

# Compute the proper scale by dividing the maximum representable value by amax.
max_representable_value = 448
d_scale = max_representable_value / amax
if rank == 0:
    print(f"d_scale = max_representable_value / amax = {max_representable_value} / {amax:.5f} = {d_scale:.5f}")

# Repeat the computation, this time using the proper scale for D.
result2 = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    quantization_scales={"a": 1, "b": 1, "d": d_scale},
)
if rank == 0:
    print(f"Result (with D scale set to {d_scale:.5f}) is:")
    # Printing the tensor synchronizes on the default CUDA stream.
    print(result2)
