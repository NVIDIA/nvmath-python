# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates usage of epilogs that generate auxiliary output.

Epilogs allow you to execute extra computations after the matrix multiplication in a single
fused kernel. In this example we'll use the RELU_AUX epilog, which generates an extra output
"relu_aux". Other examples show how to use the auxiliary output as input to other
epilogs.

$ mpiexec -n 4 python example08_epilog_relu_aux.py
"""

import cupy as cp
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import Slab

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
# cuBLASMp requires NCCL communication backend.
nvmath.distributed.initialize(device_id, comm, backends=["nccl"])

# Prepare sample input data.
m, n, k = 128, 256, 512

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

with cp.cuda.Device(device_id):
    a = cp.random.rand(*col_wise_distribution.shape(rank, (k, m)))
    b = cp.random.rand(*row_wise_distribution.shape(rank, (n, k)))

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (m, k) with row_wise_distribution
b = b.T  # b is now (k, n) with col_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [row_wise_distribution, col_wise_distribution, row_wise_distribution]

# Perform the multiplication with RELU_AUX epilog.
epilog = nvmath.distributed.linalg.advanced.MatmulEpilog.RELU_AUX
result, auxiliary = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    epilog=epilog,
)

# Note: auxiliary contains the RELU_AUX bitmask, which follows the same distribution
# as the result matrix. In this example, the result matrix is partitioned on the
# m dimension, and likewise the RELU_AUX bitmask.

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()
if rank == 0:
    print(
        f"Inputs were of types {type(a)} and {type(b)}, and the result type is {type(result)}, "
        f"and the auxiliary output is of type {type(auxiliary)}."
    )
