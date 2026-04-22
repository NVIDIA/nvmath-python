# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates distributed GEMM on CuPy ndarrays.

GEMM (General Matrix Multiply) is defined as:
alpha * A @ B + beta * C
where `@` denotes matrix multiplication.

The global operation performed in this example is: alpha * A @ B + beta * C

$ mpiexec -n 4 python example07_gemm.py
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

# The global problem size m, n, k
m, n, k = 128, 512, 1024

# See example01 for details on matrix distribution and memory layout impact and
# requirements.

# Prepare sample input data.
with cp.cuda.Device(device_id):
    a = cp.random.rand(k // nranks, m).astype(cp.float32)  # partitioned on k
    b = cp.random.rand(n, k // nranks).astype(cp.float32)  # partitioned on k
    c = cp.random.rand(n // nranks, m).astype(cp.float32)  # partitioned on n
a = a.T
b = b.T
c = c.T
alpha = 0.45
beta = 0.67

distributions = [Slab.Y, Slab.X, Slab.Y]

# Perform the distributed GEMM.
result = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    c=c,
    alpha=alpha,
    beta=beta,
    distributions=distributions,
)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()

assert result.shape == Slab.Y.shape(rank, (m, n))  # result is distributed column-wise
