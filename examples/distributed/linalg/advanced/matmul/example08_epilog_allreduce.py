# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates usage of AllReduce epilog.

With cuBLASMp's GEMM+AllReduce algorithm, each process calculates a part of the output
which will be then reduced (summed) using the AllReduce operation resulting in an output
matrix that is the same across all processes.

The global operation performed in this example is: A.T @ B
The AllReduce epilog operation is a sum reduction of the partial result of each process,
resulting in the same output matrix of shape (m, n) on all processes.

$ mpiexec -n 4 python example08_epilog_allreduce.py
"""

import cupy as cp
import numpy as np
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import BlockNonCyclic, ProcessGrid
from nvmath.distributed.linalg.advanced import matrix_qualifiers_dtype

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
# cuBLASMp requires NCCL communication backend.
nvmath.distributed.initialize(device_id, comm, backends=["nccl"])

# The global problem size m, n, k
m, n, k = 256, 512, 128

# See example01 for details on matrix distribution and memory layout impact and
# requirements.

# As of cuBLASMp 0.6, GEMM+AllReduce requires TN format with A and B distributed row-wise
# and C and D matrices distributed column-wise.

row_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(nranks, 1)))  # partitioning on rows
col_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(1, nranks)))  # partitioning on columns

with cp.cuda.Device(device_id):
    a = cp.random.rand(*col_wise_distribution.shape(rank, (m, k)))
    b = cp.random.rand(*col_wise_distribution.shape(rank, (n, k)))

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with row_wise_distribution
b = b.T  # b is now (k, n) with row_wise_distribution

# Distribution of a, b and output.
distributions = [row_wise_distribution, row_wise_distribution, col_wise_distribution]

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True  # a is transposed

epilog = nvmath.distributed.linalg.advanced.MatmulEpilog.ALLREDUCE
result = nvmath.distributed.linalg.advanced.matmul(a, b, distributions=distributions, epilog=epilog, qualifiers=qualifiers)

# AllReduce results in the same output matrix of shape (m, n) on each process.
assert result.shape == (m, n)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()
