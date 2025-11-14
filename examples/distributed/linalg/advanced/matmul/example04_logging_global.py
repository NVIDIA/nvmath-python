# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to turn on logging using the global logger.

The global operation performed in this example is: A @ B

$ mpiexec -n 2 python example04_logging_global.py
"""

import cupy as cp
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import ProcessGrid, BlockNonCyclic

# Turn on logging. Here we use the global logger, set the level to "debug", and use a custom
# format for the log.
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
# cuBLASMp requires NVSHMEM and NCCL communication backends.
nvmath.distributed.initialize(device_id, comm, backends=["nvshmem", "nccl"])

# The global problem size m, n, k
m, n, k = 64, 128, 256

# Note: see example01 for details on matrix distribution and memory layout impact and
# requirements.

# Prepare sample input data.
with cp.cuda.Device(device_id):
    # See example01_cupy_symmetric_memory.py for an example of allocating on symmetric
    # memory, which may further improve performance.
    a = cp.random.rand(k // nranks, m).astype(cp.float32)  # partitioned on k
    b = cp.random.rand(n, k // nranks).astype(cp.float32)  # partitioned on k
a = a.T
b = b.T
alpha = 0.45

row_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(nranks, 1)))  # partitioning on rows
col_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(1, nranks)))  # partitioning on columns

distributions = [col_wise_distribution, row_wise_distribution, col_wise_distribution]

# Perform the GEMM.
result = nvmath.distributed.linalg.advanced.matmul(a, b, alpha=alpha, distributions=distributions)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()
