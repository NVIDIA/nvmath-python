# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful distributed matrix multiplication objects.
Stateful objects amortize the cost of preparation across multiple executions.

The inputs as well as the result are CuPy ndarrays.

The global operation performed in this example is: A @ B

$ mpiexec -n 4 python example05_stateful_cupy.py
"""

import cupy as cp
from mpi4py import MPI

import nvmath.distributed

from nvmath.distributed.distribution import ProcessGrid, BlockNonCyclic

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
# cuBLASMp requires NVSHMEM and NCCL communication backends.
nvmath.distributed.initialize(device_id, comm, backends=["nvshmem", "nccl"])

# The global problem size m, n, k
m, n, k = 128, 512, 1024

# Note: see example01 for details on matrix distribution and memory layout impact and
# requirements.

row_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(nranks, 1)))  # partitioning on rows
col_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(1, nranks)))  # partitioning on columns

a_shape = row_wise_distribution.shape(rank, (k, m))
b_shape = col_wise_distribution.shape(rank, (n, k))
a = nvmath.distributed.allocate_symmetric_memory(a_shape, cp)
b = nvmath.distributed.allocate_symmetric_memory(b_shape, cp)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*a_shape)
    b[:] = cp.random.rand(*b_shape)

# Get a transposed view to obtain column-major Fortran memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (m, k) with col_wise_distribution
b = b.T  # b is now (k, n) with row_wise_distribution

# Distribution of a, b and output.
distributions = [col_wise_distribution, row_wise_distribution, col_wise_distribution]

# Use the stateful object as a context manager to automatically release resources.
with nvmath.distributed.linalg.advanced.Matmul(a, b, distributions=distributions) as mm:
    # Plan the matrix multiplication.
    mm.plan()

    # Execute the matrix multiplication.
    result = mm.execute()

    # Note: if all of the input operands are on symmetric memory, the result is also
    # on symmetric memory.

    # Synchronize the default stream, since by default the execution is non-blocking for GPU
    # operands.
    cp.cuda.get_current_stream().synchronize()
    print(f"Input types = {type(a), type(b)}, device = {a.device, b.device}")
    print(f"Result type = {type(result)}, device = {result.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
nvmath.distributed.free_symmetric_memory(a, b, result)
