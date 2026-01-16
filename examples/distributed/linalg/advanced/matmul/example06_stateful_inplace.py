# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of inplace update of input operands in stateful matrix
multiplication APIs.

The inputs as well as the result are CuPy ndarrays.

NOTE: The operands should be updated inplace only when they are in a memory space that is
accessible from the execution space. In this case, the operands reside on the GPU while the
execution also happens on the GPU.

The global operation performed in this example is: A @ B

$ mpiexec -n 4 python example06_stateful_inplace.py
"""

import logging

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

# Turn on logging to see what's happening.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# The global problem size m, n, k
m, n, k = 128, 512, 1024

# See example01 for details on matrix distribution and memory layout impact and
# requirements.

row_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(nranks, 1)))  # partitioning on rows
col_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(1, nranks)))  # partitioning on columns

with cp.cuda.Device(device_id):
    # See example01_cupy_symmetric_memory.py for an example of allocating on symmetric
    # memory, which may further improve performance.
    a = cp.random.rand(*row_wise_distribution.shape(rank, (k, m)))
    b = cp.random.rand(*col_wise_distribution.shape(rank, (n, k)))

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

    # Update the operand A in-place.
    print("Updating 'a' in-place.")
    with cp.cuda.Device(device_id):
        a[:] = cp.random.rand(*col_wise_distribution.shape(rank, (m, k)))

    # Execute the new matrix multiplication.
    result = mm.execute()

    # Synchronize the default stream, since by default the execution is non-blocking for GPU
    # operands.
    cp.cuda.get_current_stream().synchronize()
    print(f"Input types = {type(a), type(b)}, device = {a.device, b.device}")
    print(f"Result type = {type(result)}, device = {result.device}")
