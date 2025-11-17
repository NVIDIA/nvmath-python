# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful distributed matrix multiplication objects.
Stateful objects amortize the cost of preparation across multiple executions.

The inputs as well as the result are PyTorch tensors on the GPU.

The global operation performed in this example is: A.T @ B

$ mpiexec -n 4 python example05_stateful_torch.py
"""

import torch
import numpy as np
from mpi4py import MPI

import nvmath.distributed

from nvmath.distributed.distribution import ProcessGrid, BlockNonCyclic
from nvmath.distributed.linalg.advanced import matrix_qualifiers_dtype

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % torch.cuda.device_count()
# cuBLASMp requires NVSHMEM and NCCL communication backends.
nvmath.distributed.initialize(device_id, comm, backends=["nvshmem", "nccl"])

# The global problem size m, n, k
m, n, k = 256, 512, 256

# Note: see example01 for details on matrix distribution and memory layout impact and
# requirements.

row_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(nranks, 1)))  # partitioning on rows
col_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(1, nranks)))  # partitioning on columns

a_shape = col_wise_distribution.shape(rank, (m, k))
b_shape = col_wise_distribution.shape(rank, (n, k))
a = nvmath.distributed.allocate_symmetric_memory(a_shape, torch)
b = nvmath.distributed.allocate_symmetric_memory(b_shape, torch)
with torch.cuda.device(device_id):
    a[:] = torch.rand(*a_shape, device=f"cuda:{device_id}")
    b[:] = torch.rand(*b_shape, device=f"cuda:{device_id}")

# Get a transposed view to obtain column-major Fortran memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (k, m) with row_wise_distribution
b = b.T  # b is now (k, n) with row_wise_distribution

# Distribution of a, b and output.
distributions = [row_wise_distribution, row_wise_distribution, col_wise_distribution]

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True  # a is transposed

# Use the stateful object as a context manager to automatically release resources.
with nvmath.distributed.linalg.advanced.Matmul(a, b, distributions=distributions, qualifiers=qualifiers) as mm:
    # Plan the matrix multiplication.
    mm.plan()

    # Execute the matrix multiplication.
    result = mm.execute()

    # Note: if all of the input operands are on symmetric memory, the result is also
    # on symmetric memory.

    # Synchronize the default stream, since by default the execution is non-blocking for GPU
    # operands.
    torch.cuda.default_stream().synchronize()
    print(f"Input types = {type(a), type(b)}, device = {a.device, b.device}")
    print(f"Result type = {type(result)}, device = {result.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
nvmath.distributed.free_symmetric_memory(a, b, result)
