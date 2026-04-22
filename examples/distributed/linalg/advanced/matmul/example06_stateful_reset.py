# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to reset operands in stateful matrix multiplication APIs, and
reuse the object for multiple executions. This is needed when the memory space of the
operands is not accessible from the execution space, or if it's desired to bind new
(compatible) operands to the stateful object.

The inputs as well as the result are NumPy ndarrays.

The global operation performed in this example is: A @ B

$ mpiexec -n 4 python example06_stateful_reset.py
"""

import logging

import numpy as np

try:
    from cuda.core import system
except ImportError:
    from cuda.core.experimental import system
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import BlockNonCyclic, ProcessGrid

# Initialize nvmath.distributed.
try:
    num_devices = system.get_num_devices()
except AttributeError:
    num_devices = system.num_devices
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % num_devices
# cuBLASMp requires NCCL communication backend.
nvmath.distributed.initialize(device_id, comm, backends=["nccl"])

# Turn on logging to see what's happening.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# The global problem size m, n, k
m, n, k = 128, 512, 256

# See example01 for details on matrix distribution and memory layout impact and
# requirements.

row_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(nranks, 1)))  # partitioning on rows
col_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(1, nranks)))  # partitioning on columns

a = np.random.rand(*row_wise_distribution.shape(rank, (k, m)))
b = np.random.rand(*col_wise_distribution.shape(rank, (n, k)))

# Get a transposed view to obtain column-major memory layout. Note that this
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

    # Create new operands and bind them.
    a_new = np.random.rand(*col_wise_distribution.shape(rank, (m, k)))
    b_new = np.random.rand(*row_wise_distribution.shape(rank, (k, n)))
    mm.reset_operands(a=a_new, b=b_new)

    # Execute the new matrix multiplication.
    result = mm.execute()

    # No synchronization is needed for CPU tensors, since the execution always blocks.

    print(f"Input types = {type(a_new), type(b_new)}")
    print(f"Result type = {type(result)}")
