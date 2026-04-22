# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates distributed matrix multiplication using 2D block-cyclic
distribution.

The global operation performed in this example is: A @ B

$ mpiexec -n 4 python example02_2d_block_cyclic_4p.py
"""

import numpy as np

try:
    from cuda.core import system
except ImportError:
    from cuda.core.experimental import system
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import BlockCyclic, ProcessGrid

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

# The global problem size m, n, k
m, n, k = 256, 512, 1024

# This example uses the PBLAS 2D block-cyclic distribution.
# See example01 for details on matrix distribution and memory layout impact and
# requirements.

assert nranks == 4, "This example requires 4 processes"

# 2D 2x2 process grid (4 processes) with column-major layout:
# ---------
# | 0 | 2 |
# ---------
# | 1 | 3 |
# ---------
process_grid = ProcessGrid(shape=(2, 2), layout=ProcessGrid.Layout.COL_MAJOR)

# Cyclic distribution with 4x4 block size.
distribution = BlockCyclic(process_grid, (4, 4))

# Get the shape of inputs a and b on this rank according to this distribution.
a_shape = distribution.shape(rank, (m, k))
b_shape = distribution.shape(rank, (k, n))

# Prepare sample input data.
a = np.random.rand(*a_shape).astype(np.float32)
b = np.random.rand(*b_shape).astype(np.float32)
# cuBLASMp requires column-major (Fortran) memory layout (see example01 for details and
# alternate ways to handle).
a = np.asfortranarray(a)
b = np.asfortranarray(b)

# Matrices a, b and output use the same distribution.
distributions = [distribution, distribution, distribution]

# Perform the distributed matrix multiplication.
result = nvmath.distributed.linalg.advanced.matmul(a, b, distributions=distributions)

# No synchronization is needed for CPU tensors, since the execution always blocks.
