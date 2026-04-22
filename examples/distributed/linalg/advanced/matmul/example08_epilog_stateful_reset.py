# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to reset operands and epilog inputs in stateful matrix
multiplication APIs, and reuse the object for multiple executions. This is needed when the
memory space of the operands is not accessible from the execution space, or if
it's desired to bind new (compatible) operands to the stateful object.

The inputs as well as the result are NumPy ndarrays.

$ mpiexec -n 4 python example08_epilog_stateful_reset.py
"""

import logging

import numpy as np
from mpi4py import MPI

try:
    from cuda.core import system
except ImportError:
    from cuda.core.experimental import system

import nvmath.distributed
from nvmath.distributed.distribution import Slab

# Turn on logging to see what's happening.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

try:
    num_devices = system.get_num_devices()
except AttributeError:
    num_devices = system.num_devices

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % num_devices
# cuBLASMp requires NCCL communication backend.
nvmath.distributed.initialize(device_id, comm, backends=["nccl"])

# Prepare sample input data.
m, n, k = 128, 256, 512

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

a_shape = row_wise_distribution.shape(rank, (m, k))
b_shape = col_wise_distribution.shape(rank, (k, n))

a = np.asfortranarray(np.random.rand(*a_shape))
b = np.asfortranarray(np.random.rand(*b_shape))
# We're going to use row_wise_distribution for the result, meaning that the
# result will be partitioned on the m dimension across processes, and so the
# bias vector needs to be partitioned as well.
bias = np.random.rand(m // nranks, 1)

# Distributions for A, B, and result matrix D
distributions = [row_wise_distribution, col_wise_distribution, row_wise_distribution]

# Use the stateful object as a context manager to automatically release resources.
with nvmath.distributed.linalg.advanced.Matmul(a, b, distributions=distributions) as mm:
    # Plan the matrix multiplication for the BIAS epilog.
    epilog = nvmath.distributed.linalg.advanced.MatmulEpilog.BIAS
    mm.plan(epilog=epilog, epilog_inputs={"bias": bias})

    # Execute the matrix multiplication.
    result = mm.execute()

    # Create new operands and bind them.
    c = np.asfortranarray(np.random.rand(*a_shape))
    d = np.asfortranarray(np.random.rand(*b_shape))
    bias = np.random.rand(m // nranks, 1)
    mm.reset_operands(a=c, b=d, epilog_inputs={"bias": bias})

    # Execute the new matrix multiplication.
    result = mm.execute()

    # No synchronization is needed for CPU tensors, since the call always blocks.

    if rank == 0:
        print(f"Input types = {type(c), type(d)}")
        print(f"Result type = {type(result)}")
