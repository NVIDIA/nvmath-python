# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates the use of a user-provided logger.

The global operation performed in this example is: A @ B

$ mpiexec -n 2 python example04_logging_user.py
"""

import logging

import cupy as cp
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import Slab

# Create and configure a user logger.
# Any of the features provided by the logging module can be used.
logger = logging.getLogger("userlogger")
logging.getLogger().setLevel(logging.NOTSET)

# Create a console handler for the logger and set level.
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create a formatter and associate with handler.
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")
handler.setFormatter(formatter)

# Associate handler with logger, resulting in a logger with the desired level, format, and
# console output.
logger.addHandler(handler)

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
# cuBLASMp requires NCCL communication backend.
nvmath.distributed.initialize(device_id, comm, backends=["nccl"])

# The global problem size m, n, k
m, n, k = 64, 128, 256

# Note: see example01 for details on matrix distribution and memory layout impact and
# requirements.

# Prepare sample input data.
with cp.cuda.Device(device_id):
    a = cp.random.rand(k // nranks, m).astype(cp.float16)  # partitioned on k
    b = cp.random.rand(n, k // nranks).astype(cp.float16)  # partitioned on k
a = a.T
b = b.T
alpha = 0.45

distributions = [Slab.Y, Slab.X, Slab.Y]

# Specify the custom logger in the matrix multiplication options.
o = nvmath.distributed.linalg.advanced.MatmulOptions(logger=logger)
# Specify the options to the matrix multiplication operation.
result = nvmath.distributed.linalg.advanced.matmul(a, b, alpha=alpha, distributions=distributions, options=o)

print("---")

# Recall that the options can also be provided as a dict, so the following is an
# alternative, entirely equivalent way to specify options.
result = nvmath.distributed.linalg.advanced.matmul(a, b, alpha=alpha, distributions=distributions, options={"logger": logger})

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()
