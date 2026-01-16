# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify options to a distributed matrix multiplication
operation.

In this example, we'll use NumPy ndarrays as input, and look at two equivalent ways to
specify the compute type.

The global operation performed in this example is: A @ B

$ mpiexec -n 4 python example03_options.py
"""

import numpy as np

try:
    from cuda.core import system
except ImportError:
    from cuda.core.experimental import system
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import Slab

# Initialize nvmath.distributed.
try:
    num_devices = system.get_num_devices()
except AttributeError:
    num_devices = system.num_devices
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % num_devices
# cuBLASMp requires NVSHMEM and NCCL communication backends.
nvmath.distributed.initialize(device_id, comm, backends=["nvshmem", "nccl"])

# The global problem size m, n, k
m, n, k = 128, 512, 1024

# Note: see example01 for details on matrix distribution and memory layout impact and
# requirements.

# Prepare sample input data
a = np.random.rand(k // nranks, m).astype(np.float32)  # partitioned on k
b = np.random.rand(n, k // nranks).astype(np.float32)  # partitioned on k
a = a.T
b = b.T

distributions = [Slab.Y, Slab.X, Slab.Y]

# Here we'd like to use COMPUTE_32F_FAST_TF32 for the compute type, and we show two
# alternatives for doing so. Tip: use
# help(nvmath.distributed.linalg.advanced.MatmulComputeType) to see available
# compute types.
compute_type = nvmath.distributed.linalg.advanced.MatmulComputeType.COMPUTE_32F_FAST_TF32

# Alternative #1 for specifying options, using a dataclass.
# Tip: use help(nvmath.distributed.linalg.advanced.MatmulOptions) to see available options.
options = nvmath.distributed.linalg.advanced.MatmulOptions(compute_type=compute_type)
result = nvmath.distributed.linalg.advanced.matmul(a, b, distributions=distributions, options=options)

# Alternative #2 for specifying options, using dict. The two alternatives are entirely
# equivalent.
result = nvmath.distributed.linalg.advanced.matmul(a, b, distributions=distributions, options={"compute_type": compute_type})

# No synchronization is needed for CPU tensors, since the execution always blocks.

# Check if the result is numpy array as well.
if rank == 0:
    print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(result)}.")
    print(f"Inputs were of data types {a.dtype} and {b.dtype} and the result is of data type {result.dtype}.")
assert isinstance(result, np.ndarray)
