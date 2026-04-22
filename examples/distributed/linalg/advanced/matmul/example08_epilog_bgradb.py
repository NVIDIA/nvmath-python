# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates usage of epilogs.

Epilogs allow you to execute extra computations after the matrix multiplication in a single
fused kernel. In this example we'll use the BGRADB epilog, which generates an extra output
"bgradb" corresponding to the reduction of the B matrix.

$ mpiexec -n 4 python example08_epilog_bgradb.py
"""

import cupy as cp
import numpy as np
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import Slab
from nvmath.distributed.linalg.advanced import matrix_qualifiers_dtype

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
# cuBLASMp requires NCCL communication backend.
nvmath.distributed.initialize(device_id, comm, backends=["nccl"])

# Prepare sample input data.
m, n, k = 128, 256, 512

row_wise_distribution = Slab.X
col_wise_distribution = Slab.Y

# NOTE: cuBLAS requires transposed B for BGRADB epilogue.

with cp.cuda.Device(device_id):
    a = cp.random.rand(*col_wise_distribution.shape(rank, (k, m)))
    b = cp.random.rand(*col_wise_distribution.shape(rank, (k, n)))

# Get a transposed view to obtain column-major memory layout. Note that this
# also changes the distribution of a and b (see example01 for more information).
a = a.T  # a is now (m, k) with row_wise_distribution
b = b.T  # b is now (n, k) with row_wise_distribution

# Distributions for A, B, and result matrix D
distributions = [row_wise_distribution] * 3

qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[1]["is_transpose"] = True  # b is transposed

# Perform the multiplication with BGRADB epilog. The auxiliary output "auxiliary" is a dict
# containing the bias gradient with the key "bgradb".
epilog = nvmath.distributed.linalg.advanced.MatmulEpilog.BGRADB
result, auxiliary = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
    epilog=epilog,
)

# Note: BGRADB output is always replicated (same vector of length n on all processes).
assert auxiliary["bgradb"].shape == (n,)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()
if rank == 0:
    print(
        f"Inputs were of types {type(a)} and {type(b)}, and the result type is {type(result)}, "
        f"and the auxiliary output is of type {type(auxiliary)}."
    )
