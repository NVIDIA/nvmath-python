# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to perform distributed in-place matrix multiplication, where
the result overwrites operand `c`:

    c := a.T @ b + beta c

$ mpiexec -n 4 python example09_cupy_inplace.py
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

# The global problem size m, n, k
m, n, k = 128, 512, 1024

# Prepare sample input data (CuPy matrices, on the GPU).

# Specify distribution of input and output matrices.
# Note: The choice of distribution for a, b and output, as well as whether a and b are
# transposed influences the distributed algorithm used by cuBLASMp and can have a
# substantial impact on performance.
# Refer to https://docs.nvidia.com/cuda/cublasmp/usage/tp.html for more information.
#
# In this example we use TN layout (A transposed, B non-transposed).
# With TN, this configuration will run AllGather+GEMM.
distributions = [Slab.Y, Slab.Y, Slab.X]  # distribution of A, B and C/D

# cuBLASMp requires Fortran memory layout. CuPy allocates C-ordered arrays by default,
# so in this example we allocate transposed shapes and then take the transpose to obtain
# Fortran order (note that we use the transposed distribution to determine the shape
# since transposing a distributed matrix will transpose its distribution).
a_shape = Slab.X.shape(rank, (m, k))
b_shape = Slab.X.shape(rank, (n, k))
c_shape = Slab.Y.shape(rank, (n, m))

# Note: see example01 for more details on matrix distribution and memory layout impact and
# requirements.

with cp.cuda.Device(device_id):
    a = cp.random.rand(*a_shape).T  # a is now Slab.Y with global shape (k, m)
    b = cp.random.rand(*b_shape).T  # b is now Slab.Y with global shape (k, n)
    c = cp.random.rand(*c_shape).T  # c is now Slab.X with global shape (m, n)

beta = 1.0

# Perform the distributed matrix multiplication.
qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True  # a is transposed
result = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    c=c,
    beta=beta,
    distributions=distributions,
    qualifiers=qualifiers,
    options={"inplace": True},
)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()

# The result is stored in C.
assert result is c

if rank == 0:
    # result has global shape (m, n) and is distributed row-wise (as specified above).
    print(result.shape, result.flags)
    assert result.shape == Slab.X.shape(rank, (m, n))
