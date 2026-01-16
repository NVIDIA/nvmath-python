# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic distributed matrix multiplication of torch tensors,
using the function-form APIs.

nvmath-python accepts operands from multiple frameworks. The result of each operation
is a tensor of the same framework that was used to pass the inputs, and is located
on the same device as the inputs.

Tensors residing on CPU memory are copied transparently to symmetric GPU memory to
process them with cuBLASMp.

The global operation performed in this example is: A.T @ B.T

$ mpiexec -n 4 python example01_torch.py
"""

import numpy as np
import torch
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
m, n, k = 128, 512, 1024

# Prepare sample input data (torch tensors on the CPU).

# nvmath-python uses cuBLASMp for distributed matrix multiplication.
# cuBLASMp supports PBLAS 2D block-cyclic distribution of matrices. For simplicity, in this
# example we partition matrices on a single axis (distribution on a single dimension without
# cyclic property is a special case of 2D block-cyclic).

row_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(nranks, 1)))  # partitioning on rows
col_wise_distribution = BlockNonCyclic(ProcessGrid(shape=(1, nranks)))  # partitioning on columns

a = torch.rand(m, k // nranks)  # a is partitioned on k
b = torch.rand(k // nranks, n)  # b is partitioned on k

# In Python, the memory layout of ndarrays and tensors by default uses row-major or C
# ordering, while cuBLASMp requires column-major or Fortran ordering. To work with cuBLASMp,
# you can follow these guidelines:
# - The transpose of a C-ordered (row-major) matrix is a Fortran-ordered (column-major)
#   matrix and vice-versa.
# - In a distributed setting, a row-wise distributed matrix A is equivalent to a column-wise
#   distributed matrix A.T, and vice-versa.

# Note that numpy, cupy and torch also have functions to allocate tensors with Fortran order
# or to convert to Fortran order (see example01_cupy.py for an example).

# Get a transposed view (zero cost) of the matrices to obtain column-major ordering.
a = a.T  # a is now (k, m) with row_wise_distribution
b = b.T  # b is now (n, k) with col_wise_distribution

# Specify distribution of input and output matrices.

# Note: The choice of distribution for a, b and output as well as whether a and b are
# transposed influences the distributed algorithm used by cuBLASMp and can have a
# substantial impact on performance.
# The following configuration will run GEMM+ReduceScatter.
# Refer to https://docs.nvidia.com/cuda/cublasmp/usage/tp.html for more information.

# Distribution of a, b and output (note how transposing a and b influences their
# distribution):
distributions = [row_wise_distribution, col_wise_distribution, col_wise_distribution]

# Perform the distributed matrix multiplication.
qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
qualifiers[0]["is_transpose"] = True  # a is transposed
qualifiers[1]["is_transpose"] = True  # b is transposed
if rank == 0:
    print("Running the distributed multiplication on CPU tensors...")
result = nvmath.distributed.linalg.advanced.matmul(
    a,
    b,
    distributions=distributions,
    qualifiers=qualifiers,
)

# No synchronization is needed for CPU tensors, since the execution always blocks.

if rank == 0:
    # result has global shape (m, n) and is distributed row-wise (as specified above).
    # The memory layout of result is Fortran on each process.
    print(f"shape={result.shape} strides={result.stride()}")
    assert result.shape == (m, n // nranks)

    # result.T has global shape (n, m) and is distributed column-wise, with C memory layout.
    print(f"shape={result.T.shape} strides={result.T.stride()}")
    assert result.T.shape == (n // nranks, m)

    # Check if the result is torch tensor as well.
    print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(result)}.")
    print(f"Inputs were located on devices {a.device} and {b.device} and the result is on {result.device}")

# Now, move the tensors to the GPU and verify that the result is on the GPU as well.
a_gpu = a.cuda(device=f"cuda:{device_id}", memory_format=torch.preserve_format)
b_gpu = b.cuda(device=f"cuda:{device_id}", memory_format=torch.preserve_format)
if rank == 0:
    print("\nRunning the distributed multiplication on GPU tensors...")
result = nvmath.distributed.linalg.advanced.matmul(
    a_gpu,
    b_gpu,
    distributions=distributions,
    qualifiers=qualifiers,
)

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
torch.cuda.default_stream().synchronize()

print(f"Inputs were of types {type(a_gpu)} and {type(b_gpu)} and the result is of type {type(result)}.")
print(f"Inputs were located on devices {a_gpu.device} and {b_gpu.device} and the result is on {result.device}")
