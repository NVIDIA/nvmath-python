# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of function-form distributed Reshape APIs with CuPy
ndarrays.

The input as well as the result from the Reshape operation are CuPy ndarrays, resulting
in effortless interoperability between nvmath-python and CuPy.

In this example, given a 4x4 matrix which is initially distributed column-wise on two
processes, we redistribute it row-wise. This is illustrated below:

    0 0 | 1 1     0 0 1 1
    0 0 | 1 1  -> 0 0 1 1   P0
    0 0 | 1 1     -------
    0 0 | 1 1     0 0 1 1   P1
                  0 0 1 1
     P0   P1

where P0 and P1 refer to process 0 and 1, respectively. Initially, P0 holds the first
two columns and P1 the last two. After performing a distributed reshape, P0 holds the
first two rows and P1 the last two.

$ mpiexec -n 2 python example01_cupy.py
"""

import cupy as cp
import nvmath.distributed

# Initialize nvmath.distributed.
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
nvmath.distributed.initialize(device_id, comm)

assert nranks == 2, "Please run with two processes"

# Initialize the matrix on each process, as a CuPy ndarray (on the GPU).

# The distributed reshape implementation uses the NVSHMEM PGAS model for GPU-GPU transfers,
# which requires GPU operands to be on the symmetric heap.
A = nvmath.distributed.allocate_symmetric_memory((4, 2), cp)

# Note that the tensor is allocated on the same device on which nvmath.distributed
# was initialized.
if rank == 0:
    print("A is on device", A.device)

# A is a cupy ndarray and can be operated on using cupy operations.
with cp.cuda.Device(device_id):
    if rank == 0:
        # Initialize the sub-matrix on process 0.
        A[:] = cp.zeros((4, 2))
    else:
        # Initialize the sub-matrix on process 1.
        A[:] = cp.ones((4, 2))

# Reshape from column-wise to row-wise.
if rank == 0:
    input_box = [(0, 0), (4, 2)]
    output_box = [(0, 0), (2, 4)]
else:
    input_box = [(0, 2), (4, 4)]
    output_box = [(2, 0), (4, 4)]
# Distributed reshape returns a new operand with its own memory buffer
# on the symmetric heap.
A_reshaped = nvmath.distributed.reshape.reshape(A, input_box, output_box)

# Synchronize the default stream
with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()

# The result is a CuPy ndarray, distributed row-wise:
# [0] A_reshaped:
# [[0. 0. 1. 1.]
#  [0. 0. 1. 1.]]
#
# [1] A_reshaped:
# [[0. 0. 1. 1.]
#  [0. 0. 1. 1.]]
print(f"[{rank}] A_reshaped:\n{A_reshaped}")
if rank == 0:
    print(f"Input type = {type(A)}, device = {A.device}")
    print(f"Reshape output type = {type(A_reshaped)}, device = {A_reshaped.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
nvmath.distributed.free_symmetric_memory(A, A_reshaped)
