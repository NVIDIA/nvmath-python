# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful class-form distributed Reshape APIs with
CuPy ndarrays.

The input as well as the result from the Reshape operations are CuPy ndarrays.

$ mpiexec -n 4 python example02_stateful_cupy.py
"""

import cupy as cp
from mpi4py import MPI

import nvmath.distributed

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
nvmath.distributed.initialize(device_id, comm)

# The problem consists of a global 3-D array of size (512, 256, 512), that is
# initially partitioned on the X axis across processes.
X, Y, Z = (512, 256, 512)
shape = X // nranks, Y, Z

# The distributed reshape implementation uses the NVSHMEM PGAS model for GPU-GPU transfers,
# which requires GPU operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.float32)
# a is a cupy ndarray and can be operated on using cupy operations.
with cp.cuda.Device(device_id):
    a[:] = cp.ones(shape, dtype=cp.float32)

# We're going to redistribute the operand so that it is partitioned on the Y axis.
# We can get the offset of this process on the partitioned dimension with a prefix
# reduction.
x_offset = comm.scan(X // nranks, op=MPI.SUM)
input_box = [(x_offset - X // nranks, 0, 0), (x_offset, Y, Z)]

y_offset = comm.scan(Y // nranks, op=MPI.SUM)
output_box = [(0, y_offset - Y // nranks, 0), (X, y_offset, Z)]

# Create a stateful Reshape object 'r'.
with nvmath.distributed.reshape.Reshape(a, input_box, output_box) as r:
    # Plan the Reshape.
    r.plan()

    # Execute the Reshape. This returns a new operand with its own memory buffer
    # on the symmetric heap.
    b = r.execute()

    # Note the difference in shape of operand a and b due to the modified distribution.
    if rank == 0:
        print(f"Shape of a on rank {rank} is {a.shape}")
        print(f"Shape of b on rank {rank} is {b.shape}")

    # Synchronize the default stream.
    with cp.cuda.Device(device_id):
        cp.cuda.get_current_stream().synchronize()
    if rank == 0:
        print(f"Input type = {type(a)}, device = {a.device}")
        print(f"Reshape output type = {type(b)}, device = {b.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
nvmath.distributed.free_symmetric_memory(a, b)
