# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to reuse the stateful API to perform Reshape operations
by resetting operands inplace. It's important to note that operands reset in this
manner have to preserve their distribution.

$ mpiexec -n 4 python example05_stateful_reset_inplace.py
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

# The global 3-D problem size is (512, 512, 512), initially partitioned on the Y
# axes across processes.
X, Y, Z = (512, 512, 512)
shape = X, Y // nranks, Z

# The distributed reshape implementation uses the NVSHMEM PGAS model for GPU-GPU transfers,
# which requires GPU operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex64)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

# We're going to redistribute the operand so that it is partitioned on the X axis.
y_offset = comm.scan(Y // nranks, op=MPI.SUM)
input_box = [(0, y_offset - Y // nranks, 0), (X, y_offset, Z)]

x_offset = comm.scan(X // nranks, op=MPI.SUM)
output_box = [(x_offset - X // nranks, 0, 0), (x_offset, Y, Z)]

# Create a stateful Reshape object 'r'.
with nvmath.distributed.reshape.Reshape(a, input_box, output_box) as r:
    # Plan the Reshape.
    r.plan()

    # Execute the Reshape. Operand b will be partitioned on the X axis.
    b = r.execute()

    # Reset the operand inplace. Note that this implies maintaining the same
    # input operand distribution (partitioned on Y axis).
    with cp.cuda.Device(device_id):
        a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

    # Execute a new reshape with the modified operand. Operand c will be partitioned
    # on the X axis.
    c = r.execute()

    # Synchronize the default stream
    with cp.cuda.Device(device_id):
        cp.cuda.get_current_stream().synchronize()
    if rank == 0:
        print(f"Input type = {type(a)}, device = {a.device}")
        print(f"Reshape output type (1) = {type(b)}, device = {b.device}")
        print(f"Reshape output type (2) = {type(c)}, device = {c.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
nvmath.distributed.free_symmetric_memory(a, b, c)
