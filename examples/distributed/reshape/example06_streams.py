# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example on using multiple CUDA streams in distributed Reshape APIs.

$ mpiexec -n 4 python example06_streams.py
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

# The global 3-D problem size is (512, 256, 256), initially partitioned on the X
# axis across processes.
X, Y, Z = (512, 256, 256)
shape = X // nranks, Y, Z

# The distributed reshape implementation uses the NVSHMEM PGAS model for GPU-GPU transfers,
# which requires GPU operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex128)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

# We're going to redistribute the operand so that it is partitioned on the Y axis.
x_offset = comm.scan(X // nranks, op=MPI.SUM)
input_box = [(x_offset - X // nranks, 0, 0), (x_offset, Y, Z)]

y_offset = comm.scan(Y // nranks, op=MPI.SUM)
output_box = [(0, y_offset - Y // nranks, 0), (X, y_offset, Z)]

# Create a CUDA stream to use for instantiating, planning, and first execution of a stateful
# distributed Reshape object 'r'.
with cp.cuda.Device(device_id):
    s1 = cp.cuda.Stream()

# Create a stateful Reshape object 'r' on stream s1.
with nvmath.distributed.reshape.Reshape(a, input_box, output_box, stream=s1) as r:
    # Plan the Reshape on stream s1.
    r.plan(stream=s1)

    # Execute the Reshape on stream s1.
    b = r.execute(stream=s1)

    with cp.cuda.Device(device_id):
        # Record an event on s1 for use later.
        e1 = s1.record()

        # Create a new stream on which the new operand c for the second execution will be
        # filled.
        s2 = cp.cuda.Stream()

        # Fill c on s2.
        with s2:
            c = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex128)
            c[:] = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

        # In the following blocks, we will use stream s2 to perform subsequent operations.
        # Note that it's our responsibility as a user to ensure proper ordering, and we want
        # to order `reset_operand` after event e1 corresponding to the execute() call above.
        s2.wait_event(e1)

    # Alternatively, if we want to use stream s1 for subsequent operations (s2 only for
    # operand creation), we need to order `reset_operand` after the event for
    # cupy.random.rand on s2, e.g: e2 = s2.record() s1.wait_event(e2)

    # Set a new operand c on stream s2. Note that operand c is distributed in the same was
    # as operand a.
    r.reset_operand(c, stream=s2)

    # Execute the new Reshape on stream s2.
    d = r.execute(stream=s2)

    # Synchronize s2 at the end
    with cp.cuda.Device(device_id):
        s2.synchronize()

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
nvmath.distributed.free_symmetric_memory(a, b, c, d)
