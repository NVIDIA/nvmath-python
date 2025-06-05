# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to explicitly control when to synchronize the GPUs on
symmetric memory changes. This option is for advanced users who choose to manage the
synchronization on their own using the appropriate NVSHMEM API, or who know when GPUs
are synchronized on the input operand.

$ mpiexec -n 4 python example07_sync_symmetric_memory.py
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

# The global 3-D problem size is (256, 512, 512), initially partitioned on the Y
# axes across processes.
X, Y, Z = (256, 512, 512)
shape = X, Y // nranks, Z

# The distributed reshape implementation uses the NVSHMEM PGAS model for GPU-GPU transfers,
# which requires GPU operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex64)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

# We're going to execute a forward FFT that expects the input to be partitioned on
# the X axis according to Slab.X distribution, with reshape=False so that the forward
# transform result is distributed according to Slab.Y distribution.

# Operand a is distributed according to Slab.Y distribution, so we need to re-distribute it
# to Slab.X.
y_offset = comm.scan(Y // nranks, op=MPI.SUM)
input_box = [(0, y_offset - Y // nranks, 0), (X, y_offset, Z)]

x_offset = comm.scan(X // nranks, op=MPI.SUM)
output_box = [(x_offset - X // nranks, 0, 0), (x_offset, Y, Z)]

# Execute the Reshape.
# Before the reshape executes, the local changes to the input operand must be visible
# on all GPUs. This is achieved by issuing a NVSHMEM sync_all operation on the stream
# on which the operand is prepared. nvmath-python's distributed Reshape by default
# issues a synchronization on the execute stream. The same behavior is achieved
# with sync_symmetric_memory=True.
b = nvmath.distributed.reshape.reshape(a, input_box, output_box, sync_symmetric_memory=True)

# Now that we have operand b with Slab.X, we can run the forward FFT with Slab.X
# distribution and reshape=False to get the result in Slab.Y distribution.

# Create a stateful FFT object 'f' with Slab.X distribution.
# Note that we could also specify the permuted boxes as distribution,
# i.e. distribution=[output_box, input_box]
with nvmath.distributed.fft.FFT(b, distribution=nvmath.distributed.fft.Slab.X, options={"reshape": False}) as f:
    # Plan the FFT.
    f.plan()

    # After the library performs the reshape, it issues a symmetric memory synchronization
    # on the execute stream. As such, there is no need to issue another one here (we disable
    # it with sync_symmetric_memory=False)
    c = f.execute(sync_symmetric_memory=False)

    # Synchronize the default stream
    with cp.cuda.Device(device_id):
        cp.cuda.get_current_stream().synchronize()
    if rank == 0:
        print(f"Input type = {type(a)}, device = {a.device}")
        print(f"Reshape output type = {type(b)}, device = {b.device}")
        print(f"FFT output type = {type(c)}, device = {c.device}")

        # Note the same shapes of the operands:
        if rank == 0:
            print(f"Shape of a on rank {rank} is {a.shape}")
            print(f"Shape of b on rank {rank} is {b.shape}")
            print(f"Shape of c on rank {rank} is {c.shape}")

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
# Note that distributed FFT operations are inplace (b and c share the same memory buffer),
# so we take care to only free the buffer once.
nvmath.distributed.free_symmetric_memory(a, b)
