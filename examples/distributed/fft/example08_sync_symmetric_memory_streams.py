# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to explicitly control when to synchronize the GPUs on symmetric
memory changes in the context of multiple memory streams. This option is for advanced users
who choose to manage the synchronization on their own using the appropriate NVSHMEM API, or
who know when GPUs are synchronized on the input operand or not.

$ mpiexec -n 4 python example08_sync_symmetric_memory_streams.py
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

# The global 3-D FFT size is (512, 256, 256).
# In this example, the input data is distributed across processes according to
# the cuFFTMp Slab distribution on the X axis.
shape = 512 // nranks, 256, 256

# cuFFTMp uses the NVSHMEM PGAS model for distributed computation, which requires GPU
# operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex128)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

    # Create a CUDA stream to use for instantiating, planning, and first execution of
    # a stateful distributed FFT object 'f'.
    s1 = cp.cuda.Stream()

# Create a stateful FFT object 'f' on stream s1.
with nvmath.distributed.fft.FFT(a, nvmath.distributed.fft.Slab.X, options={"blocking": "auto"}, stream=s1) as f:
    # Plan the FFT on stream s1.
    f.plan(stream=s1)

    # Execute the FFT on stream s1.
    # Before the FFT executes, the local changes to the input operand must be visible
    # on all GPUs. This is achieved by issuing a NVSHMEM sync_all operation on the stream
    # on which the operand is prepared. nvmath-python's distributed FFT by default
    # issues a synchronization on the execute stream. The same behavior is achieved
    # with sync_symmetric_memory=True
    b = f.execute(direction="forward", stream=s1, sync_symmetric_memory=True)

    # We're using the output of the previous forward transform as input for the
    # inverse transform.
    f.reset_operand(b, nvmath.distributed.fft.Slab.X)

    # Execute the inverse FFT on stream s1.
    # Since cuFFTMp issued a symmetric memory synchronization on stream s1 after
    # the execution of the previous transform, we can avoid the synchronization here.
    c = f.execute(direction="forward", stream=s1, sync_symmetric_memory=False)

    with cp.cuda.Device(device_id):
        # Record an event on s1 for use later.
        e1 = s1.record()

        # Create a new stream on which the new operand c for the second execution will be
        # filled.
        s2 = cp.cuda.Stream()

        # Fill d on s1.
        with s1:
            d = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex128)
            d[:] = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

        # In the following blocks, we will use stream s2 to perform subsequent operations.
        # Note that it's our responsibility as a user to ensure proper ordering, and we
        # want to order `reset_operand` after event e1 corresponding to the execute() call
        # above.
        s2.wait_event(e1)

    # Set a new operand d on stream s2.
    f.reset_operand(d, nvmath.distributed.fft.Slab.X, stream=s2)

    # Execute the new FFT on stream s2.
    # Operand d was filled on stream s1, and the GPUs have not synchronized on these
    # changes. With sync_symmetric_memory=True, the FFT object will issue a symmetric
    # memory synchronization on the execute stream s2, so it is the user's responsibility
    # to ensure proper ordering of s1 and s2 (i.e. execute on s2 needs to happen after the
    # operand is filled on s1).
    d = f.execute(stream=s2, sync_symmetric_memory=True)

    # Synchronize s2 at the end
    with cp.cuda.Device(device_id):
        s2.synchronize()

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
# All cuFFTMp operations are inplace, so we take care to only free the buffer once.
nvmath.distributed.free_symmetric_memory(a, d)
