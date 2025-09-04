# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to explicitly control when to synchronize the GPUs on
symmetric memory changes. This option is for advanced users who choose to manage the
synchronization on their own using the appropriate NVSHMEM API, or who know when GPUs
are synchronized on the input operand or not.

$ mpiexec -n 4 python example08_sync_symmetric_memory.py
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

# The global 3-D FFT size is (512, 512, 512).
# In this example, the input data is distributed across processes according to
# the cuFFTMp Slab distribution on the X axis.
shape = 512 // nranks, 512, 512

# cuFFTMp uses the NVSHMEM PGAS model for distributed computation, which requires GPU
# operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex64)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

# Create a stateful FFT object 'f'.
with nvmath.distributed.fft.FFT(a, distribution=nvmath.distributed.fft.Slab.X, options={"reshape": False}) as f:
    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    # Before the FFT executes, the local changes to the input operand must be visible
    # on all GPUs. This is achieved by issuing a NVSHMEM sync_all operation on the stream
    # on which the operand is prepared. nvmath-python's distributed FFT by default
    # issues a synchronization on the execute stream. The same behavior is achieved
    # with sync_symmetric_memory=True
    b = f.execute(direction=nvmath.distributed.fft.FFTDirection.FORWARD, sync_symmetric_memory=True)

    # Reset the operand to the values in the frequency domain.
    # Note that because the FFT object is configured with reshape=False, the
    # distribution of operand b is Slab.Y
    f.reset_operand(b, distribution=nvmath.distributed.fft.Slab.Y)

    # Execute the new inverse FFT.
    # After cuFFTMp performs a transform, it issues a symmetric memory synchronization
    # on the execute stream. As such, there is no need to issue another one here (we disable
    # it with sync_symmetric_memory=False)
    c = f.execute(direction=nvmath.distributed.fft.FFTDirection.INVERSE, sync_symmetric_memory=False)

    # Synchronize the default stream
    with cp.cuda.Device(device_id):
        cp.cuda.get_current_stream().synchronize()
    if rank == 0:
        print(f"Input type = {type(a)}, device = {a.device}")
        print(f"FFT output type = {type(b)}, device = {b.device}")
        print(f"IFFT output type = {type(c)}, device = {c.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
# All cuFFTMp operations are inplace (a, b, and c share the same memory buffer), so we
# take care to only free the buffer once.
nvmath.distributed.free_symmetric_memory(a)
