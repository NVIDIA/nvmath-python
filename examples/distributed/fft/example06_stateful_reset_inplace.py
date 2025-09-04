# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to reuse the stateful API to perform FFT operations
by resetting operands inplace. It's important to note that operands reset in this
manner have to preserve their distribution.

$ mpiexec -n 4 python example06_stateful_reset_inplace.py
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
# the cuFFTMp Slab distribution on the Y axis.
shape = 512, 512 // nranks, 512

# cuFFTMp uses the NVSHMEM PGAS model for distributed computation, which requires GPU
# operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex64)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

# Create a stateful FFT object 'f'.
with nvmath.distributed.fft.FFT(a, distribution=nvmath.distributed.fft.Slab.Y) as f:
    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    # The distribution of operand b will be Slab.Y because reshape=True.
    b = f.execute()

    # Reset the operand inplace. Note that this implies maintaining the same
    # input operand distribution (Slab.Y in this example).
    with cp.cuda.Device(device_id):
        a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

    # Execute a new forward FFT with the modified operand.
    # The distribution of operand c will be Slab.Y because reshape=True.
    c = f.execute()

    # Synchronize the default stream
    with cp.cuda.Device(device_id):
        cp.cuda.get_current_stream().synchronize()
    if rank == 0:
        print(f"Input type = {type(a)}, device = {a.device}")
        print(f"FFT output type = {type(b)}, device = {b.device}")
        print(f"IFFT output type = {type(c)}, device = {c.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is responsible
# for freeing any that they own (this deallocation is a collective operation that
# must be called by all processes at the same point in the execution).
# All cuFFTMp operations are inplace (a, b, and c share the same memory buffer), so we
# take care to only free the buffer once.
nvmath.distributed.free_symmetric_memory(a)
