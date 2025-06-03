# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example on using multiple CUDA streams in distributed FFT APIs.

$ mpiexec -n 4 python example07_streams.py
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
    b = f.execute(stream=s1)

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
    f.reset_operand(c, nvmath.distributed.fft.Slab.X, stream=s2)

    # Execute the new FFT on stream s2.
    d = f.execute(stream=s2)

    # Synchronize s2 at the end
    with cp.cuda.Device(device_id):
        s2.synchronize()

# GPU operands on the symmetric heap are not garbage-collected and the user is responsible
# for freeing any that they own (this deallocation is a collective operation that
# must be called by all processes at the same point in the execution).
# All cuFFTMp operations are inplace, so we take care to only free the buffer once.
nvmath.distributed.free_symmetric_memory(a, c)
