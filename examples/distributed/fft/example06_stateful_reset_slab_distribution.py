# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to reuse the stateful API to perform FFT operations on operands
with compatible Slab distribution.

We will perform a forward and an inverse FFT operation to demonstrate how to recover
the original input operand.

$ mpiexec -n 4 python example06_stateful_reset_slab_distribution.py
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
    b = f.execute(direction=nvmath.distributed.fft.FFTDirection.FORWARD)

    # Reset the operand to the values in the frequency domain.
    # Note that because the FFT object is configured with reshape=False, the distribution of
    # operand b is Slab.Y
    # IMPORTANT: When resetting the operand, FFT expects a distribution that is compatible
    # with the distribution that was set when the FFT was planned. In this case, since the
    # original distribution was Slab.X, the reset operand is expected to have either a
    # Slab.X or Slab.Y distribution based on the same global shape, in this
    # case (512, 512, 512).
    f.reset_operand(b, distribution=nvmath.distributed.fft.Slab.Y)

    # Execute the new inverse FFT.
    # The distribution of operand c will be Slab.X
    c = f.execute(direction=nvmath.distributed.fft.FFTDirection.INVERSE)

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
