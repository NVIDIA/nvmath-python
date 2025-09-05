# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful class-form distributed FFT APIs with
CuPy ndarrays.

The input as well as the result from the FFT operations are CuPy ndarrays.

$ mpiexec -n 4 python example03_stateful_cupy.py
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
# a is a cupy ndarray and can be operated on using in-place cupy operations.
with cp.cuda.Device(device_id):
    a[:] = cp.ones(shape, dtype=cp.complex64)

# Create a stateful FFT object 'f'.
with nvmath.distributed.fft.FFT(a, distribution=nvmath.distributed.fft.Slab.Y) as f:
    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    b = f.execute()

    # Synchronize the default stream
    with cp.cuda.Device(device_id):
        cp.cuda.get_current_stream().synchronize()
    if rank == 0:
        print(f"Input type = {type(a)}, device = {a.device}")
        print(f"FFT output type = {type(b)}, device = {b.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
# All cuFFTMp operations are inplace (a and b share the same memory buffer), so we
# take care to only free the buffer once.
nvmath.distributed.free_symmetric_memory(a)
