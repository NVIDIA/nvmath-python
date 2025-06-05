# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to manage memory resources used by stateful objects. This is useful
when the distributed FFT operation needs a lot of memory and calls to execution method on
a stateful object are interleaved with calls to other operations (including another FFT)
also requiring a lot of memory.

In this example, two FFT operations are performed in a loop in an interleaved manner. We
assume that the available device memory is large enough for only one FFT at a time.

$ mpiexec -n 4 python example09_resource_mgmt.py
"""

import logging

import cupy as cp
from mpi4py import MPI

import nvmath.distributed

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
nvmath.distributed.initialize(device_id, comm)

# The global 3-D FFT size is (256, 512, 512).
# In this example, the input data is distributed across processes according to
# the cuFFTMp Slab distribution on the X axis.
shape = 256 // nranks, 512, 512

# cuFFTMp uses the NVSHMEM PGAS model for distributed computation, which requires GPU
# operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex64)
b = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex64)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)
    b[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

# Turn on logging and set the level to DEBUG to print memory management messages.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Create and prepare two FFT objects.
f1 = nvmath.distributed.fft.FFT(a, nvmath.distributed.fft.Slab.X)
f1.plan()

f2 = nvmath.distributed.fft.FFT(b, nvmath.distributed.fft.Slab.X)
f2.plan()

num_iter = 3
# Use the FFT objects as context managers so that internal library resources are properly
# cleaned up.
with f1, f2:
    for i in range(num_iter):
        if rank == 0:
            print(f"Iteration {i}")
        # Perform the first FFT, and request that the workspace be released at the
        # end of the operation so that there is enough memory for the second one.
        r = f1.execute(release_workspace=True)

        # Update f1's operands for the next iteration.
        if i < num_iter - 1:
            with cp.cuda.Device(device_id):
                a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

        # Perform the second FFT, and request that the workspace be released at the end of
        # the operation so that there is enough memory for the first FFT in the next
        # iteration.
        r = f2.execute(release_workspace=True)

        # Update f2's operands for the next iteration.
        if i < num_iter - 1:
            with cp.cuda.Device(device_id):
                b[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

        # Synchronize the default stream
        with cp.cuda.Device(device_id):
            cp.cuda.get_current_stream().synchronize()
        if rank == 0:
            print(f"Input type = {type(a)}, device = {a.device}")
            print(f"FFT output type = {type(r)}, device = {r.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is responsible
# for freeing any that they own (this deallocation is a collective operation that
# must be called by all processes at the same point in the execution).
nvmath.distributed.free_symmetric_memory(a, b)
