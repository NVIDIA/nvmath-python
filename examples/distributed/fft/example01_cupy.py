# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of function-form distributed FFT APIs with CuPy ndarrays,
using the default cuFFTMp Slab distributions.

The input as well as the result from the FFT operations are CuPy ndarrays, resulting
in effortless interoperability between nvmath-python and CuPy.

$ mpiexec -n 4 python example01_cupy.py
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

# The global 3-D FFT size is (512, 256, 512).
# In this example, the input data is distributed across processes according to
# the cuFFTMp Slab distribution on the X axis.
shape = 512 // nranks, 256, 512

# cuFFTMp uses the NVSHMEM PGAS model for distributed computation, which requires GPU
# operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex128)
# a is a cupy ndarray and can be operated on using in-place cupy operations.
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

# Forward FFT.
# In this example, the forward FFT operand is distributed according to Slab.X distribution.
# With reshape=False, the FFT result will be distributed according to Slab.Y distribution.
b = nvmath.distributed.fft.fft(a, distribution=nvmath.distributed.fft.Slab.X, options={"reshape": False})

# Distributed FFT performs computations in-place. The result is stored in the same
# buffer as operand a. Note, however, that operand b has a different shape (due
# to Slab.Y distribution).
if rank == 0:
    print(f"Shape of a on rank {rank} is {a.shape}")
    print(f"Shape of b on rank {rank} is {b.shape}")

# Inverse FFT.
# Recall from previous transform that the inverse FFT operand is distributed according to
# Slab.Y. With reshape=False, the inverse FFT result will be distributed according to
# Slab.X distribution.
c = nvmath.distributed.fft.ifft(b, distribution=nvmath.distributed.fft.Slab.Y, options={"reshape": False})

# The shape of c is the same as a (due to Slab.X distribution). Once again, note that
# a, b and c are sharing the same symmetric memory buffer (distributed FFT operations
# are in-place).
if rank == 0:
    print(f"Shape of c on rank {rank} is {c.shape}")

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
# All cuFFTMp operations are inplace (a, b, and c share the same memory buffer), so
# we take care to only free the buffer once.
nvmath.distributed.free_symmetric_memory(a)
