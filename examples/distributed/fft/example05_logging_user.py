# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of the a user-provided logger with distributed FFT.

$ mpiexec -n 4 python example05_logging_user.py
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

# The global 3-D FFT size is (512, 512, 256).
# In this example, the input data is distributed across processes according to
# the cuFFTMp Slab distribution on the X axis.
shape = 512 // nranks, 512, 256

# Create and configure a user logger.
# Any of the features provided by the logging module can be used.
logger = logging.getLogger("userlogger")
logging.getLogger().setLevel(logging.NOTSET)

# Create a file handler for the logger (one file per process) and set level.
handler = logging.FileHandler(f"example05_{rank}.log")
handler.setLevel(logging.DEBUG)

# Create a formatter and associate with handler.
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")
handler.setFormatter(formatter)

# Associate handler with logger, resulting in a logger with the desired level, format, and
# console output.
logger.addHandler(handler)

# cuFFTMp uses the NVSHMEM PGAS model for distributed computation, which requires GPU
# operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex128)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

# Specify the custom logger in the FFT options.
o = nvmath.distributed.fft.FFTOptions(logger=logger)

# Specify the options to the FFT operation.
b = nvmath.distributed.fft.fft(a, nvmath.distributed.fft.Slab.X, options=o)

if rank == 0:
    print("---")

# Synchronize the default stream
with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()
if rank == 0:
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"FFT output type = {type(b)}, device = {b.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is responsible
# for freeing any that they own (this deallocation is a collective operation that
# must be called by all processes at the same point in the execution).
# All cuFFTMp operations are inplace (a and b share the same memory buffer), so we
# take care to only free the buffer once.
nvmath.distributed.free_symmetric_memory(a)
