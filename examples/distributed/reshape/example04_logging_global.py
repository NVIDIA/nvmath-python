# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of the global Python logger with distributed Reshape.

$ mpiexec -n 4 python example04_logging_global.py
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

# The global 3-D problem size is (512, 512, 256), initially partitioned on the X axis
# across processes.
X, Y, Z = (64, 256, 128)
shape = X // nranks, Y, Z

# Turn on logging. Here we use the global logger, set the level to "debug", and use a custom
# format for the log. Any of the features provided by the logging module can be used.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# The distributed reshape implementation uses the NVSHMEM PGAS model for GPU-GPU transfers,
# which requires GPU operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex128)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

# Reshape the operand so that it is partitioned on the Y axis.

x_offset = comm.scan(X // nranks, op=MPI.SUM)
input_box = [(x_offset - X // nranks, 0, 0), (x_offset, Y, Z)]

y_offset = comm.scan(Y // nranks, op=MPI.SUM)
output_box = [(0, y_offset - Y // nranks, 0), (X, y_offset, Z)]

b = nvmath.distributed.reshape.reshape(a, input_box, output_box)

# Synchronize the default stream
with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()
if rank == 0:
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"Reshape output type = {type(b)}, device = {b.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
nvmath.distributed.free_symmetric_memory(a, b)
