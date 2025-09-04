# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify options to a distributed Reshape operation.

In this example, we will use a CuPy ndarray as input, and we will look at two equivalent
ways of providing options to control the blocking behavior of the reshape operation.

$ mpiexec -n 4 python example03_options.py
"""

import cupy as cp
import nvmath.distributed

# Initialize nvmath.distributed.
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
nvmath.distributed.initialize(device_id, comm)

# The problem consists of a global 3-D array of size (64, 256, 128), that is
# initially partitioned on the X axis across processes.
X, Y, Z = (64, 256, 128)
shape = X // nranks, Y, Z

# The distributed reshape implementation uses the NVSHMEM PGAS model for GPU-GPU transfers,
# which requires GPU operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.float64)
# a is a cupy ndarray and can be operated on using in-place cupy operations.
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float64)

# We're going to redistribute the operand so that it is partitioned on the Y axis.
x_offset = comm.scan(X // nranks, op=MPI.SUM)
input_box = [(x_offset - X // nranks, 0, 0), (x_offset, Y, Z)]

y_offset = comm.scan(Y // nranks, op=MPI.SUM)
output_box = [(0, y_offset - Y // nranks, 0), (X, y_offset, Z)]

# Execute the Reshape.

# Alternative #1 for specifying options, using dataclass.
options = nvmath.distributed.reshape.ReshapeOptions(blocking=True)
b = nvmath.distributed.reshape.reshape(a, input_box, output_box, options=options)

# The result is ready because the above operation is blocking.

# Alternative #2 for specifying options, using dict. The two alternatives are entirely
# equivalent.
c = nvmath.distributed.reshape.reshape(a, input_box, output_box, options={"blocking": "auto"})

# Because the input operand is in GPU memory, the "auto" behavior is to not block.
# Therefore, we have to synchronize before we can access the result.
with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
nvmath.distributed.free_symmetric_memory(a, b, c)
