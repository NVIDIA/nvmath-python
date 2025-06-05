# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to reuse the stateful API to perform FFT operations on operands
with compatible Slab distribution.

We will perform a forward and an inverse FFT operation to demonstrate how to recover the
original input operand.

$ mpiexec -n 4 python example06_stateful_reset_box_distribution.py
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

if nranks != 4:
    raise RuntimeError("This example requires 4 processes")

# The global 3-D FFT size is (128, 256, 128).
# In this example, the input data is distributed across 4 processes
# using a custom pencil distribution.
X, Y, Z = (128, 256, 128)
shape = X // 2, Y // 2, Z  # pencil decomposition on X and Y axes.

# cuFFTMp uses the NVSHMEM PGAS model for distributed computation, which requires GPU
# operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex64)
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

# Input and output boxes for the forward FFT.
# Input distribution is pencil decomposition on X and Y axes.
# Output distribution is pencil decomposition on Y an Z axes.
if rank == 0:
    input_box = ([0, 0, 0], [64, 128, 128])
    output_box = ([0, 0, 0], [128, 128, 64])
elif rank == 1:
    input_box = ([0, 128, 0], [64, 256, 128])
    output_box = ([0, 0, 64], [128, 128, 128])
elif rank == 2:
    input_box = ([64, 0, 0], [128, 128, 128])
    output_box = ([0, 128, 0], [128, 256, 64])
else:
    input_box = ([64, 128, 0], [128, 256, 128])
    output_box = ([0, 128, 64], [128, 256, 128])

# Create a stateful FFT object 'f'.
with nvmath.distributed.fft.FFT(a, distribution=[input_box, output_box]) as f:
    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    b = f.execute(direction=nvmath.distributed.fft.FFTDirection.FORWARD)

    # Reset the operand to the values in the frequency domain. Note that operand b is
    # distributed according to the output box provided above, so we have to swap the boxes.
    # IMPORTANT: When resetting the operand, FFT expects a distribution that is compatible
    # with the distribution that was set when the FFT was planned. For box distribution,
    # this means one of input_box -> output_box or output_box -> input_box, where input_box
    # and output_box are the boxes used at plan time, and based on the same global shape,
    # in this case (128, 256, 128).
    f.reset_operand(b, distribution=[output_box, input_box])

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

# GPU operands on the symmetric heap are not garbage-collected and the user is responsible
# for freeing any that they own (this deallocation is a collective operation that
# must be called by all processes at the same point in the execution).
# All cuFFTMp operations are inplace (a, b, and c share the same memory buffer), so we
# take care to only free the buffer once.
nvmath.distributed.free_symmetric_memory(a)
