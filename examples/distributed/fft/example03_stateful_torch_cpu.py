# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful class-form FFT APIs with Torch tensors on CPU.

Torch tensors residing in CPU memory are copied transparently to GPU symmetric memory for
processing with cuFFTMp.

The input as well as the result from the FFT operations are Torch tensors on CPU.

$ mpiexec -n 4 python example03_stateful_torch_cpu.py
"""

import torch
from mpi4py import MPI

import nvmath.distributed

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % torch.cuda.device_count()
nvmath.distributed.initialize(device_id, comm)

# The global 3-D FFT size is (512, 512, 512).
# In this example, the input data is distributed across processes according to
# the cuFFTMp Slab distribution on the Y axis.
shape = 512, 512 // nranks, 512

a = torch.ones(shape, dtype=torch.complex64)  # cpu tensor

# Create a stateful FFT object 'f'.
with nvmath.distributed.fft.FFT(a, nvmath.distributed.fft.Slab.Y) as f:
    # Plan the FFT.
    f.plan()

    # Execute the FFT.
    b = f.execute()

    if rank == 0:
        print(f"Input type = {type(a)}, FFT output type = {type(b)}")
