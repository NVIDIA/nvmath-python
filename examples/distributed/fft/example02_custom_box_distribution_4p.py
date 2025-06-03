# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of function-form distributed FFT APIs using a custom
box distribution based on pencil decomposition, using NumPy ndarrays.

The NumPy ndarrays reside in CPU memory, and are copied transparently to GPU
symmetric memory to process them with cuFFTMp.

The input as well as the result from the FFT operations are NumPy ndarrays, resulting
in effortless interoperability between nvmath-python and NumPy.

$ mpiexec -n 4 python example02_custom_box_distribution.py
"""

import numpy as np
import cuda.core.experimental
from mpi4py import MPI

import nvmath.distributed

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cuda.core.experimental.system.num_devices
nvmath.distributed.initialize(device_id, comm)

if nranks != 4:
    raise RuntimeError("This example requires 4 processes")

# The global 3-D FFT size is (64, 256, 128).
# In this example, the input data is distributed across 4 processes
# using a custom pencil distribution.
X, Y, Z = (64, 256, 128)
shape = X // 2, Y // 2, Z  # pencil decomposition on X and Y axes

# NumPy ndarray, on the CPU.
a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

# Forward FFT.
if rank == 0:
    input_box = [(0, 0, 0), (32, 128, 128)]
elif rank == 1:
    input_box = [(0, 128, 0), (32, 256, 128)]
elif rank == 2:
    input_box = [(32, 0, 0), (64, 128, 128)]
else:
    input_box = [(32, 128, 0), (64, 256, 128)]
# Use the same pencil distribution for the output.
output_box = input_box
b = nvmath.distributed.fft.fft(a, [input_box, output_box])

if rank == 0:
    # Note the same shape of a and b (they are both using the same distribution).
    print(f"Shape of a on rank {rank} is {a.shape}")
    print(f"Shape of b on rank {rank} is {b.shape}")

    print(f"Input type = {type(a)}, FFT output type = {type(b)}")
