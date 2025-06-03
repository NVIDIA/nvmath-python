# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of function-form distributed FFT APIs with NumPy ndarrays,
using the default cuFFTMp Slab distributions.

The NumPy ndarrays reside in CPU memory, and are copied transparently to GPU
symmetric memory to process them with cuFFTMp.

The input as well as the result from the FFT operations are NumPy ndarrays, resulting
in effortless interoperability between nvmath-python and NumPy.

$ mpiexec -n 4 python example01_numpy.py
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

# The global 3-D FFT size is (64, 256, 128).
# In this example, the input data is distributed across processes according to
# the cuFFTMp Slab distribution on the Y axis.
shape = 64, 256 // nranks, 128

# NumPy ndarray, on the CPU.
a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

# Forward FFT.
# By default, the reshape option is True, which means that the output of the distributed
# FFT will be re-distributed to retain the same distribution as the input (in this case
# Slab.Y).
b = nvmath.distributed.fft.fft(a, nvmath.distributed.fft.Slab.Y)

if rank == 0:
    # Note the same shape of a and b (they are both using the same distribution).
    print(f"Shape of a on rank {rank} is {a.shape}")
    print(f"Shape of b on rank {rank} is {b.shape}")

    print(f"Input type = {type(a)}, FFT output type = {type(b)}")
