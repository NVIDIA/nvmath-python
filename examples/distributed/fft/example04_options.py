# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify options to a distributed FFT operation.

In this example, we will use a NumPy ndarray as input, and we will look at two equivalent
ways of providing options to control the Slab distribution of the output across processes.

The NumPy ndarrays reside in CPU memory, and are copied transparently to GPU
symmetric memory to process them with cuFFTMp.

$ mpiexec -n 4 python example04_options.py
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
# the cuFFTMp Slab distribution on the X axis.
shape = 64 // nranks, 256, 128

# NumPy ndarray, on the CPU.
a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

# When using the cuFFTMp Slab distribution, the default in nvmath-python is to keep
# the same distribution for the result, by reshaping to the original distribution
# (see the documentation for the reshape option in distributed FFTOptions).

# In this example, we'd like to do a forward transform followed by an inverse transform.
# We'd like the input of the forward transform to have Slab.X distribution, and the output
# of the inverse transform to have Slab.X distribution. To do this without any extra
# distributed reshape operations (which incur GPU-GPU communication overhead), we will use
# the reshape=False option.

# Alternative #1 for specifying options, using dataclass.
options = nvmath.distributed.fft.FFTOptions(reshape=False)
b = nvmath.distributed.fft.fft(a, distribution=nvmath.distributed.fft.Slab.X, options=options)
if rank == 0:
    print(f"Does the forward FFT result share the same distribution as the input ? {b.shape == a.shape}")
    print(f"Input type = {type(a)}, FFT output type = {type(b)}")

# Alternative #2 for specifying options, using dict. The two alternatives are entirely
# equivalent.
c = nvmath.distributed.fft.ifft(b, distribution=nvmath.distributed.fft.Slab.Y, options={"reshape": False})
if rank == 0:
    print(f"Does the inverse FFT result share the same distribution as the forward input ? {c.shape == a.shape}")
    print(f"Input type = {type(a)}, FFT output type = {type(b)}")
