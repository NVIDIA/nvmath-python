# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows the use of the allocate_operand() helper to allocate operands when
data does not evenly divide across processes. Here the helper is used to allocate NumPy
ndarrays, but it can be used to allocate operands for any supported package, on CPU and GPU.

The NumPy ndarrays reside in CPU memory, and are copied transparently to GPU
symmetric memory to process them with cuFFTMp.

The input as well as the result from the FFT operations are NumPy ndarrays, resulting
in effortless interoperability between nvmath-python and NumPy.

$ mpiexec -n 4 python example01_numpy_uneven_4p.py
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

# The global 3-D FFT size is (35, 32, 32), running on 4 processes.
# In this example, the input data is distributed across processes according to the cuFFTMp
# Slab distribution on the X axis. Note that data doesn't evenly divide across the four
# processes.
if rank < 3:
    shape = 9, 32, 32
else:
    shape = 8, 32, 32

# When data doesn't evenly divide across processes, it's recommended to use the
# nvmath.distributed.fft.allocate_operand() helper to guarantee that the allocated
# buffer is large enough to accommodate the result on every process (accounting for
# both input and output distribution).
a = nvmath.distributed.fft.allocate_operand(
    shape,  # local shape
    np,
    input_dtype=np.complex128,
    distribution=nvmath.distributed.fft.Slab.X,
    fft_type="C2C",
)

# a is a numpy array and can be operated on using in-place numpy operations.
a[:] = np.random.rand(*shape) + 1j * np.random.rand(*shape)

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
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"FFT output type = {type(b)}, device = {b.device}")
    print(f"IFFT output type = {type(c)}, device = {c.device}")
