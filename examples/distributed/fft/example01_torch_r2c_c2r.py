# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example describes how to perform FFT on PyTorch tensors using function-form FFT APIs
for R2C and C2R transformations, using the default cuFFTMp Slab distributions.

$ mpiexec -n 4 python example01_torch_r2c_c2r.py
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

if nranks > 8:
    raise RuntimeError("This example requires <= 8 processes")

# The global *real* 3-D FFT size is (16, 16, 17).
# In this example, the input data is distributed across processes according to
# the cuFFTMp Slab distribution on the X axis.
shape = 16 // nranks, 16, 17

# For R2C and C2R, cuFFTMp requires the operand's underlying buffer to be padded and/or
# the operand to have specific strides. Also note that, because the distributed FFT
# is in-place, the operand's buffer must be large enough to hold the FFT output (which
# will have different dtype and potentially shape). The following helper allocates an
# operand with the required characteristics for the specified distributed FFT. For this
# example, we'll allocate the operand on the CPU (note that the operand's memory space
# -CPU or CUDA- can be specified).
a = nvmath.distributed.fft.allocate_operand(
    shape,
    torch,
    input_dtype=torch.float32,
    distribution=nvmath.distributed.fft.Slab.X,
    memory_space="cpu",  # allocate torch tensor on CPU
    fft_type="R2C",
)
# a is a torch tensor and can be operated on using in-place torch operations.
a[:] = torch.rand(shape, dtype=torch.float32)

# R2C (forward) FFT.
# In this example, the R2C operand is distributed according to Slab.X distribution.
# With reshape=False, the FFT result will be distributed according to Slab.Y distribution.
b = nvmath.distributed.fft.rfft(a, distribution=nvmath.distributed.fft.Slab.X, options={"reshape": False})

# Distributed FFT performs computations in-place. The result is stored in the same
# buffer as tensor a. Note, however, that tensor b has a different dtype and shape
# (because the output has complex dtype and Slab.Y distribution).
if rank == 0:
    print(f"Shape of a on rank {rank} is {a.shape}, dtype is {a.dtype}")
    print(f"Shape of b on rank {rank} is {b.shape}, dtype is {b.dtype}")

# C2R (inverse) FFT.
# Recall from previous transform that the inverse FFT operand is distributed according to
# Slab.Y. With reshape=False, the C2R result will be distributed according to
# Slab.X distribution.
# Note that to transform back to the original shape of the real operand (which has odd last
# axis length), we use the last_axis_parity="odd" option.
c = nvmath.distributed.fft.irfft(
    b, distribution=nvmath.distributed.fft.Slab.Y, options={"reshape": False, "last_axis_parity": "odd"}
)

# The shape of tensor c is the same as tensor a (due to Slab.X distribution). Once again,
# note that a, b and c are sharing the same memory buffer (distributed FFT operations are
# in-place).
if rank == 0:
    print(f"Shape of c on rank {rank} is {c.shape}, dtype is {c.dtype}")

if rank == 0:
    print(f"Input type = {type(a)}, dtype = {a.dtype}, device = {a.device}, data_ptr = {a.data_ptr()}")
    print(f"FFT output type = {type(b)}, dtype = {b.dtype}, device = {b.device}, data_ptr = {b.data_ptr()}")
    print(f"IFFT output type = {type(c)}, dtype = {c.dtype}, device = {c.device}, data_ptr = {c.data_ptr()}")
