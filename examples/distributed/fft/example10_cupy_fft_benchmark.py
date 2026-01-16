# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
A simple distributed FFT example benchmark using the default Slab distribution.

$ mpiexec -n 4 python example10_cupy_fft_benchmark.py
"""

import cupy as cp
import cupyx
import numpy as np

try:
    from cuda.core import system
except ImportError:
    from cuda.core.experimental import system
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import Slab

try:
    num_devices = system.get_num_devices()
except AttributeError:
    num_devices = system.num_devices
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % num_devices
nvmath.distributed.initialize(device_id, comm, backends=["nvshmem"])

# The global 3-D FFT size is (N, N, N)
N = 512
dtype = cp.complex64
shape = N // nranks, N, N

cp.cuda.runtime.setDevice(device_id)

# Create local FFT operand.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=dtype)
a[:] = cp.random.rand(*shape, dtype=cp.float32) + 1j * cp.random.rand(*shape, dtype=cp.float32)

print(f"[{rank}] The local operand shape = {a.shape}, with data type {dtype} running on {nranks} processes.")

# Create the distributed FFT op, plan, and benchmark.
with nvmath.distributed.fft.FFT(a, distribution=Slab.X, options={"reshape": False}) as fftobj:
    fftobj.plan()
    b = cupyx.profiler.benchmark(fftobj.execute, n_repeat=10)
    print(f"[{rank}] {b}")
    if rank == 0:
        median = np.median(b.gpu_times)
        print(f"Rank 0: {b}")
        print(f"Rank 0 median GPU time: {median * 1000:0.2f} ms.")

nvmath.distributed.free_symmetric_memory(a)
