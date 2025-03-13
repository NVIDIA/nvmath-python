# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
An example illustrating user-provided memory allocator.
"""

import cupy as cp

import nvmath
from nvmath.memory import BaseCUDAMemoryManager, MemoryPointer


class RawCUDAMemoryManager(BaseCUDAMemoryManager):
    """
    A simple allocator using cudaMalloc and cudaFree, instead of CuPy's memory pool.
    """

    def __init__(self, device_id):
        self.device_id = device_id

    def memalloc(self, size):
        with cp.cuda.Device(self.device_id):
            device_ptr = cp.cuda.runtime.malloc(size)
        print(f"Allocated memory of size {size} bytes using {type(self).__name__}.")

        def create_finalizer():
            def finalizer():
                cp.cuda.runtime.free(device_ptr)
                print(f"Free'd allocated memory using {type(self).__name__}.")

            return finalizer

        return MemoryPointer(device_ptr, size, finalizer=create_finalizer())


shape = 512, 256, 512
axes = 0, 1

a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

# Forward FFT along (0,1), batched along axis=2 with user-provided memory allocator.
b = nvmath.fft.fft(a, axes=axes, options={"allocator": RawCUDAMemoryManager(a.device.id)})

# Synchronize the default stream
cp.cuda.get_current_stream().synchronize()
print(f"Input type = {type(a)}, device = {a.device}")
print(f"FFT output type = {type(b)}, device = {b.device}")
