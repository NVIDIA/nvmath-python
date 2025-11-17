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


a = cp.random.rand(4, 4, 12, 12)
b = cp.random.rand(12, 12, 8, 8)

c = cp.random.rand(4, 4, 8, 8)

# result[i,j,m,n] = \sum_{k,l} a[i,j,k,l] * b[k,l,m,n] + c[i,j,m,n]
result = nvmath.tensor.binary_contraction(
    "ijkl,klmn->ijmn", a, b, c=c, beta=1, options={"allocator": RawCUDAMemoryManager(a.device.id)}
)

assert cp.allclose(result, cp.einsum("ijkl,klmn->ijmn", a, b) + c)

print(f"Input type = {type(a), type(b), type(c)}, contraction result type = {type(result)}")
