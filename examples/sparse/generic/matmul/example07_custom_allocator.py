# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates the use of a custom memory allocator.

nvmath-python supports multiple sparse formats, frameworks, memory spaces, and
execution spaces. The sparse operand can be provided from SciPy, CuPy, PyTorch
in a variety of supported formats such as BSR, BSC, COO, CSR, CSC, DIA or
as a universal sparse tensor (UST), which supports custom formats in addition to
the standard named formats.
"""

import logging

import numpy as np
import scipy.sparse as sp

try:
    from cuda.core import Device, DeviceMemoryResource
except ImportError:
    from cuda.core.experimental import Device, DeviceMemoryResource

import nvmath
from nvmath.memory import BaseCUDAMemoryManager, MemoryPointer


class CustomMemoryManager(BaseCUDAMemoryManager):
    """
    A simple allocator using the cuda.core DeviceMemoryResource.
    """

    def __init__(self, device_id, logger):
        # Create the device context, if needed.
        Device(device_id).set_current()
        self.mem_resource = DeviceMemoryResource(device_id)
        self.logger = logger

    def memalloc(self, size):
        self.buffer = self.mem_resource.allocate(size)
        device_ptr = self.buffer.handle
        self.logger.info(f"[CUSTOM MEMALLOC] Allocated memory of size {size} bytes using {type(self).__name__}.")

        def create_finalizer():
            def finalizer():
                self.mem_resource.deallocate(device_ptr, size)
                self.logger.info(f"[CUSTOM FREE] Free'd allocated memory using {type(self).__name__}.")

            return finalizer

        return MemoryPointer(device_ptr, size, finalizer=create_finalizer())

    def free(self):
        self.mem_resource.close()
        self.logger.info("Free'd the custom memory manager.")


# Turn on logging to see what's happening under the hood.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

dtype = np.float64
shape = 1024, 1024

# Create a SciPy CSR array.
a = sp.random(*shape, density=0.2, format="csr", dtype=dtype)

# Dense 'b' and 'c', from NumPy.
b = np.ones(shape, dtype=dtype)
c = np.zeros(shape, dtype=dtype)

# Create a custom allocator.
allocator = CustomMemoryManager(device_id=0, logger=logging.getLogger())

# c := a @ b + c
r = nvmath.sparse.matmul(a, b, c, beta=1.0, options={"allocator": allocator})
