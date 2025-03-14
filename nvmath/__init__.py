# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

from nvmath import bindings
from nvmath._utils import ComputeType
from nvmath._utils import CudaDataType
from nvmath._utils import LibraryPropertyType

from nvmath import fft, linalg
from nvmath.memory import BaseCUDAMemoryManager, MemoryPointer


# Attempt to preload libraries.  Fail silently if preload fails.
def _force_lib_load():
    from nvmath._utils import module_init_force_cupy_lib_load

    module_init_force_cupy_lib_load()


_force_lib_load()

__all__ = [
    "BaseCUDAMemoryManager",
    "bindings",
    "ComputeType",
    "CudaDataType",
    "fft",
    "LibraryPropertyType",
    "linalg",
    "MemoryPointer",
]

__version__ = importlib.metadata.version("nvmath-python")
