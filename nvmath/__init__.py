# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata


# Attempt to preload libraries.  Fail silently if preload fails.
def _force_lib_load():
    from nvmath._utils import module_init_force_cupy_lib_load

    module_init_force_cupy_lib_load()


_force_lib_load()

from nvmath import bindings  # noqa: E402
from nvmath._utils import ComputeType  # noqa: E402
from nvmath._utils import CudaDataType  # noqa: E402
from nvmath._utils import LibraryPropertyType  # noqa: E402

from nvmath import fft, linalg, sparse  # noqa: E402
from nvmath.memory import BaseCUDAMemoryManager, MemoryPointer  # noqa: E402

__all__ = [
    "BaseCUDAMemoryManager",
    "bindings",
    "ComputeType",
    "CudaDataType",
    "fft",
    "LibraryPropertyType",
    "linalg",
    "MemoryPointer",
    "sparse",
]

__version__ = importlib.metadata.version("nvmath-python")
