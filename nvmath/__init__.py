# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata


# Attempt to preload libraries.  Fail silently if preload fails.
def _force_lib_load():
    from nvmath._utils import module_init_force_cupy_lib_load

    module_init_force_cupy_lib_load()


_force_lib_load()

from nvmath import (  # noqa: E402
    bindings,  # noqa: E402
    fft,
    linalg,
    sparse,
    tensor,
)
from nvmath._utils import (  # noqa: E402
    ComputeType,
    CudaDataType,
    LibraryPropertyType,
)
from nvmath.memory import BaseCUDAMemoryManager, BaseCUDAMemoryManagerAsync, MemoryPointer  # noqa: E402

__all__ = [
    "BaseCUDAMemoryManager",
    "BaseCUDAMemoryManagerAsync",
    "bindings",
    "ComputeType",
    "CudaDataType",
    "fft",
    "LibraryPropertyType",
    "linalg",
    "MemoryPointer",
    "sparse",
    "tensor",
]

__version__ = importlib.metadata.version("nvmath-python")
