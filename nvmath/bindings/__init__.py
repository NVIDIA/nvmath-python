# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# type: ignore

from nvmath.bindings import cublas
from nvmath.bindings import cudss
from nvmath.bindings import cufft
from nvmath.bindings import curand
from nvmath.bindings import cusolver
from nvmath.bindings import cusolverDn
from nvmath.bindings import cusparse

try:
    # nvpl is Linux-only.
    from nvmath.bindings import nvpl
except ImportError:
    nvpl = None

try:
    # cufftMp is Linux-only.
    from nvmath.bindings import cufftMp
except ImportError:
    cufftMp = None

try:
    # nvshmem is Linux-only.
    from nvmath.bindings import nvshmem
except ImportError:
    nvshmem = None

__all__ = [
    "cublas",
    "cudss",
    "cufft",
    "cufftMp",
    "curand",
    "cusolver",
    "cusolverDn",
    "cusparse",
    "nvpl",
    "nvshmem",
]
