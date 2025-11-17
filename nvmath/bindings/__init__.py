# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# type: ignore

from nvmath.bindings import cublas
from nvmath.bindings import cublasLt
from nvmath.bindings import cudss
from nvmath.bindings import cufft
from nvmath.bindings import curand
from nvmath.bindings import cusolver
from nvmath.bindings import cusolverDn
from nvmath.bindings import cusparse

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

try:
    # NCCL is Linux-only.
    from nvmath.bindings import nccl
except ImportError:
    nccl = None

try:
    # cublasMp is Linux-only.
    from nvmath.bindings import cublasMp
except ImportError:
    cublasMp = None

try:
    # cutensor binding is Linux-only for nvmath-python beta7.0.
    from nvmath.bindings import cutensor
except ImportError:
    cutensor = None

__all__ = [
    "cublas",
    "cublasLt",
    "cublasMp",
    "cudss",
    "cufft",
    "cufftMp",
    "curand",
    "cusolver",
    "cusolverDn",
    "cusparse",
    "cutensor",
    "nccl",
    "nvpl",
    "nvshmem",
]
