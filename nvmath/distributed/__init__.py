# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

try:
    import mpi4py  # noqa: F401
except ImportError as e:
    # TODO: point to documentation with ways to install mpi4py
    raise ImportError("nvmath.distributed requires mpi4py for bootstrapping.") from e

import atexit
import numpy as np
import re
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from ._internal import nvshmem
from ._utils import allocate_symmetric_memory, free_symmetric_memory

from . import distribution

from nvmath.bindings import nccl  # type: ignore
from nvmath.internal.utils import device_ctx

from nvmath.distributed import fft, linalg, reshape  # noqa: E402

__all__ = [
    "initialize",
    "finalize",
    "get_context",
    "allocate_symmetric_memory",
    "free_symmetric_memory",
    "distribution",
    "fft",
    "linalg",
    "reshape",
]

_initialize_mutex = Lock()
_atexit_registered = False
_ctx = None


@dataclass(frozen=True)
class DistributedContext:
    """
    Context of initialized nvmath.distributed runtime.

    Attributes:
        device_id: CUDA device ID associated with the distributed runtime
            on this process.

        communicator: MPI communicator of participating processes.

        nvshmem_available: True if NVSHMEM backend was selected at initialization.

        nccl_comm: pointer to NCCL communicator if NCCL backend was selected at
            initialization, None otherwise.
    """

    device_id: int
    communicator: mpi4py.MPI.Comm
    nvshmem_available: bool
    nccl_comm: int | None


def initialize(
    device_id: int,
    communicator: mpi4py.MPI.Comm,
    backends: Sequence[Literal["nvshmem", "nccl"]],
) -> None:
    """Initialize nvmath.distributed runtime. This is required before any distributed
    operations can be performed. **Note that this is a collective operation and must be
    called by all processes.**

    If the runtime is already initialized this function will raise an error. If you need
    to reinitialize the runtime (for example with different backends) you have to finalize
    it first.

    NCCL doesn't allow assigning more than one process to the same GPU.

    Args:
        device_id: CUDA device ID to associate with the nvmath.distributed runtime on this
            process.

        communicator: MPI communicator specifying the participating processes. If None, will
                      use MPI.COMM_WORLD. MPI is used for setup and not for communication
                      during compute.

        backends: Communication backends to use in distributed computations. Valid values
            are "nvshmem" and "nccl". Note that specific libraries (cuFFTMp, cuBLASMp, ...)
            have specific required backends.
    """
    if not isinstance(device_id, int):
        raise TypeError(
            "The device ID used to initialize the nvmath.distributed module "
            f"must be an integer. The provided device ID is {device_id}."
        )

    valid_backends = ("nvshmem", "nccl")
    for backend in backends:
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got {backend}")

    if len(backends) == 0:
        raise ValueError("Need to specify at least one backend")

    with _initialize_mutex:
        global _atexit_registered, _ctx

        if _ctx is not None:
            raise RuntimeError("nvmath.distributed has already been initialized")

        try:
            # This initializes MPI if it hasn't been initialized yet.
            from mpi4py import MPI
        except RuntimeError as e:
            if re.search(r"[Cc]annot load MPI library", str(e)):
                raise RuntimeError("mpi4py could not find the MPI library. See [LINK] for installation tips") from e
            raise

        if not _atexit_registered:
            # mpi4py's atexit handler is registered during `import MPI`, and to finalize
            # NVSHMEM *before* MPI, we need to make sure that we register our exit handler
            # *after* mpi4py's.
            atexit.register(finalize)
            _atexit_registered = True

        if communicator is None:
            communicator = MPI.COMM_WORLD
        elif not isinstance(communicator, mpi4py.MPI.Comm):
            raise TypeError(
                "The provided communicator object should be of type mpi4py.MPI communicator, but "
                f"got object of type {type(communicator)}."
            )

        rank = communicator.Get_rank()
        nranks = communicator.Get_size()

        nvshmem_available = False
        if "nvshmem" in backends:
            nvshmem.initialize(device_id, communicator)
            nvshmem_available = True

        nccl_comm = None
        if "nccl" in backends:
            # Create NCCL communicator.
            unique_id = nccl.UniqueId()
            if rank == 0:
                nccl.get_unique_id(unique_id.ptr)
            # PE 0 broadcasts the unique ID.
            communicator.Bcast(unique_id._data.view(np.int8), root=0)
            with device_ctx(device_id):
                nccl_comm = nccl.comm_init_rank(nranks, unique_id.ptr, rank)

        _ctx = DistributedContext(
            device_id=device_id, communicator=communicator, nvshmem_available=nvshmem_available, nccl_comm=nccl_comm
        )


def finalize() -> None:
    """Finalize nvmath.distributed runtime (this is called automatically at exit
    if the runtime is initialized). **Note that this is a collective operation and
    must be called by all processes.**"""
    global _ctx
    with _initialize_mutex:
        if _ctx is None:
            return

        linalg.advanced.matmulmod._grid_cache.clear()

        if _ctx.nccl_comm is not None:
            with device_ctx(_ctx.device_id):
                nccl.comm_destroy(_ctx.nccl_comm)

        if _ctx.nvshmem_available:
            nvshmem.finalize(_ctx.device_id)

        _ctx = None


def get_context() -> DistributedContext | None:
    """Return the distributed runtime's context or None if not initialized."""
    return _ctx
