# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

try:
    import mpi4py  # noqa: F401
except ImportError as e:
    # TODO: point to documentation with ways to install mpi4py
    raise ImportError("nvmath.distributed requires mpi4py for bootstrapping. See [LINK] for installation guide.") from e

import atexit
import re
from dataclasses import dataclass
from threading import Lock

from ._internal import nvshmem
from ._utils import allocate_symmetric_memory, free_symmetric_memory

from nvmath.distributed import fft
from nvmath.distributed import reshape

__all__ = [
    "initialize",
    "finalize",
    "get_context",
    "allocate_symmetric_memory",
    "free_symmetric_memory",
    "fft",
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
        device_id: CUDA device ID associated with the distributed runtime.
        communicator: MPI communicator of participating processes.
    """

    device_id: int
    communicator: mpi4py.MPI.Comm


def initialize(device_id: int, communicator: mpi4py.MPI.Comm | None = None) -> None:
    """Initialize nvmath.distributed. This is required before any distributed operations can
    be performed. **Note that this is a collective operation and must be called by all
    processes.**

    Args:
        device_id: CUDA device ID to associate with the nvmath.distributed runtime.
        communicator: MPI communicator specifying the participating processes. If None, will
                      use MPI.COMM_WORLD.
    """
    if not isinstance(device_id, int):
        raise TypeError(
            "The device ID used to initialize the nvmath.distributed module "
            f"must be an integer. The provided device ID is {device_id}."
        )

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

        nvshmem.initialize(device_id, communicator)

        _ctx = DistributedContext(device_id=device_id, communicator=communicator)


def finalize() -> None:
    """Finalize nvmath.distributed runtime. **Note that this is a collective operation and
    must be called by all processes.**"""
    global _ctx
    with _initialize_mutex:
        if _ctx is None:
            return
        nvshmem.finalize(_ctx.device_id)

        _ctx = None


def get_context() -> DistributedContext | None:
    """Return the distributed runtime's context or None if not initialized."""
    return _ctx
