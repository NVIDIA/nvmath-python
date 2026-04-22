# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

if sys.platform != "linux":
    raise ImportError("nvmath.distributed is only supported on Linux.")

import atexit
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device

if TYPE_CHECKING:
    import mpi4py

from nvmath.distributed import fft, linalg, reshape  # noqa: E402
from nvmath.internal.utils import device_ctx

from . import distribution
from ._internal import nvshmem
from ._utils import allocate_symmetric_memory, free_symmetric_memory
from .process_group import MPIProcessGroup, ProcessGroup, TorchProcessGroup

__all__ = [
    "initialize",
    "finalize",
    "get_context",
    "ProcessGroup",
    "MPIProcessGroup",
    "TorchProcessGroup",
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
    Context of initialized ``nvmath.distributed`` runtime.

    Attributes:
        device_id: CUDA device ID associated with the distributed runtime
            on this process.

        process_group: ``nvmath.distributed`` participating processes.

        nvshmem_available: True if NVSHMEM backend was selected at initialization.

        nccl_comm: nccl4py communicator if NCCL backend was selected at
            initialization, None otherwise.
    """

    device_id: int
    process_group: ProcessGroup
    nvshmem_available: bool
    nccl_comm: Any | None


def initialize(
    device_id: int,
    process_group: ProcessGroup | mpi4py.MPI.Comm,
    backends: Sequence[Literal["nvshmem", "nccl"]],
) -> None:
    """Initialize ``nvmath.distributed`` runtime. This is required before any distributed
    operations can be performed. **Note that this is a collective operation and must be
    called by all processes.**

    If the runtime is already initialized this function will raise an error. If you need
    to reinitialize the runtime (for example with different backends) you have to finalize
    it first.

    Note: NCCL doesn't allow assigning more than one process to the same GPU.

    Args:
        device_id: CUDA device ID to associate with the ``nvmath.distributed`` runtime
            on this process.

        process_group: ProcessGroup (or mpi4py communicator) specifying the participating
            processes. This is used for setup and not for communication during compute.

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
        raise ValueError(f"Need to specify at least one of {valid_backends} communication backends")

    if process_group is None:
        raise ValueError("process_group cannot be None")

    with _initialize_mutex:
        global _atexit_registered, _ctx

        if _ctx is not None:
            raise RuntimeError("nvmath.distributed has already been initialized")

        if not _atexit_registered:
            atexit.register(finalize)
            _atexit_registered = True

        if not isinstance(process_group, ProcessGroup):
            valid_process_group = False
            if "mpi4py" in sys.modules:
                import mpi4py.MPI

                if isinstance(process_group, mpi4py.MPI.Comm):
                    process_group = MPIProcessGroup(process_group)
                    valid_process_group = True
            if not valid_process_group:
                raise TypeError(
                    f"Unrecognized process group type ({process_group}). "
                    "Need nvmath.distributed.ProcessGroup or mpi4py communicator."
                )

        assert isinstance(process_group, ProcessGroup)  # for type-checker
        rank = process_group.rank
        nranks = process_group.nranks

        # Set the device for NVSHMEM and NCCL initialization, but also need to make sure
        # that a CUDA context has been created. We can't rely on `device_ctx` to do it
        # since it's not guaranteed to make a runtime API call.
        old_device = Device()
        device = Device(device_id)
        device.set_current()

        try:
            nvshmem_available = False
            if "nvshmem" in backends:
                nvshmem.initialize(device_id, process_group)
                nvshmem_available = True

            nccl_comm = None
            if "nccl" in backends:
                # Don't want to import this at the global scope yet because
                # `import nccl.core` imports cupy and torch (with nccl4py 0.1.1)
                import nccl.core as nccl  # type: ignore

                # Create NCCL communicator.
                unique_id = nccl.get_unique_id()
                process_group.broadcast_buffer(unique_id.as_ndarray.view(np.int8), root=0)
                nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

            _ctx = DistributedContext(
                device_id=device_id, process_group=process_group, nvshmem_available=nvshmem_available, nccl_comm=nccl_comm
            )
        finally:
            old_device.set_current()


def finalize() -> None:
    """Finalize ``nvmath.distributed`` runtime (this is called automatically at exit
    if the runtime is initialized). **Note that this is a collective operation and
    must be called by all processes.**"""
    global _ctx
    with _initialize_mutex:
        if _ctx is None:
            return

        linalg.advanced.matmulmod._grid_cache.clear()

        if _ctx.nccl_comm is not None:
            with device_ctx(_ctx.device_id):
                _ctx.nccl_comm.finalize()
                _ctx.nccl_comm.destroy()

        if _ctx.nvshmem_available:
            nvshmem.finalize(_ctx.device_id)

        _ctx = None


def get_context() -> DistributedContext | None:
    """Return the distributed runtime's context or None if not initialized."""
    return _ctx
