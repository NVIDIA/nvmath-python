# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "initialize",
    "finalize",
    "is_initialized",
    "nvshmem_empty_dlpack",
    "free",
    "NvshmemMemoryManager",
    "NvshmemNDBufferAllocator",
]

import atexit
import logging

import numpy as np

try:
    from cuda.core import Buffer, Device, MemoryResource
except ImportError:
    from cuda.core.experimental import Buffer, Device, MemoryResource

from nvmath import memory
from nvmath.bindings import nvshmem  # type: ignore
from nvmath.distributed.process_group import ProcessGroup, ReductionOp
from nvmath.internal.utils import device_ctx

# Indicates if this module has initialized NVSHMEM
_nvshmem_initialized_here = False

_atexit_registered = False
_exiting = False


def initialize(device_id: int, process_group: ProcessGroup) -> None:
    """Initialize NVSHMEM runtime if not initialized, otherwise do nothing.
    NOTE: device_id must be set as the current device, and a CUDA context must have
    been created for this device.
    """

    global _nvshmem_initialized_here

    rank = process_group.rank
    nranks = process_group.nranks

    status = nvshmem.init_status()
    if status > nvshmem.STATUS_IS_BOOTSTRAPPED:
        # NVSHMEM is already initialized
        # NOTE: We assume that the user has passed the same communicator and device_id
        # on which NVSHMEM was initialized (we have no way of checking here).
        assert nvshmem.n_pes() == nranks
        return
    elif status == nvshmem.STATUS_IS_BOOTSTRAPPED:
        # A value of 0 indicates an initialization that is similar to when nvshmem_init
        # is used.
        nvshmem.hostlib_init_attr(0, 0)
        assert nvshmem.n_pes() == nranks
        # Sanity check, can eventually remove
        assert nvshmem.init_status() > nvshmem.STATUS_IS_BOOTSTRAPPED
        _nvshmem_initialized_here = True
        return

    attr = nvshmem.InitAttr()
    unique_id = nvshmem.UniqueId()
    # PE 0 queries the unique ID
    if rank == 0:
        nvshmem.get_uniqueid(unique_id.ptr)
    # PE 0 broadcasts the unique ID
    process_group.broadcast_buffer(unique_id._data.view(np.int8), root=0)  # type: ignore[attr-defined]
    nvshmem.set_attr_uniqueid_args(rank, nranks, unique_id.ptr, attr.ptr)
    nvshmem.hostlib_init_attr(nvshmem.Flags.INIT_WITH_UNIQUEID, attr.ptr)

    # sanity check
    assert nvshmem.init_status() > nvshmem.STATUS_IS_BOOTSTRAPPED
    _nvshmem_initialized_here = True
    _register_atexit_maybe()


def _register_atexit_maybe() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_detect_exit)
        _atexit_registered = True


def _detect_exit() -> None:
    global _exiting
    _exiting = True


def finalize(device_id: int) -> None:
    """Finalize NVSHMEM runtime if initialized"""
    global _nvshmem_initialized_here
    if not _nvshmem_initialized_here or (nvshmem.init_status() < nvshmem.STATUS_IS_INITIALIZED):
        return

    with device_ctx(device_id):
        nvshmem.hostlib_finalize()
    _nvshmem_initialized_here = False


def is_initialized() -> bool:
    return nvshmem.init_status() >= nvshmem.STATUS_IS_INITIALIZED


def _check_initialized():
    if nvshmem.init_status() < nvshmem.STATUS_IS_INITIALIZED:
        raise RuntimeError("NVSHMEM is not initialized. Please initialize nvmath.distributed with NVSHMEM backend")
    _register_atexit_maybe()


# Keeps track of memory allocated with nvshmem_empty_dlpack. This is used to report memory
# leaks and double free errors. This functionality might be handled by nvshmem4py in the
# future. More precisely, how it works is that when a buffer allocated via
# nvshmem_empty_dlpack is garbage-collected and found to not have been freed explicitly
# with the `free` function in this module, we report it.
_resource_registry = {}


class _NvshmemResource(MemoryResource):
    def __init__(self, device):
        self.device = device

    def allocate(self, size, stream=None) -> Buffer:
        # NOTE: setting the device is left to the caller
        ptr = nvshmem.malloc(size)
        if ptr == 0 and size != 0:
            raise MemoryError("nvshmem_malloc returned NULL")
        self.freed = False
        return Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None, manual=False):
        # NOTE: setting the device is left to the caller
        if not manual:
            if self.freed:
                return  # this is fine
            else:
                # We can't call nvshmem_free when deallocate is triggered by the GC, since
                # the GC has non-deterministic behavior and nvshmem_free is a collective
                # call.
                if not _exiting:
                    logging.error("Symmetric heap memory needs to be deallocated explicitly")
                else:
                    logging.error(
                        "Symmetric heap memory was not deallocated explicitly (you may have "
                        "forgotten to clean up before exit, or an unrelated exception "
                        "crashed the program)"
                    )
                return
        if self.freed:
            raise RuntimeError("This memory resource was already deallocated")
        nvshmem.free(ptr)
        self.freed = True

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return self.device.device_id


def nvshmem_empty_dlpack(size, device_id, process_group, make_symmetric=False, skip_symmetric_check=False, logger=None):
    """Return uninitialized DLPack buffer of given size in bytes, allocated using
    nvshmem_malloc (which makes this a *collective* call). Note that the DLPack
    buffer currently does not include any shape, dtype, or stride information.

    IMPORTANT: device_id must be the one with which NVSHMEM was initialized, and
    setting the device is left to the caller.
    """

    _check_initialized()

    global _resource_registry

    logger = logger if logger is not None else logging.getLogger()

    if make_symmetric and skip_symmetric_check:
        raise ValueError("skip_symmetric_check is incompatible with make_symmetric=True")

    if not skip_symmetric_check:
        max_size = np.array([-size, size], dtype=np.int64)
        process_group.allreduce_buffer(max_size, op=ReductionOp.MAX)
        if -max_size[0] != max_size[1]:
            # The buffer size is not the same on all processes.
            if make_symmetric:
                logger.info(
                    "Symmetric memory allocator: the buffer will be padded on some processes to "
                    f"satisfy symmetric requirement (make_symmetric=True), size={size} max_size={max_size[1]}."
                )
            else:
                raise ValueError(
                    "The buffer size for symmetric memory allocation is not the same on all processes. "
                    "Consider using make_symmetric=True if you have uneven data distribution."
                )
        else:
            logger.info(f"Symmetric memory allocator: the requested buffer size ({size}) is the same on all processes.")
        # Sizes are equal or make_symmetric=True.
        size = max_size[1]

    mem = _NvshmemResource(Device(device_id))
    mem_buffer = mem.allocate(size)
    pointer = int(mem_buffer.handle)
    assert pointer not in _resource_registry
    _resource_registry[pointer] = (mem, size)
    return mem_buffer


def free(pointer):
    """This is a *collective* call (invokes nvshmem_free), used to free buffers
    allocated with nvshmem_empty_dlpack.
    Setting the device is left to the caller"""

    _check_initialized()

    global _resource_registry

    try:
        resource, size = _resource_registry.pop(pointer)
    except KeyError as e:
        raise RuntimeError(
            "Unknown pointer to free. Possible causes:\n"
            " - This memory was not allocated with nvmath.distributed helpers.\n"
            " - This memory was already freed. Possible causes:\n"
            "     - Free was called multiple times on the same tensor.\n"
            "     - You have multiple tensors sharing the same symmetric memory buffer,\n"
            "       e.g. as a result of inplace operations, or tensor operations that\n"
            "       result in views such as slicing."
        ) from e
    resource.deallocate(pointer, size, manual=True)


class NvshmemMemoryManager(memory.BaseCUDAMemoryManager):
    """
    Nvshmem memory allocator.

    Args:
        device_id: The ID (int) of the device on which memory is to be allocated.
        logger (logging.Logger): Python Logger object.
    """

    def __init__(self, device_id, logger):
        """
        __init__(device_id)
        """
        _check_initialized()
        self.device_id = device_id
        self.logger = logger

    def memalloc(self, size):
        """This is a *collective* call (invokes nvshmem_malloc)"""
        mem = _NvshmemResource(Device(self.device_id))
        mem_buffer = mem.allocate(size)
        pointer = int(mem_buffer.handle)
        assert pointer not in _resource_registry
        _resource_registry[pointer] = (mem, size)

        self.logger.debug(
            f"NvshmemMemoryManager (allocate memory): size = {size}, pointer = {pointer}, device_id = {self.device_id}"
        )

        return SymmetricMemoryPointer(mem_buffer)


class NvshmemNDBufferAllocator:
    __slots__ = ("ctx", "make_symmetric", "skip_symmetric_check")

    def __init__(self, device_id, ctx, make_symmetric, skip_symmetric_check):
        assert ctx.device_id == device_id, (
            "Internal error: attempting to allocate symmetric memory on a device not used "
            "by the NVSHMEM runtime on this process"
        )
        self.ctx = ctx
        self.make_symmetric = make_symmetric
        self.skip_symmetric_check = skip_symmetric_check

    def allocate(self, size, stream, logger=None):
        return nvshmem_empty_dlpack(
            size,
            self.ctx.device_id,
            self.ctx.process_group,
            make_symmetric=self.make_symmetric,
            skip_symmetric_check=self.skip_symmetric_check,
            logger=logger,
        )


class SymmetricMemoryPointer(memory.MemoryPointer):
    def __init__(self, mem_buffer):
        super().__init__(int(mem_buffer.handle), mem_buffer.size, finalizer=None)
        self.mem_buffer = mem_buffer

    def free(self):
        """This is a *collective* call (invokes nvshmem_free)"""
        free(self.device_ptr)
