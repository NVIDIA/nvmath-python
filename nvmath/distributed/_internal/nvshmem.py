# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["initialize", "finalize", "is_initialized", "nvshmem_empty_dlpack", "free", "NvshmemMemoryManager"]

import logging
import numpy as np
import cuda.core.experimental as ccx

from nvmath import memory
from nvmath.bindings import nvshmem  # type: ignore
from nvmath.internal.utils import device_ctx

# Indicates if this module has initialized NVSHMEM
_nvshmem_initialized_here = False


def initialize(device_id: int, mpi_comm) -> None:
    """Initialize NVSHMEM runtime if not initialized, otherwise do nothing."""

    global _nvshmem_initialized_here

    rank = mpi_comm.Get_rank()
    nranks = mpi_comm.Get_size()

    # Here we set the device for NVSHMEM initialization, but we also need to make sure that
    # a CUDA context has been created before initializing NVSHMEM. We can't rely on
    # `device_ctx` to do it since it's not guaranteed to make a runtime API call.
    old_device = ccx.Device()
    ccx.Device(device_id).set_current()

    try:
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
        mpi_comm.Bcast(unique_id._data.view(np.int8), root=0)
        nvshmem.set_attr_uniqueid_args(rank, nranks, unique_id.ptr, attr.ptr)
        nvshmem.hostlib_init_attr(nvshmem.Flags.INIT_WITH_UNIQUEID, attr.ptr)

        # sanity check
        assert nvshmem.init_status() > nvshmem.STATUS_IS_BOOTSTRAPPED
        _nvshmem_initialized_here = True
    finally:
        old_device.set_current()


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
        raise RuntimeError("NVSHMEM is not initialized. Please initialize nvmath.distributed")


# Keeps track of memory allocated with nvshmem_empty_dlpack. This is used to report memory
# leaks and double free errors. This functionality might be handled by nvshmem4py in the
# future. More precisely, how it works is that when a buffer allocated via
# nvshmem_empty_dlpack is garbage-collected and found to not have been freed explicitly
# with the `free` function in this module, we report it.
_resource_registry = {}


class _NvshmemResource(ccx.MemoryResource):
    def __init__(self, device):
        self.device = device

    def allocate(self, size, stream=None) -> ccx.Buffer:
        # NOTE: setting the device is left to the caller
        ptr = nvshmem.malloc(size)
        if ptr == 0 and size != 0:
            raise MemoryError("nvshmem_malloc returned NULL")
        self.freed = False
        return ccx.Buffer.from_handle(ptr=ptr, size=size, mr=self)

    def deallocate(self, ptr, size, stream=None, manual=False):
        # NOTE: setting the device is left to the caller
        if not manual:
            if self.freed:
                return  # this is fine
            else:
                # We can't call nvshmem_free when deallocate is triggered by the GC, since
                # the GC has non-deterministic behavior and nvshmem_free is a collective
                # call.
                raise RuntimeError("Symmetric heap memory needs to be deallocated explicitly")
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


def nvshmem_empty_dlpack(size, device_id, comm, make_symmetric=False, logger=None):
    """Return uninitialized DLPack buffer of given size in bytes, allocated using
    nvshmem_malloc (which makes this a *collective* call). Note that the DLPack
    buffer currently does not include any shape, dtype, or stride information.

    IMPORTANT: device_id must be the one with which NVSHMEM was initialized, and
    setting the device is left to the caller.
    """

    _check_initialized()

    global _resource_registry

    logger = logger if logger is not None else logging.getLogger()

    from mpi4py import MPI

    max_size = np.array([-size, size], dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, max_size, MPI.MAX)
    if -max_size[0] != max_size[1]:
        # The buffer size is not the same on all processes.
        if not make_symmetric:
            raise ValueError(
                "The buffer size for symmetric memory allocation is not the same on all processes. "
                "Consider using make_symmetric=True if you have uneven data distribution."
            )
        else:
            logger.info(
                "Symmetric memory allocator: the buffer will be padded on some processes to "
                f"satisfy symmetric requirement (make_symmetric=True), size={size} max_size={max_size[1]}."
            )
    else:
        logger.info(f"Symmetric memory allocator: the requested buffer size ({size}) is the same on all processes.")
    # Sizes are equal or make_symmetric=True.
    size = max_size[1]

    mem = _NvshmemResource(ccx.Device(device_id))
    mem_buffer = mem.allocate(size)
    pointer = mem_buffer._mnff.ptr
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
    except KeyError:
        raise RuntimeError(
            "Unknown pointer to free. Possible causes:\n"
            " - This memory was not allocated with nvmath.distributed helpers.\n"
            " - This memory was already freed. Possible causes:\n"
            "     - Free was called multiple times on the same tensor.\n"
            "     - You have multiple tensors sharing the same symmetric memory buffer,\n"
            "       e.g. as a result of inplace operations, or tensor operations that\n"
            "       result in views such as slicing."
        )
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
        mem = _NvshmemResource(ccx.Device(self.device_id))
        mem_buffer = mem.allocate(size)
        pointer = mem_buffer._mnff.ptr
        assert pointer not in _resource_registry
        _resource_registry[pointer] = (mem, size)

        self.logger.debug(
            f"NvshmemMemoryManager (allocate memory): size = {size}, pointer = {pointer}, device_id = {self.device_id}"
        )

        return SymmetricMemoryPointer(mem_buffer)


class SymmetricMemoryPointer(memory.MemoryPointer):
    def __init__(self, mem_buffer):
        super().__init__(mem_buffer._mnff.ptr, mem_buffer.size, finalizer=None)
        self.mem_buffer = mem_buffer

    def free(self):
        """This is a *collective* call (invokes nvshmem_free)"""
        free(self.device_ptr)
