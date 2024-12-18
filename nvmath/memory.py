# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""Interface for pluggable memory handlers."""

__all__ = ["BaseCUDAMemoryManager", "MemoryPointer"]

from abc import abstractmethod
from typing import Protocol, runtime_checkable
import weakref

try:
    import cupy as cp
except ImportError:
    cp = None

from nvmath._internal import utils


class MemoryPointer:
    """
    An RAII class for a device memory buffer.

    Args:
        device_ptr: The address of the device memory buffer.
        size: The size of the memory buffer in bytes.
        finalizer: A nullary callable that will be called when the buffer is to be freed.

    .. seealso:: :class:`numba.cuda.MemoryPointer`
    """

    def __init__(self, device_ptr, size, finalizer):
        self.device_ptr = device_ptr
        self.size = size
        if finalizer is not None:
            self._finalizer = weakref.finalize(self, finalizer)
        else:
            self._finalizer = None

    def free(self):
        """
        "Frees" the memory buffer by calling the finalizer.
        """
        if self._finalizer is None:
            return

        if not self._finalizer.alive:
            raise RuntimeError("The buffer has already been freed.")
        self._finalizer()


@runtime_checkable
class BaseCUDAMemoryManager(Protocol):
    """
    Protocol for memory manager plugins.

    .. seealso:: :class:`numba.cuda.BaseCUDAMemoryManager`
    """

    @abstractmethod
    def memalloc(self, size):
        """
        Allocate device memory.

        Args:
            size: The size of the memory buffer in bytes.

        Returns:
            An object that owns the allocated memory and is responsible for releasing it (to
            the OS or a pool). The object must have an attribute named ``device_ptr``,
            ``device_pointer``, or ``ptr`` specifying the pointer to the allocated memory
            buffer. See :class:`MemoryPointer` for an example interface.

        Note:
            Objects of type :class:`numba.cuda.MemoryPointer` as well as
            :class:`cupy.cuda.MemoryPointer` meet the requirements listed above for the
            device memory pointer object.
        """
        raise NotImplementedError


class _RawCUDAMemoryManager(BaseCUDAMemoryManager):
    """
    Raw device memory allocator.

    Args:
        device_id: The ID (int) of the device on which memory is to be allocated.
        logger (logging.Logger): Python Logger object.
    """

    def __init__(self, device_id, logger):
        """
        __init__(device_id)
        """
        self.device_id = device_id
        self.logger = logger

    def memalloc(self, size):
        with utils.device_ctx(self.device_id):
            device_ptr = cp.cuda.runtime.malloc(size)

        self.logger.debug(
            f"_RawCUDAMemoryManager (allocate memory): size = {size}, ptr = {device_ptr}, "
            f"device = {self.device_id}, stream={cp.cuda.get_current_stream()}"
        )

        def create_finalizer():
            def finalizer():
                # Note: With UVA there is no need to switch context to the device the memory
                # belongs to before calling free().
                cp.cuda.runtime.free(device_ptr)
                self.logger.debug(f"_RawCUDAMemoryManager (release memory): ptr = {device_ptr}")

            return finalizer

        return MemoryPointer(device_ptr, size, finalizer=create_finalizer())


class _CupyCUDAMemoryManager(BaseCUDAMemoryManager):
    """
    CuPy device memory allocator.

    Args:
        device_id: The ID (int) of the device on which memory is to be allocated.
        logger (logging.Logger): Python Logger object.
    """

    def __init__(self, device_id, logger):
        """
        __init__(device_id)
        """
        self.device_id = device_id
        self.logger = logger

    def memalloc(self, size):
        with utils.device_ctx(self.device_id):
            cp_mem_ptr = cp.cuda.alloc(size)
            device_ptr = cp_mem_ptr.ptr

        self.logger.debug(
            f"_CupyCUDAMemoryManager (allocate memory): size = {size}, ptr = {device_ptr}, "
            f"device = {self.device_id}, stream={cp.cuda.get_current_stream()}"
        )

        return cp_mem_ptr


class _TorchCUDAMemoryManager(BaseCUDAMemoryManager):
    """
    Torch caching memory allocator.

    Args:
        device_id: The ID (int) of the device on which memory is to be allocated.
        logger (logging.Logger): Python Logger object.
    """

    def __init__(self, device_id, logger):
        """
        __init__(device_id)
        """
        self.device_id = device_id
        self.logger = logger

    def memalloc(self, size):
        from torch.cuda import caching_allocator_alloc, caching_allocator_delete, current_stream

        device_ptr = caching_allocator_alloc(size, device=self.device_id)

        self.logger.debug(
            f"_TorchCUDAMemoryManager (allocate memory): size = {size}, ptr = {device_ptr}, "
            f"device_id = {self.device_id}, stream={current_stream()}"
        )

        def create_finalizer():
            def finalizer():
                caching_allocator_delete(device_ptr)
                self.logger.debug(f"_TorchCUDAMemoryManager (release memory): ptr = {device_ptr}")

            return finalizer

        return MemoryPointer(device_ptr, size, finalizer=create_finalizer())


_MEMORY_MANAGER = {"_raw": _RawCUDAMemoryManager, "cupy": _CupyCUDAMemoryManager, "torch": _TorchCUDAMemoryManager}
