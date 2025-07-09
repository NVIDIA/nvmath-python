# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""Interface for pluggable memory handlers."""

__all__ = ["BaseCUDAMemoryManager", "MemoryPointer"]

from abc import abstractmethod
from collections.abc import Callable
from typing import Protocol, runtime_checkable
import logging
import weakref

import cuda.core.experimental as ccx

from nvmath.internal import utils
from nvmath.internal.package_ifc_cuda import CUDAPackage


class MemoryPointer:
    """
    An RAII class for a device memory buffer.

    Args:
        device_ptr: The address of the device memory buffer.
        size: The size of the memory buffer in bytes.
        finalizer: A nullary callable that will be called when the buffer is to be freed.

    .. seealso:: :class:`numba.cuda.MemoryPointer`
    """

    def __init__(self, device_ptr: int, size: int, finalizer: None | Callable[[], None]):
        self.device_ptr = device_ptr
        self.size = size
        self._finalizer: weakref.finalize | None
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
    def __init__(self, device_id: int, logger: logging.Logger):
        raise NotImplementedError

    @abstractmethod
    def memalloc(self, size: int) -> MemoryPointer:
        """
        Allocate device memory synchronously or on the current stream.

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


@runtime_checkable
class BaseCUDAMemoryManagerAsync(Protocol):
    """
    Protocol for async memory manager plugins.

    .. seealso:: :class:`BaseCUDAMemoryManager`
    """

    @abstractmethod
    def __init__(self, device_id: int, logger: logging.Logger):
        raise NotImplementedError

    @abstractmethod
    def memalloc_async(self, size: int, stream: ccx.Stream) -> MemoryPointer:
        """
        Allocate device memory asynchronously on the provided stream.

        Args:
            size: The size of the memory buffer in bytes.
            stream: A cuda.core.Stream object on which the allocation will be performed.

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


class _RawCUDAMemoryManager(BaseCUDAMemoryManagerAsync):
    """
    Raw device memory allocator.

    Args:
        device_id: The ID (int) of the device on which memory is to be allocated.
        logger (logging.Logger): Python Logger object.
    """

    def __init__(self, device_id: int, logger: logging.Logger):
        """
        __init__(device_id)
        """
        self.device_id = device_id
        self.logger = logger

    def memalloc_async(self, size: int, stream: ccx.Stream) -> MemoryPointer:
        with utils.device_ctx(self.device_id) as device:
            buffer = device.allocate(size=size, stream=stream)
            device_ptr = int(buffer.handle)

        self.logger.debug(
            "_RawCUDAMemoryManager (allocate memory): size = %d, ptr = %d, device_id = %d, stream = %s",
            size,
            device_ptr,
            self.device_id,
            stream,
        )

        def finalizer():
            nonlocal buffer, stream, device_ptr
            self.logger.debug(
                "_RawCUDAMemoryManager (release memory): ptr = %d, device_id = %d, stream = %s",
                device_ptr,
                self.device_id,
                stream,
            )
            with utils.device_ctx(self.device_id):
                buffer.close(stream=stream)

        return MemoryPointer(device_ptr, size, finalizer=finalizer)


_MEMORY_MANAGER: dict[str, type[BaseCUDAMemoryManager] | type[BaseCUDAMemoryManagerAsync]] = {
    "_raw": _RawCUDAMemoryManager,
}


def lazy_load_cupy():
    global _MEMORY_MANAGER
    import cupy as cp
    from nvmath.internal.package_ifc_cupy import CupyPackage

    class _CupyCUDAMemoryManager(BaseCUDAMemoryManagerAsync):
        """
        CuPy device memory allocator.

        Args:
            device_id: The ID (int) of the device on which memory is to be allocated.
            logger (logging.Logger): Python Logger object.
        """

        def __init__(self, device_id: int, logger: logging.Logger):
            """
            __init__(device_id)
            """
            self.device_id = device_id
            self.logger = logger

        def memalloc_async(self, size: int, stream) -> MemoryPointer:
            stream_ctx = CupyPackage.to_stream_context(
                CupyPackage.create_external_stream(self.device_id, CUDAPackage.to_stream_pointer(stream))
            )
            with utils.device_ctx(self.device_id), stream_ctx:
                cp_mem_ptr = cp.cuda.alloc(size)
                device_ptr = cp_mem_ptr.ptr

            self.logger.debug(
                "_CupyCUDAMemoryManager (allocate memory): size = %d, ptr = %d, device_id = %d, stream = %s",
                size,
                device_ptr,
                self.device_id,
                stream,
            )

            def finalizer():
                # The cupy MemoryPointer object is RAII, so we keep a reference to it
                # until we don't need it anymore.
                nonlocal cp_mem_ptr, device_ptr
                self.logger.debug("_CupyCUDAMemoryManager (release memory): ptr = %d", device_ptr)
                del cp_mem_ptr

            return MemoryPointer(device_ptr, size, finalizer=finalizer)

    _MEMORY_MANAGER["cupy"] = _CupyCUDAMemoryManager


def lazy_load_torch():
    global _MEMORY_MANAGER

    from torch.cuda import caching_allocator_alloc, caching_allocator_delete
    from nvmath.internal.package_ifc_torch import TorchPackage

    class _TorchCUDAMemoryManager(BaseCUDAMemoryManagerAsync):
        """
        Torch caching memory allocator.

        Args:
            device_id: The ID (int) of the device on which memory is to be allocated.
            logger (logging.Logger): Python Logger object.
        """

        def __init__(self, device_id: int, logger: logging.Logger):
            """
            __init__(device_id)
            """
            self.device_id = device_id
            self.logger = logger

        def memalloc_async(self, size: int, stream: ccx.Stream) -> MemoryPointer:
            torch_stream = TorchPackage.create_external_stream(self.device_id, CUDAPackage.to_stream_pointer(stream))
            device_ptr = caching_allocator_alloc(size, device=self.device_id, stream=torch_stream)

            self.logger.debug(
                "_TorchCUDAMemoryManager (allocate memory): size = %d, ptr = %d, device_id = %d, stream = %s",
                size,
                device_ptr,
                self.device_id,
                stream,
            )

            def finalizer():
                nonlocal device_ptr, stream
                self.logger.debug(
                    "_TorchCUDAMemoryManager (release memory): ptr = %d, device_id = %d, stream = %s",
                    device_ptr,
                    self.device_id,
                    stream,
                )
                caching_allocator_delete(device_ptr)

            return MemoryPointer(device_ptr, size, finalizer=finalizer)

    _MEMORY_MANAGER["torch"] = _TorchCUDAMemoryManager
