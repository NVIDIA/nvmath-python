# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import _cython_3_2_4
from typing import Any, ClassVar

__pyx_capi__: dict
__test__: dict
allocate_from_mr: _cython_3_2_4.cython_function_or_method
free_reserved_memory: _cython_3_2_4.cython_function_or_method
get_device_memory_resource: _cython_3_2_4.cython_function_or_method

class _MemoryPointer:
    """
    Temporary internal NDBuffer allocation adapter class. NDBuffer expects
    custom allocator to return a cuda.core.Buffer instance. Until all supported
    cuda.core versions have unified support for wrapping external allocations
    with Buffer.from_handle, this class servers as an adapter/workaround.

    WARNING: This is internal tool subject to change/removal without notice.

    Internally, it is used conditionally if any of the following is needed:
        * wrap external allocations that don't come as cuda.core.Buffer (e.g. from cupy).
        This is needed because prior to cuda.core 0.5.0, it's not possible to
        pass reference to externall RAII object as ``owner`` parameter to Buffer.from_handle.
        * provide debug logging on deallocation. In the future, we can use
        Buffer.from_handle(owner=...) to inject deallocation callback (cuda.core >= 0.5.0)
        or weakref.finalize for Buffer (cuda.core >= 0.6.0).

    The only publicly exposed field is the handle - a base pointer to the allocated memory.
    """
    from_handle: ClassVar[method] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    handle: handle
    owner: owner
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __del__(self, *args, **kwargs) -> None: ...
    def __reduce__(self):
        """_MemoryPointer.__reduce_cython__(self)"""
    def __reduce_cython__(self) -> Any:
        """_MemoryPointer.__reduce_cython__(self)"""
    def __setstate_cython__(self, __pyx_state) -> Any:
        """_MemoryPointer.__setstate_cython__(self, __pyx_state)"""
