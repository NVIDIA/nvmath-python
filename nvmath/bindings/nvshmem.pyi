# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import _cython_3_2_4
import enum
import numpy.dtypes
from _typeshed import Incomplete
from typing import Any, Callable, ClassVar

STATUS_FULL_MPG: int
STATUS_INVALID: int
STATUS_IS_BOOTSTRAPPED: int
STATUS_IS_INITIALIZED: int
STATUS_LIMITED_MPG: int
STATUS_NOT_INITIALIZED: int
TEAMS_MIN: int
TEAM_GPU_LEADERS: int
TEAM_GPU_LEADERS_INDEX: int
TEAM_INDEX_MAX: int
TEAM_INVALID: int
TEAM_NODE: int
TEAM_NODE_INDEX: int
TEAM_SAME_GPU: int
TEAM_SAME_GPU_INDEX: int
TEAM_SAME_MYPE_NODE: int
TEAM_SAME_MYPE_NODE_INDEX: int
TEAM_SHARED: int
TEAM_SHARED_INDEX: int
TEAM_WORLD: int
TEAM_WORLD_INDEX: int
__pyx_capi__: dict
__test__: dict
align: _cython_3_2_4.cython_function_or_method
barrier_all_on_stream: _cython_3_2_4.cython_function_or_method
calloc: _cython_3_2_4.cython_function_or_method
check_status: _cython_3_2_4.cython_function_or_method
free: _cython_3_2_4.cython_function_or_method
get_uniqueid: _cython_3_2_4.cython_function_or_method
hostlib_finalize: _cython_3_2_4.cython_function_or_method
hostlib_init_attr: _cython_3_2_4.cython_function_or_method
init_attr_dtype: numpy.dtypes.VoidDType
init_status: _cython_3_2_4.cython_function_or_method
int_p: _cython_3_2_4.cython_function_or_method
malloc: _cython_3_2_4.cython_function_or_method
my_pe: _cython_3_2_4.cython_function_or_method
n_pes: _cython_3_2_4.cython_function_or_method
ptr: _cython_3_2_4.cython_function_or_method
set_attr_uniqueid_args: _cython_3_2_4.cython_function_or_method
sync_all_on_stream: _cython_3_2_4.cython_function_or_method
team_my_pe: _cython_3_2_4.cython_function_or_method
uniqueid_dtype: numpy.dtypes.VoidDType

class Flags(enum.IntEnum):
    """See `flags`."""
    __new__: ClassVar[Callable] = ...
    INIT_MAX: ClassVar[Flags] = ...
    INIT_THREAD_PES: ClassVar[Flags] = ...
    INIT_WITH_MPI_COMM: ClassVar[Flags] = ...
    INIT_WITH_SHMEM: ClassVar[Flags] = ...
    INIT_WITH_UNIQUEID: ClassVar[Flags] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class InitAttr:
    """InitAttr()"""
    mpi_comm: Incomplete
    ptr: Incomplete
    version: Incomplete
    def __init__(self) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def __reduce__(self):
        """InitAttr.__reduce_cython__(self)"""
    def __reduce_cython__(self) -> Any:
        """InitAttr.__reduce_cython__(self)"""
    def __setstate_cython__(self, __pyx_state) -> Any:
        """InitAttr.__setstate_cython__(self, __pyx_state)"""

class NvshmemError(Exception):
    def __init__(self, status) -> Any:
        """NvshmemError.__init__(self, status)"""
    def __reduce__(self) -> Any:
        """NvshmemError.__reduce__(self)"""

class Status(enum.IntEnum):
    """See `nvshmemx_status`."""
    __new__: ClassVar[Callable] = ...
    ERROR_COLLECTIVE_LAUNCH_FAILED: ClassVar[Status] = ...
    ERROR_GPU_NOT_SELECTED: ClassVar[Status] = ...
    ERROR_INTERNAL: ClassVar[Status] = ...
    ERROR_INVALID_VALUE: ClassVar[Status] = ...
    ERROR_NOT_SUPPORTED: ClassVar[Status] = ...
    ERROR_OUT_OF_MEMORY: ClassVar[Status] = ...
    ERROR_SENTINEL: ClassVar[Status] = ...
    ERROR_SYMMETRY: ClassVar[Status] = ...
    SUCCESS: ClassVar[Status] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class UniqueId(uniqueid):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """UniqueId.__reduce_cython__(self)"""
    def __reduce_cython__(self) -> Any:
        """UniqueId.__reduce_cython__(self)"""
    def __setstate_cython__(self, __pyx_state) -> Any:
        """UniqueId.__setstate_cython__(self, __pyx_state)"""

class uniqueid:
    """uniqueid(size=1)

    Empty-initialize an array of `nvshmemx_uniqueid_v1`.

    The resulting object is of length `size` and of dtype `uniqueid_dtype`.
    If default-constructed, the instance represents a single struct.

    Args:
        size (int): number of structs, default=1.


    .. seealso:: `nvshmemx_uniqueid_v1`"""
    ptr: Incomplete
    version: Incomplete
    def __init__(self, size=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def from_data(data) -> Any:
        """uniqueid.from_data(data)

        Create an uniqueid instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `uniqueid_dtype` holding the data."""
    @staticmethod
    def from_ptr(intptr_tptr, size_tsize=..., boolreadonly=...) -> Any:
        """uniqueid.from_ptr(intptr_t ptr, size_t size=1, bool readonly=False)

        Create an uniqueid instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            size (int): number of structs, default=1.
            readonly (bool): whether the data is read-only (to the user). default is `False`."""
    def __delitem__(self, other) -> None:
        """Delete self[key]."""
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __getitem__(self, index):
        """Return self[key]."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    def __int__(self) -> int:
        """int(self)"""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""
    def __reduce__(self):
        """uniqueid.__reduce_cython__(self)"""
    def __reduce_cython__(self) -> Any:
        """uniqueid.__reduce_cython__(self)"""
    def __setitem__(self, index, object) -> None:
        """Set self[key] to value."""
    def __setstate_cython__(self, __pyx_state) -> Any:
        """uniqueid.__setstate_cython__(self, __pyx_state)"""
