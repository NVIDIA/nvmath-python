# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 3.1.7. Do not modify it directly.

cimport cython  # NOQA
from cpython cimport buffer as _buffer
from cpython.memoryview cimport PyMemoryView_FromMemory

from enum import IntEnum as _IntEnum

import numpy as _numpy


###############################################################################
# POD
###############################################################################

uniqueid_dtype = _numpy.dtype([
    ("version", _numpy.int32, ),
    ("internal", _numpy.int8, (124,)),
    ], align=True)


cdef class uniqueid:
    """Empty-initialize an array of `nvshmemx_uniqueid_v1`.

    The resulting object is of length `size` and of dtype `uniqueid_dtype`.
    If default-constructed, the instance represents a single struct.

    Args:
        size (int): number of structs, default=1.


    .. seealso:: `nvshmemx_uniqueid_v1`
    """
    cdef:
        readonly object _data

    def __init__(self, size=1):
        arr = _numpy.empty(size, dtype=uniqueid_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(nvshmemx_uniqueid_v1), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(nvshmemx_uniqueid_v1)}"

    def __repr__(self):
        if self._data.size > 1:
            return f"<{__name__}.uniqueid_Array_{self._data.size} object at {hex(id(self))}>"
        else:
            return f"<{__name__}.uniqueid object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        if self._data.size > 1:
            raise TypeError("int() argument must be a bytes-like object of size 1. "
                            "To get the pointer address of an array, use .ptr")
        return self._data.ctypes.data

    def __len__(self):
        return self._data.size

    def __eq__(self, other):
        if not isinstance(other, uniqueid):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def version(self):
        """Union[~_numpy.int32, int]: """
        if self._data.size == 1:
            return int(self._data.version[0])
        return self._data.version

    @version.setter
    def version(self, val):
        self._data.version = val

    def __getitem__(self, key):
        if isinstance(key, int):
            size = self._data.size
            if key >= size or key <= -(size+1):
                raise IndexError("index is out of bounds")
            if key < 0:
                key += size
            return uniqueid.from_data(self._data[key:key+1])
        out = self._data[key]
        if isinstance(out, _numpy.recarray) and out.dtype == uniqueid_dtype:
            return uniqueid.from_data(out)
        return out

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an uniqueid instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `uniqueid_dtype` holding the data.
        """
        cdef uniqueid obj = uniqueid.__new__(uniqueid)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != uniqueid_dtype:
            raise ValueError("data array must be of dtype uniqueid_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, size_t size=1, bint readonly=False):
        """Create an uniqueid instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            size (int): number of structs, default=1.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef uniqueid obj = uniqueid.__new__(uniqueid)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(nvshmemx_uniqueid_v1) * size, flag)
        data = _numpy.ndarray((size,), buffer=buf,
                              dtype=uniqueid_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


cdef class UniqueId(uniqueid): pass

# POD wrapper for nvshmemx_init_attr_t. cybind can't generate this automatically
# because it doesn't fully support nested structs (https://gitlab-master.nvidia.com/leof/cybind/-/issues/67).
# The nested structure is made opaque.
# TODO: remove this once cybind supports nested structs.

init_attr_dtype = _numpy.dtype([
    ("version", _numpy.int32, ),
    ("mpi_comm", _numpy.intp, ),
    ("args", _numpy.int8, (sizeof(nvshmemx_init_args_t),)),  # opaque
    ], align=True)


cdef class InitAttr:

    cdef:
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=init_attr_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(nvshmemx_init_attr_t), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(nvshmemx_init_attr_t)}"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    @property
    def version(self):
        """version (~_numpy.int32): """
        return int(self._data.version[0])

    @version.setter
    def version(self, val):
        self._data.version = val

    @property
    def mpi_comm(self):
        """mpi_comm (~_numpy.intp): """
        return int(self._data.mpi_comm[0])

    @mpi_comm.setter
    def mpi_comm(self, val):
        self._data.mpi_comm = val


###############################################################################
# Enum
###############################################################################

_PROXY_GLOBAL_EXIT_NOT_REQUESTED = PROXY_GLOBAL_EXIT_NOT_REQUESTED
_PROXY_GLOBAL_EXIT_INIT = PROXY_GLOBAL_EXIT_INIT
_PROXY_GLOBAL_EXIT_REQUESTED = PROXY_GLOBAL_EXIT_REQUESTED
_PROXY_GLOBAL_EXIT_FINISHED = PROXY_GLOBAL_EXIT_FINISHED
_PROXY_GLOBAL_EXIT_MAX_STATE = PROXY_GLOBAL_EXIT_MAX_STATE

STATUS_NOT_INITIALIZED = NVSHMEM_STATUS_NOT_INITIALIZED
STATUS_IS_BOOTSTRAPPED = NVSHMEM_STATUS_IS_BOOTSTRAPPED
STATUS_IS_INITIALIZED = NVSHMEM_STATUS_IS_INITIALIZED
STATUS_LIMITED_MPG = NVSHMEM_STATUS_LIMITED_MPG
STATUS_FULL_MPG = NVSHMEM_STATUS_FULL_MPG
STATUS_INVALID = NVSHMEM_STATUS_INVALID

TEAM_INVALID = NVSHMEM_TEAM_INVALID
TEAM_WORLD = NVSHMEM_TEAM_WORLD
TEAM_WORLD_INDEX = NVSHMEM_TEAM_WORLD_INDEX
TEAM_SHARED = NVSHMEM_TEAM_SHARED
TEAM_SHARED_INDEX = NVSHMEM_TEAM_SHARED_INDEX
TEAM_NODE = NVSHMEMX_TEAM_NODE
TEAM_NODE_INDEX = NVSHMEM_TEAM_NODE_INDEX
TEAM_SAME_MYPE_NODE = NVSHMEMX_TEAM_SAME_MYPE_NODE
TEAM_SAME_MYPE_NODE_INDEX = NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX
TEAM_SAME_GPU = NVSHMEMI_TEAM_SAME_GPU
TEAM_SAME_GPU_INDEX = NVSHMEM_TEAM_SAME_GPU_INDEX
TEAM_GPU_LEADERS = NVSHMEMI_TEAM_GPU_LEADERS
TEAM_GPU_LEADERS_INDEX = NVSHMEM_TEAM_GPU_LEADERS_INDEX
TEAMS_MIN = NVSHMEM_TEAMS_MIN
TEAM_INDEX_MAX = NVSHMEM_TEAM_INDEX_MAX

class Status(_IntEnum):
    """See `nvshmemx_status`."""
    SUCCESS = NVSHMEMX_SUCCESS
    ERROR_INVALID_VALUE = NVSHMEMX_ERROR_INVALID_VALUE
    ERROR_OUT_OF_MEMORY = NVSHMEMX_ERROR_OUT_OF_MEMORY
    ERROR_NOT_SUPPORTED = NVSHMEMX_ERROR_NOT_SUPPORTED
    ERROR_SYMMETRY = NVSHMEMX_ERROR_SYMMETRY
    ERROR_GPU_NOT_SELECTED = NVSHMEMX_ERROR_GPU_NOT_SELECTED
    ERROR_COLLECTIVE_LAUNCH_FAILED = NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED
    ERROR_INTERNAL = NVSHMEMX_ERROR_INTERNAL
    ERROR_SENTINEL = NVSHMEMX_ERROR_SENTINEL

class Flags(_IntEnum):
    """See `flags`."""
    INIT_THREAD_PES = NVSHMEMX_INIT_THREAD_PES
    INIT_WITH_MPI_COMM = NVSHMEMX_INIT_WITH_MPI_COMM
    INIT_WITH_SHMEM = NVSHMEMX_INIT_WITH_SHMEM
    INIT_WITH_UNIQUEID = NVSHMEMX_INIT_WITH_UNIQUEID
    INIT_MAX = NVSHMEMX_INIT_MAX


###############################################################################
# Error handling
###############################################################################

class NvshmemError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value})"
        super(NvshmemError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise NvshmemError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef int init_status() except? -1:
    return nvshmemx_init_status()


cpdef int my_pe() except? -1:
    return nvshmem_my_pe()


cpdef int n_pes() except? -1:
    return nvshmem_n_pes()


cpdef intptr_t malloc(size_t size) except? 0:
    return <intptr_t>nvshmem_malloc(size)


cpdef intptr_t calloc(size_t count, size_t size) except? 0:
    return <intptr_t>nvshmem_calloc(count, size)


cpdef intptr_t align(size_t alignment, size_t size) except? 0:
    return <intptr_t>nvshmem_align(alignment, size)


cpdef void free(intptr_t ptr) except*:
    nvshmem_free(<void*>ptr)


cpdef intptr_t ptr(intptr_t dest, int pe) except? 0:
    return <intptr_t>nvshmem_ptr(<const void*>dest, pe)


cpdef void int_p(intptr_t dest, int value, int pe) except*:
    nvshmem_int_p(<int*>dest, <const int>value, pe)


cpdef int team_my_pe(int32_t team) except? -1:
    return nvshmem_team_my_pe(<nvshmem_team_t>team)


cpdef void barrier_all_on_stream(intptr_t stream) except*:
    nvshmemx_barrier_all_on_stream(<Stream>stream)


cpdef void sync_all_on_stream(intptr_t stream) except*:
    nvshmemx_sync_all_on_stream(<Stream>stream)


cpdef hostlib_init_attr(unsigned int flags, intptr_t attr):
    with nogil:
        status = nvshmemx_hostlib_init_attr(flags, <nvshmemx_init_attr_t*>attr)
    check_status(status)


cpdef void hostlib_finalize() except*:
    nvshmemx_hostlib_finalize()


cpdef set_attr_uniqueid_args(int myrank, int nranks, intptr_t uniqueid, intptr_t attr):
    with nogil:
        status = nvshmemx_set_attr_uniqueid_args(<const int>myrank, <const int>nranks, <const nvshmemx_uniqueid_t*>uniqueid, <nvshmemx_init_attr_t*>attr)
    check_status(status)


cpdef get_uniqueid(intptr_t uniqueid):
    with nogil:
        status = nvshmemx_get_uniqueid(<nvshmemx_uniqueid_t*>uniqueid)
    check_status(status)
