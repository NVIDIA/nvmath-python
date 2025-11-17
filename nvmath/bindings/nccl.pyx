# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 2.11.4 to 2.28.3. Do not modify it directly.

cimport cython  # NOQA
from cpython cimport buffer as _buffer
from cpython.memoryview cimport PyMemoryView_FromMemory

from ._internal.utils cimport get_buffer_pointer

from enum import IntEnum as _IntEnum

import numpy as _numpy


###############################################################################
# POD
###############################################################################

unique_id_dtype = _numpy.dtype([
    ("internal", _numpy.int8, (128,)),
    ], align=True)


cdef class UniqueId:
    """Empty-initialize an array of `ncclUniqueId`.

    The resulting object is of length `size` and of dtype `unique_id_dtype`.
    If default-constructed, the instance represents a single struct.

    Args:
        size (int): number of structs, default=1.


    .. seealso:: `ncclUniqueId`
    """
    cdef:
        readonly object _data

    def __init__(self, size=1):
        arr = _numpy.empty(size, dtype=unique_id_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(ncclUniqueId), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(ncclUniqueId)}"

    def __repr__(self):
        if self._data.size > 1:
            return f"<{__name__}.UniqueId_Array_{self._data.size} object at {hex(id(self))}>"
        else:
            return f"<{__name__}.UniqueId object at {hex(id(self))}>"

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
        if not isinstance(other, UniqueId):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    def __getitem__(self, key):
        if isinstance(key, int):
            size = self._data.size
            if key >= size or key <= -(size+1):
                raise IndexError("index is out of bounds")
            if key < 0:
                key += size
            return UniqueId.from_data(self._data[key:key+1])
        out = self._data[key]
        if isinstance(out, _numpy.recarray) and out.dtype == unique_id_dtype:
            return UniqueId.from_data(out)
        return out

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an UniqueId instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `unique_id_dtype` holding the data.
        """
        cdef UniqueId obj = UniqueId.__new__(UniqueId)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != unique_id_dtype:
            raise ValueError("data array must be of dtype unique_id_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, size_t size=1, bint readonly=False):
        """Create an UniqueId instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            size (int): number of structs, default=1.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef UniqueId obj = UniqueId.__new__(UniqueId)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(ncclUniqueId) * size, flag)
        data = _numpy.ndarray((size,), buffer=buf,
                              dtype=unique_id_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj



###############################################################################
# Enum
###############################################################################

class Result(_IntEnum):
    """See `ncclResult_t`."""
    Success = ncclSuccess
    UnhandledCudaError = ncclUnhandledCudaError
    SystemError = ncclSystemError
    InternalError = ncclInternalError
    InvalidArgument = ncclInvalidArgument
    InvalidUsage = ncclInvalidUsage
    RemoteError = ncclRemoteError
    InProgress = ncclInProgress
    NumResults = ncclNumResults


###############################################################################
# Error handling
###############################################################################

class NCCLError(Exception):

    def __init__(self, status):
        self.status = status
        s = Result(status)
        cdef str err = f"{s.name} ({s.value}): {get_error_string(status)}"
        super(NCCLError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise NCCLError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef int get_version() except? -1:
    cdef int version
    with nogil:
        status = ncclGetVersion(&version)
    check_status(status)
    return version


cpdef get_unique_id(intptr_t unique_id):
    with nogil:
        status = ncclGetUniqueId(<ncclUniqueId*>unique_id)
    check_status(status)


cpdef intptr_t comm_init_rank(int nranks, comm_id, int rank) except? 0:
    cdef void* _comm_id_ = get_buffer_pointer(comm_id, -1, readonly=False)
    cdef Comm comm
    with nogil:
        status = ncclCommInitRank(&comm, nranks, (<ncclUniqueId*>(_comm_id_))[0], rank)
    check_status(status)
    return <intptr_t>comm


cpdef comm_destroy(intptr_t comm):
    with nogil:
        status = ncclCommDestroy(<Comm>comm)
    check_status(status)


cpdef comm_abort(intptr_t comm):
    with nogil:
        status = ncclCommAbort(<Comm>comm)
    check_status(status)


cpdef str get_error_string(int result):
    cdef bytes _output_
    _output_ = ncclGetErrorString(<_Result>result)
    return _output_.decode()


cpdef int comm_count(intptr_t comm) except? -1:
    cdef int count
    with nogil:
        status = ncclCommCount(<const Comm>comm, &count)
    check_status(status)
    return count


cpdef int comm_cu_device(intptr_t comm) except? -1:
    cdef int device
    with nogil:
        status = ncclCommCuDevice(<const Comm>comm, &device)
    check_status(status)
    return device


cpdef int comm_user_rank(intptr_t comm) except? -1:
    cdef int rank
    with nogil:
        status = ncclCommUserRank(<const Comm>comm, &rank)
    check_status(status)
    return rank


cpdef str get_last_error(intptr_t comm):
    cdef bytes _output_
    _output_ = ncclGetLastError(<Comm>comm)
    return _output_.decode()


cpdef comm_finalize(intptr_t comm):
    with nogil:
        status = ncclCommFinalize(<Comm>comm)
    check_status(status)
