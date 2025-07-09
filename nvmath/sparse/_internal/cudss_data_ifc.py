# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface classes to encapsulate low-level calls to get or set cuDSS "data" attributes.
"""

__all__ = ["FactorizationInfo", "PlanInfo", "memory_estimates_dtype"]

import re
import threading
from typing import TypeAlias

import numpy as np

from nvmath.bindings import cudss
from nvmath.internal import utils


DataParamEnum: TypeAlias = cudss.DataParam

# Future-proofing - cuDSS is currently not thread-safe.
# https://docs.nvidia.com/cuda/cudss/general.html#thread-safety
_tls = threading.local()
_tls.size_written = np.empty((1,), dtype=np.uint64)


def _get_attribute(handle, data_ptr, name, attribute, length=1):
    """
    name      = cudss enumerator for the attribute.
    attribute = numpy ndarray object into which the value is stored.
    """
    cudss.data_get(
        handle, data_ptr, name, attribute.ctypes.data, length * attribute.dtype.itemsize, _tls.size_written.ctypes.data
    )
    assert _tls.size_written[0] <= length * attribute.dtype.itemsize, "Internal error."


def _check_valid_solver(info):
    """
    Check if the configuration points to a valid DirectSolver object.
    """
    if not info._solver.valid_state:
        m = re.match(r"<nvmath\..*\.(.*Info) object at (.*)>$", str(info))
        assert m is not None, "Internal error."
        name = m.group(1)
        address = m.group(2)

        raise RuntimeError(f"The {name} object at {address} cannot be used after it's solver object is free'd.")


memory_estimates_dtype = np.dtype(
    [
        ("permanent_device_memory", "<u8"),
        ("peak_device_memory", "<u8"),
        ("permanent_host_memory", "<u8"),
        ("peak_host_memory", "<u8"),
        ("hybrid_min_device_memory", "<u8"),
        ("hybrid_max_device_memory", "<u8"),
        ("reserved", "<u8", (10,)),
    ]
)


class PlanInfo:
    """
    An interface to query information returned by
    :meth:`nvmath.sparse.advanced.DirectSolver.plan`.
    """

    def __init__(self, solver):
        """
        ctor for internal use only.
        """
        self._solver = solver
        self._handle = self._solver.handle
        self._data_ptr = self._solver.data_ptr
        assert self._data_ptr is not None, "Internal error."

        self._N = self._solver._N
        self._batched = self._solver.batched

        # Allocate permutation arrays lazily, and only if not batched.
        self._perm_reorder_col = None
        self._perm_reorder_row = None

        self._memory_estimates = np.zeros((), dtype=memory_estimates_dtype).view(np.recarray)

    def _check_valid_solver_wrapper(self, *args, **kwargs):
        _check_valid_solver(self)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def memory_estimates(self):
        """
        Query the memory estimates. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>`_
        for more information.
        """
        _get_attribute(self._handle, self._data_ptr, DataParamEnum.MEMORY_ESTIMATES, self._memory_estimates, length=1)

        return self._memory_estimates

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def col_permutation(self):
        """
        Query the column permutation after planning (reordering). See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>`_
        for more information.
        """
        if self._batched:
            raise RuntimeError("Column permutation is not available for batched systems.")

        if self._perm_reorder_col is None:
            get_dtype = cudss.get_data_param_dtype
            self._perm_reorder_col = np.empty((self._N,), dtype=get_dtype(DataParamEnum.PERM_REORDER_COL))

        _get_attribute(self._handle, self._data_ptr, DataParamEnum.PERM_REORDER_COL, self._perm_reorder_col, length=self._N)

        return self._perm_reorder_col

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def row_permutation(self):
        """
        Query the row permutation after planning (reordering). See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>`_
        for more information.
        """
        if self._batched:
            raise RuntimeError("Row permutation is not available for batched systems.")

        if self._perm_reorder_row is None:
            get_dtype = cudss.get_data_param_dtype
            self._perm_reorder_row = np.empty((self._N,), dtype=get_dtype(DataParamEnum.PERM_REORDER_ROW))

        _get_attribute(self._handle, self._data_ptr, DataParamEnum.PERM_REORDER_ROW, self._perm_reorder_row, length=self._N)

        return self._perm_reorder_row


class FactorizationInfo:
    """
    An interface to query information returned by
    :meth:`nvmath.sparse.advanced.DirectSolver.factorize`.
    """

    def __init__(self, solver):
        """
        ctor for internal use only.
        """
        self._solver = solver
        self._handle = self._solver.handle
        self._data_ptr = self._solver.data_ptr
        assert self._data_ptr is not None, "Internal error"

        self._N = self._solver._N
        self._batched = self._solver.batched
        self._value_type = self._solver.value_type

        get_dtype = cudss.get_data_param_dtype

        self._info = np.zeros((1,), dtype=get_dtype(DataParamEnum.INFO))
        self._lu_nnz = np.zeros((1,), dtype=get_dtype(DataParamEnum.LU_NNZ))
        self._npivots = np.zeros((1,), dtype=get_dtype(DataParamEnum.NPIVOTS))
        self._inertia = np.zeros((2,), dtype=get_dtype(DataParamEnum.INERTIA))

        # Allocate permutation and diagonal arrays lazily, and only if not batched.
        self._perm_col = None
        self._perm_row = None
        self._diag = None

    def _check_valid_solver_wrapper(self, *args, **kwargs):
        _check_valid_solver(self)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def info(self):
        """
        Query the error info after factorization. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>`_
        for more information.
        """
        _get_attribute(self._handle, self._data_ptr, DataParamEnum.INFO, self._info)

        return self._info.item()

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def lu_nnz(self):
        """
        Query the number of non-zeros  in the LU factorization. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>`_
        for more information.
        """
        _get_attribute(self._handle, self._data_ptr, DataParamEnum.LU_NNZ, self._lu_nnz)

        return self._lu_nnz.item()

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def npivots(self):
        """
        Query the number of pivots encountered in the LU factorization. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>`_
        for more information.
        """
        _get_attribute(self._handle, self._data_ptr, DataParamEnum.NPIVOTS, self._npivots)

        return self._npivots.item()

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def inertia(self):
        """
        Query the inertia (number of positive and negative pivots) encountered in the
        LU factorization. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>`_
        for more information.
        """
        _get_attribute(self._handle, self._data_ptr, DataParamEnum.INERTIA, self._inertia, length=2)

        return self._inertia

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def col_permutation(self):
        """
        Query the column permutation after factorization. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>`_
        for more information.
        """
        if self._batched:
            raise RuntimeError("Column permutation is not available for batched systems.")

        if self._perm_col is None:
            get_dtype = cudss.get_data_param_dtype
            self._perm_col = np.empty((self._N,), dtype=get_dtype(DataParamEnum.PERM_COL))

        _get_attribute(self._handle, self._data_ptr, DataParamEnum.PERM_COL, self._perm_col, length=self._N)

        return self._perm_col

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def row_permutation(self):
        """
        Query the row permutation after factorization. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>`_
        for more information.
        """
        if self._batched:
            raise RuntimeError("Row permutation is not available for batched systems.")

        if self._perm_row is None:
            get_dtype = cudss.get_data_param_dtype
            self._perm_row = np.empty((self._N,), dtype=get_dtype(DataParamEnum.PERM_ROW))

        _get_attribute(self._handle, self._data_ptr, DataParamEnum.PERM_ROW, self._perm_row, length=self._N)

        return self._perm_row

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def diag(self):
        """
        Query the diagonal of the factorized system. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>`_
        for more information.
        """
        if self._batched:
            raise RuntimeError("The factorized system's diagonal is not available for batched systems.")

        if self._diag is None:
            self._diag = np.empty((self._N,), dtype=self._value_type)

        _get_attribute(self._handle, self._data_ptr, DataParamEnum.DIAG, self._diag, length=self._N)

        return self._diag
