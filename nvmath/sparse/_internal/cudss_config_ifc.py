# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface classes to encapsulate low-level calls to get or set configuration information.
"""

__all__ = ["FactorizationConfig", "PlanConfig", "SolutionConfig"]

import re
import threading
from typing import TypeAlias

import numpy as np

from nvmath.bindings import cudss
from nvmath.internal import utils


ConfigParamEnum: TypeAlias = cudss.ConfigParam

# Future-proofing - cuDSS is currently not thread-safe.
# https://docs.nvidia.com/cuda/cudss/general.html#thread-safety
_tls = threading.local()
_tls.size_written = np.empty((1,), dtype=np.uint64)


def _get_scalar_attribute(config_ptr, name, attribute):
    """
    name      = cudss enumerator for the attribute.
    attribute = numpy ndarray object into which the value is stored.
    """
    cudss.config_get(config_ptr, name, attribute.ctypes.data, attribute.dtype.itemsize, _tls.size_written.ctypes.data)
    assert _tls.size_written[0] <= attribute.dtype.itemsize, "Internal error."


def _set_scalar_attribute(config_ptr, name, attribute, value):
    """
    name      = cudss enumerator for the attribute.
    attribute = numpy ndarray object into which the value is stored.
    value     = the value to set the the attribute to.
    """
    attribute[0] = value
    cudss.config_set(config_ptr, name, attribute.ctypes.data, attribute.dtype.itemsize)


def _check_valid_solver(config):
    """
    Check if the configuration points to a valid DirectSolver object.
    """
    if not config._solver.valid_state:
        m = re.match(r"<nvmath\..*\.(.*Config) object at (.*)>$", str(config))
        assert m is not None, "Internal error."
        name = m.group(1)
        address = m.group(2)

        raise RuntimeError(f"The {name} object at {address} cannot be used after it's solver object is free'd.")


# TODO: Set user permutation as part of PlanConfig, even though it sets the data object.
class PlanConfig:
    """
    An interface to configure :meth:`nvmath.sparse.advanced.DirectSolver.plan`. The
    current configuration can also be queried.
    """

    def __init__(self, solver):
        """
        ctor for internal use only.
        """
        self._solver = solver
        self._config_ptr = self._solver.config_ptr
        assert self._config_ptr is not None, "Internal error"

        get_dtype = cudss.get_config_param_dtype

        self._host_nthreads = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.HOST_NTHREADS))
        self._reordering_alg = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.REORDERING_ALG))
        self._pivot_type = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.PIVOT_TYPE))
        self._pivot_threshold = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.PIVOT_THRESHOLD))
        self._max_lu_nnz = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.MAX_LU_NNZ))

    def _check_valid_solver_wrapper(self, *args, **kwargs):
        _check_valid_solver(self)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def host_nthreads(self):
        """
        Query or set the number of host threads. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.HOST_NTHREADS, self._host_nthreads)

        return self._host_nthreads.item()

    @host_nthreads.setter
    @utils.precondition(_check_valid_solver_wrapper)
    def host_nthreads(self, nthreads):
        """
        Set the number of host threads. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.

        Args:
            nthreads: The number of host threads as Python `int`.

        """
        if not self._solver.multithreading:
            raise ValueError(
                "The number of host threads cannot be set if a multithreading library was not provided during problem \
specification."
            )
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.HOST_NTHREADS, self._host_nthreads, nthreads)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def reordering_algorithm(self):
        """
        Query or set the reordering algorithm used. See
        :class:`nvmath.bindings.cudss.AlgType` and the `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_ for more
        information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.REORDERING_ALG, self._reordering_alg)

        return cudss.AlgType(self._reordering_alg.item())

    @reordering_algorithm.setter
    @utils.precondition(_check_valid_solver_wrapper)
    def reordering_algorithm(self, algorithm):
        """
        Set the reordering algorithm to use. See :class:`nvmath.bindings.cudss.AlgType` and
        the `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_ for more
        information.

        Args:
            algorithm: The reordering algorithm of type
            :class:`nvmath.bindings.cudss.AlgType` or Python `int`.

        """
        algorithm = cudss.AlgType(algorithm)
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.REORDERING_ALG, self._reordering_alg, algorithm)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def pivot_type(self):
        """
        Query or set the type of pivoting. See
        :class:`nvmath.bindings.cudss.PivotType` and the `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.PIVOT_TYPE, self._pivot_type)

        return cudss.PivotType(self._pivot_type.item())

    @pivot_type.setter
    @utils.precondition(_check_valid_solver_wrapper)
    def pivot_type(self, pivot_type):
        """
        Set the type of pivoting. See :class:`nvmath.bindings.cudss.PivotType` and the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_ for more
        information.

        Args:
            pivot_type: The type of pivoting (:class:`nvmath.bindings.cudss.PivotType` or
            Python `int`).

        """
        pivot_type = cudss.PivotType(pivot_type)
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.PIVOT_TYPE, self._pivot_type, pivot_type)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def pivot_threshold(self):
        """
        Query or set the pivot threshold. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.PIVOT_THRESHOLD, self._pivot_threshold)

        return self._pivot_threshold.item()

    @pivot_threshold.setter
    @utils.precondition(_check_valid_solver_wrapper)
    def pivot_threshold(self, pivot_threshold):
        """
        Set the pivot threshold. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.

        Args:
            pivot_threshold: The threshold for pivoting (Python `float`).

        """
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.PIVOT_THRESHOLD, self._pivot_threshold, pivot_threshold)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def max_nnz(self):
        """
        Query or set the maximum limit for non-zero factors in the LU decomposition. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.MAX_LU_NNZ, self._max_lu_nnz)

        return self._max_lu_nnz.item()

    @max_nnz.setter
    @utils.precondition(_check_valid_solver_wrapper)
    def max_nnz(self, max_nnz):
        """
        Set the maximum limit for non-zero factors in the LU decomposition. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.

        Args:
            max_nnz: The maximum limit for non-zero factors in the LU decomposition
                     (Python `int`).

        """
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.MAX_LU_NNZ, self._max_lu_nnz, max_nnz)


class FactorizationConfig:
    """
    An interface to configure :meth:`nvmath.sparse.advanced.DirectSolver.factorize`. The
    current configuration can also be queried.
    """

    def __init__(self, solver):
        """
        ctor for internal use only.
        """
        self._solver = solver
        self._config_ptr = self._solver.config_ptr
        assert self._config_ptr is not None, "Internal error"

        get_dtype = cudss.get_config_param_dtype

        self._factorization_alg = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.FACTORIZATION_ALG))
        self._pivot_epsilon_alg = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.PIVOT_EPSILON_ALG))
        self._pivot_epsilon = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.PIVOT_EPSILON))

    def _check_valid_solver_wrapper(self, *args, **kwargs):
        _check_valid_solver(self)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def factorization_algorithm(self):
        """
        Query or set the factorization algorithm used. See
        :class:`nvmath.bindings.cudss.AlgType` and the `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_ for more
        information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.FACTORIZATION_ALG, self._factorization_alg)

        return cudss.AlgType(self._factorization_alg.item())

    @factorization_algorithm.setter
    @utils.precondition(_check_valid_solver_wrapper)
    def factorization_algorithm(self, algorithm):
        """
        Set the factorization algorithm to use. See :class:`nvmath.bindings.cudss.AlgType`
        and the `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_ for more
        information.

        Args:
            algorithm: The factorization algorithm of type
            :class:`nvmath.bindings.cudss.AlgType` or Python `int`.

        """
        algorithm = cudss.AlgType(algorithm)
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.FACTORIZATION_ALG, self._factorization_alg, algorithm)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def pivot_eps_algorithm(self):
        """
        Query or set the algorithm used for pivot epsilon calculation. See
        :class:`nvmath.bindings.cudss.AlgType` and the `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_ for more
        information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.PIVOT_EPSILON_ALG, self._pivot_epsilon_alg)

        return cudss.AlgType(self._factorization_alg.item())

    @pivot_eps_algorithm.setter
    @utils.precondition(_check_valid_solver_wrapper)
    def pivot_eps_algorithm(self, algorithm):
        """
        Set the algorithm to use for pivot epsilon calculation. See
        :class:`nvmath.bindings.cudss.AlgType` and the `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_ for more
        information.

        Args:
            algorithm: The pivot epsilon algorithm of type
            :class:`nvmath.bindings.cudss.AlgType` or Python `int`.

        """
        algorithm = cudss.AlgType(algorithm)
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.PIVOT_EPSILON_ALG, self._pivot_epsilon_alg, algorithm)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def pivot_eps(self):
        """
        Query or set the pivot epsilon value. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.PIVOT_EPSILON, self._pivot_epsilon)

        return self._factorization_alg.item()

    @pivot_eps.setter
    @utils.precondition(_check_valid_solver_wrapper)
    def pivot_eps(self, epsilon):
        """
        Set the pivot epsilon value. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.

        Args:
            epsilon: The pivot epsilon value (Python numerical type).

        """
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.PIVOT_EPSILON, self._pivot_epsilon, epsilon)


class SolutionConfig:
    """
    An interface to configure :meth:`nvmath.sparse.advanced.DirectSolver.solve`. The
    current configuration can also be queried.
    """

    def __init__(self, solver):
        """
        ctor for internal use only.
        """
        self._solver = solver
        self._config_ptr = self._solver.config_ptr
        assert self._config_ptr is not None, "Internal error"

        get_dtype = cudss.get_config_param_dtype

        self._solve_alg = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.SOLVE_ALG))
        self._ir_n_steps = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.IR_N_STEPS))

    def _check_valid_solver_wrapper(self, *args, **kwargs):
        _check_valid_solver(self)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def solution_algorithm(self):
        """
        Query or set the solution algorithm used. See
        :class:`nvmath.bindings.cudss.AlgType` and the `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_ for more
        information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.SOLVE_ALG, self._solve_alg)

        return cudss.AlgType(self._solve_alg.item())

    @solution_algorithm.setter
    @utils.precondition(_check_valid_solver_wrapper)
    def solution_algorithm(self, algorithm):
        """
        Set the algorithm to use for solving. See :class:`nvmath.bindings.cudss.AlgType` and
        the `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_ for more
        information.

        Args:
            algorithm: The solution algorithm of type :class:`nvmath.bindings.cudss.AlgType`
            or Python `int`.

        """
        algorithm = cudss.AlgType(algorithm)
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.SOLVE_ALG, self._solve_alg, algorithm)

    @property
    @utils.precondition(_check_valid_solver_wrapper)
    def ir_num_steps(self):
        """
        Query or set the number of steps used for iterative refinement. Set the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.IR_N_STEPS, self._ir_n_steps)

        return self._ir_n_steps.item()

    @ir_num_steps.setter
    @utils.precondition(_check_valid_solver_wrapper)
    def ir_num_steps(self, num_steps):
        """
        Set the  number of steps to use for iterative refinement. Set the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.

        Args:
            num_steps: The number of steps to use for iterative refinement as a
                       Python `int`.

        """
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.IR_N_STEPS, self._ir_n_steps, num_steps)


class InternalConfig:
    """
    Interface class for internal use only. No precondition for `@property` since this is
    used in the ctor before `solver.valid_state` is set.
    """

    def __init__(self, solver):
        """
        ctor for internal use only.
        """
        self._solver = solver
        self._config_ptr = self._solver.config_ptr
        assert self._config_ptr is not None, "Internal error"

        get_dtype = cudss.get_config_param_dtype

        self._hybrid_mode = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.HYBRID_MODE))
        self._hybrid_device_memory_limit = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.HYBRID_DEVICE_MEMORY_LIMIT))
        self._use_cuda_register_memory = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.USE_CUDA_REGISTER_MEMORY))
        self._host_nthreads = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.HOST_NTHREADS))
        self._hybrid_execute_mode = np.zeros((1,), dtype=get_dtype(ConfigParamEnum.HYBRID_EXECUTE_MODE))

    @property
    def hybrid_mode(self):
        """
        Query or set the hybrid memory mode. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.HYBRID_MODE, self._hybrid_mode)

        return self._hybrid_mode.item()

    @hybrid_mode.setter
    def hybrid_mode(self, mode):
        """
        Set the hybrid memory mode. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.

        Args:
            mode: The hybrid memory mode as Python `int` (0=device or 1=hybrid).

        """
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.HYBRID_MODE, self._hybrid_mode, mode)

    @property
    def hybrid_device_memory_limit(self):
        """
        Query or set the hybrid device memory limit. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.HYBRID_DEVICE_MEMORY_LIMIT, self._hybrid_device_memory_limit)

        return self._hybrid_mode.item()

    @hybrid_device_memory_limit.setter
    def hybrid_device_memory_limit(self, limit):
        """
        Set the hybrid device memory limit. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.

        Args:
            limit: The hybrid memory limit as Python `int`.

        """
        _set_scalar_attribute(
            self._config_ptr, ConfigParamEnum.HYBRID_DEVICE_MEMORY_LIMIT, self._hybrid_device_memory_limit, limit
        )

    @property
    def use_cuda_register_memory(self):
        """
        Query or set the CUDA memory registration flag. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.USE_CUDA_REGISTER_MEMORY, self._use_cuda_register_memory)

        return self._use_cuda_register_memory.item()

    @use_cuda_register_memory.setter
    def use_cuda_register_memory(self, flag):
        """
        Set the CUDA memory registration flag. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.

        Args:
            flag: The memory registration flag as Python `int` (0 or 1).

        """
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.USE_CUDA_REGISTER_MEMORY, self._use_cuda_register_memory, flag)

    @property
    def host_nthreads(self):
        """
        Query or set the number of host threads. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.HOST_NTHREADS, self._host_nthreads)

        return self._host_nthreads.item()

    @host_nthreads.setter
    def host_nthreads(self, nthreads):
        """
        Set the number of host threads. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.

        Args:
            nthreads: The number of host threads as Python `int`.

        """
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.HOST_NTHREADS, self._host_nthreads, nthreads)

    @property
    def hybrid_execute_mode(self):
        """
        Query or set the hybrid execution mode. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.
        """
        _get_scalar_attribute(self._config_ptr, ConfigParamEnum.HYBRID_MODE, self._hybrid_mode)

        return self._hybrid_mode.item()

    @hybrid_execute_mode.setter
    def hybrid_execute_mode(self, mode):
        """
        Set the hybrid execution mode. See the
        `cuDSS documentation
        <https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t>`_
        for more information.

        Args:
            mode: The hybrid execution mode as Python `int` (0=device or 1=hybrid).

        """
        _set_scalar_attribute(self._config_ptr, ConfigParamEnum.HYBRID_EXECUTE_MODE, self._hybrid_execute_mode, mode)
