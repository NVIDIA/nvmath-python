# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "DirectSolverOptions",
    "DirectSolverMatrixType",
    "DirectSolverMatrixViewType",
    "ExecutionCUDA",
    "ExecutionHybrid",
    "HybridMemoryModeOptions",
]


from dataclasses import dataclass, field
from logging import Logger
from typing import ClassVar, Literal, TypeAlias

from nvmath.bindings import cudss
from nvmath.internal import mem_limit

DirectSolverMatrixType: TypeAlias = cudss.MatrixType
DirectSolverMatrixViewType: TypeAlias = cudss.MatrixViewType


@dataclass
class HybridMemoryModeOptions:
    """
    A data class for providing options related to the use of hybrid (CPU-GPU) memory to
    those execution spaces that support it.

    Attributes:
        hybrid_memory_mode: If set to True, use CPU memory to store factors (default:
            ``False``). See :attr:`nvmath.bindings.cudss.ConfigParam.HYBRID_MODE`.
        device_memory_limit: The maximum device memory available for execution. It can
            be specified as a value (with optional suffix like K[iB], M[iB], G[iB]) or as a
            percentage. The default is based on internal heuristics. See
            :attr:`nvmath.bindings.cudss.ConfigParam.HYBRID_DEVICE_MEMORY_LIMIT`.
        register_cuda_memory: Specify whether to register memory using
            ``cudaHostRegister()`` if hybrid memory mode is used. The default is
            ``True``. See
            :attr:`nvmath.bindings.cudss.ConfigParam.USE_CUDA_REGISTER_MEMORY`.

    See Also:
       :class:`ExecutionHybrid`, :class:`DirectSolver`, :func:`direct_solver`.
    """

    hybrid_memory_mode: bool = False
    hybrid_device_memory_limit: int | str | None = None  # Internal heuristic.
    register_cuda_memory: bool = True

    def __post_init__(self):
        if self.hybrid_device_memory_limit is not None:
            mem_limit.check_memory_str(self.hybrid_device_memory_limit, "hybrid device memory limit")


@dataclass
class ExecutionCUDA:
    """
    A data class for providing GPU execution options to the :class:`DirectSolver`
    object and the wrapper function :func:`direct_solver`.

    Attributes:
        device_id: CUDA device ordinal (only used if the operand resides on the CPU). The
            default value is 0.
        hybrid_memory_mode_options: Options controlling the use of hybrid (CPU-GPU) memory
            as an object of type :class:`HybridMemoryModeOptions` or a `dict`.

    See Also:
       :class:`ExecutionHybrid`, :class:`HybridMemoryModeOptions`, :class:`DirectSolver`,
       :func:`direct_solver`.
    """

    name: ClassVar[Literal["cuda"]] = "cuda"
    device_id: int = 0
    hybrid_memory_mode_options: object = field(default_factory=HybridMemoryModeOptions)


@dataclass
class ExecutionHybrid:
    """
    A data class for providing hybrid (CPU-GPU) execution options to the
    :class:`DirectSolver` object and the wrapper function :func:`direct_solver`.

    Attributes:
        device_id: CUDA device ordinal (only used if the operand resides on the CPU). The
            default value is 0.
        num_threads: The number of CPU threads used to execute the plan.
            If not specified, defaults to the number of CPU cores available to the process.

    See Also:
       :class:`ExecutionCUDA`, :class:`DirectSolver`, :func:`direct_solver`.
    """

    name: ClassVar[Literal["hybrid"]] = "hybrid"
    device_id: int = 0
    num_threads: int | None = None


# TODO: docstring
@dataclass
class DirectSolverOptions:
    """
    A data class for providing options to the :class:`DirectSolver` object and the wrapper
    function :func:`direct_solver`.

    Attributes:
        sparse_system_type (:class:`DirectSolverMatrixType`): The type of the sparse
            system of equations (general, symmetric, symmetric positive definite, etc).
            The default is ``DirectSolverMatrixType.GENERAL``.

        sparse_system_view (:class:`DirectSolverMatrixViewType`): The desired view of the
            sparse system of equations (full, upper, lower). The default is
            ``DirectSolverMatrixViewType.FULL``.

        multithreading_lib: The location (full path) to the library implementing the
            threading layer interface. **TODO**: link to docs and default pip path.

        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.

        blocking: A flag specifying the behavior of the execution functions and methods,
            such as :func:`direct_solver` and :meth:`DirectSolver.solve`.
            When ``blocking`` is `True`, the execution methods do not return until the
            operation is complete. When
            ``blocking`` is ``"auto"``, the methods return immediately when the input tensor
            is on the GPU and ``execution`` is set to 'CUDA'. The execution methods always
            block when the input tensor is on the CPU or ``execution`` is specified to 'CPU'
            to ensure that the user doesn't inadvertently use the result before it becomes
            available. The default is ``"auto"``.

        handle: cuDSS library handle. A handle will be created if one is not provided.

    See Also:
        :class:`ExecutionCUDA`, :class:`ExecutionHybrid`, :class:`DirectSolver`, and
        :func:`direct_solver`.
    """

    sparse_system_type: int = DirectSolverMatrixType.GENERAL
    sparse_system_view: int = DirectSolverMatrixViewType.FULL
    multithreading_lib: str | None = None
    logger: Logger | None = None
    blocking: Literal[True, "auto"] = "auto"
    handle: int | None = None

    def __post_init__(self):
        if self.blocking not in (True, "auto"):
            raise ValueError("The value specified for 'blocking' must be either True or 'auto'.")
        self.sparse_system_type = DirectSolverMatrixType(self.sparse_system_type)
        self.sparse_system_view = DirectSolverMatrixViewType(self.sparse_system_view)
