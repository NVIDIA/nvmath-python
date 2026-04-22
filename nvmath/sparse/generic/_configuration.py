# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "ExecutionCUDA",
    "MatmulOptions",
    "matrix_qualifiers_dtype",
    "ComputeType",
]

import dataclasses
from enum import IntEnum
from logging import Logger
from typing import Literal

import numpy as _np

from nvmath._internal import templates
from nvmath._utils import CudaDataType
from nvmath.internal.mem_limit import check_memory_str
from nvmath.memory import BaseCUDAMemoryManager, BaseCUDAMemoryManagerAsync


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionCUDA(templates.ExecutionCUDA):
    """
    A data class for providing GPU execution options.

    Attributes:
        device_id: CUDA device ordinal (only used if the operand resides on the CPU). The
            default value is 0.

    .. seealso::
        :class:`ExecutionCPU`, :class:`Matmul`, :func:`matmul`
    """

    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionCPU(templates.ExecutionCPU):
    """
    A data class for providing CPU execution options.

    Attributes:
        num_threads: The number of CPU threads used to execute the operation.
                     If not specified, defaults to the number of CPU cores available to the
                     process.

    .. seealso::
       :class:`ExecutionCUDA`, :class:`Matmul`, :func:`matmul`,
    """

    pass


@dataclasses.dataclass
class MatmulOptions:
    """A data class for providing options to the :class:`Matmul` object and the wrapper
    function :func:`matmul`.

    Attributes:
        codegen (bool): If ``True`` the operation will not be dispatched to cuSPARSE
            (even if it is supported) for UST operands. Instead, the kernel will be
            generated, compiled just-in-time, and used. The default is ``False``.

        compute_type (nvmath.sparse.ComputeType): CUDA compute type. A suitable compute type
            will be selected if not specified.

        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.

        memory_limit: Maximum memory available to the MM operation. It can be specified as a
            value (with optional suffix like K[iB], M[iB], G[iB]) or as a percentage. The
            default is 80% of the device memory.

        blocking: A flag specifying the behavior of the execution functions and methods,
            such as :func:`matmul` and :meth:`Matmul.execute`. When ``blocking`` is `True`,
            the execution methods do not return until the operation is complete. When
            ``blocking`` is ``"auto"``, the methods return immediately when the inputs are
            on the GPU. The execution methods always block when the operands are on the CPU
            to ensure that the user doesn't inadvertently use the result before it becomes
            available. The default is ``"auto"``.

        allocator: An object that supports the :class:`BaseCUDAMemoryManager` protocol, used
            to draw device memory. If an allocator is not provided, a memory allocator from
            the library package will be used (:func:`torch.cuda.caching_allocator_alloc` for
            PyTorch operands, :func:`cupy.cuda.alloc` otherwise).

    .. seealso::
       :class:`Matmul`, :func:`matmul`
    """

    codegen: bool = False
    compute_type: int | None = None
    logger: Logger | None = None
    memory_limit: int | str | None = r"80%"
    blocking: Literal[True, "auto"] = "auto"
    allocator: BaseCUDAMemoryManager | None = None

    def __post_init__(self):
        #  Defer computing the memory limit till we know the device the operands are on.

        if self.compute_type is not None:
            self.compute_type = ComputeType(self.compute_type)

        if self.memory_limit is None:
            self.memory_limit = MatmulOptions.memory_limit

        check_memory_str(self.memory_limit, "memory limit")

        if self.blocking not in (True, "auto"):
            raise ValueError("The value specified for blocking must be either True or 'auto'.")

        if self.allocator is not None and not isinstance(self.allocator, BaseCUDAMemoryManager | BaseCUDAMemoryManagerAsync):
            raise TypeError("The allocator must be an object of type that fulfills the BaseCUDAMemoryManager protocol.")


matrix_qualifiers_dtype = _np.dtype([("is_transpose", "<i4"), ("is_conjugate", "<i4")])


class ComputeType(IntEnum):
    """The subset of `cudaDataType_t` that is supported by the generic sparse APIs."""

    CUDA_R_32F = CudaDataType.CUDA_R_32F
    CUDA_C_32F = CudaDataType.CUDA_C_32F
    CUDA_R_64F = CudaDataType.CUDA_R_64F
    CUDA_C_64F = CudaDataType.CUDA_C_64F
