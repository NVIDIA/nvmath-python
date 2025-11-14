# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "ContractionAlgo",
    "ContractionAutotuneMode",
    "ContractionJitMode",
    "ContractionCacheMode",
    "ContractionOptions",
    "ExecutionCUDA",
]


from dataclasses import dataclass
from logging import Logger
from typing import ClassVar, Literal

from nvmath.bindings import cutensor
from nvmath.memory import BaseCUDAMemoryManager


ContractionAlgo = cutensor.Algo
ContractionAutotuneMode = cutensor.AutotuneMode
ContractionJitMode = cutensor.JitMode
ContractionCacheMode = cutensor.CacheMode


@dataclass
class ContractionOptions:
    """
    A data class for providing options to the :class:`BinaryContraction` and
    :class:`TernaryContraction` objects, or the wrapper functions
    :func:`binary_contraction`and :func:`ternary_contraction`.

    Attributes:
        compute_type: The compute type to use for the contraction.
            See :class:`~nvmath.tensor.ComputeDesc` for available compute types.
        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.

        blocking: A flag specifying the behavior of the execution functions and methods,
            such as :func:`binary_contraction` and :meth:`TernaryContraction.execute`.
            When ``blocking`` is `True`, the execution methods do not return until the
            operation is complete. When
            ``blocking`` is ``"auto"``, the methods return immediately when the input tensor
            is on the GPU. The execution methods always block when the input tensor is
            on the CPU to ensure that the user doesn't inadvertently use the result
            before it becomes available. The default is ``"auto"``.

        handle: cuTensor library handle. A handle will be created if one is not provided.

        allocator: An object that supports the :class:`BaseCUDAMemoryManager` protocol, used
            to draw device memory. If an allocator is not provided, a memory allocator from
            the library package will be used (:func:`torch.cuda.caching_allocator_alloc` for
            PyTorch operands, :func:`cupy.cuda.alloc` otherwise).

        memory_limit: Maximum memory available to the contraction operation.
            It can be specified as a value (with optional suffix like K[iB], M[iB],
            G[iB]) or as a percentage. The default is 80% of the device memory.

    """

    compute_type: int | None = None
    logger: Logger | None = None
    blocking: Literal[True, "auto"] = "auto"
    handle: int | None = None
    allocator: BaseCUDAMemoryManager | None = None
    memory_limit: int | str | None = r"80%"

    def __post_init__(self):
        if self.blocking not in (True, "auto"):
            raise ValueError("The value specified for 'blocking' must be either True or 'auto'.")


@dataclass
class ExecutionCUDA:
    """
    A data class for providing GPU execution options to the :class:`BinaryContraction` and
    :class:`TernaryContraction` objects, or the wrapper functions
    :func:`binary_contraction`and :func:`ternary_contraction`.

    Attributes:
        device_id: CUDA device ordinal (only used if the operand resides on the CPU). The
            default value is 0.

    """

    name: ClassVar[Literal["cuda"]] = "cuda"
    device_id: int = 0
