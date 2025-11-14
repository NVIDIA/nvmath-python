# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "MatmulEpilog",
    "MatmulAlgoType",
    "MatmulOptions",
    "MatmulEpilogPreferences",
    "MatmulPlanPreferences",
    "matrix_qualifiers_dtype",
]

import dataclasses
from logging import Logger
from typing import Literal

import numpy as _np

from nvmath.bindings import cublas  # type: ignore
from nvmath.bindings import cublasMp  # type: ignore
from nvmath.internal.utils import check_or_create_options
from nvmath._utils import CudaDataType

MatmulEpilog = cublasMp.MatmulEpilogue
MatmulAlgoType = cublasMp.MatmulAlgoType


@dataclasses.dataclass
class MatmulOptions:
    """A data class for providing options to the :class:`Matmul` object and the wrapper
    function :func:`matmul`.

    Attributes:
        compute_type (nvmath.distributed.linalg.ComputeType): CUDA compute type. A suitable
            compute type will be selected if not specified.

        scale_type (nvmath.CudaDataType): CUDA data type. A suitable data type consistent
            with the compute type will be selected if not specified.

        result_type (nvmath.CudaDataType): CUDA data type. A requested datatype of the
            result. If not specified, this type will be determined based on the input types.

        algo_type (nvmath.distributed.linalg.advanced.MatmulAlgoType): Hints the algorithm
            type to be used. If not supported, cuBLASMp will fallback to the default
            algorithm.

        sm_count_communication (int) : The number of SMs to use for communication. This is
            only relevant for some algorithms (please consult cuBLASMp documentation).

        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.

        blocking: A flag specifying the behavior of the execution functions and methods,
            such as :func:`matmul` and :meth:`Matmul.execute`. When ``blocking`` is `True`,
            the execution methods do not return until the operation is complete. When
            ``blocking`` is ``"auto"``, the methods return immediately when the inputs are
            on the GPU. The execution methods always block when the operands are on the CPU
            to ensure that the user doesn't inadvertently use the result before it becomes
            available. The default is ``"auto"``.

    .. seealso::
       :class:`Matmul`, :func:`matmul`
    """

    compute_type: int | None = None
    scale_type: int | None = None
    result_type: int | None = None
    algo_type: int | None = None
    sm_count_communication: int | None = None
    logger: Logger | None = None
    blocking: Literal[True, "auto"] = "auto"

    def __post_init__(self):
        if self.compute_type is not None:
            self.compute_type = cublas.ComputeType(self.compute_type)

        if self.scale_type is not None:
            self.scale_type = CudaDataType(self.scale_type)

        if self.algo_type is not None:
            self.algo_type = MatmulAlgoType(self.algo_type)

        if self.sm_count_communication is not None and not (
            isinstance(self.sm_count_communication, int) and self.sm_count_communication > 0
        ):
            raise ValueError("sm_count_communication must be a positive integer")

        if self.blocking not in (True, "auto"):
            raise ValueError("The value specified for blocking must be either True or 'auto'.")


matrix_qualifiers_dtype = _np.dtype([("structure", object), ("is_transpose", "<i1"), ("is_conjugate", "<i1")])


@dataclasses.dataclass
class MatmulEpilogPreferences:
    """A data class for providing epilog options as part of ``preferences`` to the
    :meth:`Matmul.plan` method and the wrapper function :func:`matmul`.

    Attributes:
        aux_type (nvmath.CudaDataType): The requested datatype of the
            epilog auxiliary output. If not specified, this type will be determined based on
            the input types. Non-default auxiliary output types are only supported for
            certain epilogs. For more details on the supported combinations, see
            ``CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_DATA_TYPE`` in cuBLASMp
            documentation.

    .. seealso::
       :meth:`Matmul.plan`, :func:`matmul`, :class:`MatmulPlanPreferences`
    """

    aux_type: int | None = None


@dataclasses.dataclass
class MatmulPlanPreferences:
    """A data class for providing options to the :meth:`Matmul.plan` method and the
    wrapper function :func:`matmul`.

    Attributes:
        epilog (:class:`nvmath.distributed.linalg.advanced.MatmulEpilogPreferences`):
            Epilog preferences (as an object of class
            :class:`~nvmath.distributed.linalg.advanced.MatmulEpilogPreferences`
            or a `dict`).

    .. seealso::
       :meth:`Matmul.plan`, :func:`matmul`
    """

    epilog: MatmulEpilogPreferences | None = None

    def __post_init__(self):
        self.epilog = check_or_create_options(MatmulEpilogPreferences, self.epilog, "epilog preferences")
