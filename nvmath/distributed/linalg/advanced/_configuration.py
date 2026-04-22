# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "MatmulEpilog",
    "MatmulAlgoType",
    "MatmulOptions",
    "MatmulEpilogPreferences",
    "MatmulPlanPreferences",
    "MatmulQuantizationScales",
    "matrix_qualifiers_dtype",
]

import dataclasses
from logging import Logger
from typing import Literal

import numpy as _np

from nvmath._utils import CudaDataType
from nvmath.bindings import (
    cublas,  # type: ignore
    cublasMp,  # type: ignore
)
from nvmath.internal.utils import check_or_create_options
from nvmath.linalg.advanced import MatmulQuantizationScales

MatmulEpilog = cublasMp.MatmulEpilogue
MatmulAlgoType = cublasMp.MatmulAlgoType


@dataclasses.dataclass
class MatmulOptions:
    """A data class for providing options to the :class:`Matmul` object and the wrapper
    function :func:`matmul`.

    Attributes:
        inplace: Whether the matrix multiplication is performed in-place (operand C is
            overwritten). The default is ``inplace=False``.

        compute_type (nvmath.distributed.linalg.ComputeType): CUDA compute type. A suitable
            compute type will be selected if not specified.

        scale_type (nvmath.CudaDataType): CUDA data type. A suitable data type consistent
            with the compute type will be selected if not specified.

        result_type (nvmath.CudaDataType): CUDA data type. A requested datatype of the
            result. If not specified, this type will be determined based on the input types.
            Non-default result types are only supported for narrow-precision (FP8 and lower)
            operations.

        algo_type (nvmath.distributed.linalg.advanced.MatmulAlgoType): Hints the algorithm
            type to be used. If not supported, cuBLASMp will fallback to the default
            algorithm.

        result_amax (bool): If set, the absolute maximum (amax) of the result will be
            returned in the auxiliary output tensor. Only supported for narrow-precision
            (FP8 and lower) operations.

        block_scaling (bool): If set, block scaling (MXFP8) will be used instead of
            tensor-wide scaling for FP8 operations. If the result is a narrow-precision
            (FP8 and lower) data type, scales used for result quantization will be returned
            in the auxiliary output tensor as ``"d_out_scale"`` in UE8M0 format. For more
            information on UE8M0 format, see the documentation of
            :class:`~linalg.advanced.MatmulQuantizationScales`.
            This option is only supported for narrow-precision (FP8 and lower) operations.

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

    inplace: bool = False
    compute_type: int | None = None
    scale_type: int | None = None
    result_type: int | None = None
    algo_type: int | None = None
    result_amax: bool = False
    block_scaling: bool = False
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
            narrow-precision operations and certain epilogs. For more details on the
            supported combinations, see
            ``CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_DATA_TYPE`` in cuBLASMp
            documentation. If this option is set to a narrow-precision data type, an
            additional epilog input ``"aux_quantization_scale"`` needs to be specified.

        aux_amax (bool): If set, the absolute maximum (amax) of the epilog
            auxiliary output will be returned in the auxiliary output tensor.
            Only supported when ``aux_type`` option is set to a narrow-precision
            data type.

    .. seealso::
       :meth:`Matmul.plan`, :func:`matmul`, :class:`MatmulPlanPreferences`
    """

    aux_type: int | None = None
    aux_amax: bool = False


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
