# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "MatmulEpilog",
    "MatmulInnerShape",
    "MatmulNumericalImplFlags",
    "MatmulOptions",
    "MatmulEpilogPreferences",
    "MatmulPlanPreferences",
    "MatmulQuantizationScales",
    "MatmulReductionScheme",
    "matrix_qualifiers_dtype",
]

import dataclasses
from enum import IntEnum
from logging import Logger
from typing import Literal

import numpy as _np

from nvmath.bindings import cublas  # type: ignore
from nvmath.bindings import cublasLt as cublaslt  # type: ignore
from nvmath.internal import enum_utils
from nvmath.internal.utils import check_or_create_options
from nvmath.internal.mem_limit import check_memory_str
from nvmath.memory import BaseCUDAMemoryManager, BaseCUDAMemoryManagerAsync
from nvmath._utils import CudaDataType

MatmulEpilog = cublaslt.Epilogue
MatmulInnerShape = cublaslt.MatmulInnerShape
MatmulReductionScheme = cublaslt.ReductionScheme


@dataclasses.dataclass
class MatmulOptions:
    """A data class for providing options to the :class:`Matmul` object and the wrapper
    function :func:`matmul`.

    Attributes:
        compute_type (nvmath.linalg.ComputeType): CUDA compute type. A suitable compute type
            will be selected if not specified.

        scale_type (nvmath.CudaDataType): CUDA data type. A suitable data type consistent
            with the compute type will be selected if not specified.

        result_type (nvmath.CudaDataType): CUDA data type. A requested datatype of the
            result. If not specified, this type will be determined based on the input types.
            Non-default result types are only supported for narrow-precision (FP8 and lower)
            operations.

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

        sm_count_target (int) : The number of SMs to use for execution. The default is 0,
            corresponding to all available SMs.

        fast_accumulation (bool) : Enable or disable FP8 fast accumulation mode. The default
            is False (disabled).

        device_id: CUDA device ordinal (only used if the operand resides on the CPU). The
            default value is 0.

        handle: Linear algebra library handle. A handle will be created if one is not
            provided.

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

    See Also:
       :class:`Matmul`, :func:`matmul`
    """

    compute_type: int | None = None
    scale_type: int | None = None
    result_type: int | None = None
    result_amax: bool = False
    block_scaling: bool = False
    sm_count_target: int | None = 0
    fast_accumulation: bool | None = False
    device_id: int = 0
    handle: int | None = None
    logger: Logger | None = None
    memory_limit: int | str | None = r"80%"
    blocking: Literal[True, "auto"] = "auto"
    allocator: BaseCUDAMemoryManager | None = None

    def __post_init__(self):
        #  Defer computing the memory limit till we know the device the network is on.

        if self.compute_type is not None:
            self.compute_type = cublas.ComputeType(self.compute_type)

        if self.scale_type is not None:
            self.scale_type = CudaDataType(self.scale_type)

        if self.sm_count_target is None:
            self.sm_count_target = MatmulOptions.sm_count_target

        if self.fast_accumulation is None:
            self.fast_accumulation = MatmulOptions.fast_accumulation

        if self.device_id is None:
            self.device_id = 0

        if self.memory_limit is None:
            self.memory_limit = MatmulOptions.memory_limit

        check_memory_str(self.memory_limit, "memory limit")

        if self.blocking not in (True, "auto"):
            raise ValueError("The value specified for blocking must be either True or 'auto'.")

        if self.allocator is not None and not isinstance(self.allocator, BaseCUDAMemoryManager | BaseCUDAMemoryManagerAsync):
            raise TypeError("The allocator must be an object of type that fulfils the BaseCUDAMemoryManager protocol.")


matrix_qualifiers_dtype = _np.dtype([("structure", object), ("is_conjugate", "<i4")])


class MatmulNumericalImplFlags(IntEnum):
    """
    These flags can be combined with the | operator: OP_TYPE_FMA | OP_TYPE_TENSOR_HMMA ...
    """

    OP_TYPE_FMA = 0x01 << 0
    OP_TYPE_TENSOR_HMMA = 0x02 << 0
    OP_TYPE_TENSOR_IMMA = 0x04 << 0
    OP_TYPE_TENSOR_DMMA = 0x08 << 0
    OP_TYPE_TENSOR_MASK = 0xFE << 0
    OP_TYPE_MASK = 0xFF << 0

    ACCUMULATOR_16F = 0x01 << 8
    ACCUMULATOR_32F = 0x02 << 8
    ACCUMULATOR_64F = 0x04 << 8
    ACCUMULATOR_32I = 0x08 << 8
    ACCUMULATOR_TYPE_MASK = 0xFF << 8

    INPUT_TYPE_16F = 0x01 << 16
    INPUT_TYPE_16BF = 0x02 << 16
    INPUT_TYPE_TF32 = 0x04 << 16
    INPUT_TYPE_32F = 0x08 << 16
    INPUT_TYPE_64F = 0x10 << 16
    INPUT_TYPE_8I = 0x20 << 16
    INPUT_TYPE_8F_E4M3 = 0x40 << 16
    INPUT_TYPE_8F_E5M2 = 0x80 << 16
    INPUT_TYPE_MASK = 0xFF << 16

    GAUSSIAN = 0x01 << 32

    ALL = (1 << 64) - 1


@dataclasses.dataclass
class MatmulEpilogPreferences:
    """A data class for providing epilog options as part of ``preferences`` to the
    :meth:`Matmul.plan` method and the wrapper function :func:`matmul`.

    Attributes:
        aux_type (nvmath.CudaDataType): The requested datatype of the
            epilog auxiliary output. If not specified, this type will be determined based on
            the input types. Non-default auxiliary output types are only supported for
            narrow-precision operations and certain epilogs. For more details on the
            supported combinations, see ``CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`` in
            cuBLAS documentation. If this option is set to a narrow-precision data type,
            an additional epilog input ``"aux_quantization_scale"`` needs to be specified.

        aux_amax (bool): If set, the absolute maximum (amax) of the epilog
            auxiliary output will be returned in the auxiliary output tensor.
            Only supported when ``aux_type`` option is set to a narrow-precision
            data type.

    See Also:
       :meth:`Matmul.plan`, :func:`matmul`, :class:`MatmulPlanPreferences`
    """

    aux_type: int | None = None
    aux_amax: bool = False


@dataclasses.dataclass
class MatmulPlanPreferences:
    """A data class for providing options to the :meth:`Matmul.plan` method and the
    wrapper function :func:`matmul`.

    Attributes:
        reduction_scheme_mask (:class:`nvmath.linalg.advanced.MatmulReductionScheme`):
            Enumerators from
            :class:`nvmath.linalg.advanced.MatmulReductionScheme` combined with
            bitwise operator ``|``. The default is all reduction schemes.

        max_waves_count (float) : The maximum wave count. Selecting a value greater than 0.
            will exclude algorithms with device utilization greater than specified. The
            default is 0.

        numerical_impl_mask (:class:`nvmath.linalg.advanced.MatmulNumericalImplFlags`):
            Enumerators from
            :class:`nvmath.nvmath.linalg.advanced.MatmulNumericalImplFlags` combined with
            bitwise operator ``|``. The default is all numerical implementation flag
            choices.

        limit (int) : The number of algorithms to consider. If not specified, a suitable
            default will be chosen.

        epilog (:class:`nvmath.linalg.advanced.MatmulEpilogPreferences`):
            Epilog preferences (as an object of class
            :class:`~nvmath.linalg.advanced.MatmulEpilogPreferences` or a `dict`).

    See Also:
       :meth:`Matmul.plan`, :func:`matmul`
    """

    reduction_scheme_mask: cublaslt.ReductionScheme | None = cublaslt.ReductionScheme.MASK
    max_waves_count: float | None = 0.0
    numerical_impl_mask: MatmulNumericalImplFlags | None = MatmulNumericalImplFlags.ALL
    limit: int = 8
    epilog: MatmulEpilogPreferences | None = None

    def __post_init__(self):
        if self.reduction_scheme_mask is None:
            self.reduction_scheme_mask = MatmulPlanPreferences.reduction_scheme_mask
        else:
            self.reduction_scheme_mask = cublaslt.ReductionScheme(self.reduction_scheme_mask)

        if self.max_waves_count is None:
            self.max_waves_count = MatmulPlanPreferences.max_waves_count

        if self.numerical_impl_mask is None:
            self.numerical_impl_mask = MatmulPlanPreferences.numerical_impl_mask

        if self.limit is None:
            self.limit = MatmulPlanPreferences.limit

        self.epilog = check_or_create_options(MatmulEpilogPreferences, self.epilog, "epilog preferences")


@dataclasses.dataclass
class MatmulQuantizationScales:
    """A data class for providing quantization_scales to :class:`Matmul` constructor and the
    wrapper function :func:`matmul`.

    Scales can only be set for narrow-precision (FP8 and lower) matrices.

    When ``MatmulOptions.block_scaling=False``, each scale can either be a scalar (integer
    or float) or a single-element tensor of shape ``()`` or ``(1,)``.

    When ``MatmulOptions.block_scaling=True``, each scale should be a 1D ``uint8`` tensor
    with layout matching the requirements of cuBLAS MXFP8 scaling tensor. Values in the
    tensor will be interpreted as UE8M0 values. This means that a value :math:`x` in the
    scaling tensor will cause cuBLAS to multiply the respective block by :math:`2^{x-127}`.

    Attributes:
        a (float or Tensor) : Scale for matrix A.

        b (float or Tensor) : Scale for matrix B.

        c (float or Tensor) : Scale for matrix C.

        d (float or Tensor) : Scale for matrix D.

    See Also:
       :class:`Matmul`, :func:`matmul`
    """

    a: float | None = None
    b: float | None = None
    c: float | None = None
    d: float | None = None


_create_options = enum_utils.create_options_class_from_enum
_algo_cap_enum = cublaslt.MatmulAlgoCapAttribute
_get_dtype = cublaslt.get_matmul_algo_cap_attribute_dtype

AlgorithmCapabilities = _create_options(
    "AlgorithmCapabilities", _algo_cap_enum, _get_dtype, "algorithm capabilities", "(?P<option_name>.*)"
)


def algorithm_capabilities_str(self):
    names = [field.name for field in dataclasses.fields(self)]
    width = max(len(n) for n in names)
    s = """Algorithm Capabilities (refer to `MatmulAlgoCapAttribute` for documentation):
"""
    for name in names:
        s += f"    {name:{width}} = {getattr(self, name)}\n"
    return s


AlgorithmCapabilities.__str__ = algorithm_capabilities_str

del _create_options, _algo_cap_enum, _get_dtype
