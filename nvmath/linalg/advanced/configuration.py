# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ['MatmulEpilog', 'MatmulInnerShape','MatmulNumericalImplFlags', 'MatmulOptions', 'MatmulPlanPreferences', 'MatmulReductionScheme', 'matrix_qualifiers_dtype']

import dataclasses
from enum import IntEnum
from logging import Logger
from typing import Literal, Optional, Union

import numpy as _np

from nvmath.bindings import cublas
from nvmath.bindings import cublasLt as cublaslt
from nvmath._internal import enum_utils
from nvmath._internal.mem_limit import check_memory_str
from nvmath._internal.mem_limit import MEM_LIMIT_RE_PCT, MEM_LIMIT_RE_VAL, MEM_LIMIT_DOC
from nvmath.memory import BaseCUDAMemoryManager
from nvmath._utils import CudaDataType

MatmulEpilog = cublaslt.Epilogue
MatmulInnerShape = cublaslt.MatmulInnerShape
MatmulReductionScheme = cublaslt.ReductionScheme

@dataclasses.dataclass
class MatmulOptions:
    """A data class for providing options to the :class:`Matmul` object and the wrapper function :func:`matmul`.

    Attributes:
        compute_type (nvmath.linalg.ComputeType): CUDA compute type. A suitable compute type will be selected if not specified.
        scale_type (nvmath.CudaDataType): CUDA data type. A suitable data type consistent with the compute type will be
            selected if not specified.
        sm_count_target (int) : The number of SMs to use for execution. The default is 0, corresponding to all available SMs.
        fast_accumulation (bool) : Enable or disable FP8 fast accumulation mode. The default is False (disabled).
        device_id: CUDA device ordinal (used if the MM operands reside on the CPU). Device 0 will be used if not specified.
        handle: Linear algebra library handle. A handle will be created if one is not provided.
        logger (logging.Logger): Python Logger object. The root logger will be used if a logger object is not provided.
        memory_limit: Maximum memory available to the MM operation. It can be specified as a value (with optional suffix like
            K[iB], M[iB], G[iB]) or as a percentage. The default is 80% of the device memory.
        blocking: A flag specifying the behavior of the execution functions and methods, such as :func:`matmul` and :meth:`Matmul.execute`.
            When ``blocking`` is `True`, the execution methods do not return until the operation is complete. When ``blocking`` is
            ``"auto"``, the methods return immediately when the inputs are on the GPU. The execution methods always block
            when the operands are on the CPU to ensure that the user doesn't inadvertently use the result before it becomes
            available. The default is ``"auto"``.
        allocator: An object that supports the :class:`BaseCUDAMemoryManager` protocol, used to draw device memory. If an
            allocator is not provided, a memory allocator from the library package will be used
            (:func:`torch.cuda.caching_allocator_alloc` for PyTorch operands, :func:`cupy.cuda.alloc` otherwise).

    See Also:
       :class:`Matmul`, :func:`matmul`
    """
    compute_type : Optional[int] = None
    scale_type : Optional[int] = None
    sm_count_target : Optional[int] = 0
    fast_accumulation : Optional[bool] = False
    device_id : Optional[int] = None
    handle : Optional[int] = None
    logger : Optional[Logger] = None
    memory_limit : Optional[Union[int, str]] = r'80%'
    blocking : Literal[True, "auto"] = "auto"
    allocator : Optional[BaseCUDAMemoryManager] = None

    def __post_init__(self):
        #  Defer computing the memory limit till we know the device the network is on.

        if self.compute_type is not None:
            self.compute_type = cublas.ComputeType(self.compute_type)

        if self.scale_type is not None:
            self.scale_type = CudaDataType(self.scale_type)

        if self.device_id is None:
            self.device_id = 0

        check_memory_str(self.memory_limit, "memory limit")

        if self.blocking != True and self.blocking != "auto":
            raise ValueError("The value specified for blocking must be either True or 'auto'.")

        if self.allocator is not None and not isinstance(self.allocator, BaseCUDAMemoryManager):
            raise TypeError("The allocator must be an object of type that fulfils the BaseCUDAMemoryManager protocol.")

matrix_qualifiers_dtype = _np.dtype([("structure", object), ("is_conjugate", "<i4")])

class MatmulNumericalImplFlags(IntEnum):
    """
    These flags can be combined with the | operator: OP_TYPE_FMA | OP_TYPE_TENSOR_HMMA ...
    """
    OP_TYPE_FMA         = 0x01 << 0
    OP_TYPE_TENSOR_HMMA = 0x02 << 0
    OP_TYPE_TENSOR_IMMA = 0x04 << 0
    OP_TYPE_TENSOR_DMMA = 0x08 << 0
    OP_TYPE_TENSOR_MASK = 0xfe << 0
    OP_TYPE_MASK        = 0xff << 0

    ACCUMULATOR_16F       = 0x01 << 8
    ACCUMULATOR_32F       = 0x02 << 8
    ACCUMULATOR_64F       = 0x04 << 8
    ACCUMULATOR_32I       = 0x08 << 8
    ACCUMULATOR_TYPE_MASK = 0xff << 8

    INPUT_TYPE_16F     = 0x01 << 16
    INPUT_TYPE_16BF    = 0x02 << 16
    INPUT_TYPE_TF32    = 0x04 << 16
    INPUT_TYPE_32F     = 0x08 << 16
    INPUT_TYPE_64F     = 0x10 << 16
    INPUT_TYPE_8I      = 0x20 << 16
    INPUT_TYPE_8F_E4M3 = 0x40 << 16
    INPUT_TYPE_8F_E5M2 = 0x80 << 16
    INPUT_TYPE_MASK    = 0xff << 16

    GAUSSIAN = 0x01 << 32

@dataclasses.dataclass
class MatmulPlanPreferences:
    """A data class for providing options to the :meth:`Matmul.plan` method and the wrapper function :func:`matmul`.

    Attributes:
        reduction_scheme_mask (object of type :class:`linalg.advanced.MatmulReductionScheme`) : Enumerators from :class:`linalg.advanced.MatmulReductionScheme`
            combined with bitwise operator ``|``. The default is all reduction schemes.
        max_waves_count (float) : The maximum wave count. Selecting a value greater than 0. will exclude algorithms with
            device utilization greater than specified. The default is 0.
        numerical_impl_mask (object of type :class:`linalg.advanced.MatmulNumericalImplFlags`) : Enumerators from :class:`linalg.advanced.MatmulNumericalImplFlags`
            combined with bitwise operator ``|``. The default is all numerical implementation flag choices.
        limit (int) : The number of algorithms to consider. If not specified, a suitable default will be chosen.

    See Also:
       :meth:`Matmul.plan`, :func:`matmul`
    """

    reduction_scheme_mask : Optional[cublaslt.ReductionScheme] = cublaslt.ReductionScheme.MASK
    max_waves_count : Optional[float] = 0.
    numerical_impl_mask : Optional[MatmulNumericalImplFlags] = (1 << 64) - 1
    limit : int = 8

    def __post_init__(self):
        if self.reduction_scheme_mask is not None:
            self.reduction_scheme_mask = cublaslt.ReductionScheme(self.reduction_scheme_mask)


_create_options = enum_utils.create_options_class_from_enum
_algo_cap_enum = cublaslt.MatmulAlgoCapAttribute
_get_dtype = cublaslt.get_matmul_algo_cap_attribute_dtype

AlgorithmCapabilities = _create_options('AlgorithmCapabilities', _algo_cap_enum, _get_dtype, "algorithm capabilities", '(?P<option_name>.*)')
def algorithm_capabilities_str(self):
    names = [field.name for field in dataclasses.fields(self)]
    width = max(len(n) for n in names)
    s = f"""Algorithm Capabilities (refer to `MatmulAlgoCapAttribute` for documentation):
"""
    for name in names:
        s += f"    {name:{width}} = {getattr(self, name)}\n"
    return s
AlgorithmCapabilities.__str__ = algorithm_capabilities_str

del _create_options, _algo_cap_enum, _get_dtype
