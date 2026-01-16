# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["LeadingDimension", "TransposeMode", "Arrangement"]

from collections import namedtuple
from functools import lru_cache
from typing import NamedTuple, Protocol
from collections.abc import Sequence
import weakref

import numpy as np


from .common import check_in, check_sm
from .common_backend import (
    EXECUTION_STR_TO_MATHDX,
    NP_TYPES_TO_MATHDX_PRECISION,
    NP_TYPES_TO_MATHDX_TYPES,
    NVARG_GEN_OPT_LTO,
    build_get_int_traits,
    build_get_str_trait,
    get_isa_version,
    get_lto,
    get_mathdx_sm,
    set_code_target_sm,
)
from .common_cuda import Code, CodeType, Dim3, ComputeCapability, ISAVersion
from .common_backend import DescriptorWrapper
from .types import REAL_NP_TYPES, INT_NP_TYPES

from nvmath.bindings import mathdx

try:
    from cuda.core import (
        ObjectCode,
        ProgramOptions,
        LinkerOptions,
        Program,
        Linker,
        Kernel,
    )
except ImportError:
    from cuda.core.experimental import (
        ObjectCode,
        ProgramOptions,
        LinkerOptions,
        Program,
        Linker,
        Kernel,
    )


_BLAS_API_STR_TO_MATHDX = {
    "static_leading_dimensions": mathdx.CublasdxApi.SMEM,
    "dynamic_leading_dimensions": mathdx.CublasdxApi.SMEM_DYNAMIC_LD,
    "tensors": mathdx.CublasdxApi.TENSORS,
}

_BLAS_ARRENGEMENT_STR_TO_MATHDX = {a.name.lower(): a for a in mathdx.CublasdxArrangement}

_BLAS_TYPE_STR_TO_MATHDX = {t.name.lower(): t for t in mathdx.CublasdxType}

_BLAS_TRANSPOSE_STR_TO_MATHDX = {tm.name.lower(): tm for tm in mathdx.CublasdxTransposeMode}

_BLAS_FUNCTION_STR_TO_MATHDX = {
    "MM": mathdx.CublasdxFunction.MM,
}

_TENSOR_TYPE_STR_TO_MATHDX = {t.name.lower(): t for t in mathdx.CublasdxTensorType}


def check_blas_sm(sm, library_name: str, var_name: str = "sm"):
    check_sm(sm, library_name, var_name)
    if sm.arch == "a" and sm.integer not in {900, 1000, 1030, 1100}:
        raise ValueError(f"SM modifiers other than generic are only supported for SM 900, 1000, 1030, 1100, got {sm}")


class LeadingDimension(namedtuple("LeadingDimension", ["a", "b", "c"])):
    """
    A namedtuple class that encapsulates the three leading dimensions in matrix
    multiplication :math:`C = \\alpha Op(A) Op(B) + \\beta C`.

    Attributes:
        a (int): The leading dimension of two-dimensional array used to store the matrix
            ``A``.

        b (int): The leading dimension of two-dimensional array used to store the matrix
            ``B``.

        c (int): The leading dimension of two-dimensional array used to store the matrix
            ``C``.
    """

    __slots__ = ()


class TransposeMode(namedtuple("TransposeMode", ["a", "b"])):
    """
    A namedtuple class that encapsulates the transpose mode for input matrices ``A`` and
    ``B`` in matrix multiplication.

    Attributes:
        a: The operation that needs to be performed with input matrix ``A``, currently
            supports ``'non_transposed'``, ``'transposed'`` and ``'conj_transposed'``.

        b: The operation that needs to be performed with input matrix ``B``, currently
            supports ``'non_transposed'``, ``'transposed'`` and ``'conj_transposed'``.
    """

    __slots__ = ()


class Arrangement(NamedTuple):
    """
    A namedtuple class that encapsulates the three arrangements in matrix allocation.

    Attributes:
        a (str): The arrangement of two-dimensional array used to store the matrix
            ``A``.

        b (str): The arrangement of two-dimensional array used to store the matrix
            ``B``.

        c (str): The arrangement of two-dimensional array used to store the matrix
            ``C``.
    """

    a: str
    b: str
    c: str


class Precision(NamedTuple):
    """
    A namedtuple class that encapsulates the three precisions in matrix
    multiplication :math:`C = \\alpha Op(A) Op(B) + \\beta C`.

    Attributes:
        a (type): The precision of two-dimensional array used to store the matrix
            ``A``.

        b (type): The precision of two-dimensional array used to store the matrix
            ``B``.

        c (type): The precision of two-dimensional array used to store the matrix
            ``C``.
    """

    a: type[object]
    b: type[object]
    c: type[object]


class Alignment(NamedTuple):
    """
    A type to encapsulate the memory alignment in bytes of the matrix operands
    A, B, and C.

    Attributes:
        a (int): The alignment of two-dimensional array used to store the matrix
            ``A``.

        b (int): The alignment of two-dimensional array used to store the matrix
            ``B``.

        c (int): The alignment of two-dimensional array used to store the matrix
            ``C``.
    """

    a: int
    b: int
    c: int


class CublasdxTensors(NamedTuple):
    a: int
    b: int
    c: int


class CublasdxTensorsResponse:
    target: CublasdxTensors

    def __init__(self, target: CublasdxTensors):
        self.target = target


MAX_ALIGNMENT = Alignment(16, 16, 16)


class CallableGetIntTraits(Protocol):
    def __call__(self, handle: int, trait_type: mathdx.CublasdxTraitType, size: int) -> tuple: ...


class CallableGetStrTrait(Protocol):
    def __call__(self, handle: int, trait_type: mathdx.CublasdxTraitType) -> str: ...


get_int_traits: CallableGetIntTraits = build_get_int_traits(mathdx.cublasdx_get_trait_int64s)

get_str_trait: CallableGetStrTrait = build_get_str_trait(mathdx.cublasdx_get_trait_str_size, mathdx.cublasdx_get_trait_str)

get_str_device_trait: CallableGetStrTrait = build_get_str_trait(
    mathdx.cublasdx_get_device_function_trait_str_size, mathdx.cublasdx_get_device_function_trait_str
)


def get_tensor_int_traits(tensors: CublasdxTensors, trait: mathdx.CublasdxTensorTrait):
    return (
        int(mathdx.cublasdx_get_tensor_trait_int64(tensors.a, trait)),
        int(mathdx.cublasdx_get_tensor_trait_int64(tensors.b, trait)),
        int(mathdx.cublasdx_get_tensor_trait_int64(tensors.c, trait)),
    )


_ACCEPTED_PRECISION = REAL_NP_TYPES + INT_NP_TYPES


def validate(
    size,
    data_type,
    precision,
    execution,
    transpose_mode,
    arrangement,
    alignment,
    block_dim,
    sm,
    function,
    leading_dimension,
    static_block_dim,
    with_pipeline,
    enable_input_streaming,
):
    (m, n, k) = size
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"m, n, k must be > 0. Got {m}, {n}, {k}")
    if isinstance(precision, Precision):
        for m in ["a", "b", "c"]:
            check_in(f"precision.{m}", getattr(precision, m), _ACCEPTED_PRECISION)
    else:
        raise ValueError(
            f"precision should be an instance of {Precision} or a 3-sequence, and individual fields "
            f"should be one of {_ACCEPTED_PRECISION}. Instead got precision = {precision}"
        )
    check_blas_sm(sm, "sm")
    check_in("data_type", data_type, ["real", "complex"])
    check_in("execution", execution, ["Block", "Thread"])
    check_in("function", function, ["MM"])
    check_in("static_block_dim", static_block_dim, [True, False])
    if transpose_mode is not None:
        allowed_values = ["non_transposed", "transposed", "conj_transposed"]
        if isinstance(transpose_mode, TransposeMode):
            check_in("transpose_mode.a", transpose_mode.a, allowed_values)
            check_in("transpose_mode.b", transpose_mode.b, allowed_values)
        else:
            raise ValueError(
                f"transpose_mode should be an instance of {TransposeMode} or a 2-tuple, and individual fields "
                f"should be one of {allowed_values}. Instead got transpose_mode = {transpose_mode}"
            )
    if arrangement is not None:
        allowed_values = _BLAS_ARRENGEMENT_STR_TO_MATHDX.keys()
        if isinstance(arrangement, Arrangement):
            check_in("arrangement.a", arrangement.a, allowed_values)
            check_in("arrangement.b", arrangement.b, allowed_values)
            check_in("arrangement.c", arrangement.c, allowed_values)
        else:
            raise ValueError(
                f"arrangement should be an instance of {Arrangement} or a 3-tuple, and individual fields "
                f"should be one of {allowed_values}. Instead got arrangement = {arrangement}"
            )
    if arrangement is not None and transpose_mode is not None:
        raise ValueError("only arrangement or transpose_mode must be provide. Instead got both")
    if arrangement is None and transpose_mode is None:
        raise ValueError("arrangement or transpose_mode must be provide. Instead got nothing")
    if alignment is not None:
        validate_alignment(alignment, precision, data_type)
    if block_dim in (None, "suggested"):
        pass
    elif isinstance(block_dim, Dim3):
        prod = block_dim[0] * block_dim[1] * block_dim[2]
        if prod <= 0 or prod > 1024:
            raise ValueError(
                f"The product of the entries in block_dim should be between 1 and 1024 ; got block_dim = {block_dim}"
            )
    else:
        raise ValueError(f"block_dim should be None, a Dim3 instance or 'suggested'; got block_dim = {block_dim}")
    if leading_dimension in (None, "suggested") or isinstance(leading_dimension, LeadingDimension):
        pass
    else:
        raise ValueError(
            f"leading_dimension should be None, a LeadingDimension instance or 'suggested'; "
            f"got leading_dimension = {leading_dimension}"
        )
    if not isinstance(with_pipeline, bool):
        raise ValueError(f"with_pipeline should be a bool; got with_pipeline = {with_pipeline}")
    if not isinstance(enable_input_streaming, bool):
        raise ValueError(f"enable_input_streaming should be a bool; got enable_input_streaming = {enable_input_streaming}")


def validate_execute_api(execute_api):
    """
    Validate the execute_api argument.
    """
    check_in("execute_api", execute_api, _BLAS_API_STR_TO_MATHDX.keys())


def validate_tensor_types(tensor_types):
    """
    Validate the tensor_types argument.
    """
    allowed_values = _TENSOR_TYPE_STR_TO_MATHDX.keys()
    if isinstance(tensor_types, Sequence) and len(tensor_types) == 3:
        check_in("tensor_types[0]", tensor_types[0], allowed_values)
        check_in("tensor_types[1]", tensor_types[1], allowed_values)
        check_in("tensor_types[2]", tensor_types[2], allowed_values)
    else:
        raise ValueError(
            f"tensor_types should be a 3-tuple, and individual fields "
            f"should be one of {allowed_values}. Instead got tensor_types = {tensor_types}"
        )


def validate_alignment(alignment: Alignment, precision: Precision, data_type: str, gmem: bool = False):
    """
    Validates alignment. precision and data_type must be already validated.
    """
    name = "alignment" if not gmem else "global_memory_alignment"
    default_a = default_alignment(precision, data_type)
    for m in ["a", "b", "c"]:
        a = getattr(alignment, m)
        max_a = getattr(MAX_ALIGNMENT, m)
        def_a = getattr(default_a, m)
        if a <= 0:
            raise ValueError(f"{name}.{m} must be > 0. Got {a}")
        if a > max_a:
            raise ValueError(f"{name}.{m} must be less than maximum alignment {max_a}. Got {a}")
        if a % def_a != 0:
            raise ValueError(f"{name}.{m} must be a multiple of input value type {def_a}. Got {a}")


def default_alignment(precision: Precision, data_type: str):
    # supported precision's itemsize is always power of two, so we are safe
    mul = 1 if data_type == "real" else 2

    alignment = tuple(np.dtype(p).itemsize * mul for p in precision)
    return Alignment(*alignment)


@lru_cache
def generate_MM(
    size: tuple[int, int, int],
    precision: Precision,
    data_type: str,
    function: str,
    transpose_mode: TransposeMode | None,
    arrangement: Arrangement | None,
    alignment: Alignment | None,
    sm: ComputeCapability,
    block_dim: Dim3 | None,
    static_block_dim: bool,
    execution: str,
    leading_dimension: LeadingDimension | None = None,
    execute_api: str | None = None,
    tensor_types: tuple[str, str, str] | None = None,
    with_pipeline: bool = False,
    enable_input_streaming: bool = False,
):
    """
    Generate a cuBLASDx descriptor for matrix multiplication.

    tenosr_types is only used to create cache properly.
    """
    h = mathdx.cublasdx_create_descriptor()

    (m, n, k) = size

    # TODO: remove once libmathdx supports it
    if execute_api is None:
        execute_api = "static_leading_dimensions"

    if execute_api is not None:
        mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.API, _BLAS_API_STR_TO_MATHDX[execute_api])

    mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.SIZE, 3, [m, n, k])
    mathdx.cublasdx_set_operator_int64s(
        h, mathdx.CublasdxOperatorType.PRECISION, len(precision), [NP_TYPES_TO_MATHDX_PRECISION[p] for p in precision]
    )

    mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.FUNCTION, _BLAS_FUNCTION_STR_TO_MATHDX[function])
    mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.EXECUTION, EXECUTION_STR_TO_MATHDX[execution])
    mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.TYPE, _BLAS_TYPE_STR_TO_MATHDX[data_type])

    sm_mathdx = get_mathdx_sm(sm)
    mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.SM, len(sm_mathdx), sm_mathdx)

    if block_dim:
        mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.BLOCK_DIM, 3, block_dim)

    if static_block_dim:
        mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.STATIC_BLOCK_DIM, 1)

    if leading_dimension:
        mathdx.cublasdx_set_operator_int64s(
            h, mathdx.CublasdxOperatorType.LEADING_DIMENSION, 3, [leading_dimension.a, leading_dimension.b, leading_dimension.c]
        )

    if transpose_mode:
        mathdx.cublasdx_set_operator_int64s(
            h,
            mathdx.CublasdxOperatorType.TRANSPOSE_MODE,
            2,
            [_BLAS_TRANSPOSE_STR_TO_MATHDX[transpose_mode.a], _BLAS_TRANSPOSE_STR_TO_MATHDX[transpose_mode.b]],
        )

    if arrangement:
        mathdx.cublasdx_set_operator_int64s(
            h,
            mathdx.CublasdxOperatorType.ARRANGEMENT,
            3,
            [_BLAS_ARRENGEMENT_STR_TO_MATHDX[a] for a in arrangement],
        )

    if alignment:
        mathdx.cublasdx_set_operator_int64s(
            h,
            mathdx.CublasdxOperatorType.ALIGNMENT,
            3,
            alignment,
        )

    if with_pipeline:
        mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.WITH_PIPELINE, 1)

    if enable_input_streaming:
        mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.ENABLE_INPUT_STREAMING, 1)

    return DescriptorWrapper(h, mathdx.cublasdx_destroy_descriptor)


@lru_cache
def generate_device_pipeline(
    handle,
    depth: int,
    value_type_a: mathdx.CommondxValueType,
    value_type_b: mathdx.CommondxValueType,
    shape_a: list[int],
    shape_b: list[int],
    strides_a: list[int],
    strides_b: list[int],
) -> DescriptorWrapper:
    assert len(shape_a) == len(strides_a)
    assert len(shape_b) == len(strides_b)

    big_gmem_a = mathdx.cublasdx_create_tensor_strided(
        mathdx.CublasdxMemorySpace.GMEM, value_type_a, 0, len(shape_a), shape_a, strides_a
    )
    big_gmem_b = mathdx.cublasdx_create_tensor_strided(
        mathdx.CublasdxMemorySpace.GMEM, value_type_b, 0, len(shape_b), shape_b, strides_b
    )

    device_pipeline = mathdx.cublasdx_create_device_pipeline(
        handle,
        mathdx.CublasdxDevicePipelineType.SUGGESTED,
        depth,
        # FIXED for accumulator, HEURISTIC for epilogue
        mathdx.CublasdxBlockSizeStrategy.FIXED,
        big_gmem_a,
        big_gmem_b,
    )

    tensors = [big_gmem_a, big_gmem_b]
    pipelines = [device_pipeline]

    mathdx.cublasdx_finalize_pipelines(len(pipelines), pipelines)
    mathdx.cublasdx_finalize_tensors(handle, len(tensors), tensors)

    return DescriptorWrapper(device_pipeline, mathdx.cublasdx_destroy_pipeline)


@lru_cache
def generate_tile_pipeline(
    handle,
    device_pipeline_handle,
):
    tile_pipeline = mathdx.cublasdx_create_tile_pipeline(
        handle,
        mathdx.CublasdxTilePipelineType.PIPELINE_DEFAULT,
        device_pipeline_handle,
    )

    pipelines = [tile_pipeline]

    mathdx.cublasdx_finalize_pipelines(len(pipelines), pipelines)

    return DescriptorWrapper(tile_pipeline, mathdx.cublasdx_destroy_pipeline)


@lru_cache
def generate_code(handle, version: ComputeCapability, device_functions: tuple | None = None):
    code = mathdx.commondx_create_code()

    set_code_target_sm(code, version)
    mathdx.commondx_set_code_option_str(code, mathdx.CommondxOption.EXTRA_NVTRC_ARGS, NVARG_GEN_OPT_LTO)
    if device_functions:
        mathdx.cublasdx_finalize_device_functions(code, len(device_functions), list(device_functions))
    else:
        mathdx.cublasdx_finalize_code(code, handle)

    return DescriptorWrapper(code, mathdx.commondx_destroy_code)


@lru_cache
def generate_tensor(h: int, tensor_type: str, gmem_alignment: int | None = None) -> DescriptorWrapper:
    tensor = mathdx.cublasdx_create_tensor(h, _TENSOR_TYPE_STR_TO_MATHDX[tensor_type])

    if gmem_alignment is not None:
        try:
            mathdx.cublasdx_set_tensor_option_int64(
                tensor,
                mathdx.CublasdxTensorOption.ALIGNMENT_BYTES,
                gmem_alignment,
            )
        except Exception:
            print(f"[WARN] Failed to set {tensor_type} tensor alignment {gmem_alignment}")

    mathdx.cublasdx_finalize_tensors(h, 1, [tensor])

    return DescriptorWrapper(tensor, mathdx.cublasdx_destroy_tensor)


@lru_cache
def generate_tensor_like(h: int, src_tensor: int, dtype: np.number) -> DescriptorWrapper:
    dx_type = NP_TYPES_TO_MATHDX_TYPES[dtype]
    dst_tensor = mathdx.cublasdx_make_tensor_like(src_tensor, dx_type)
    mathdx.cublasdx_finalize_tensors(h, 1, [dst_tensor])

    return DescriptorWrapper(dst_tensor, mathdx.cublasdx_destroy_tensor)


@lru_cache
def get_tensor_traits(tensor: int) -> tuple[int, int, int, int]:
    """Get tensor traits: (uid, logical_size, storage_bytes, alignment)"""
    return (
        int(mathdx.cublasdx_get_tensor_trait_int64(tensor, mathdx.CublasdxTensorTrait.UID)),
        int(mathdx.cublasdx_get_tensor_trait_int64(tensor, mathdx.CublasdxTensorTrait.LOGICAL_SIZE)),
        int(mathdx.cublasdx_get_tensor_trait_int64(tensor, mathdx.CublasdxTensorTrait.STORAGE_BYTES)),
        int(mathdx.cublasdx_get_tensor_trait_int64(tensor, mathdx.CublasdxTensorTrait.ALIGNMENT_BYTES)),
    )


def generate_function_code(
    MM_handler: int, function_type: mathdx.CublasdxDeviceFunctionType, args: Sequence[int], version: ISAVersion
):
    function_handler = mathdx.cublasdx_create_device_function(MM_handler, function_type, len(args), list(args))
    symbol = get_str_device_trait(function_handler, mathdx.CublasdxDeviceFunctionTrait.SYMBOL)
    code = generate_code(MM_handler, version, (function_handler,))

    return code, symbol


@lru_cache
def generate_function_with_pipelines_code(
    MM_handler: int,
    function_type: mathdx.CublasdxDeviceFunctionType,
    tensors: tuple[int],
    pipelines: tuple[int],
    version: ISAVersion,
):
    function_handler = mathdx.cublasdx_create_device_function_with_pipelines(
        MM_handler,
        function_type,
        len(tensors),
        list(tensors),
        len(pipelines),
        list(pipelines),
    )
    symbol = get_str_device_trait(function_handler, mathdx.CublasdxDeviceFunctionTrait.SYMBOL)
    code = generate_code(MM_handler, version, (function_handler,))

    return code, symbol


def get_function_code(
    MM_handler: int,
    function_type: mathdx.CublasdxDeviceFunctionType,
    args: Sequence[int],
    code_type: CodeType,
):
    code, symbol = generate_function_code(MM_handler, function_type, args, code_type.cc)

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    lto = Code(code_type, isa_version, lto_fn)

    return lto, symbol


@lru_cache
def generate_tensors(h, tensor_types):
    type_mem_a = mathdx.cublasdx_create_tensor(h, _TENSOR_TYPE_STR_TO_MATHDX[tensor_types[0]])
    type_mem_b = mathdx.cublasdx_create_tensor(h, _TENSOR_TYPE_STR_TO_MATHDX[tensor_types[1]])
    type_mem_c = mathdx.cublasdx_create_tensor(h, _TENSOR_TYPE_STR_TO_MATHDX[tensor_types[2]])

    tensors = [
        type_mem_a,
        type_mem_b,
        type_mem_c,
    ]

    mathdx.cublasdx_finalize_tensors(h, len(tensors), tensors)

    target_tensors = CublasdxTensors(type_mem_a, type_mem_b, type_mem_c)
    resp = CublasdxTensorsResponse(target_tensors)

    weakref.finalize(resp, destroy_tensors, resp.target)

    return resp


def destroy_tensors(tensors: CublasdxTensors):
    mathdx.cublasdx_destroy_tensor(tensors.a)
    mathdx.cublasdx_destroy_tensor(tensors.b)
    mathdx.cublasdx_destroy_tensor(tensors.c)


def generate_copy_wait_lto(compute_capability: ComputeCapability):
    return generate_device_function_lto(
        compute_capability,
        mathdx.CublasdxDeviceFunctionType.COPY_WAIT,
        (),
    )


@lru_cache
def generate_device_function_lto(compute_capability: ComputeCapability, function_type: mathdx.CublasdxDeviceFunctionType, args):
    # Create the cuBLASDx descriptor
    h = mathdx.cublasdx_create_descriptor()
    mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.FUNCTION, mathdx.CublasdxFunction.MM)
    mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.EXECUTION, mathdx.CommondxExecution.BLOCK)
    mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.API, mathdx.CublasdxApi.TENSORS)
    mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.PRECISION, mathdx.CommondxPrecision.F32)
    sm_mathdx = get_mathdx_sm(compute_capability)
    mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.SM, len(sm_mathdx), sm_mathdx)
    mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.TYPE, mathdx.CublasdxType.REAL)
    mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.BLOCK_DIM, 3, [32, 1, 1])
    mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.SIZE, 3, [1, 1, 1])

    function = mathdx.cublasdx_create_device_function(h, function_type, len(args), [*args])
    symbol = get_str_device_trait(function, mathdx.CublasdxDeviceFunctionTrait.SYMBOL)

    # Compile the device function to lto
    code = mathdx.commondx_create_code()
    set_code_target_sm(code, compute_capability)
    mathdx.commondx_set_code_option_str(code, mathdx.CommondxOption.EXTRA_NVTRC_ARGS, NVARG_GEN_OPT_LTO)
    mathdx.cublasdx_finalize_device_functions(code, 1, [function])

    # Extract the LTOIR
    lto_size = mathdx.commondx_get_code_ltoir_size(code)
    lto = bytearray(lto_size)
    mathdx.commondx_get_code_ltoir(code, lto_size, lto)

    mathdx.commondx_destroy_code(code)
    mathdx.cublasdx_destroy_device_function(function)
    mathdx.cublasdx_destroy_descriptor(h)

    return symbol, bytes(lto)


def _compile_blas_device_pipeline_init(
    mm_descriptor: int,
    pipeline_descriptor: int,
    code_type: CodeType,
) -> tuple[Code, str]:
    assert isinstance(code_type, CodeType)

    code, symbol = generate_function_with_pipelines_code(
        mm_descriptor,
        mathdx.CublasdxDeviceFunctionType.CREATE,
        (),
        (pipeline_descriptor,),
        code_type.cc,
    )

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    lto = Code(code_type, isa_version, lto_fn)

    return lto, symbol


@lru_cache
def _compile_blas_device_pipeline_init_kernel(
    mm_descriptor: int,
    pipeline_descriptor: int,
    code_type: CodeType,
) -> Kernel:
    lto, sym = _compile_blas_device_pipeline_init(mm_descriptor, pipeline_descriptor, code_type=code_type)
    init_pipeline_func = ObjectCode.from_ltoir(lto.data)

    # CUDA C source code for our kernel
    init_pipeline_source = (
        f"#define create_dev_pipe {sym}\n"
        + """
    struct libmathdx_tensor_0s_0s { void* ptr; };
    struct libmathdx_pipeline { void* ptr; };

    extern "C" __device__ void create_dev_pipe(libmathdx_pipeline, libmathdx_tensor_0s_0s, libmathdx_tensor_0s_0s);

    extern "C" __global__ void create_device_pipeline(void* device_pipeline_ptr, void* ga_storage, void* gb_storage) {
        auto ga = libmathdx_tensor_0s_0s { ga_storage };
        auto gb = libmathdx_tensor_0s_0s { gb_storage };
        auto device_pipeline = libmathdx_pipeline { device_pipeline_ptr };
        create_dev_pipe(device_pipeline, ga, gb);
    }
    """
    )

    # Compiler arguments
    arch = f"sm_{code_type.cc.major}{code_type.cc.minor}{code_type.cc.arch}"
    program_options = ProgramOptions(link_time_optimization=True, arch=arch)
    linker_options = LinkerOptions(link_time_optimization=True, arch=arch)

    # Compile the CUDA code into a program
    program = Program(init_pipeline_source, code_type="c++", options=program_options)
    compiled_program = program.compile(target_type="ltoir")

    linker = Linker(compiled_program, init_pipeline_func, options=linker_options)
    linked_program = linker.link("cubin")

    # Get the specific kernel function we want to use
    kernel = linked_program.get_kernel("create_device_pipeline")

    return kernel


def _compile_blas_device_pipeline_destroy(
    mm_descriptor: int,
    pipeline_descriptor: int,
    code_type: CodeType,
) -> tuple[Code, str]:
    code, symbol = generate_function_with_pipelines_code(
        mm_descriptor,
        mathdx.CublasdxDeviceFunctionType.DESTROY,
        (),
        (pipeline_descriptor,),
        code_type.cc,
    )

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    lto = Code(code_type, isa_version, lto_fn)

    return lto, symbol


@lru_cache
def _compile_blas_device_pipeline_destroy_kernel(
    mm_descriptor: int,
    pipeline_descriptor: int,
    code_type: CodeType,
) -> Kernel:
    lto, sym = _compile_blas_device_pipeline_destroy(
        mm_descriptor=mm_descriptor,
        pipeline_descriptor=pipeline_descriptor,
        code_type=code_type,
    )
    destroy_pipeline_func = ObjectCode.from_ltoir(lto.data)

    # CUDA C source code for our kernel
    destroy_pipeline_source = (
        f"#define destroy_dev_pipe {sym}\n"
        + """
    struct libmathdx_pipeline { void* ptr; };

    extern "C" __device__ void destroy_dev_pipe(libmathdx_pipeline);

    extern "C" __global__ void destroy_device_pipeline(void* device_pipeline_ptr) {
        auto device_pipeline = libmathdx_pipeline { device_pipeline_ptr };
        destroy_dev_pipe(device_pipeline);
    }
    """
    )

    # Compiler arguments
    arch = f"sm_{code_type.cc.major}{code_type.cc.minor}{code_type.cc.arch}"
    program_options = ProgramOptions(link_time_optimization=True, arch=arch)
    linker_options = LinkerOptions(link_time_optimization=True, arch=arch)

    # Compile the CUDA code into a program
    program = Program(destroy_pipeline_source, code_type="c++", options=program_options)
    compiled_program = program.compile(target_type="ltoir")

    linker = Linker(compiled_program, destroy_pipeline_func, options=linker_options)
    linked_program = linker.link("cubin")

    # Get the specific kernel function we want to use
    kernel = linked_program.get_kernel("destroy_device_pipeline")

    return kernel
