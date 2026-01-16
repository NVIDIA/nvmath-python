# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from collections.abc import Callable
import numbers
import operator
from numba import cuda
import numba
from numba.core import typing, cgutils
from numba.core.extending import models
from numba.core.errors import TypingError
from numba.extending import types, utils
from numba.cuda.extending import overload_method, overload, intrinsic, typeof_impl
from numba.cuda.cudaimpl import lower_constant, registry as cuda_registry
from numba.cuda.models import register_model, StructModel

from numba.np import numpy_support
import numpy

from nvmath.device.common_cuda import get_default_code_type
from nvmath.device.cublasdx_backend import generate_copy_wait_lto

from .common import axpby, copy, copy_fragment, clear, copy_wait, make_tensor, make_fragment_like, OpaqueTensor
from .common_numba import (
    NUMBA_FE_TYPES_TO_NUMBA_IR,
    OpaquePointerType,
    declare_cabi_device,
    get_array_ptr,
    get_opaque_pointer,
    get_value_ptr,
    overload_type_attribute,
    EmptyStructModel,
)
from .cublasdx import (
    _BlasMatmulLikeLayout,
    DevicePipeline,
    TilePipeline,
    Matmul,
    _BlasMatmulLayout,
    Partitioner,
    compile_blas_accumulator_init,
    compile_blas_axpby,
    compile_blas_clear,
    compile_blas_copy,
    compile_blas_execute,
    compile_blas_is_index_in_bounds,
    compile_blas_is_predicated,
    compile_blas_is_thread_active,
    compile_blas_map_idx2crd_partitioner,
    compile_blas_tile_pipeline_execute,
    compile_blas_tile_pipeline_init,
    compile_blas_device_pipeline_reset_tile,
    compile_blas_tile_pipeline_destroy,
)
from .common_opaque_tensor import (
    LayoutModel,
    LayoutType,
    OpaqueTensorModel,
    OpaqueTensorType,
)

import llvmlite.ir as llvmir
from numba.core.base import BaseContext

_BLAS_DEFINITION_ARGS = [
    "size",
    "precision",
    "data_type",
    "sm",
    "block_size",
    "block_dim",
    "leading_dimension",
    "transpose_mode",
    "arrangement",
    "function",
    "execution",
    "alignment",
]

_BLAS_COMPILED_ARGS = [
    "a_value_type",
    "b_value_type",
    "c_value_type",
    # value_type, input_type, and output_type not included intentionally,
    # because they are deprecated.
    "a_dim",
    "b_dim",
    "c_dim",
    "a_size",
    "b_size",
    "c_size",
    "leading_dimension",
    "shared_memory_size",
    "max_threads_per_block",
]

_DEVICE_PIPELINE_DEFINITION_ARGS = [
    "mm",
    "pipeline_depth",
    "a",
    "b",
]

_DEVICE_PIPELINE_COMPILED_ARGS = [
    "buffer_alignment",
    "buffer_size",
    "storage_bytes",
    "storage_alignment",
]

_TILE_PIPELINE_DEFINITION_ARGS = [
    "device_pipeline",
]

_TILE_PIPELINE_COMPILED_ARGS = [
    "storage_bytes",
    "storage_alignment",
]


class BlasType(types.Type):
    """
    Type class associated with the `cublasdx.Matmul`.
    """

    def __init__(self, blas: Matmul):
        assert isinstance(blas, Matmul)
        self._blas = blas
        attributes = [f"{attr}={getattr(blas, attr)}" for attr in _BLAS_DEFINITION_ARGS if getattr(blas, attr)]
        attributes.sort()

        self.name = "BlasNumba(" + ",".join(attributes) + ")"

    @property
    def blas(self) -> Matmul:
        return self._blas


register_model(BlasType)(EmptyStructModel)


@lower_constant(BlasType)
def constant_dummy(context, builder, typ, pyval):
    struct_ptr = cgutils.create_struct_proxy(typ)(context, builder)
    return struct_ptr._getvalue()


@typeof_impl.register(Matmul)
def typeof_blas_numba(val: Matmul, c: typing.Context) -> BlasType:
    return BlasType(val)


for attribute in _BLAS_COMPILED_ARGS + _BLAS_DEFINITION_ARGS:
    overload_type_attribute(BlasType, "blas", attribute)


class DevicePipelineType(types.Type):
    """
    Type class associated with the `cublasdx.DevicePipeline`.
    """

    def __init__(self, pipeline: DevicePipeline):
        assert isinstance(pipeline, DevicePipeline)
        self._pipeline = pipeline
        MM_type = BlasType(pipeline.mm)
        attributes = [
            f"a_layout=(dtype={pipeline.a.dtype},shape={pipeline.a.shape},strides={pipeline.a_strides})",
            f"b_layout=(dtype={pipeline.b.dtype},shape={pipeline.b.shape},strides={pipeline.b_strides})",
            f"pipeline_depth={pipeline.pipeline_depth}",
            f"mm={MM_type}",
        ]
        attributes.sort()

        self.name = "DevicePipeline(" + ",".join(attributes) + ")"

    @property
    def _buffer(self):
        return self._pipeline._storage

    @property
    def pipeline(self) -> DevicePipeline:
        return self._pipeline


class PipelineModel(models.StructModel):
    def __init__(self, dmm, fe_type: DevicePipelineType):
        members = [
            ("ptr", types.voidptr),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


register_model(DevicePipelineType)(PipelineModel)


@typeof_impl.register(DevicePipeline)
def typeof_device_pipeline_numba(val: DevicePipeline, c: typing.Context) -> DevicePipelineType:
    return DevicePipelineType(val)


for attribute in _DEVICE_PIPELINE_DEFINITION_ARGS + _DEVICE_PIPELINE_COMPILED_ARGS:
    overload_type_attribute(DevicePipelineType, "pipeline", attribute)


class TilePipelineType(types.Type):
    """
    Type class associated with the `cublasdx.DevicePipeline`.
    """

    def __init__(self, pipeline: TilePipeline):
        assert isinstance(pipeline, TilePipeline)
        self._pipeline = pipeline
        device_pipeline_type = DevicePipelineType(pipeline.device_pipeline)
        self.name = f"TilePipeline(device_pipeline={device_pipeline_type})"

    @property
    def pipeline(self) -> TilePipeline:
        return self._pipeline


register_model(TilePipelineType)(PipelineModel)


@typeof_impl.register(TilePipeline)
def typeof_tile_pipeline_numba(val: TilePipeline, c: typing.Context) -> TilePipelineType:
    return TilePipelineType(val)


for attribute in _TILE_PIPELINE_DEFINITION_ARGS + _TILE_PIPELINE_COMPILED_ARGS:
    overload_type_attribute(TilePipelineType, "pipeline", attribute)


@intrinsic
def _create_tile_pipeline(typingctx: typing.Context, device_pipeline_type: DevicePipelineType):
    """Create a new BLAS tile pipeline object."""

    tile_pipeline = TilePipeline(device_pipeline_type.pipeline)
    tile_pipeline_type = TilePipelineType(tile_pipeline)

    return_ty = tile_pipeline_type
    sig = typing.signature(return_ty, device_pipeline_type)

    def codegen(context: BaseContext, builder, sig, args):
        tile_pipeline_type = sig.return_type
        struct_ptr = cgutils.create_struct_proxy(tile_pipeline_type)(context, builder)
        return struct_ptr._getvalue()

    return sig, codegen


@intrinsic
def _set_tile_pipeline_buffer(typingctx: typing.Context, tile_pipeline_type: TilePipelineType, buffer: types.Array):
    """Create a new BLAS tile pipeline object."""

    return_ty = tile_pipeline_type
    sig = typing.signature(return_ty, tile_pipeline_type, buffer)

    def codegen(context: BaseContext, builder, sig, args):
        tile_pipeline_type, tile_pipeline = sig.args[0], args[0]
        buffer_type, buffer = sig.args[1], args[1]
        buffer_ptr = cgutils.create_struct_proxy(buffer_type)(context, builder, buffer).data
        struct_ptr = cgutils.create_struct_proxy(tile_pipeline_type)(context, builder, tile_pipeline)
        struct_ptr.ptr = buffer_ptr

        return struct_ptr._getvalue()

    return sig, codegen


# Numba does not support method overload or variadic arguments, so we using
# default values as a workaround
# https://github.com/numba/numba/issues/9980
# https://github.com/numba/numba/issues/9979
# https://github.com/numba/numba/issues/10143
@overload_method(BlasType, "execute", jit_options={"forceinline": True}, strict=False)
def ol_blas_numba_execute(
    blas_numba: BlasType, _arg1, _arg2, _arg3, _arg4=None, _arg5=None, _arg6=None, _arg7=None, _arg8=None
):
    return ol_blas_numba(blas_numba, _arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8)


@overload_method(BlasType, "__call__", strict=False)
def ol_blas_numba_call(blas_numba: BlasType, _arg1, _arg2, _arg3, _arg4=None, _arg5=None, _arg6=None, _arg7=None, _arg8=None):
    return ol_blas_numba(blas_numba, _arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8)


def ol_blas_numba(blas_numba: BlasType, _arg1, _arg2, _arg3, _arg4=None, _arg5=None, _arg6=None, _arg7=None, _arg8=None):
    none_set = {None, types.Omitted(None)}
    if {_arg4, _arg5, _arg6, _arg7, _arg8} <= none_set:
        return lambda _, a, b, c, _arg4=None, _arg5=None, _arg6=None, _arg7=None, _arg8=None: _bals_type___call__(_, a, b, c)
    elif {_arg6, _arg7, _arg8} <= none_set:
        return lambda _, alpha, a, b, beta, c, _arg6=None, _arg7=None, _arg8=None: _bals_type___call__(_, alpha, a, b, beta, c)
    else:
        return lambda _, alpha, a, lda, b, ldb, beta, c, ldc: _bals_type___call__(_, alpha, a, lda, b, ldb, beta, c, ldc)


# TODO: use overload_method when supported
def _bals_type___call__(*args):
    raise Exception("Stub for overloads")


def assert_suggested_tensors(tensor_types: tuple[types.Type, ...]):
    """
    Verify that all tensors are suggested or not suggested at the same time.
    """
    suggested_flags = []
    for t in tensor_types:
        assert isinstance(t, (OpaqueTensorType, BlasAccumulatorType))
        layout_type = t.layout
        assert isinstance(layout_type, BlasLayoutType)
        suggested_flags.append(layout_type.layout.suggested)

    if not all(suggested_flags) and any(suggested_flags):
        raise TypingError("All tensors must be either suggested or not suggested at the same time.")


@overload(_bals_type___call__, jit_options={"forceinline": True}, strict=False)
def ol_blas_type___call___tensors_rmem(
    blas_numba: BlasType,
    a: OpaqueTensorType,
    b: OpaqueTensorType,
    c: OpaqueTensorType,
):
    if not isinstance(blas_numba, BlasType):
        return
    MM = blas_numba.blas
    if not all(isinstance(t, OpaqueTensorType) for t in (a, b)):
        return
    if not (isinstance(c, (OpaqueTensorType, BlasAccumulatorType))):
        return

    a_layout, b_layout, c_layout = a.layout.layout, b.layout.layout, c.layout.layout
    assert isinstance(a_layout, _BlasMatmulLayout)
    assert isinstance(b_layout, _BlasMatmulLayout)
    assert isinstance(c_layout, _BlasMatmulLayout)

    if c_layout.memory_space != "r" and not c_layout.accumulator:
        return

    assert_suggested_tensors((a, b, c))

    return_type = types.void
    sig = typing.signature(return_type, a, b, c)

    code, symbol = compile_blas_execute(
        MM,
        code_type=get_default_code_type(),
        execute_api="tensors",
        tensor_types=(a_layout.tensor_type, b_layout.tensor_type, c_layout.tensor_type),
    )

    lto = cuda.LTOIR(code.data)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    def impl(_, a, b, c):
        blas_device_func(a, b, c)

    return impl


@overload(_bals_type___call__, jit_options={"forceinline": True}, strict=False)
def ol_blas_type___call___tensors_smem(
    blas_numba: BlasType,
    alpha: types.Number,
    a: OpaqueTensorType,
    b: OpaqueTensorType,
    beta: types.Number,
    c: OpaqueTensorType,
):
    if not isinstance(blas_numba, BlasType):
        return
    MM = blas_numba.blas
    if not all(isinstance(a, types.Number) for a in (alpha, beta)):
        return
    if not all(isinstance(a, OpaqueTensorType) for a in (a, b, c)):
        return
    a_layout, b_layout, c_layout = a.layout.layout, b.layout.layout, c.layout.layout
    assert isinstance(a_layout, _BlasMatmulLayout)
    assert isinstance(b_layout, _BlasMatmulLayout)
    assert isinstance(c_layout, _BlasMatmulLayout)
    if c_layout.memory_space != "s":
        return

    return_type = types.void
    c_ptr = types.CPointer(c.dtype)
    sig = typing.signature(return_type, c_ptr, a, b, c_ptr, c)

    code, symbol = compile_blas_execute(
        MM,
        code_type=get_default_code_type(),
        execute_api="tensors",
        tensor_types=(a_layout.tensor_type, b_layout.tensor_type, c_layout.tensor_type),
    )

    lto = cuda.LTOIR(code.data)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    dtype = c.dtype

    def impl(_, alpha, a, b, beta, c):
        alpha_ptr = get_value_ptr(dtype(alpha))
        beta_ptr = get_value_ptr(dtype(beta))

        blas_device_func(alpha_ptr, a, b, beta_ptr, c)

    return impl


@overload(_bals_type___call__, jit_options={"forceinline": True}, strict=False)
def ol_blas_type___call___basic(
    blas_numba: BlasType,
    alpha: types.Number,
    a: types.Array,
    b: types.Array,
    beta: types.Number,
    c: types.Array,
):
    if not isinstance(blas_numba, BlasType):
        return
    MM = blas_numba.blas
    if not all(isinstance(a, types.Number) for a in (alpha, beta)):
        return
    if not all(isinstance(a, types.Array) for a in (a, b, c)):
        return
    if (a.dtype, b.dtype, c.dtype) != tuple(NUMBA_FE_TYPES_TO_NUMBA_IR[vt] for vt in MM._traits.value_types):
        return

    # setting signature for intrinsic to much calling conventions. Numba will
    # automatically cast to desired values.
    return_type = types.void
    c_ptr = types.CPointer(c.dtype)
    sig = typing.signature(return_type, c_ptr, types.CPointer(a.dtype), types.CPointer(b.dtype), c_ptr, c_ptr)

    code, symbol = compile_blas_execute(
        MM,
        code_type=get_default_code_type(),
        execute_api="static_leading_dimensions",
    )

    lto = cuda.LTOIR(code.data)

    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    def impl(_, alpha, a, b, beta, c):
        aptr = get_array_ptr(a)
        bptr = get_array_ptr(b)
        cptr = get_array_ptr(c)
        alpha = get_value_ptr(c.dtype.type(alpha))
        beta = get_value_ptr(c.dtype.type(beta))

        blas_device_func(alpha, aptr, bptr, beta, cptr)

    return impl


@overload(_bals_type___call__, jit_options={"forceinline": True}, strict=False)
def ol_blas_type___call___ldabc(
    blas_numba: BlasType,
    alpha: types.Number,
    a: types.Array,
    lda: types.Integer,
    b: types.Array,
    ldb: types.Integer,
    beta: types.Number,
    c: types.Array,
    ldc: types.Integer,
):
    if not isinstance(blas_numba, BlasType):
        return
    MM = blas_numba.blas
    if not all(isinstance(a, types.Number) for a in (alpha, beta)):
        return
    if not all(isinstance(a, types.Array) for a in (a, b, c)):
        return
    if (a.dtype, b.dtype, c.dtype) != tuple(NUMBA_FE_TYPES_TO_NUMBA_IR[vt] for vt in MM._traits.value_types):
        return
    if not all(isinstance(a, types.Integer) for a in (lda, ldb, ldc)):
        return

    ld_type = types.uint32
    ld_ptr = types.CPointer(ld_type)
    return_type = types.void
    c_ptr = types.CPointer(c.dtype)
    sig = typing.signature(
        return_type, c_ptr, types.CPointer(a.dtype), ld_ptr, types.CPointer(b.dtype), ld_ptr, c_ptr, c_ptr, ld_ptr
    )

    code, symbol = compile_blas_execute(
        MM,
        code_type=get_default_code_type(),
        execute_api="dynamic_leading_dimensions",
    )

    lto = cuda.LTOIR(code.data)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    def impl(_, alpha, a, lda, b, ldb, beta, c, ldc):
        aptr = get_array_ptr(a)
        bptr = get_array_ptr(b)
        cptr = get_array_ptr(c)
        alpha = get_value_ptr(c.dtype.type(alpha))
        beta = get_value_ptr(c.dtype.type(beta))
        lda = get_value_ptr(ld_type(lda))
        ldb = get_value_ptr(ld_type(ldb))
        ldc = get_value_ptr(ld_type(ldc))

        blas_device_func(alpha, aptr, lda, bptr, ldb, beta, cptr, ldc)

    return impl


# __call__ overload is not supported by numba, however adding this overload
# kind of activates proper behaviour and works like magic.
# Issue reference: https://github.com/numba/numba/issues/5885
# TODO: remove once supported
@cuda_registry.lower(BlasType, BlasType, types.VarArg(types.Any))
def method_impl(context, builder, sig, args):
    typing_context = context.typing_context
    fnty = typing_context.resolve_value_type(ol_blas_numba_call)
    sig = fnty.get_call_type(typing_context, sig.args, {})
    sig = sig.replace(pysig=utils.pysignature(ol_blas_numba_call))

    call = context.get_function(fnty, sig)
    # Link dependent library
    context.add_linking_libs(getattr(call, "libs", ()))
    return call(builder, args)


@overload(copy, jit_options={"forceinline": True}, strict=False)
def ol_blas_copy(src: OpaqueTensorType, dst: OpaqueTensorType, alignment=None):
    return ol_blas_copy_generic(src, dst, alignment, "copy")


@overload(copy_fragment, jit_options={"forceinline": True}, strict=False)
def ol_blas_copy_fragment(src: OpaqueTensorType, dst: OpaqueTensorType, alignment=None):
    return ol_blas_copy_generic(src, dst, alignment, "copy_fragment")


def ol_blas_copy_generic(src: OpaqueTensorType, dst: OpaqueTensorType, alignment_ty: types.Type | None, func: str):
    assert isinstance(src, (OpaqueTensorType, BlasAccumulatorType))
    assert isinstance(dst, OpaqueTensorType)
    src_layout_ty = src.layout
    dst_layout_ty = dst.layout
    assert isinstance(src_layout_ty, BlasLayoutType)
    assert isinstance(dst_layout_ty, BlasLayoutType)

    alignment: int | None = None
    if alignment_ty not in {None, types.Omitted(None)}:
        if not isinstance(alignment_ty, types.Literal):
            return lambda src, dst, alignment: numba.literally(alignment)
        alignment = alignment_ty.literal_value
        if alignment not in {1, 2, 4, 8, 16}:
            raise TypingError(f"Alignment must be one of (1, 2, 4, 8, 16), got {alignment}")

    rmem = dst_layout_ty.layout.memory_space == "r" or src_layout_ty.layout.memory_space == "r"

    if func == "copy_fragment":
        if not rmem:
            raise TypingError("copy_fragment is only supported for rmem tensors. Please use copy instead.")
    else:
        assert func == "copy"
        if rmem:
            raise TypingError("copy is not supported for rmem tensors. Please use copy_fragment instead.")

    if alignment is not None:
        dtype = numpy_support.as_dtype(src.layout.dtype)
        if alignment < dtype.itemsize:
            raise TypingError(f"Alignment must be at least the size of the data type {dtype.itemsize}, got {alignment}")

    code, symbol = compile_blas_copy(
        src_tensor=src_layout_ty.layout,
        dst_tensor=dst_layout_ty.layout,
        code_type=get_default_code_type(),
        alignment=alignment,
    )

    return_type = types.void
    sig = typing.signature(return_type, src, dst)

    lto = cuda.LTOIR(code.data)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    def impl(src, dst, alignment=None):
        return blas_device_func(src, dst)

    return impl


@overload(clear, jit_options={"forceinline": True}, strict=False)
def ol_blas_clear(arr: OpaqueTensorType):
    assert isinstance(arr, OpaqueTensorType)
    assert isinstance(arr.layout, BlasLayoutType)

    code, symbol = compile_blas_clear(
        tensor=arr.layout.layout,
        code_type=get_default_code_type(),
    )

    lto = cuda.LTOIR(code.data)

    return_type = types.void
    sig = typing.signature(return_type, arr)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    def impl(arr):
        return blas_device_func(arr)

    return impl


class BlasLayoutType(LayoutType):
    """
    Type class associated with opaque tensor layouts.
    """

    def __init__(self, layout: _BlasMatmulLayout):
        assert isinstance(layout, _BlasMatmulLayout)
        super().__init__(layout)
        # Using handle descriptor in the type name to avoid symbol copy caching
        # by numba.
        MM_type = BlasType(layout.MM)
        self.name = f"BlasLayout(uid={layout.uid},layout={layout.layout},dtype={layout.dtype},MM={MM_type})"

    @property
    def layout(self) -> _BlasMatmulLayout:
        assert isinstance(self._layout, _BlasMatmulLayout)
        return self._layout

    def make_layout_like(self, dtype: numpy.number) -> "BlasLayoutType":
        src_layout = self.layout
        dst_layout = _BlasMatmulLikeLayout(
            src_layout.MM,
            src_layout.layout,
            dtype,
            leading_dimension=src_layout._default_ld,
        )

        return BlasLayoutType(dst_layout)


register_model(BlasLayoutType)(LayoutModel)


def lower_blas_layout_codegen(
    context: BaseContext,
    builder: llvmir.IRBuilder,
    layout_type: BlasLayoutType,
    ld_val: llvmir.NamedValue | numbers.Number | None,
    ld_type: types.Type | None = None,
):
    layout = cgutils.create_struct_proxy(layout_type)(context, builder)
    if layout_type.layout.dynamic_strides_size > 0:
        assert layout_type.layout.dynamic_strides_size == 1
        if ld_val is None:
            # Use default MM ld when value is not provided
            ld_val = layout_type.layout.MM.leading_dimension[layout_type.layout.tensor_index]
        if isinstance(ld_val, numbers.Number):
            ld = context.get_constant(types.int64, ld_val)
        else:
            ld = context.cast(builder, ld_val, ld_type, types.int64)
        layout.strides = cgutils.pack_array(builder, [ld])
    else:
        assert ld_val is None
    return layout._getvalue()


@lower_constant(BlasLayoutType)
def constant_blas_layout(context, builder, typ, pyval):
    assert isinstance(typ, BlasLayoutType)
    assert isinstance(pyval, _BlasMatmulLayout)
    return lower_blas_layout_codegen(context, builder, typ, pyval.default_ld)


@typeof_impl.register(_BlasMatmulLayout)
def typeof_blas_mm_layout(layout: _BlasMatmulLayout, c: typing.Context) -> BlasLayoutType:
    # We are clearing ld from the layout because it is a runtime value.
    layout_no_ld = _BlasMatmulLayout(layout.MM, layout.layout)
    return BlasLayoutType(layout_no_ld)


@typeof_impl.register(_BlasMatmulLikeLayout)
def typeof_blas_mm_like_layout(layout: _BlasMatmulLikeLayout, c: typing.Context) -> BlasLayoutType:
    # We are clearing ld from the layout because it is a runtime value.
    layout_no_ld = _BlasMatmulLikeLayout(layout.MM, layout.layout, layout.dtype)
    return BlasLayoutType(layout_no_ld)


for attribute in ["size", "cosize", "alignment"]:
    overload_type_attribute(BlasLayoutType, "_layout", attribute)


def ol_blas_layout(blas_numba: BlasType, method: str, leading_dimension: types.Number | None = None):
    # leading_dimension is available only for global memory
    if ("gmem" not in method) and (leading_dimension not in {None, types.Omitted(None)}):
        return
    MM = blas_numba.blas

    layout = getattr(MM, method)()
    return_type = BlasLayoutType(layout)

    @intrinsic
    def _intrinsic(typingctx, leading_dimension=None):
        def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args):
            ld = args[0] if not isinstance(leading_dimension, types.NoneType) else None
            ld_type = signature.args[0] if ld is not None else None
            return lower_blas_layout_codegen(context, builder, return_type, ld, ld_type)

        return typing.signature(return_type, leading_dimension), codegen

    return lambda blas_numba, leading_dimension=None: _intrinsic(leading_dimension)


def overload_blas_layout_method(method: str):
    overload_method(
        BlasType,
        method,
        jit_options={"forceinline": True},
        strict=False,
    )(lambda blas_numba, leading_dimension=None: ol_blas_layout(blas_numba, method, leading_dimension))


for method in [
    "get_layout_smem_a",
    "get_layout_smem_b",
    "get_layout_smem_c",
    "get_layout_gmem_a",
    "get_layout_gmem_b",
    "get_layout_gmem_c",
    "suggest_layout_smem_a",
    "suggest_layout_smem_b",
    "suggest_layout_smem_c",
    "suggest_layout_rmem_c",
    "get_layout_rmem_c",
    "_get_accumulator_c",
    "_suggest_accumulator_c",
]:
    overload_blas_layout_method(method)


@overload(make_tensor, jit_options={"forceinline": True}, strict=False)
def ol_make_tensor(array, layout):
    assert isinstance(array, types.Array)
    assert isinstance(layout, BlasLayoutType)

    if array.dtype == layout.dtype:
        return lambda array, layout: OpaqueTensor(array, layout)
    else:

        @intrinsic
        def copy_strides(typingctx, array_type, layout_type):
            def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args):
                array_type, layout_type = signature.args
                array = cgutils.create_struct_proxy(array_type)(context, builder, args[0])
                layout = cgutils.create_struct_proxy(layout_type)(context, builder, args[1])
                if array_type.layout.layout.dynamic_strides_size > 0:
                    array.strides = layout.strides
                return array._getvalue()

            return typing.signature(array_type, array_type, layout_type), codegen

        np_dtype = numpy.dtype(numpy_support.as_dtype(array.dtype)).type
        dst_layout_ty = layout.make_layout_like(np_dtype)
        dst_layout = dst_layout_ty.layout

        def impl(array, layout):
            tensor = OpaqueTensor(array, dst_layout)
            return copy_strides(tensor, layout)

        return impl


@overload(make_fragment_like, inline="always", strict=False)
def ol_make_fragment_like(tensor, dtype):
    assert isinstance(tensor, OpaqueTensorType)
    layout = tensor.layout
    assert isinstance(layout, BlasLayoutType)
    assert layout.layout.memory_space == "r"

    assert isinstance(dtype, types.NumberClass)
    np_dtype = numpy.dtype(numpy_support.as_dtype(dtype)).type

    dst_layout_ty = layout.make_layout_like(np_dtype)
    dst_layout = dst_layout_ty.layout

    def impl(tensor, dtype):
        buff = cuda.local.array(dst_layout.cosize, dtype=dtype, alignment=dst_layout.alignment)
        return OpaqueTensor(buff, dst_layout)

    return impl


@overload(copy_wait, jit_options={"forceinline": True}, strict=False)
def ol_copy_wait():
    # numba has cache per compute capability, so the function won't end up
    # cached for the wrong compute capability.
    return_type = types.void
    sig = typing.signature(return_type)

    ct = get_default_code_type()
    symbol, code = generate_copy_wait_lto(ct.cc)

    lto = cuda.LTOIR(code)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    return lambda: blas_device_func()


@overload(axpby, jit_options={"forceinline": True}, strict=False)
def ol_axpby(a, x, b, y):
    if not isinstance(a, types.Number):
        return
    if not isinstance(x, OpaqueTensorType):
        return
    if not isinstance(b, types.Number):
        return
    if not isinstance(y, OpaqueTensorType):
        return
    if x.layout.layout.memory_space != "r":
        raise TypeError("axpby is only supported for rmem tensors. x is not")
    if y.layout.layout.memory_space != "r":
        raise TypeError("axpby is only supported for rmem tensors. y is not")

    code, symbol = compile_blas_axpby(
        x_tensor=x.layout.layout,
        y_tensor=y.layout.layout,
        code_type=get_default_code_type(),
    )

    lto = cuda.LTOIR(code.data)

    return_type = types.void
    sig = typing.signature(return_type, types.CPointer(x.dtype), x, types.CPointer(y.dtype), y)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    x_dtype = x.dtype
    y_dtype = y.dtype

    def impl(a, x, b, y):
        a_ptr = get_value_ptr(x_dtype(a))
        b_ptr = get_value_ptr(y_dtype(b))
        return blas_device_func(a_ptr, x, b_ptr, y)

    return impl


class BlasPartitionerType(types.Type):
    """
    Type class for Blas partitioner.
    """

    def __init__(self, MM: Matmul, suggested: bool):
        assert isinstance(MM, Matmul)
        self._MM = MM
        mm_type = BlasType(MM)
        self._suggested = suggested
        super().__init__(f"BlasPartitioner(MM={mm_type}, suggested={suggested})")

    @property
    def MM(self) -> Matmul:
        return self._MM

    @property
    def suggested(self) -> bool:
        return self._suggested

    @property
    def dtype(self) -> types.Number:
        return NUMBA_FE_TYPES_TO_NUMBA_IR[self._MM._traits.value_types[2]]

    @property
    def fragment_layout(self) -> BlasLayoutType:
        MM = self.MM
        if self.suggested:
            layout = MM.suggest_layout_rmem_c()
        else:
            layout = MM.get_layout_rmem_c()

        assert isinstance(layout, _BlasMatmulLayout)

        return BlasLayoutType(layout)


@register_model(BlasPartitionerType)
class PartitionerModel(StructModel):
    def __init__(self, dmm, fe_type: BlasPartitionerType):
        StructModel.__init__(self, dmm, fe_type, [])


class BlasAccumulatorType(BlasPartitionerType):
    """
    Type class for Blas partitioner.
    """

    def __init__(self, MM: Matmul, suggested: bool):
        super().__init__(MM, suggested)
        mm = BlasType(MM)
        self.name = f"BlasAccumulator(MM={mm}, suggested={suggested})"

    @cached_property
    def layout(self) -> BlasLayoutType:
        if self._suggested:
            layout = self._MM._suggest_accumulator_c()
        else:
            layout = self._MM._get_accumulator_c()

        assert isinstance(layout, _BlasMatmulLayout)

        return BlasLayoutType(layout)


register_model(BlasAccumulatorType)(OpaqueTensorModel)


def ol_blas_get_accumulator_generic(blas_numba: BlasType, suggested: bool):
    assert isinstance(blas_numba, BlasType)

    MM = blas_numba.blas
    return_type = BlasAccumulatorType(MM, suggested=suggested)
    opaque_tensor_type = OpaqueTensorType(return_type.layout)

    @intrinsic
    def _type_cast_ot_acc(typingctx, opaque_tensor):
        def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args):
            return args[0]

        return typing.signature(return_type, opaque_tensor_type), codegen

    if suggested:
        acc_layout = MM._suggest_accumulator_c()
    else:
        acc_layout = MM._get_accumulator_c()

    alignment = max(acc_layout.alignment, 8)

    def impl(mm: Matmul):
        buff = cuda.local.array(acc_layout.cosize, dtype=mm.c_value_type, alignment=alignment)
        tensor = make_tensor(buff, acc_layout)
        clear(tensor)
        acc = _type_cast_ot_acc(tensor)
        acc._init()
        return acc

    return impl


@overload_method(BlasType, "suggest_accumulator", inline="always", strict=False)
def ol_blas_suggest_accumulator(blas_numba: BlasType):
    return ol_blas_get_accumulator_generic(blas_numba, suggested=True)


@overload_method(BlasType, "get_accumulator", inline="always", strict=False)
def ol_blas_get_accumulator(blas_numba: BlasType):
    return ol_blas_get_accumulator_generic(blas_numba, suggested=False)


@overload_method(BlasPartitionerType, "partition_like_C", jit_options={"forceinline": True}, strict=False)
def ol_blas_partition_like_C(partitioner: BlasPartitionerType, tensor: OpaqueTensorType):
    assert isinstance(partitioner, BlasPartitionerType)
    assert isinstance(tensor, OpaqueTensorType)
    assert isinstance(tensor.layout, BlasLayoutType)
    assert tensor.layout.layout.tensor_type == "gmem_c"

    raise NotImplementedError("Blas partition_like_C is not yet implemented, please use map_fragment_index instead")


def get_map_idx2crd_partitioner(symbol: str, lto: cuda.LTOIR):
    assert isinstance(symbol, str)
    return_type = types.Tuple((types.int32, types.int32))

    @intrinsic
    def map_idx2crd_partitioner(typingctx, index):
        def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args):
            context.active_code_library.add_linking_file(lto)
            idx: llvmir.AllocaInstr = args[0]
            i_ptr = cgutils.alloca_once(builder, cgutils.int32_t)
            j_ptr = cgutils.alloca_once(builder, cgutils.int32_t)
            idx_ptr = cgutils.alloca_once_value(builder, idx)

            int32_ptr = cgutils.int32_t.as_pointer()

            fnTy = llvmir.FunctionType(cgutils.voidptr_t, [int32_ptr, int32_ptr, int32_ptr])
            fn = cgutils.get_or_insert_function(builder.module, fnTy, symbol)
            builder.call(fn, [idx_ptr, i_ptr, j_ptr])

            return context.make_tuple(builder, return_type, (builder.load(i_ptr), builder.load(j_ptr)))

        return typing.signature(return_type, types.int32), codegen

    return map_idx2crd_partitioner


@overload_method(
    BlasPartitionerType,
    "map_fragment_index",
    jit_options={"forceinline": True},
    strict=False,
)
def ol_blas_partitioner_map_fragment_index(
    partitioner: BlasPartitionerType,
    index: types.Integer,
):
    if not isinstance(partitioner, BlasPartitionerType):
        return
    if not isinstance(index, types.Integer):
        return

    code, symbol = compile_blas_map_idx2crd_partitioner(partitioner.MM, code_type=get_default_code_type())

    lto = cuda.LTOIR(code.data)

    map_idx2crd_partitioner = get_map_idx2crd_partitioner(symbol, lto=lto)

    def map_fragment_index_impl(obj, idx):
        i, j = map_idx2crd_partitioner(idx)
        return (i, j)

    return map_fragment_index_impl


def get_bool_return_intrinsic(symbol: str, index: bool = False, lto=None):
    return_type = types.bool

    def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args: list):
        if lto is not None:
            context.active_code_library.add_linking_file(lto)
        active = cgutils.alloca_once(builder, cgutils.int32_t)
        int32_ptr = cgutils.int32_t.as_pointer()

        fn_args, fn_args_ty = [], []
        if index:
            idx_ptr = cgutils.alloca_once_value(builder, args[0])
            fn_args += [idx_ptr]
            fn_args_ty += [int32_ptr]
        fn_args += [active]
        fn_args_ty += [int32_ptr]

        fnTy = llvmir.FunctionType(cgutils.voidptr_t, fn_args_ty)
        fn = cgutils.get_or_insert_function(builder.module, fnTy, symbol)
        builder.call(fn, fn_args)

        res = builder.icmp_signed("!=", builder.load(active), llvmir.Constant(cgutils.int32_t, 0))
        return res

    _intrinsic: Callable = lambda typingctx: (typing.signature(return_type), codegen)
    if index:
        _intrinsic = lambda typingctx, index: (typing.signature(return_type, types.int32), codegen)

    return intrinsic(_intrinsic)


@overload_method(
    BlasPartitionerType,
    "is_thread_active",
    jit_options={"forceinline": True},
    strict=False,
)
def ol_blas_partitioner_is_thread_active(
    partitioner: BlasPartitionerType,
):
    if not isinstance(partitioner, BlasPartitionerType):
        return

    code, symbol = compile_blas_is_thread_active(partitioner.MM, code_type=get_default_code_type())
    lto = cuda.LTOIR(code.data)
    is_thread_active = get_bool_return_intrinsic(symbol, lto=lto)

    return lambda partitioner: is_thread_active()


@overload_method(
    BlasPartitionerType,
    "is_predicated",
    jit_options={"forceinline": True},
    strict=False,
)
def ol_blas_partitioner_is_predicated(
    partitioner: BlasPartitionerType,
):
    if not isinstance(partitioner, BlasPartitionerType):
        return

    code, symbol = compile_blas_is_predicated(partitioner.MM, code_type=get_default_code_type())
    lto = cuda.LTOIR(code.data)
    is_predicated = get_bool_return_intrinsic(symbol, lto=lto)

    return lambda partitioner: is_predicated()


@overload_method(
    BlasPartitionerType,
    "is_index_in_bounds",
    jit_options={"forceinline": True},
    strict=False,
)
def ol_blas_partition_is_index_in_bounds(
    partitioner: BlasPartitionerType,
    index: types.Integer,
):
    if not isinstance(partitioner, BlasPartitionerType):
        return
    if not isinstance(index, types.Integer):
        return

    code, symbol = compile_blas_is_index_in_bounds(partitioner.MM, code_type=get_default_code_type())
    lto = cuda.LTOIR(code.data)
    is_index_in_bounds = get_bool_return_intrinsic(symbol, index=True, lto=lto)

    return lambda partitioner, idx: is_index_in_bounds(idx)


@overload_method(
    BlasPartitionerType,
    "partition_and_copy",
    inline="always",
    strict=False,
)
def ol_blas_partitioner_partition_and_copy(
    partitioner: BlasPartitionerType,
    src: OpaqueTensorType,
    dst: OpaqueTensorType,
):
    if not isinstance(partitioner, BlasPartitionerType):
        return
    if not isinstance(src, OpaqueTensorType):
        return
    if not isinstance(dst, OpaqueTensorType):
        return
    assert isinstance(src.layout, BlasLayoutType)
    assert isinstance(dst.layout, BlasLayoutType)
    src_mem = src.layout.layout.memory_space
    dst_mem = dst.layout.layout.memory_space
    assert src_mem == "r" and dst_mem == "g" or src_mem == "g" and dst_mem == "r"

    alignment = max(partitioner.MM.alignment.c, src.layout.layout.alignment, dst.layout.layout.alignment)

    def impl(partitioner: Partitioner, src, dst):
        if partitioner.is_thread_active():
            copy_fragment(src, dst, alignment=alignment)

    return impl


@overload_method(
    BlasPartitionerType,
    "make_empty_fragment",
    inline="always",
    strict=False,
)
def ol_blas_partitioner_make_empty_fragment(
    partitioner: BlasPartitionerType,
):
    MM = partitioner.MM
    layout = partitioner.fragment_layout.layout

    alignment = max(8, layout.alignment, MM.alignment.c)

    def impl(partitioner):
        buff = cuda.local.array(layout.cosize, dtype=MM.c_value_type, alignment=alignment)
        frag = make_tensor(buff, layout)
        return frag

    return impl


@overload_method(
    BlasPartitionerType,
    "make_partition_and_copy",
    inline="always",
    strict=False,
)
def ol_blas_partitioner_make_partition_and_copy(
    partitioner: BlasPartitionerType,
    tensor: OpaqueTensorType,
):
    if not isinstance(tensor, OpaqueTensorType):
        return

    layout_ty = partitioner.fragment_layout
    layout = layout_ty.layout
    if layout_ty.dtype != tensor.layout.dtype:
        layout = layout_ty.make_layout_like(tensor.layout.layout.dtype).layout
    assert isinstance(layout, _BlasMatmulLayout)

    MM = partitioner.MM
    alignment = max(8, layout.alignment, MM.alignment.c)
    dtype = layout.dtype

    def impl(partitioner, tensor):
        buff = cuda.local.array(layout.cosize, dtype=dtype, alignment=alignment)
        frag = make_tensor(buff, layout)
        partitioner.partition_and_copy(tensor, frag)
        return frag

    return impl


@overload_method(
    BlasAccumulatorType,
    "_init",
    target="cuda",
    jit_options={"forceinline": True},
    strict=False,
)
def ol_opaque_tensor_init(
    accumulator: BlasAccumulatorType,
):
    if not isinstance(accumulator, BlasAccumulatorType):
        return

    layout = accumulator.layout
    assert isinstance(layout, BlasLayoutType)
    code, symbol = compile_blas_accumulator_init(layout.layout, code_type=get_default_code_type())
    lto = cuda.LTOIR(code.data)

    return_type = types.void
    sig = typing.signature(return_type, accumulator)

    cublasdx_init_accumulator = declare_cabi_device(symbol, sig, link=lto)

    def impl(accumulator):
        cublasdx_init_accumulator(accumulator)

    return impl


@overload_method(
    BlasAccumulatorType,
    "get_results",
    inline="always",
    strict=False,
)
def ol_blas_accumulator_get_results(
    accumulator: BlasAccumulatorType,
):
    if not isinstance(accumulator, BlasAccumulatorType):
        return

    def impl(accumulator):
        frag = accumulator.make_empty_fragment()
        copy_fragment(accumulator, frag)

        return frag

    return impl


@overload_method(BlasAccumulatorType, "clear", jit_options={"forceinline": True}, strict=False)
def ol_blas_accumulator_clear(accumulator: BlasAccumulatorType):
    opaque_tensor_type = OpaqueTensorType(accumulator.layout)

    @intrinsic
    def _type_cast_ot_tensor(typingctx, opaque_tensor):
        def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args):
            return args[0]

        return_type = opaque_tensor_type
        return typing.signature(return_type, accumulator), codegen

    def impl(accumulator: BlasAccumulatorType):
        tensor = _type_cast_ot_tensor(accumulator)
        clear(tensor)

    return impl


@overload_method(DevicePipelineType, "get_tile", target="cuda", inline="always", strict=False)
def ol_device_pipeline_get_tile(
    device_pipeline: DevicePipelineType,
    smem: types.Array,
    idx: types.Number,
    idy: types.Number,
):
    if not isinstance(device_pipeline, DevicePipelineType):
        return
    if not isinstance(smem, types.Array):
        return
    if not isinstance(idx, types.Number) and not isinstance(idx, types.UniTuple):
        return
    if not isinstance(idy, types.Number) and not isinstance(idy, types.UniTuple):
        return

    assert (
        isinstance(idx, types.Number)
        and isinstance(idy, types.Number)
        or isinstance(idx, types.UniTuple)
        and isinstance(idy, types.UniTuple)
    )

    def impl(device_pipeline, smem, idx, idy):
        tile_pipeline: TilePipeline = _create_tile_pipeline(device_pipeline)
        tile_pipeline_buffer = cuda.local.array(
            shape=(tile_pipeline.storage_bytes,), dtype=numpy.byte, alignment=tile_pipeline.storage_alignment
        )
        tile_pipeline = _set_tile_pipeline_buffer(tile_pipeline, tile_pipeline_buffer)
        tile_pipeline._init(device_pipeline, smem, idx, idy)

        return tile_pipeline

    return impl


@overload_method(DevicePipelineType, "reset_tile", target="cuda", inline="always", strict=False)
def ol_tile_pipeline_reset_tile(
    device_pipeline: DevicePipelineType,
    tile_pipeline: TilePipelineType,
    idx: types.Number,
    idy: types.Number,
):
    if not isinstance(device_pipeline, DevicePipelineType):
        return
    if not isinstance(tile_pipeline, TilePipelineType):
        return
    if not isinstance(idx, types.Number) and not isinstance(idx, types.UniTuple):
        return
    if not isinstance(idy, types.Number) and not isinstance(idy, types.UniTuple):
        return

    assert tile_pipeline.pipeline.device_pipeline == device_pipeline.pipeline

    code, symbol = compile_blas_device_pipeline_reset_tile(
        device_pipeline.pipeline, tile_pipeline.pipeline, code_type=get_default_code_type()
    )
    lto = cuda.LTOIR(code.data)

    return_type = types.void
    sig = typing.signature(
        return_type,
        OpaquePointerType(),
        OpaquePointerType(),
        types.CPointer(types.int32),
        types.CPointer(types.int32),
    )

    cublasdx_reset_tile = declare_cabi_device(symbol, sig, link=lto)

    if isinstance(idx, types.Number):

        def impl(device_pipeline, tile_pipeline, idx, idy):
            idx = get_value_ptr(types.int32(idx))
            idy = get_value_ptr(types.int32(idy))
            cublasdx_reset_tile(get_opaque_pointer(device_pipeline), get_opaque_pointer(tile_pipeline), idx, idy)

        return impl
    else:

        def impl(device_pipeline, tile_pipeline, idx, idy):
            idx_arr = cuda.local.array(shape=(2,), dtype=numpy.int32)
            idx_arr[0], idx_arr[1] = idx
            idy_arr = cuda.local.array(shape=(2,), dtype=numpy.int32)
            idy_arr[0], idy_arr[1] = idy
            idx_ptr = get_array_ptr(idx_arr)
            idy_ptr = get_array_ptr(idy_arr)
            cublasdx_reset_tile(get_opaque_pointer(device_pipeline), get_opaque_pointer(tile_pipeline), idx_ptr, idy_ptr)

        return impl


@overload_method(TilePipelineType, "_init", target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_tile_pipeline_init(
    tile_pipeline: TilePipelineType,
    device_pipeline: DevicePipelineType,
    smem: types.Array,
    idx: types.Number,
    idy: types.Number,
):
    if not isinstance(tile_pipeline, TilePipelineType):
        return
    if not isinstance(device_pipeline, DevicePipelineType):
        return
    if not isinstance(smem, types.Array):
        return
    if not isinstance(idx, types.Number) and not isinstance(idx, types.UniTuple):
        return
    if not isinstance(idy, types.Number) and not isinstance(idy, types.UniTuple):
        return

    assert tile_pipeline.pipeline.device_pipeline == device_pipeline.pipeline

    code, symbol = compile_blas_tile_pipeline_init(tile_pipeline.pipeline, code_type=get_default_code_type())
    lto = cuda.LTOIR(code.data)

    return_type = types.void
    sig = typing.signature(
        return_type,
        OpaquePointerType(),
        OpaquePointerType(),
        types.CPointer(smem.dtype),
        types.CPointer(types.int32),
        types.CPointer(types.int32),
    )

    cublasdx_init_pipeline = declare_cabi_device(symbol, sig, link=lto)

    if isinstance(idx, types.Number):

        def impl(tile_pipeline, device_pipeline, smem, idx, idy):
            smem_ptr = get_array_ptr(smem)
            idx = get_value_ptr(types.int32(idx))
            idy = get_value_ptr(types.int32(idy))

            cublasdx_init_pipeline(get_opaque_pointer(device_pipeline), get_opaque_pointer(tile_pipeline), smem_ptr, idx, idy)

        return impl
    else:

        def impl(tile_pipeline, device_pipeline, smem, idx, idy):
            smem_ptr = get_array_ptr(smem)
            idx_arr = cuda.local.array(shape=(2,), dtype=numpy.int32)
            idx_arr[0], idx_arr[1] = idx
            idy_arr = cuda.local.array(shape=(2,), dtype=numpy.int32)
            idy_arr[0], idy_arr[1] = idy
            idx_ptr = get_array_ptr(idx_arr)
            idy_ptr = get_array_ptr(idy_arr)
            cublasdx_init_pipeline(
                get_opaque_pointer(device_pipeline), get_opaque_pointer(tile_pipeline), smem_ptr, idx_ptr, idy_ptr
            )

        return impl


@overload_method(TilePipelineType, "_del", target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_tile_pipeline_destroy(tile_pipeline: TilePipelineType):
    if not isinstance(tile_pipeline, TilePipelineType):
        return

    code, symbol = compile_blas_tile_pipeline_destroy(tile_pipeline.pipeline, code_type=get_default_code_type())
    lto = cuda.LTOIR(code.data)

    return_type = types.void
    sig = typing.signature(return_type, tile_pipeline)

    cublasdx_destroy_pipeline = declare_cabi_device(symbol, sig, link=lto)

    def impl(tile_pipeline):
        cublasdx_destroy_pipeline(tile_pipeline)

    return impl


@overload_method(TilePipelineType, "execute", target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_tile_pipeline_execute(
    tile_pipeline: TilePipelineType,
    accumulator: BlasAccumulatorType,
):
    assert isinstance(accumulator, BlasAccumulatorType)
    acc_layout = accumulator.layout.layout
    assert acc_layout.accumulator

    code, symbol = compile_blas_tile_pipeline_execute(tile_pipeline.pipeline, acc_layout, code_type=get_default_code_type())
    lto = cuda.LTOIR(code.data)

    return_type = types.void
    sig = typing.signature(return_type, OpaquePointerType(), accumulator)

    cublasdx_tile_pipeline_execute = declare_cabi_device(symbol, sig, link=lto)

    def impl(tile_pipeline, accumulator):
        cublasdx_tile_pipeline_execute(get_opaque_pointer(tile_pipeline), accumulator)

    return impl


@intrinsic
def _get_set_ot_item(typingctx, tensor: OpaqueTensorType, index: types.Integer, val=None):
    def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args):
        tensor_ty = signature.args[0]
        tensor, idx = args[0], args[1]

        tensor_struct = cgutils.create_struct_proxy(tensor_ty)(context, builder, tensor)
        buff_void_ptr = tensor_struct.ptr

        assert isinstance(buff_void_ptr.type, llvmir.PointerType)

        llvm_val_ty = context.get_value_type(tensor_ty.dtype)

        data_ptr_ty = llvmir.PointerType(llvm_val_ty, buff_void_ptr.type.addrspace)
        buff_ptr = builder.bitcast(buff_void_ptr, data_ptr_ty)

        val_ptr = builder.gep(buff_ptr, [idx], inbounds=True)

        if val is None:
            return builder.load(val_ptr)
        else:
            val_arg = args[2]
            builder.store(val_arg, val_ptr)

    if val is None:
        ret_type = tensor.dtype
        sig = typing.signature(ret_type, tensor, index, types.none)
    else:
        ret_type = types.void
        sig = typing.signature(ret_type, tensor, index, tensor.dtype)
    return sig, codegen


def _validate_fragment_tensor(tensor: OpaqueTensorType) -> bool:
    if not isinstance(tensor, OpaqueTensorType):
        return False

    layout_ty = tensor.layout
    if not isinstance(layout_ty, BlasLayoutType):
        return False

    layout = layout_ty.layout
    if not isinstance(layout, _BlasMatmulLayout):
        return False
    return layout.memory_space == "r"


@overload(operator.getitem, jit_options={"forceinline": True}, strict=False)
def ol_blas_partition_getitem(tensor: OpaqueTensorType, index: types.Integer):
    if not _validate_fragment_tensor(tensor):
        return
    if not isinstance(index, types.Integer):
        return

    return lambda tensor, index: _get_set_ot_item(tensor, index)


@overload(operator.setitem, jit_options={"forceinline": True}, strict=False)
def ol_blas_partition_setitem(tensor: OpaqueTensorType, index: types.Integer, value):
    if not _validate_fragment_tensor(tensor):
        return
    if not isinstance(index, types.Integer):
        return
    if not isinstance(value, types.Number):
        return

    dtype = tensor.dtype

    return lambda tensor, index, value: _get_set_ot_item(tensor, index, dtype(value))


class _PipelineExtension:
    def prepare_args(self, ty, val, **kwargs):
        if isinstance(val, DevicePipeline):
            assert isinstance(ty, DevicePipelineType)
            c_ptr = types.CPointer(types.void)
            return c_ptr, int(val._storage.handle)
        else:
            return ty, val


# TODO: make implicit, once numba-cuda supports it
#  https://github.com/NVIDIA/numba-cuda/pull/504
pipeline_extensions = [_PipelineExtension()]
