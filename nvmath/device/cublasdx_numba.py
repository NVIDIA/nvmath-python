# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
import operator
from collections.abc import Callable
from numba import cuda
import numba
from numba.core import typing, cgutils
from numba.extending import typeof_impl, overload_method, intrinsic, types, utils, overload
from numba.cuda.cudaimpl import lower_constant, registry as cuda_registry
from numba.cuda.models import register_model
from numba.core.errors import TypingError
from numba.np import numpy_support

from nvmath.device.common_cuda import get_default_code_type
from nvmath.device.cublasdx_backend import generate_copy_wait_lto

from .common import axpby, copy, copy_fragment, clear, copy_wait, make_tensor, OpaqueTensor
from .common_numba import (
    NUMBA_FE_TYPES_TO_NUMBA_IR,
    declare_cabi_device,
    get_array_ptr,
    get_opaque_tensor,
    get_value_ptr,
    overload_type_attribute,
    EmptyStructModel,
)
from .cublasdx import (
    Matmul,
    _BlasLayout,
    compile_blas_axpby,
    compile_blas_clear,
    compile_blas_copy,
    compile_blas_execute,
    compile_blas_is_index_in_bounds,
    compile_blas_is_predicated,
    compile_blas_is_thread_active,
    compile_blas_map_idx2crd_partitioner,
)
from .common_opaque_tensor import (
    LayoutModel,
    LayoutType,
    OpaqueTensorType,
    PartitionModel,
    PartitionType,
    PartitionerModel,
    PartitionerType,
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


# Numba does not support method overload or variadic arguments, so we using
# default values as a workaround
# https://github.com/numba/numba/issues/9980
# https://github.com/numba/numba/issues/9979
# https://github.com/numba/numba/issues/10143
@overload_method(BlasType, "execute", target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_blas_numba_execute(
    blas_numba: BlasType, _arg1, _arg2, _arg3, _arg4=None, _arg5=None, _arg6=None, _arg7=None, _arg8=None
):
    return ol_blas_numba(blas_numba, _arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8)


@overload_method(BlasType, "__call__", target="cuda", strict=False)
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
    if not all(isinstance(t, OpaqueTensorType) for t in (a, b, c)):
        return
    if not all(isinstance(t.layout, BlasLayoutType) for t in (a, b, c)):
        return
    if "rmem" not in c.layout.blas_layout._tensor_type:
        return

    return_type = types.void
    sig = typing.signature(return_type, a._capi_type, b._capi_type, c._capi_type)

    code, symbol = compile_blas_execute(
        MM,
        code_type=get_default_code_type(),
        execute_api="tensors",
        tensor_types=(a.layout.blas_layout._tensor_type, b.layout.blas_layout._tensor_type, c.layout.blas_layout._tensor_type),
    )

    lto = cuda.LTOIR(code.data)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    def impl(_, a, b, c):
        a_struct = get_opaque_tensor(a)
        b_struct = get_opaque_tensor(b)
        c_struct = get_opaque_tensor(c)

        blas_device_func(a_struct, b_struct, c_struct)

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
    if not all(isinstance(t.layout, BlasLayoutType) for t in (a, b, c)):
        return
    if "smem" not in c.layout.blas_layout._tensor_type:
        return

    return_type = types.void
    c_ptr = types.CPointer(c.dtype)
    sig = typing.signature(return_type, c_ptr, a._capi_type, b._capi_type, c_ptr, c._capi_type)

    code, symbol = compile_blas_execute(
        MM,
        code_type=get_default_code_type(),
        execute_api="tensors",
        tensor_types=(a.layout.blas_layout._tensor_type, b.layout.blas_layout._tensor_type, c.layout.blas_layout._tensor_type),
    )

    lto = cuda.LTOIR(code.data)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    def impl(_, alpha, a, b, beta, c):
        a_struct = get_opaque_tensor(a)
        b_struct = get_opaque_tensor(b)
        c_struct = get_opaque_tensor(c)
        alpha_ptr = get_value_ptr(c.buffer.dtype.type(alpha))
        beta_ptr = get_value_ptr(c.buffer.dtype.type(beta))

        blas_device_func(alpha_ptr, a_struct, b_struct, beta_ptr, c_struct)

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


@overload(copy, target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_blas_copy(src: OpaqueTensorType, dst: OpaqueTensorType, alignment=None):
    return ol_blas_copy_generic(src, dst, alignment, "copy")


@overload(copy_fragment, target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_blas_copy_fragment(src: OpaqueTensorType, dst: OpaqueTensorType, alignment=None):
    return ol_blas_copy_generic(src, dst, alignment, "copy_fragment")


def ol_blas_copy_generic(src: OpaqueTensorType, dst: OpaqueTensorType, alignment_ty: types.Type | None, func: str):
    assert isinstance(src, OpaqueTensorType)
    assert isinstance(src.layout, BlasLayoutType)
    assert isinstance(dst, OpaqueTensorType)
    assert isinstance(dst.layout, BlasLayoutType)
    assert src.dtype == dst.dtype

    alignment: int | None = None
    if alignment_ty not in {None, types.Omitted(None)}:
        if not isinstance(alignment_ty, types.Literal):
            return lambda src, dst, alignment: numba.literally(alignment)
        alignment = alignment_ty.literal_value
        if alignment not in {1, 2, 4, 8, 16}:
            raise TypingError(f"Alignment must be one of (1, 2, 4, 8, 16), got {alignment}")

    rmem = "rmem" in dst.layout.layout or "rmem" in src.layout.layout

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
        src_tensor=src.layout.blas_layout,
        dst_tensor=dst.layout.blas_layout,
        code_type=get_default_code_type(),
        alignment=alignment,
    )

    return_type = types.void
    sig = typing.signature(return_type, src._capi_type, dst._capi_type)

    lto = cuda.LTOIR(code.data)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    def impl(src, dst, alignment=None):
        src_struct = get_opaque_tensor(src)
        dst_struct = get_opaque_tensor(dst)
        return blas_device_func(src_struct, dst_struct)

    return impl


@overload(clear, target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_blas_clear(arr: OpaqueTensorType):
    assert isinstance(arr, OpaqueTensorType)
    assert isinstance(arr.layout, BlasLayoutType)
    assert arr.buffer_type

    code, symbol = compile_blas_clear(
        tensor=arr.layout.blas_layout,
        code_type=get_default_code_type(),
    )

    lto = cuda.LTOIR(code.data)

    return_type = types.void
    sig = typing.signature(return_type, arr._capi_type)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    def impl(arr):
        arr_struct = get_opaque_tensor(arr)
        return blas_device_func(arr_struct)

    return impl


class BlasLayoutType(LayoutType):
    """
    Type class associated with opaque tensor layouts.
    """

    def __init__(self, blas_layout: _BlasLayout):
        assert isinstance(blas_layout, _BlasLayout)

        self._blas_layout = blas_layout

        # Using handle descriptor in the type name to avoid symbol copy caching
        # by numba.
        self.name = f"Layout(uid={blas_layout._uid},layout={blas_layout._layout},MM={blas_layout._MM})"

    @property
    def blas_layout(self) -> _BlasLayout:
        return self._blas_layout

    @property
    def layout(self) -> str:
        return self._blas_layout._layout

    @property
    def uid(self) -> int:
        return self._blas_layout._uid

    @property
    def tensor_index(self) -> int:
        """Tensor index is 0 for A, 1 for B and 2 for C."""
        return self._blas_layout._tensor_index

    @cached_property
    def dtype(self) -> types.Number:
        return NUMBA_FE_TYPES_TO_NUMBA_IR[self._blas_layout.dtype]

    @property
    def size(self) -> int:
        return self._blas_layout._size

    @property
    def cosize(self) -> int:
        return self._blas_layout._cosize

    @property
    def dynamic_ld(self) -> bool:
        return self._blas_layout._dynamic_ld


register_model(BlasLayoutType)(LayoutModel)


@lower_constant(BlasLayoutType)
def constant_blas_layout(context, builder, typ, pyval):
    struct_ptr = cgutils.create_struct_proxy(typ)(context, builder)
    return struct_ptr._getvalue()


@typeof_impl.register(_BlasLayout)
def typeof_blas_layout(blas_layout: _BlasLayout, c: typing.Context) -> BlasLayoutType:
    return BlasLayoutType(blas_layout)


for attribute in ["size", "cosize"]:
    overload_type_attribute(BlasLayoutType, "", attribute)


def ol_blas_layout(blas_numba: BlasType, method: str, leading_dimension: types.Number | None = None):
    # leading_dimension is available only for global memory
    if ("gmem" not in method) and (leading_dimension not in {None, types.Omitted(None)}):
        return
    MM = blas_numba.blas

    blas_layout = getattr(MM, method)()
    return_type = BlasLayoutType(blas_layout)

    @intrinsic
    def _intrinsic(typingctx, leading_dimension=None):
        def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args):
            # Create empty struct to avoid runtime memory usage
            layout = cgutils.create_struct_proxy(return_type)(context, builder)
            if return_type.dynamic_ld:
                if isinstance(leading_dimension, types.NoneType):
                    default_ld = MM.leading_dimension[return_type.tensor_index]
                    ld = context.get_constant(types.int64, default_ld)
                else:
                    ld = args[0]
                    ld_ty = signature.args[0]
                    ld = context.cast(builder, ld, ld_ty, types.int64)
                layout.leading_dimension = ld
            return layout._getvalue()

        return typing.signature(return_type, leading_dimension), codegen

    return lambda blas_numba, leading_dimension=None: _intrinsic(leading_dimension)


def overload_blas_layout_method(method: str):
    overload_method(
        BlasType,
        method,
        target="cuda",
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
]:
    overload_blas_layout_method(method)


@overload(make_tensor, target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_make_tensor(array, layout):
    assert isinstance(array, types.Array)
    assert isinstance(layout, BlasLayoutType)
    assert array.dtype == layout.dtype

    return lambda array, layout: OpaqueTensor(array, layout)


@overload(copy_wait, target="cuda", jit_options={"forceinline": True}, strict=False)
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


@overload(axpby, target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_axpby(a, x, b, y):
    if not isinstance(a, types.Number):
        return
    if not isinstance(x, OpaqueTensorType):
        return
    if not isinstance(b, types.Number):
        return
    if not isinstance(y, OpaqueTensorType):
        return
    if x != y:
        raise TypeError("x and y must be the same tensor type")
    if "rmem" not in x.layout.layout:
        raise TypeError("axpby is only supported for rmem tensors")

    code, symbol = compile_blas_axpby(
        x_tensor=x.layout.blas_layout,
        y_tensor=y.layout.blas_layout,
        code_type=get_default_code_type(),
    )

    lto = cuda.LTOIR(code.data)

    return_type = types.void
    sig = typing.signature(return_type, types.CPointer(x.dtype), x._capi_type, types.CPointer(y.dtype), y._capi_type)
    blas_device_func = declare_cabi_device(symbol, sig, link=lto)

    def impl(a, x, b, y):
        x_struct = get_opaque_tensor(x)
        y_struct = get_opaque_tensor(y)
        a_ptr = get_value_ptr(x.buffer.dtype.type(a))
        b_ptr = get_value_ptr(y.buffer.dtype.type(b))
        return blas_device_func(a_ptr, x_struct, b_ptr, y_struct)

    return impl


class BlasPartitionerType(PartitionerType):
    """
    Type class for Blas partitioner.
    """

    def __init__(self, MM: Matmul):
        assert isinstance(MM, Matmul)
        self._MM = MM
        super().__init__(f"BlasPartitioner(MM={MM})")

    @property
    def MM(self) -> Matmul:
        return self._MM


register_model(BlasPartitionerType)(PartitionerModel)


class BlasPartitionType(PartitionType):
    """
    Type class for Blas partitioner.
    """

    def __init__(self, partitioner: BlasPartitionerType, tensor: OpaqueTensorType):
        assert isinstance(partitioner, BlasPartitionerType)
        super().__init__(partitioner, tensor)
        self.name = f"BlasPartitionType(partitioner={partitioner}, tensor={tensor})"


register_model(BlasPartitionType)(PartitionModel)


@overload_method(BlasType, "suggest_partitioner", target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_blas_suggest_partitioner(blas_numba: BlasType):
    assert isinstance(blas_numba, BlasType)

    MM = blas_numba.blas
    return_type = BlasPartitionerType(MM)

    @intrinsic
    def _intrinsic(typingctx):
        def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args):
            # Create empty struct to avoid runtime memory usage
            layout = cgutils.create_struct_proxy(return_type)(context, builder)
            return layout._getvalue()

        return typing.signature(return_type), codegen

    return lambda blas_numba: _intrinsic()


@overload_method(BlasPartitionerType, "partition_like_C", target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_blas_partition_like_C(partitioner: BlasPartitionerType, tensor: OpaqueTensorType):
    assert isinstance(partitioner, BlasPartitionerType)
    assert isinstance(tensor, OpaqueTensorType)
    assert tensor.layout.blas_layout._tensor_type == "gmem_c"

    return_type = BlasPartitionType(partitioner, tensor)

    @intrinsic
    def _intrinsic(typingctx, partitioner, tensor):
        def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args):
            partition = cgutils.create_struct_proxy(return_type)(context, builder)
            partition.partitioner = args[0]
            partition.tensor = args[1]
            return partition._getvalue()

        return typing.signature(return_type, partitioner, tensor), codegen

    return lambda partitioner, tensor: _intrinsic(partitioner, tensor)


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
    target="cuda",
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
    target="cuda",
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
    target="cuda",
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
    target="cuda",
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


@intrinsic
def extract_partition(typingctx, ty_partition: BlasPartitionType):
    assert isinstance(ty_partition, BlasPartitionType)
    return_type = types.Tuple((ty_partition.partitioner, ty_partition.tensor))

    def codegen(context: BaseContext, builder: llvmir.IRBuilder, signature, args):
        partition = cgutils.create_struct_proxy(ty_partition)(context, builder, value=args[0])
        return context.make_tuple(builder, return_type, (partition.partitioner, partition.tensor))

    return typing.signature(return_type, ty_partition), codegen


@overload(operator.getitem, target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_blas_partition_getitem(partition: BlasPartitionType, index: types.Integer):
    if not isinstance(partition, BlasPartitionType):
        return
    if not isinstance(index, types.Integer):
        return

    def dummy_getitem_impl(obj, idx):
        partitioner, tensor = extract_partition(obj)
        i, j = partitioner.map_fragment_index(idx)
        return tensor.buffer[i, j]

    return dummy_getitem_impl


@overload(operator.setitem, target="cuda", jit_options={"forceinline": True}, strict=False)
def ol_blas_partition_setitem(partition: BlasPartitionType, index: types.Integer, value: types.Number):
    if not isinstance(partition, BlasPartitionType):
        return
    if not isinstance(index, types.Integer):
        return
    if not isinstance(value, types.Number):
        return

    def dummy_setitem_impl(obj, idx, value):
        partitioner, tensor = extract_partition(obj)
        i, j = partitioner.map_fragment_index(idx)
        tensor.buffer[i, j] = value

    return dummy_setitem_impl
