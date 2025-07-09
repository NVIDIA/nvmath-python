# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import typing, cgutils
from numba.extending import typeof_impl, overload_method, intrinsic, types, utils, overload
from numba.cuda.cudaimpl import lower_constant, registry as cuda_registry
from numba.cuda.models import register_model

from nvmath.device.common_cuda import get_default_code_type
from nvmath.device.cublasdx_backend import generate_copy_wait_lto

from .common import axpby, copy, copy_fragment, clear, copy_wait, make_tensor, OpaqueTensor
from .common_numba import NUMBA_FE_TYPES_TO_NUMBA_IR, make_function_call, overload_type_attribute, EmptyStructModel
from .cublasdx import BlasNumba, _BlasLayout
from .common_opaque_tensor import LayoutModel, LayoutType, OpaqueTensorType

import llvmlite.ir as llvmir
from numba.core.base import BaseContext

_BLAS_DEFINITION_ARGS = [
    "size",
    "precision",
    "data_type",
    "code_type",
    "block_size",
    "block_dim",
    "leading_dimension",
    "transpose_mode",
    "arrangement",
    "function",
    "execution",
    "execute_api",
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
    Type class associated with the `cublasdx.BlasNumba`.
    """

    def __init__(self, blas: BlasNumba):
        assert isinstance(blas, BlasNumba)
        self._blas = blas
        attributes = [f"{attr}={getattr(blas, attr)}" for attr in _BLAS_DEFINITION_ARGS if getattr(blas, attr)]
        if blas._tensor_types:
            attributes += [f"tensor_types={blas._tensor_types}"]
        attributes.sort()

        self.name = "BlasNumba(" + ",".join(attributes) + ")"

    @property
    def blas(self) -> BlasNumba:
        return self._blas


register_model(BlasType)(EmptyStructModel)


@lower_constant(BlasType)
def constant_dummy(context, builder, typ, pyval):
    struct_ptr = cgutils.create_struct_proxy(typ)(context, builder)
    return struct_ptr._getvalue()


@typeof_impl.register(BlasNumba)
def typeof_blas_numba(val: BlasNumba, c: typing.Context) -> BlasType:
    return BlasType(val)


for attribute in _BLAS_COMPILED_ARGS + _BLAS_DEFINITION_ARGS:
    overload_type_attribute(BlasType, "blas", attribute)


# Numba does not support method overload or variadic arguments, so we using
# default values as a workaround
# https://github.com/numba/numba/issues/9980
# https://github.com/numba/numba/issues/9979
@overload_method(BlasType, "execute", target="cuda", inline="always", strict=False)
def ol_blas_numba_execute(*args):
    return ol_blas_numba(*args)


@overload_method(BlasType, "__call__", target="cuda", strict=False)
def ol_blas_numba_call(blas_numba: BlasType, _arg1, _arg2, _arg3, _arg4=None, _arg5=None, _arg6=None, _arg7=None, _arg8=None):
    return ol_blas_numba(blas_numba, _arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8)


def ol_blas_numba(blas_numba: BlasType, _arg1, _arg2, _arg3, _arg4=None, _arg5=None, _arg6=None, _arg7=None, _arg8=None):
    if blas_numba.blas.execute_api == "tensors" and blas_numba.blas._tensor_types[2] == "suggested_rmem_c":
        assert _arg4 in {None, types.Omitted(None)}
        assert _arg5 in {None, types.Omitted(None)}
        assert _arg6 in {None, types.Omitted(None)}
        assert _arg7 in {None, types.Omitted(None)}
        assert _arg8 in {None, types.Omitted(None)}

        return lambda _, a, b, c, _arg4=None, _arg5=None, _arg6=None, _arg7=None, _arg8=None: _bals_type___call__(_, a, b, c)
    elif (
        blas_numba.blas.execute_api == "static_leading_dimensions"
        or blas_numba.blas.execute_api == "tensors"
        and blas_numba.blas._tensor_types[2] != "suggested_rmem_c"
    ):
        assert _arg6 in {None, types.Omitted(None)}
        assert _arg7 in {None, types.Omitted(None)}
        assert _arg8 in {None, types.Omitted(None)}

        return lambda _, alpha, a, b, beta, c, _arg6=None, _arg7=None, _arg8=None: _bals_type___call__(_, alpha, a, b, beta, c)
    elif blas_numba.blas.execute_api == "dynamic_leading_dimensions":
        return lambda _, alpha, a, lda, b, ldb, beta, c, ldc: _bals_type___call__(_, alpha, a, lda, b, ldb, beta, c, ldc)
    else:
        return  # no implementation


# TODO: use overload_method when supported
def _bals_type___call__(*args):
    raise Exception("Stub for overloads")


@overload(_bals_type___call__, inline="always", strict=False)
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
    if (a.uid, b.uid, c.uid) != MM._target_tensor_uids:
        return

    @intrinsic
    def sym_call(typingctx, a, b, c):
        return_type = types.void
        sig = typing.signature(return_type, a, b, c)
        return sig, make_function_call(MM._tensor_api_symbols.gemm)

    return lambda _, a, b, c: sym_call(a, b, c)


@overload(_bals_type___call__, inline="always", strict=False)
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
    if (a.uid, b.uid, c.uid) != MM._target_tensor_uids:
        return

    @intrinsic
    def sym_call(typingctx, alpha, a, b, beta, c):
        return_type = types.void
        sig = typing.signature(return_type, c.dtype, a, b, c.dtype, c)
        return sig, make_function_call(MM._tensor_api_symbols.gemm)

    return lambda _, alpha, a, b, beta, c: sym_call(alpha, a, b, beta, c)


@overload(_bals_type___call__, inline="always", strict=False)
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
    if (a.dtype, b.dtype, c.dtype) != tuple(NUMBA_FE_TYPES_TO_NUMBA_IR[vt] for vt in MM._numba_value_types):
        return

    # setting signature for intrinsic to much calling conventions. Numba will
    # automatically cast to desired values.
    return_type = types.void
    sig = typing.signature(return_type, c.dtype, a, b, c.dtype, c)

    symbol = MM.symbol

    @intrinsic
    def sym_call(typingctx, alpha, a, b, beta, c):
        return sig, make_function_call(symbol)

    return lambda _, alpha, a, b, beta, c: sym_call(alpha, a, b, beta, c)


@overload(_bals_type___call__, inline="always", strict=False)
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
    if (a.dtype, b.dtype, c.dtype) != tuple(NUMBA_FE_TYPES_TO_NUMBA_IR[vt] for vt in MM._numba_value_types):
        return
    if not all(isinstance(a, types.Integer) for a in (lda, ldb, ldc)):
        return

    # setting signature for intrinsic to much calling conventions. Numba will
    # automatically cast to desired values.
    ld_type = types.uint32
    return_type = types.void
    sig = typing.signature(return_type, c.dtype, a, ld_type, b, ld_type, c.dtype, c, ld_type)

    @intrinsic
    def sym_call(typingctx, alpha, a, lda, b, ldb, beta, c, ldc):
        return sig, make_function_call(MM.symbol)

    return lambda _, alpha, a, lda, b, ldb, beta, c, ldc: sym_call(alpha, a, lda, b, ldb, beta, c, ldc)


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


@overload(copy, target="cuda", inline="always", strict=False)
def ol_blas_copy(src: OpaqueTensorType, dst: OpaqueTensorType):
    return ol_blas_copy_generic(src, dst, "copy")


@overload(copy_fragment, target="cuda", inline="always", strict=False)
def ol_blas_copy_fragment(src: OpaqueTensorType, dst: OpaqueTensorType):
    return ol_blas_copy_generic(src, dst, "copy_fragment")


def ol_blas_copy_generic(src: OpaqueTensorType, dst: OpaqueTensorType, func: str):
    assert isinstance(src, OpaqueTensorType)
    assert isinstance(src.layout, BlasLayoutType)
    assert isinstance(dst, OpaqueTensorType)
    assert isinstance(dst.layout, BlasLayoutType)

    rmem = "rmem" in dst.layout.layout or "rmem" in src.layout.layout

    if func == "copy_fragment":
        assert rmem
    else:
        assert func == "copy"
        assert not rmem

    symbol = src.layout.copy_to_symbol(dst.layout)

    @intrinsic
    def _intrinsic(typingctx, src, dst):
        assert isinstance(src, OpaqueTensorType)
        assert isinstance(dst, OpaqueTensorType)
        assert src.dtype == dst.dtype

        return_type = types.void
        return typing.signature(return_type, src, dst), make_function_call(symbol)

    def impl(src, dst):
        return _intrinsic(src, dst)

    return impl


@overload(clear, target="cuda", inline="always", strict=False)
def ol_blas_clear(arr: OpaqueTensorType):
    assert isinstance(arr, OpaqueTensorType)
    assert isinstance(arr.layout, BlasLayoutType)
    assert arr.buffer_type

    symbol = arr.layout.clear_symbol

    assert symbol

    @intrinsic
    def _intrinsic(typingctx, arr):
        return_type = types.void
        return typing.signature(return_type, arr), make_function_call(symbol)

    def impl(arr):
        return _intrinsic(arr)

    return impl


class BlasLayoutType(LayoutType):
    """
    Type class associated with opaque tensor layouts.
    """

    def __init__(self, MM: BlasNumba, layout: str):
        assert isinstance(MM, BlasNumba)
        assert isinstance(layout, str)

        blas_layout = _BlasLayout(MM, layout)

        self._uid = blas_layout._uid
        self._size = blas_layout._size
        self._cosize = blas_layout._cosize
        self._tensor_index = blas_layout._tensor_index
        self._dynamic_ld = blas_layout._dynamic_ld
        self._dtype = NUMBA_FE_TYPES_TO_NUMBA_IR[MM._value_types[self._tensor_index]]
        self._layout = layout

        self._copy_symbols_map = MM._copy_symbols_map
        self._clear_symbol = MM._tensor_api_symbols.clear_c if blas_layout._is_register else None
        self._axpby_symbol = MM._tensor_api_symbols.axpby if blas_layout._is_register else None

        # Using handle descriptor in the type name to avoid symbol copy caching
        # by numba.
        self.name = f"Layout(uid={self._uid},layout={self._layout},handle={MM._handle.descriptor})"

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def uid(self) -> int:
        return self._uid

    @property
    def tensor_index(self) -> int:
        """Tensor index is 0 for A, 1 for B and 2 for C."""
        return self._tensor_index

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def size(self) -> int:
        return self._size

    @property
    def cosize(self) -> int:
        return self._cosize

    @property
    def dynamic_ld(self) -> bool:
        return self._dynamic_ld

    def copy_to_symbol(self, dst: "BlasLayoutType") -> str:
        return self._copy_symbols_map[(self.uid, dst.uid)]

    @property
    def clear_symbol(self) -> str | None:
        return self._clear_symbol

    @property
    def axpby_symbol(self) -> str | None:
        return self._axpby_symbol


register_model(BlasLayoutType)(LayoutModel)


@lower_constant(BlasLayoutType)
def constant_blas_layout(context, builder, typ, pyval):
    struct_ptr = cgutils.create_struct_proxy(typ)(context, builder)
    return struct_ptr._getvalue()


@typeof_impl.register(_BlasLayout)
def typeof_blas_layout(val: _BlasLayout, c: typing.Context) -> BlasLayoutType:
    assert val._MM is not None
    return BlasLayoutType(val._MM, val._layout)


for attribute in ["size", "cosize"]:
    overload_type_attribute(BlasLayoutType, "", attribute)


def ol_blas_layout(blas_numba: BlasType, method: str, leading_dimension: types.Number | None = None):
    # leading_dimension is available only for global memory
    if ("gmem" not in method) and (leading_dimension not in {None, types.Omitted(None)}):
        return
    MM = blas_numba.blas
    if not MM._tensor_types:
        return

    return_type = BlasLayoutType(MM, method)

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
        inline="always",
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


@overload(make_tensor, target="cuda", inline="always", strict=False)
def ol_make_tensor(array, layout):
    assert isinstance(array, types.Array)
    assert isinstance(layout, BlasLayoutType)
    assert array.dtype == layout.dtype

    return lambda array, layout: OpaqueTensor(array, layout)


@overload(copy_wait, target="cuda", inline="always", strict=False)
def ol_copy_wait():
    # numba has cache per compute capability, so the function won't end up
    # cached for the wrong compute capability.
    ct = get_default_code_type()
    symbol, _ = generate_copy_wait_lto(ct.cc)

    @intrinsic
    def _intrinsic(typingctx):
        return_type = types.void
        return typing.signature(return_type), make_function_call(symbol)

    return lambda: _intrinsic()


@overload(axpby, target="cuda", inline="always", strict=False)
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

    symbol = x.layout.axpby_symbol

    assert symbol is not None

    @intrinsic
    def _intrinsic(typingctx, a, x, b, y):
        return_type = types.void
        return typing.signature(return_type, x.dtype, x, y.dtype, y), make_function_call(symbol)

    return lambda a, x, b, y: _intrinsic(a, x, b, y)
