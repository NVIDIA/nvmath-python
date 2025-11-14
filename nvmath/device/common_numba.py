# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from collections.abc import Callable
from llvmlite import ir
from numba import types
from numba.core import cgutils, typing
from numba.core.base import BaseContext
from numba.extending import models, overload, overload_attribute, typeof_impl, intrinsic
from numba.core.errors import TypingError
from nvmath.bindings import mathdx
import numpy as np


from .vector_types_numba import float16x2_type, float16x4_type, float32x2_type, float64x2_type
from .types import np_float16x2, np_float16x4, complex32, complex64, complex128, half2, half4, Complex, Vector
from .common_opaque_tensor import OpaqueTensorType

NP_TYPES_TO_NUMBA_FE_TYPES = {
    np.float16: np.float16,
    np.float32: np.float32,
    np.float64: np.float64,
    np_float16x2: float16x2_type,
    np_float16x4: float16x4_type,
    np.complex64: float32x2_type,
    np.complex128: float64x2_type,
    np.int8: np.int8,
    np.int16: np.int16,
    np.int32: np.int32,
    np.int64: np.int64,
    np.uint8: np.uint8,
    np.uint16: np.uint16,
    np.uint32: np.uint32,
    np.uint64: np.uint64,
}

NUMBA_FE_TYPES_TO_NUMBA_IR = {
    np.float16: types.float16,
    np.float32: types.float32,
    np.float64: types.float64,
    complex32: float16x2_type,
    complex64: float32x2_type,
    complex128: float64x2_type,
    half2: float16x2_type,
    half4: float16x4_type,
    float16x2_type: float16x2_type,
    float16x4_type: float16x4_type,
    float32x2_type: float32x2_type,
    float64x2_type: float64x2_type,
    np.int8: types.int8,
    np.int16: types.int16,
    np.int32: types.int32,
    np.int64: types.int64,
    np.uint8: types.uint8,
    np.uint16: types.uint16,
    np.uint32: types.uint32,
    np.uint64: types.uint64,
}


class EmptyStructModel(models.StructModel):
    """Data model that does not take space. Intended to be used with types that
    are presented only at typing stage and not represented in memory."""

    def __init__(self, dmm, fe_type):
        members = []
        super().__init__(dmm, fe_type, members)


def overload_type_attribute(numba_type, attribute_base, attribute):
    """Make type attribute available inside jitted code."""
    assert issubclass(numba_type, types.Type)

    @overload_attribute(numba_type, attribute, jit_options={"forceinline": True}, target="cuda")
    def ol_blas_attribute(blas_numba):
        tp = blas_numba
        if attribute_base != "":
            tp = getattr(tp, attribute_base)
        val = getattr(tp, attribute)
        return lambda blas_numba: val


@typeof_impl.register(Complex)
def typeof_complex(val: Complex, c: typing.Context) -> Any:
    if val.real_dtype == np.float16:
        return types.NumberClass(float16x2_type)
    elif val.real_dtype == np.float32:
        return types.NumberClass(float32x2_type)
    elif val.real_dtype == np.float64:
        return types.NumberClass(float64x2_type)

    raise RuntimeError(f"Unsupported complex real dtype {val.real_dtype}")


@typeof_impl.register(Vector)
def typeof_vector(val: Vector, c: typing.Context) -> Any:
    if val.real_dtype != np.float16 or val.size not in (2, 4):
        raise RuntimeError(f"Unsupported vector type {val.real_dtype}x{val.size}")

    return types.NumberClass(float16x2_type if val.size == 2 else float16x4_type)


@intrinsic
def get_array_ptr(typingctx: typing.Context, arr):
    """Get raw pointer to the data of a Numba array."""
    assert isinstance(arr, types.Array)

    sig = typing.signature(types.CPointer(arr.dtype), arr)

    def codegen(context: BaseContext, builder, sig, args):
        arrTy = sig.args[0]
        arr = args[0]

        dtype = arrTy.dtype
        valueTy = context.get_value_type(dtype)
        ptrTy = ir.PointerType(valueTy)
        if arr is None:
            ptr = ptrTy(None)
        else:
            ptr = cgutils.create_struct_proxy(arrTy)(context, builder, arr).data

        # Future release of numba-cuda may have support for address spaces.
        # It is not supported to pass a non generic pointer to device function
        # call.
        if ptr.type.addrspace != 0:
            ptr = builder.addrspacecast(ptr, ir.PointerType(ptr.type.pointee), "generic")

        return ptr

    return sig, codegen


@intrinsic
def get_value_ptr(typingctx: typing.Context, value):
    """Get raw pointer to the value."""
    if value not in [float16x2_type, float16x4_type, float32x2_type, float64x2_type] and not isinstance(  # noqa: UP038
        value, (types.Float, types.Complex, types.Integer)
    ):
        raise TypingError(f"get_value_ptr does not support type {value}")

    sig = typing.signature(types.CPointer(value), value)

    def codegen(context: BaseContext, builder, sig, args):
        return cgutils.alloca_once_value(builder, args[0])

    return sig, codegen


@intrinsic
def get_opaque_tensor(typingctx: typing.Context, value: OpaqueTensorType):
    """Get raw pointer to the value."""
    if not isinstance(value, OpaqueTensorType):
        raise TypingError(f"get_opaque_tensor does not support type {value}")

    sig = typing.signature(value._capi_type, value)

    def codegen(context: BaseContext, builder, sig, args):
        ptrTy = ir.PointerType(ir.IntType(8))
        ldTy = ir.IntType(64)

        opaque_tensor = cgutils.create_struct_proxy(value)(context, builder, args[0])
        ptr = cgutils.create_struct_proxy(value.buffer_type)(context, builder, opaque_tensor.buffer).data

        # Future release of numba-cuda may have support for address spaces.
        # It is not supported to pass a non generic pointer to device function
        # call.
        if ptr.type.addrspace != 0:
            ptr = builder.addrspacecast(ptr, ir.PointerType(ptr.type.pointee), "generic")

        ptr = builder.bitcast(ptr, ptrTy)

        layout = cgutils.create_struct_proxy(value.layout)(context, builder, opaque_tensor.layout)

        member_values = [ptr]
        if value.layout.dynamic_ld:
            ld = builder.bitcast(layout.leading_dimension, ldTy)
            member_values += [ld]

        return cgutils.pack_struct(builder, member_values)

    return sig, codegen


_cabi_device_registry: dict[str | tuple[str, typing.Signature], Callable] = {}


def declare_cabi_device(symbol: str, sig: typing.Signature, link=None):
    """declare_cabi_device is an analog of cuda.declare_device but uses C ABI
    calling convention instead of Numba ABI calling convention. It means that
    the first argument is not a return value pointer."""
    key: str | tuple[str, typing.Signature] = symbol
    if mathdx.get_version_ex() < (0, 3, 0):
        key = (symbol, sig)
    device_func = _cabi_device_registry.get(key)
    if device_func is None:
        device_func = _declare_cabi_device(symbol, sig, link)
        _cabi_device_registry[key] = device_func

    return device_func


def _declare_cabi_device(symbol: str, sig: typing.Signature, link=None):
    intrinsic_sig = sig.return_type(types.Tuple(sig.args))

    @intrinsic
    def call_device(typingctx: typing.Context, args):
        def codegen(context: BaseContext, builder, sig: typing.Signature, args):
            if link is not None:
                context.active_code_library.add_linking_file(link)

            args = cgutils.unpack_tuple(builder, args[0])
            argTypes = sig.args[0]

            assert len(args) == len(argTypes)

            retTy = context.get_value_type(sig.return_type)
            fnTy = ir.FunctionType(retTy, [context.get_value_type(argTy) for argTy in argTypes])
            fn = cgutils.get_or_insert_function(builder.module, fnTy, symbol)
            builder.call(fn, args)

        return intrinsic_sig, codegen

    def device_func():
        pass

    @overload(device_func, jit_options={"forceinline": True}, target="cuda")
    def ol_device_func(*args):
        if len(args) != len(sig.args):
            raise RuntimeError(
                f"Invalid number of arguments for device function {symbol}: expected {len(sig.args)}, got {len(args)}"
            )
        for expected_t, provided_t in zip(sig.args, args, strict=True):
            if expected_t == provided_t:
                continue
            raise RuntimeError(f"Invalid argument type for device function {symbol}: expected {expected_t}, got {provided_t}")
        return lambda *args: call_device(tuple(args))

    return device_func
