# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

import llvmlite.ir as llvmir
import numpy as np
from llvmlite import ir
from numba import types
from numba.core import cgutils, typing
from numba.core.base import BaseContext
from numba.core.errors import TypingError
from numba.cuda.cudaimpl import lower_constant
from numba.cuda.extending import intrinsic, overload, overload_attribute, register_model
from numba.extending import models, typeof_impl

from nvmath.bindings import mathdx

from .llvm_array import LLVMArray
from .types import Complex, Vector, complex32, complex64, complex128, half2, half4, np_float16x2, np_float16x4
from .vector_types_numba import float16x2_type, float16x4_type, float32x2_type, float64x2_type

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

    @overload_attribute(numba_type, attribute, inline="always")
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


def build_get_value_ptr(target_type=None):
    @intrinsic
    def get_value_ptr_impl(typingctx: typing.Context, value):
        """Get raw pointer to the value."""
        if value not in [float16x2_type, float16x4_type, float32x2_type, float64x2_type] and not isinstance(  # noqa: UP038
            value, (types.Float, types.Complex, types.Integer)
        ):
            raise TypingError(f"get_value_ptr does not support type {value}")

        return_type = target_type if target_type is not None else value
        sig = typing.signature(types.CPointer(return_type), value)

        def codegen(context: BaseContext, builder, sig, args):
            if target_type is not None and sig.args[0] != target_type:
                value = context.cast(builder, args[0], sig.args[0], target_type)
            else:
                value = args[0]
            return cgutils.alloca_once_value(builder, value)

        return sig, codegen

    return get_value_ptr_impl


get_value_ptr = build_get_value_ptr()
get_uint32_value_ptr = build_get_value_ptr(target_type=types.uint32)


class MathdxOpaqueTensorType(types.Type):
    def __init__(self, dynamic_strides_size=0, dynamic_shape_size=0):
        self.dynamic_strides_size = dynamic_strides_size
        self.dynamic_shape_size = dynamic_shape_size
        super().__init__(f"MathdxOpaqueTensor({dynamic_strides_size}, {dynamic_shape_size})")


@register_model(MathdxOpaqueTensorType)
class MathdxOpaqueTensorModel(models.StructModel):
    def __init__(self, dmm, fe_type: MathdxOpaqueTensorType):
        members = [("ptr", types.voidptr)]

        if fe_type.dynamic_strides_size > 0:
            array_type = LLVMArray(types.int64, fe_type.dynamic_strides_size)
            members += [("strides", array_type)]

        if fe_type.dynamic_shape_size > 0:
            array_type = LLVMArray(types.int64, fe_type.dynamic_shape_size)
            members += [("shape", array_type)]

        models.StructModel.__init__(self, dmm, fe_type, members)

    def get_value_type(self):
        if self._value_type is None:
            name = f"struct.libmathdx_tensor_{self._fe_type.dynamic_strides_size}s_{self._fe_type.dynamic_shape_size}s"
            t = llvmir.global_context.get_identified_type(name)
            if t.is_opaque:
                t.set_body(*[m.get_value_type() for m in self._models])
            self._value_type = t
        return self._value_type


class MathdxPipelineType(types.Type):
    def __init__(self):
        super().__init__("MathdxPipeline()")


@register_model(MathdxPipelineType)
class MathdxPipelineModel(models.StructModel):
    def __init__(self, dmm, fe_type: MathdxPipelineType):
        members = [("ptr", types.voidptr)]
        models.StructModel.__init__(self, dmm, fe_type, members)

    def get_value_type(self):
        if self._value_type is None:
            name = "struct.libmathdx_pipeline"
            t = llvmir.global_context.get_identified_type(name)
            if t.is_opaque:
                t.set_body(*[m.get_value_type() for m in self._models])
            self._value_type = t
        return self._value_type


@intrinsic
def cast_to_void_pointer(typingctx: typing.Context, ptr_ty):
    if not isinstance(ptr_ty, types.CPointer):
        raise TypingError("cast_to_void_pointer expects a CPointer type")
    i8_ptr_ty = types.CPointer(types.uint8)
    sig = i8_ptr_ty(ptr_ty)

    def codegen(context: BaseContext, builder, sig, args):
        return builder.bitcast(args[0], context.get_value_type(i8_ptr_ty))

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

            # CUDA target maps types.void to i8*, but we need void for abi correctness.
            if sig.return_type == types.void:
                retTy = ir.VoidType()
            else:
                retTy = context.get_value_type(sig.return_type)
            fnTy = ir.FunctionType(retTy, [context.get_value_type(argTy) for argTy in argTypes])
            fn = cgutils.get_or_insert_function(builder.module, fnTy, symbol)
            builder.call(fn, args)

        return intrinsic_sig, codegen

    def device_func():
        pass

    @overload(device_func, jit_options={"forceinline": True})
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


def register_dummy_numba_type(numba_type, base_type, name, attributes):
    """Register and lower a dummy Numba type that does not take space in memory.
    Intended for types that are present only at typing stage and not represented in memory,
    but are allowed to be accessed inside kernel."""
    for attribute in attributes:
        overload_type_attribute(numba_type, name, attribute)

    @typeof_impl.register(base_type)
    def typeof_numba(val, context):
        return numba_type(val)

    register_model(numba_type)(EmptyStructModel)

    @lower_constant(numba_type)
    def constant_dummy(context, builder, typ, pyval):
        struct_ptr = cgutils.create_struct_proxy(typ)(context, builder)
        return struct_ptr._getvalue()
