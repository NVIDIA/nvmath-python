# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import models, overload_attribute
import numpy as np

from .vector_types_numba import float16x2_type, float16x4_type, float32x2_type, float64x2_type
from .types import np_float16x2, np_float16x4
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


def make_dx_codegen_one_arg(context, builder: ir.IRBuilder, type_, arg):
    # mathdx expects everything to be passed by pointers.
    # Arrays are lowered by passing a pointer to the data
    # arg can be None, in which case a null pointer is passed
    if isinstance(type_, OpaqueTensorType):
        ptrTy = ir.PointerType(ir.IntType(8))
        ldTy = ir.IntType(64)

        opaque_tensor = cgutils.create_struct_proxy(type_)(context, builder, arg)
        ptr = cgutils.create_struct_proxy(type_.buffer_type)(context, builder, opaque_tensor.buffer).data

        # Future release of numba-cuda may have support for address spaces.
        # It is not supported to pass a non generic pointer to device function
        # call.
        if ptr.type.addrspace != 0:
            ptr = builder.addrspacecast(ptr, ir.PointerType(ptr.type.pointee), "generic")

        ptr = builder.bitcast(ptr, ptrTy)

        layout = cgutils.create_struct_proxy(type_.layout)(context, builder, opaque_tensor.layout)

        member_values = [ptr]
        member_types = [ptrTy]
        if type_.layout.dynamic_ld:
            member_types += [ldTy]
            ld = builder.bitcast(layout.leading_dimension, ldTy)
            member_values += [ld]

        structTy = ir.LiteralStructType(member_types)
        val = cgutils.pack_struct(builder, member_values)

        return (structTy, val)
    elif isinstance(type_, types.Array):
        dtype = type_.dtype
        valueTy = context.get_value_type(dtype)
        ptrTy = ir.PointerType(valueTy)
        if arg is None:
            ptr = ptrTy(None)
        else:
            ptr = cgutils.create_struct_proxy(type_)(context, builder, arg).data

        # Future release of numba-cuda may have support for address spaces.
        # It is not supported to pass a non generic pointer to device function
        # call.
        if ptr.type.addrspace != 0:
            ptr = builder.addrspacecast(ptr, ir.PointerType(ptr.type.pointee), "generic")

        return (ptrTy, ptr)
    # Integers are passed as-pointers
    # arg can be None, in which case a 0 is passed
    elif isinstance(type_, types.Integer):
        intTy = context.get_value_type(type_)
        if arg is None:
            val = intTy(0)
        else:
            val = arg
        ptrTy = ir.PointerType(intTy)
        ptr = cgutils.alloca_once_value(builder, val)
        return (ptrTy, ptr)
    # Floats and Complex are passed by reference (pointer) This is because some CUDA C++
    # types, such as __half2 are non-trivial, and those must be passed by reference. For
    # consistency we pass everything by reference.
    elif type_ in [float16x2_type, float16x4_type, float32x2_type, float64x2_type] or isinstance(  # noqa: UP038
        type_, (types.Float, types.Complex)
    ):
        assert arg is not None
        valueTy = context.get_value_type(type_)
        ptrTy = ir.PointerType(valueTy)
        ptr = cgutils.alloca_once_value(builder, arg)
        return (ptrTy, ptr)
    else:
        raise RuntimeError(f"Unsupported lowering for type {type_} for arg {arg}")


def make_function_call(symbol):
    def codegen(context, builder, sig, args):
        assert len(sig.args) == len(args)
        argsTyAndArgs = [make_dx_codegen_one_arg(context, builder, t, a) for (t, a) in zip(sig.args, args, strict=True)]
        argsTy = [t for (t, _) in argsTyAndArgs]
        args = [v for (_, v) in argsTyAndArgs]
        retTy = context.get_value_type(sig.return_type)
        fnTy = ir.FunctionType(retTy, argsTy)
        fn = cgutils.get_or_insert_function(builder.module, fnTy, symbol)
        builder.call(fn, args)

    return codegen


class EmptyStructModel(models.StructModel):
    """Data model that does not take space. Intended to be used with types that
    are presented only at typing stage and not represented in memory."""

    def __init__(self, dmm, fe_type):
        members = []
        super().__init__(dmm, fe_type, members)


def overload_type_attribute(numba_type, attribute_base, attribute):
    """Make type attribute available inside jitted code."""
    assert issubclass(numba_type, types.Type)

    @overload_attribute(numba_type, attribute, inline="always", target="cuda")
    def ol_blas_attribute(blas_numba):
        tp = blas_numba
        if attribute_base != "":
            tp = getattr(tp, attribute_base)
        val = getattr(tp, attribute)
        return lambda blas_numba: val
