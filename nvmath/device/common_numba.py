# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir
from numba import types
from numba.core import cgutils
import numpy as np
from .vector_types_numba import float16x2_type, float16x4_type, float32x2_type, float64x2_type
from .types import np_float16x2, np_float16x4

NP_TYPES_TO_NUMBA_FE_TYPES = {
    np.float16: np.float16,
    np.float32: np.float32,
    np.float64: np.float64,
    np_float16x2: float16x2_type,
    np_float16x4: float16x4_type,
    np.complex64: float32x2_type,
    np.complex128: float64x2_type,
}

NUMBA_FE_TYPES_TO_NUMBA_IR = {
    np.float16: types.float16,
    np.float32: types.float32,
    np.float64: types.float64,
    float16x2_type: float16x2_type,
    float16x4_type: float16x4_type,
    float32x2_type: float32x2_type,
    float64x2_type: float64x2_type,
}


def make_dx_codegen_one_arg(context, builder, type_, arg):
    # Arrays are lowered by passing a pointer to the data
    # arg can be None, in which case a null pointer is passed
    if isinstance(type_, types.Array):
        dtype = type_.dtype
        valueTy = context.get_value_type(dtype)
        ptrTy = ir.PointerType(valueTy)
        if arg is None:
            ptr = ptrTy(None)
        else:
            ptr = cgutils.create_struct_proxy(type_)(context, builder, arg).data
        return (ptrTy, ptr)
    # Integers are passed as-is
    # arg can be None, in which case a 0 is passed
    elif isinstance(type_, types.Integer):
        intTy = context.get_value_type(type_)
        if arg is None:
            val = intTy(0)
        else:
            val = arg
        return (intTy, val)
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
