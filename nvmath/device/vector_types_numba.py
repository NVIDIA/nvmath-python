# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "float16x2",
    "float16x4",
    "float32x2",
    "float64x2",
    "float16x2_type",
    "float16x4_type",
    "float32x2_type",
    "float64x2_type",
]

import typing

from numba.core import types

from numba.extending import overload, models, register_model, lower_builtin, as_numba_type, type_callable, lower_cast
from numba.core.typeconv import Conversion
from llvmlite import ir
from numba.core.typing.templates import AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import lower_attr as cuda_lower_attr


def make_vector(
    float_bitwidth: int, numba_float_type: types.scalars.Float, llvm_float_type, vector_length: typing.Literal[2, 4]
):
    assert vector_length == 2 or vector_length == 4
    vector_bitwidth = vector_length * float_bitwidth

    # User visible types and functions
    class vector:
        def __init__(self, *args):
            pass

    def create(input):
        pass

    # FE types
    class vectorType(types.Number):
        def __init__(self):
            super().__init__(name=f"vector({numba_float_type.name}x{vector_length})")
            self.dtype = numba_float_type
            self.count = vector_length
            self.bitwidth = vector_bitwidth
            self.make = vector

        def can_convert_from(self, typingctx, other):
            if vector_length == 2 and isinstance(other, types.Complex):
                if other.bitwidth == vector_bitwidth:
                    return Conversion.exact
                elif other.bitwidth < vector_bitwidth:
                    return Conversion.safe
                elif other.bitwidth > vector_bitwidth:
                    return Conversion.unsafe

        def can_convert_to(self, typingctx, other):
            if vector_length == 2 and isinstance(other, types.Complex):
                if other.bitwidth == vector_bitwidth:
                    return Conversion.exact
                elif other.bitwidth < vector_bitwidth:
                    return Conversion.unsafe
                elif other.bitwidth > vector_bitwidth:
                    return Conversion.safe

        def __name__(self):
            return f"vector({numba_float_type.name}x{vector_length})"

    vector_type = vectorType()

    types.number_domain = types.number_domain | frozenset([vector_type])

    # Type inference (Python --> FE type)
    as_numba_type.register(vector, vector_type)

    # How to build a vectorType from individual elements
    @type_callable(vector)
    def type(context):
        def typer(x, y=None, z=None, w=None):
            if (
                (vector_length == 2 and isinstance(x, types.Complex) and y is None and z is None and w is None)
                or (vector_length == 2 and all(isinstance(v, types.Float) for v in [x, y]) and z is None and w is None)
                or (vector_length == 4 and all(isinstance(v, types.Float) for v in [x, y, z, w]))
            ):
                return vector_type

        return typer

    # How to represent vectorType in Memory?
    @register_model(vectorType)
    class vectorTypeModel(models.PrimitiveModel):
        def __init__(self, dmm, fe_type):
            be_type = ir.IntType(vector_bitwidth)
            super().__init__(dmm, fe_type, be_type)

    # How to build a vectorType from individual float values?
    @lower_builtin(vector, *(vector_length * [types.Float]))  # FIXME
    def vector_ctor(context, builder, sig, args):
        typ = sig.return_type
        args = list(args)
        assert len(args) == vector_length
        for i in range(vector_length):
            args[i] = context.cast(builder, args[i], sig.args[i], numba_float_type)

        vt = ir.VectorType(llvm_float_type, vector_length)
        val = ir.Constant(vt, ir.Undefined)

        for i in range(vector_length):
            val = builder.insert_element(val, args[i], context.get_constant(types.int32, i))

        return builder.bitcast(val, context.get_value_type(typ))

    # How to access .x, .y, .z, .w
    @cuda_registry.register_attr
    class complex_attrs(AttributeTemplate):
        key = vectorType

        def resolve_x(self, mod):
            return numba_float_type

        def resolve_y(self, mod):
            return numba_float_type

        def resolve_z(self, mod):
            return numba_float_type

        def resolve_w(self, mod):
            return numba_float_type

    def complex_get(context, builder, typ, val, i):
        vt = ir.VectorType(llvm_float_type, vector_length)
        vec = builder.bitcast(val, vt)
        index = context.get_constant(types.int32, i)
        return builder.extract_element(vec, index)

    # This cannot be done with a loop
    @cuda_lower_attr(vectorType, "x")
    def complex_get_x(context, builder, typ, val):
        return complex_get(context, builder, typ, val, 0)

    @cuda_lower_attr(vectorType, "y")
    def complex_get_y(context, builder, typ, val):
        return complex_get(context, builder, typ, val, 1)

    if vector_length == 4:

        @cuda_lower_attr(vectorType, "z")
        def complex_get_z(context, builder, typ, val):
            return complex_get(context, builder, typ, val, 2)

        @cuda_lower_attr(vectorType, "w")
        def complex_get_w(context, builder, typ, val):
            return complex_get(context, builder, typ, val, 3)

    # Conversions from types.Complex
    if vector_length == 2:

        @overload(create)
        def make_complex(cplx):
            if isinstance(cplx, types.Complex):

                def impl(cplx):
                    return vector(cplx.real, cplx.imag)

                return impl

        @lower_cast(types.Complex, vector_type)
        def np_complex_to_complex(context, builder, fromty, toty, val):
            src = context.make_complex(builder, fromty, value=val)
            src_float_ty = fromty.underlying_float
            ctor_sig = toty(src_float_ty, src_float_ty)
            ctor_args = (src.real, src.imag)
            return vector_ctor(context, builder, ctor_sig, ctor_args)

        @lower_cast(vector_type, types.Complex)
        def complex_to_np_complex(context, builder, fromty, toty, val):
            dst_float_ty = toty.underlying_float
            x = complex_get(context, builder, fromty, val, 0)
            y = complex_get(context, builder, fromty, val, 1)
            x = context.cast(builder, x, numba_float_type, dst_float_ty)
            y = context.cast(builder, y, numba_float_type, dst_float_ty)
            vt = context.get_value_type(toty)
            val = ir.Constant(vt, ir.Undefined)
            val = builder.insert_value(val, x, [0])
            val = builder.insert_value(val, y, [1])
            return val

    vector.__name__ = f"float{float_bitwidth}x{vector_length}"
    vector.__doc__ = f"""
    float{float_bitwidth}x{vector_length}({"x, y" if vector_length == 2 else "x, y, z, w"})

    Create a Numba compliant vector object for float{float_bitwidth} with vector length {vector_length}.
    """
    return vector, vector_type


float16x2, float16x2_type = make_vector(16, types.float16, ir.IntType(16), 2)
float16x4, float16x4_type = make_vector(16, types.float16, ir.IntType(16), 4)
float32x2, float32x2_type = make_vector(32, types.float32, ir.FloatType(), 2)
float64x2, float64x2_type = make_vector(64, types.float64, ir.DoubleType(), 2)
