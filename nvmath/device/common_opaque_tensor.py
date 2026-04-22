# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property

import numpy
from llvmlite import ir as llir
from numba.core import cgutils
from numba.core.base import BaseContext
from numba.cuda.extending import lower_builtin, make_attribute_wrapper, type_callable
from numba.cuda.models import register_model
from numba.extending import models, types

from nvmath.device.common import Layout, OpaqueTensor
from nvmath.device.common_numba import NUMBA_FE_TYPES_TO_NUMBA_IR, overload_type_attribute
from nvmath.device.llvm_array import LLVMArray

_LIBMATHDX_RUNTIME = -9223372036854775808


class OpaqueLayout(Layout):
    """
    A generic tensor layout that can represent arbitrary shapes and strides.
    It is the same as libmathdx opaque tensor, but without the pointer
    to data.
    """

    _dtype: numpy.number
    _shape: tuple[int, ...]
    _strides: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...], dtype: numpy.number):
        assert len(shape) == len(strides)
        self._shape = shape
        self._strides = strides
        self._dtype = dtype

    @cached_property
    def dynamic_strides_size(self) -> int:
        return self._strides.count(_LIBMATHDX_RUNTIME)

    @cached_property
    def dynamic_shape_size(self) -> int:
        return self._shape.count(_LIBMATHDX_RUNTIME)

    @property
    def dtype(self) -> numpy.number:
        return self._dtype

    @cached_property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides


class LayoutType(types.Type):
    """
    Type class associated with opaque tensor layouts.
    """

    def __init__(
        self,
        layout: OpaqueLayout,
    ):
        self._layout = layout
        super().__init__(
            f"Layout(dynamic_shape_size={layout.dynamic_shape_size}, dynamic_strides_size={layout.dynamic_strides_size})"
        )

    @property
    def layout(self) -> OpaqueLayout:
        return self._layout

    @cached_property
    def dtype(self) -> types.Number:
        return NUMBA_FE_TYPES_TO_NUMBA_IR[self._layout.dtype]


@register_model(LayoutType)
class LayoutModel(models.StructModel):
    def __init__(self, dmm, fe_type: LayoutType):
        members = []

        strides_size = fe_type.layout.dynamic_strides_size
        if strides_size > 0:
            array_type = LLVMArray(types.int64, strides_size)
            members += [("strides", array_type)]

        shape_size = fe_type.layout.dynamic_shape_size
        if shape_size > 0:
            array_type = LLVMArray(types.int64, shape_size)
            members += [("shape", array_type)]

        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(LayoutType, "strides", "strides")
make_attribute_wrapper(LayoutType, "shape", "shape")

for attribute in ("uid", "alignment", "storage_bytes"):
    overload_type_attribute(LayoutType, "", attribute)


class OpaqueTensorType(types.Type):
    def __init__(self, layout: LayoutType):
        super().__init__(f"OpaqueTensor(layout={layout})")
        self._layout = layout

    @property
    def uid(self):
        return self._layout.layout.uid

    @property
    def layout(self) -> LayoutType:
        return self._layout

    @cached_property
    def ndim(self):
        return 2

    @cached_property
    def dtype(self):
        return self._layout.dtype


class OpaqueTensorModel(models.StructModel):
    def __init__(self, dmm, fe_type: OpaqueTensorType):
        members = [
            ("ptr", types.voidptr),
        ]
        layout = fe_type.layout.layout

        strides_size = layout.dynamic_strides_size
        if strides_size > 0:
            array_type = LLVMArray(types.int64, strides_size)
            members += [("strides", array_type)]

        shape_size = layout.dynamic_shape_size
        if shape_size > 0:
            array_type = LLVMArray(types.int64, shape_size)
            members += [("shape", array_type)]

        models.StructModel.__init__(self, dmm, fe_type, members)


register_model(OpaqueTensorType)(OpaqueTensorModel)


@type_callable(OpaqueTensor)
def type_callable_OpaqueTensor(context):
    def typer(buffer_ty, layout_ty):
        if not isinstance(buffer_ty, types.Array):
            return
        if not isinstance(layout_ty, LayoutType):
            return
        assert buffer_ty.dtype == layout_ty.dtype
        return OpaqueTensorType(layout_ty)

    return typer


@lower_builtin(OpaqueTensor, types.Array, LayoutType)
def impl_opaque_tensor(context: BaseContext, builder: llir.IRBuilder, sig, args):
    tensor_ty: OpaqueTensorType = sig.return_type
    buffer, buffer_ty = args[0], sig.args[0]
    layout, layout_ty = args[1], sig.args[1]

    ptr = cgutils.create_struct_proxy(buffer_ty)(context, builder, buffer).data
    # Future release of numba-cuda may have support for address spaces.
    # It is not supported to pass a non generic pointer to device function
    # call.
    if ptr.type.addrspace != 0:
        ptr = builder.addrspacecast(ptr, llir.PointerType(ptr.type.pointee), "generic")

    voidPtrTy = llir.PointerType(llir.IntType(8))
    ptr = builder.bitcast(ptr, voidPtrTy)

    opaque_tensor = cgutils.create_struct_proxy(tensor_ty)(context, builder)
    opaque_tensor.ptr = ptr

    layout_struct = cgutils.create_struct_proxy(layout_ty)(context, builder, layout)

    if tensor_ty.layout.layout.dynamic_shape_size > 0:
        opaque_tensor.shape = layout_struct.shape

    if tensor_ty.layout.layout.dynamic_strides_size > 0:
        opaque_tensor.strides = layout_struct.strides

    return opaque_tensor._getvalue()


make_attribute_wrapper(OpaqueTensorType, "ptr", "ptr")
overload_type_attribute(OpaqueTensorType, "layout", "layout")
