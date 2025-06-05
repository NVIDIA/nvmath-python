# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from functools import cached_property

from numba.extending import types, models, type_callable, lower_builtin, overload_attribute
from numba.cuda.models import register_model

from nvmath.device.common import OpaqueTensor
from numba.cuda.extending import make_attribute_wrapper

from numba.core import cgutils

from llvmlite.ir import IRBuilder
from numba.core.base import BaseContext


class LayoutType(types.Type):
    """
    Type class associated with opaque tensor layouts.
    """

    @property
    @abstractmethod
    def uid(self) -> int:
        pass

    @property
    @abstractmethod
    def dtype(self) -> str:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    @abstractmethod
    def dynamic_ld(self) -> bool:
        pass


@register_model(LayoutType)
class LayoutModel(models.StructModel):
    def __init__(self, dmm, fe_type: LayoutType):
        members = []
        if fe_type.dynamic_ld:
            members += [("leading_dimension", types.int64)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(LayoutType, "leading_dimension", "leading_dimension")


@overload_attribute(LayoutType, "size", inline="always", strict=False)
def ol_layout_size(layout: LayoutType):
    assert isinstance(layout, LayoutType)
    size = layout.size
    return lambda _: size


class OpaqueTensorType(types.Type):
    def __init__(self, buffer_type: types.Array, layout: LayoutType):
        assert layout.dtype == buffer_type.dtype
        super().__init__(f"OpaqueTensor(layout={layout})")
        self._buffer_type = buffer_type
        self._layout = layout

    @property
    def uid(self):
        return self._layout.uid

    @property
    def buffer_type(self):
        return self._buffer_type

    @property
    def layout(self) -> LayoutType:
        return self._layout

    @cached_property
    def ndim(self):
        return 2

    @cached_property
    def dtype(self):
        return self.buffer_type.dtype


@register_model(OpaqueTensorType)
class OpaqueTensorModel(models.StructModel):
    def __init__(self, dmm, fe_type: OpaqueTensorType):
        members = [
            ("buffer", fe_type.buffer_type),
            ("layout", fe_type.layout),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(OpaqueTensorType, "buffer", "buffer")
make_attribute_wrapper(OpaqueTensorType, "layout", "layout")


@type_callable(OpaqueTensor)
def type_callable_OpaqueTensor(context):
    def typer(buffer_ty, layout_ty):
        if not isinstance(buffer_ty, types.Array):
            return
        if not isinstance(layout_ty, LayoutType):
            return
        return OpaqueTensorType(buffer_ty, layout_ty)

    return typer


@lower_builtin(OpaqueTensor, types.Array, LayoutType)
def impl_interval(context: BaseContext, builder: IRBuilder, sig, args):
    typ = sig.return_type
    buffer = args[0]
    layout = args[1]
    opaque_tensor = cgutils.create_struct_proxy(typ)(context, builder)
    opaque_tensor.buffer = buffer
    opaque_tensor.layout = layout

    return opaque_tensor._getvalue()
