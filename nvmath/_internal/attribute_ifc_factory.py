# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Generic factory that builds a Python class with one ``property`` per
enum member, plus optional ``set_<attr>_unchecked`` methods, from
caller-supplied closures.

The low-level :func:`_make_attribute_interface_impl` is generic:
it receives *make_getter* / *make_setter* / *make_setter_unchecked*
callables that produce the per-attribute closures, then attaches them
as properties.

Library-specific wrappers (e.g. :func:`make_cublas_attribute_interface`)
supply the closure factories that encode a particular C API convention
(argument order, ``size_written`` type, logging, etc.).
"""

__all__ = ["make_cublas_attribute_interface"]

import ctypes
import enum
import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger()

# A closure factory accepted by _impl.  Called once per attribute name;
# returns the bound method to attach as a getter, setter, etc.
_ClosureFactory = Callable[[str], Callable]


def _make_attribute_interface_impl(
    *,
    class_module: str,
    class_name: str,
    attribute_names: list[str],
    make_getter: _ClosureFactory,
    make_setter: _ClosureFactory | None = None,
    make_setter_unchecked: _ClosureFactory | None = None,
) -> type:
    """Build and return an attribute interface class.

    This is the low-level implementation behind library-specific wrappers
    such as :func:`make_cublas_attribute_interface`.  It performs no
    caller-site validation and is completely agnostic to the underlying
    C API convention -- all library-specific logic lives in the closure
    factories passed by the caller.

    Args:
        class_module: ``__module__`` for the generated class.
        class_name: ``__name__`` and ``__qualname__`` for the generated
            class.
        attribute_names: Uppercase attribute names (e.g.
            ``["POINTER_MODE", "EPILOGUE"]``).
        make_getter: ``(name) -> getter`` factory that returns a
            property getter closure.
        make_setter: ``(name) -> setter`` factory that returns a
            property setter closure, or ``None`` for read-only
            interfaces.
        make_setter_unchecked: ``(name) -> setter`` factory for
            hot-path setters without logging, or ``None`` to skip.
    """

    class AttributeInterface:  # noqa: D101
        if TYPE_CHECKING:

            def __getattr__(self, name: str) -> Any: ...
            def __setattr__(self, name: str, value: Any) -> None: ...

        def __init__(self, obj: Any) -> None:
            self._handle = obj

    AttributeInterface.__name__ = class_name
    AttributeInterface.__qualname__ = class_name
    AttributeInterface.__module__ = class_module

    for _name in attribute_names:
        _lower = _name.lower()

        getter = make_getter(_name)

        if make_setter is not None:
            setter = make_setter(_name)
            if make_setter_unchecked is not None:
                setattr(
                    AttributeInterface,
                    f"set_{_lower}_unchecked",
                    make_setter_unchecked(_name),
                )
        else:
            setter = None
        setattr(AttributeInterface, _lower, property(getter, setter))

    return AttributeInterface


# ---------------------------------------------------------------------------
# cuBLAS-specific wrapper
# ---------------------------------------------------------------------------


def _build_enum_to_ctype_map(
    attribute_enum: type[enum.IntEnum],
    get_attribute_dtype_fn: Callable[[enum.IntEnum], np.dtype],
    exclude: set[str] | None = None,
) -> dict[str, tuple[int, type, int]]:
    """
    Return ``{NAME: (enum_int_value, ctypes_type, sizeof)}`` for every
    non-deprecated enum member whose name is not in *exclude*.
    """
    result: dict[str, tuple[int, type, int]] = {}
    for member in attribute_enum:
        name: str = member.name

        if exclude and name in exclude:
            continue

        try:
            dtype: np.dtype = get_attribute_dtype_fn(member)
        except ValueError:
            continue

        try:
            ctype = np.ctypeslib.as_ctypes_type(dtype)
        except NotImplementedError as exc:
            raise TypeError(
                f"get_attribute_dtype_fn returned a dtype for {name!r} that cannot be converted to a ctypes type"
            ) from exc

        result[name] = (member.value, ctype, ctypes.sizeof(ctype))

    return result


def _cublas_interface_class_doc(
    *,
    attribute_enum: type[enum.IntEnum],
    get_attribute_fn: Callable[..., None],
    set_attribute_fn: Callable[..., None] | None,
    with_unchecked: bool,
    attribute_names: list[str],
) -> str:
    enum_qualname = f"{attribute_enum.__module__}.{attribute_enum.__qualname__}"
    getter_qualname = f"{get_attribute_fn.__module__}.{get_attribute_fn.__qualname__}"

    names_sorted = sorted(attribute_names)
    if names_sorted:
        example_upper = names_sorted[0]
        example_lower = example_upper.lower()
    else:
        example_upper = "UPPER_CASE"
        example_lower = "lower_case"

    properties_extra = (
        " (and their ``set_<attr>_unchecked`` variants)" if set_attribute_fn is not None and with_unchecked else ""
    )

    lines = [
        f"Property-based access to attributes enumerated in\n``{enum_qualname}``.",
        "",
        f"Properties{properties_extra} are generated \ndynamically at import time by ``make_cublas_attribute_interface``.",
        "",
        "This avoids hand-writing a property for every entry in "
        f"``{enum_qualname}``, \nkeeps the interface in sync automatically when new\n"
        "attributes are added to the binding, and normalises the UPPER_CASE enum\n"
        "names to lowercase property names "
        f"(e.g. ``{example_upper}`` becomes ``{example_lower}``).",
        "",
    ]

    if set_attribute_fn is not None:
        set_qualname = f"{set_attribute_fn.__module__}.{set_attribute_fn.__qualname__}"
        delegates = f"``{getter_qualname}`` / ``{set_qualname}``"
    else:
        delegates = f"``{getter_qualname}``"
    lines.append(
        "Each generated property delegates to the corresponding\n"
        f"{delegates} \n"
        "call with the correct enum value and ctypes type\n"
        "already baked in via closure."
    )

    if set_attribute_fn is not None and with_unchecked:
        lines.extend(
            [
                "",
                "The ``set_<attr>_unchecked`` methods are identical to the property\n"
                "setters but skip debug logging. They are useful in low-overhead code paths \n"
                "where the attribute type has already been validated during initialization.",
            ]
        )

    return "\n".join(lines)


def make_cublas_attribute_interface(
    *,
    class_module: str,
    class_name: str,
    attribute_enum: type[enum.IntEnum],
    get_attribute_dtype_fn: Callable[[enum.IntEnum], np.dtype],
    get_attribute_fn: Callable[..., None],
    set_attribute_fn: Callable[..., None] | None = None,
    with_unchecked: bool = False,
    exclude: set[str] | None = None,
) -> type:
    """
    Build and return an attribute interface class for a cuBLAS-style
    descriptor whose attributes are accessed through a uniform
    ``*_get_attribute`` / ``*_set_attribute`` C API pattern.

    The returned class's docstring is generated automatically from the
    enum and the getter/setter callables (see *Returns*).

    This is the public entry point for all cublasLt and cublasMp
    descriptor / algorithm interface modules.
    For example, the entire ``MatmulDescInterface`` for
    ``nvmath.linalg`` is reduced to::

        from nvmath.bindings import cublasLt as cublaslt

        MatmulDescInterface = make_cublas_attribute_interface(
            class_module=__name__,
            class_name="MatmulDescInterface",
            attribute_enum=cublaslt.MatmulDescAttribute,
            get_attribute_dtype_fn=cublaslt.get_matmul_desc_attribute_dtype,
            get_attribute_fn=cublaslt.matmul_desc_get_attribute,
            set_attribute_fn=cublaslt.matmul_desc_set_attribute,
        )

    The opaque library handle passed to ``__init__`` is stored as
    ``self._handle``. Subclasses that need direct access to it (e.g.
    for variable-length attribute queries) should use ``self._handle``.

    Args:
        class_module: The ``__module__`` to assign to the generated
            class.  Callers should pass ``__name__``.
        class_name: The ``__name__`` and ``__qualname__`` to assign to
            the generated class.  Must match the module-level variable
            name the caller assigns the result to.
        attribute_enum: The binding's enum type whose members enumerate
            all gettable/settable attributes.
        get_attribute_dtype_fn: Callable that maps an enum member to its
            numpy dtype.
        get_attribute_fn: Callable invoked as
            ``(handle, enum_value, buf_addr, buf_size, size_written_addr)``
            that must store the attribute's value into *buf_addr*.
        set_attribute_fn: Callable invoked as
            ``(handle, enum_value, buf_addr, buf_size)`` that must apply
            the new value from *buf_addr*.  ``None`` for read-only
            interfaces.
        with_unchecked: If ``True``, generate ``set_<attr>_unchecked``
            methods (no debug logging).  Requires *set_attribute_fn*.
        exclude: Enum member names to skip.

    Returns:
        A class with one *lowercase* property per enum member.
        Its ``__doc__`` is built automatically from the enum
        and the getter/setter callables.

    .. note::
        This function **must** be called at module scope (i.e. as a
        top-level statement in the module body). A ``RuntimeError`` is
        raised otherwise. This constraint ensures that the generated
        class receives a correct ``__qualname__``.
    """

    if with_unchecked and set_attribute_fn is None:
        raise ValueError("with_unchecked=True requires set_attribute_fn")

    caller_frame = sys._getframe(1)
    if caller_frame.f_code.co_name != "<module>":
        raise RuntimeError("make_cublas_attribute_interface must be called at module level")

    attr_info = _build_enum_to_ctype_map(attribute_enum, get_attribute_dtype_fn, exclude)

    def _cublas_make_getter(name):
        enum_value, ctype, sizeof = attr_info[name]

        def getter(self):
            logger.debug("Getting attribute %s.", name)
            buf = ctype()
            sw = ctypes.c_uint64()
            get_attribute_fn(self._handle, enum_value, ctypes.addressof(buf), sizeof, ctypes.addressof(sw))
            return buf.value

        return getter

    def _cublas_make_setter(name):
        enum_value, ctype, sizeof = attr_info[name]

        def setter(self, value):
            logger.debug("Setting attribute %s to %s.", name, value)
            v = ctype(value)
            set_attribute_fn(self._handle, enum_value, ctypes.addressof(v), sizeof)

        return setter

    def _cublas_make_setter_unchecked(name):
        enum_value, ctype, sizeof = attr_info[name]

        def setter_unc(self, value):
            v = ctype(value)
            set_attribute_fn(self._handle, enum_value, ctypes.addressof(v), sizeof)

        return setter_unc

    attr_names = list(attr_info)
    iface_cls = _make_attribute_interface_impl(
        class_module=class_module,
        class_name=class_name,
        attribute_names=attr_names,
        make_getter=_cublas_make_getter,
        make_setter=_cublas_make_setter if set_attribute_fn is not None else None,
        make_setter_unchecked=(_cublas_make_setter_unchecked if (set_attribute_fn is not None and with_unchecked) else None),
    )
    iface_cls.__doc__ = _cublas_interface_class_doc(
        attribute_enum=attribute_enum,
        get_attribute_fn=get_attribute_fn,
        set_attribute_fn=set_attribute_fn,
        with_unchecked=with_unchecked,
        attribute_names=attr_names,
    )
    return iface_cls
