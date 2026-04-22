# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for :func:`make_cublas_attribute_interface` and
:func:`_make_attribute_interface_impl` using a fake binding layer that
mimics the cuBLAS get/set attribute C API pattern without requiring
CUDA or any real library.
"""

import ctypes
import enum

import numpy as np
import pytest

from nvmath._internal.attribute_ifc_factory import (
    _build_enum_to_ctype_map,
    _make_attribute_interface_impl,
    make_cublas_attribute_interface,
)

# Fake binding layer
#
# Simulates the three things every real CUDA binding provides:
#   1. An IntEnum of attribute identifiers  (FakeAttr)
#   2. A dtype function: enum member -> numpy dtype  (fake_get_attribute_dtype)
#   3. A get/set pair that read/write attribute values through raw pointers
#      into an in-memory store  (fake_get_attribute / fake_set_attribute)
#
# The in-memory store (_attribute_store) is a dict keyed by
# (descriptor_handle, enum_value) that holds raw bytes, mimicking
# what the real C functions would write into a caller-provided buffer.


class FakeAttr(enum.IntEnum):
    """Two attributes with different dtypes to exercise type handling."""

    ALPHA = 0
    BETA = 1


ATTRIBUTE_DTYPES = {
    FakeAttr.ALPHA: np.dtype(np.int32),
    FakeAttr.BETA: np.dtype(np.float64),
}


def fake_get_attribute_dtype(member):
    """Return the numpy dtype for *member*, like ``cublaslt.get_*_attribute_dtype``."""
    return ATTRIBUTE_DTYPES[member]


_attribute_store: dict[tuple[int, int], bytes] = {}


def fake_get_attribute(handle, enum_value, buf_addr, buf_size, size_written_addr):
    """Read attribute bytes from the store into the caller's buffer.

    Mirrors the ``cublasLt*GetAttribute(handle, attr, buf, size, &written)``
    C API signature.
    """
    data = _attribute_store[(handle, enum_value)]
    ctypes.memmove(buf_addr, data, buf_size)
    ctypes.c_uint64.from_address(size_written_addr).value = len(data)


def fake_set_attribute(handle, enum_value, buf_addr, buf_size):
    """Write attribute bytes from the caller's buffer into the store.

    Mirrors the ``cublasLt*SetAttribute(handle, attr, buf, size)``
    C API signature.
    """
    _attribute_store[(handle, enum_value)] = bytes((ctypes.c_char * buf_size).from_address(buf_addr))


@pytest.fixture(autouse=True)
def _clear_attribute_store():
    """Reset the fake store between tests so they are independent."""
    _attribute_store.clear()


# ---------------------------------------------------------------------------
# Interfaces built from the fake binding (module-level, as required by the
# factory's module-level precondition).
# ---------------------------------------------------------------------------

Ifc = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="Ifc",
    attribute_enum=FakeAttr,
    get_attribute_dtype_fn=fake_get_attribute_dtype,
    get_attribute_fn=fake_get_attribute,
    set_attribute_fn=fake_set_attribute,
    with_unchecked=True,
)

ReadOnlyIfc = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="ReadOnlyIfc",
    attribute_enum=FakeAttr,
    get_attribute_dtype_fn=fake_get_attribute_dtype,
    get_attribute_fn=fake_get_attribute,
)

SetterNoUncheckedIfc = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="SetterNoUncheckedIfc",
    attribute_enum=FakeAttr,
    get_attribute_dtype_fn=fake_get_attribute_dtype,
    get_attribute_fn=fake_get_attribute,
    set_attribute_fn=fake_set_attribute,
)

ExcludeIfc = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="ExcludeIfc",
    attribute_enum=FakeAttr,
    get_attribute_dtype_fn=fake_get_attribute_dtype,
    get_attribute_fn=fake_get_attribute,
    set_attribute_fn=fake_set_attribute,
    exclude={"BETA"},
)


class _DeprecatedAttr(enum.IntEnum):
    CURRENT = 0
    DEPRECATED = 1


def _deprecated_dtype_fn(m):
    if m == _DeprecatedAttr.DEPRECATED:
        raise ValueError("deprecated")
    return np.dtype(np.int32)


DeprecatedIfc = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="DeprecatedIfc",
    attribute_enum=_DeprecatedAttr,
    get_attribute_dtype_fn=_deprecated_dtype_fn,
    get_attribute_fn=fake_get_attribute,
    set_attribute_fn=fake_set_attribute,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_properties_exist():
    """Each enum member should produce a lowercase property on the class."""
    assert isinstance(Ifc.alpha, property)
    assert isinstance(Ifc.beta, property)


def test_class_name_reflects_caller():
    """The generated class metadata should match what the caller passed."""
    assert Ifc.__name__ == "Ifc"
    assert Ifc.__qualname__ == "Ifc"
    assert Ifc.__module__ == __name__


def test_cublas_factory_class_docstring():
    """Check generated docstring."""
    doc = Ifc.__doc__
    assert doc is not None
    assert "Property-based access to attributes enumerated in" in doc
    assert f"``{__name__}.FakeAttr``." in doc
    assert "make_cublas_attribute_interface" in doc
    assert f"{__name__}.FakeAttr" in doc
    assert f"{__name__}.fake_get_attribute" in doc
    assert f"{__name__}.fake_set_attribute" in doc
    assert "``ALPHA`` becomes ``alpha``" in doc
    assert "set_<attr>_unchecked" in doc


def test_read_only_class_docstring():
    """Read-only interfaces should not mention setters or unchecked methods."""
    doc = ReadOnlyIfc.__doc__
    assert doc is not None
    assert "fake_set_attribute" not in doc
    assert "set_<attr>_unchecked" not in doc
    assert f"{__name__}.fake_get_attribute" in doc


def test_set_and_get():
    ifc = Ifc(1)
    ifc.alpha = 42
    ifc.beta = 3.14
    assert ifc.alpha == 42
    assert ifc.beta == pytest.approx(3.14)


def test_unchecked_setter():
    """``set_<attr>_unchecked`` should behave like the property setter."""
    ifc = Ifc(1)
    ifc.set_alpha_unchecked(99)
    assert ifc.alpha == 99


def test_unchecked_setter_absent_by_default():
    """When with_unchecked is False (default), no unchecked setters should
    be generated even when set_attribute_fn is provided."""
    assert not hasattr(SetterNoUncheckedIfc, "set_alpha_unchecked")
    assert not hasattr(SetterNoUncheckedIfc, "set_beta_unchecked")


def test_independent_instances():
    """Two instances with different handles must not share attribute values."""
    a, b = Ifc(1), Ifc(2)
    a.alpha = 10
    b.alpha = 20
    assert a.alpha == 10
    assert b.alpha == 20


def test_deprecated_members_skipped():
    """Enum members whose dtype function raises ValueError should be silently
    skipped, they must not appear as properties on the interface."""
    assert hasattr(DeprecatedIfc, "current")
    assert not hasattr(DeprecatedIfc, "deprecated")


def test_read_only_interface():
    """When set_attribute_fn is None, properties should be read-only and no
    setter or unchecked setter methods should be generated."""
    assert isinstance(ReadOnlyIfc.alpha, property)
    assert ReadOnlyIfc.alpha.fset is None
    with pytest.raises(AttributeError):
        ReadOnlyIfc(1).alpha = 42
    assert not hasattr(ReadOnlyIfc, "set_alpha_unchecked")
    assert not hasattr(ReadOnlyIfc, "set_beta_unchecked")


def test_exclude_skips_members():
    """Members listed in *exclude* should not appear as properties."""
    assert hasattr(ExcludeIfc, "alpha")
    assert not hasattr(ExcludeIfc, "beta")
    assert not hasattr(ExcludeIfc, "set_beta_unchecked")


def test_unconvertible_dtype_raises():
    """If get_attribute_dtype_fn returns a dtype that cannot be converted to a
    ctypes type, the factory must raise TypeError at class construction time."""

    class Attr(enum.IntEnum):
        SOMETHING = 0

    with pytest.raises(TypeError, match="cannot be converted to a ctypes type"):
        _build_enum_to_ctype_map(Attr, lambda m: np.dtype(object))


def test_with_unchecked_without_set_attribute_raises():
    """with_unchecked=True is only meaningful when setters exist."""
    with pytest.raises(ValueError, match="with_unchecked=True requires set_attribute_fn"):
        make_cublas_attribute_interface(
            class_module=__name__,
            class_name="BadUnchecked",
            attribute_enum=FakeAttr,
            get_attribute_dtype_fn=fake_get_attribute_dtype,
            get_attribute_fn=fake_get_attribute,
            set_attribute_fn=None,
            with_unchecked=True,
        )


def test_rejects_call_from_nested_scope():
    """The cuBLAS wrapper must reject calls from inside a function or class body."""
    with pytest.raises(RuntimeError, match="must be called at module level"):
        make_cublas_attribute_interface(
            class_module=__name__,
            class_name="Nested",
            attribute_enum=FakeAttr,
            get_attribute_dtype_fn=fake_get_attribute_dtype,
            get_attribute_fn=fake_get_attribute,
        )


# ---------------------------------------------------------------------------
# Tests for _make_attribute_interface_impl (generic layer)
# ---------------------------------------------------------------------------


def test_impl_with_custom_closure_factories():
    """_make_attribute_interface_impl should accept arbitrary closure
    factories, decoupled from any cuBLAS-specific convention."""
    store: dict[tuple[int, int], int] = {}
    key_map = {"ALPHA": 0, "BETA": 1}

    def my_make_getter(name):
        key = key_map[name]

        def getter(self):
            return store.get((id(self), key), 0)

        return getter

    def my_make_setter(name):
        key = key_map[name]

        def setter(self, value):
            store[(id(self), key)] = value

        return setter

    Cls = _make_attribute_interface_impl(
        class_module=__name__,
        class_name="CustomImpl",
        attribute_names=["ALPHA", "BETA"],
        make_getter=my_make_getter,
        make_setter=my_make_setter,
    )

    obj = Cls(None)
    obj.alpha = 42
    assert obj.alpha == 42
    assert Cls.__name__ == "CustomImpl"


def test_impl_no_module_level_restriction():
    """_make_attribute_interface_impl should work fine inside a function,
    unlike make_cublas_attribute_interface which requires module level."""

    def trivial_getter(name):
        def getter(self):
            return 0

        return getter

    Cls = _make_attribute_interface_impl(
        class_module=__name__,
        class_name="InsideFunc",
        attribute_names=["ALPHA", "BETA"],
        make_getter=trivial_getter,
    )
    assert Cls.__name__ == "InsideFunc"
    assert hasattr(Cls, "alpha")
    assert hasattr(Cls, "beta")
