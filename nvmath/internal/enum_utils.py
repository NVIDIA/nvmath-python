# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Factories for create options dataclasses, as well as utilities to add docstring to enum
classes.
"""

import dataclasses
from enum import IntEnum
import re
from typing import Any, ClassVar
from collections.abc import Callable

import numpy


def create_options_class_from_enum(
    options_class_name: str,
    enum_class: IntEnum,
    get_attr_dtype: Callable,
    description: str,
    filter_re: str = r"(?P<option_name>.*)",
):
    """
    Create an options dataclass from a Python enum class. Names can be filtered if desired.

    Args:
       options_class_name: Name of the dataclass that will be created.

       enum_class: The IntEnum class that contains the options for the dataclass.

       get_attr_dtype: A callable that takes in an enum value as the argument and returns
            the size in bytes of the cuTensorNet.

       filter_re: A re definition that defines the match named 'option_name'.
    """
    if r"(?P<option_name>" not in filter_re:
        message = """Incorrect re.
The re for the filter must contain the named group 'option_name'."""
        raise ValueError(message)

    # Helper vars for creating attribute docstring.
    doc = f"""A data class for capturing the {description} options.

    Attributes:
"""
    indent = " " * 8
    prefix = determine_enum_prefix(enum_class, "_ATTRIBUTE")

    filter_pattern = re.compile(filter_re)
    option_to_enum: dict[str, IntEnum] = dict()
    option_to_dtype: dict[str, Any] = dict()
    # mypy claims that IntEnum is not iterable
    for e in enum_class:  # type: ignore
        m = filter_pattern.match(e.name)
        if not m:
            continue
        option_name = m.group("option_name").lower()
        option_to_enum[option_name] = e
        option_to_dtype[option_name] = get_attr_dtype(e)

        # Add docstring for this attribute.
        doc += indent + option_name + ":" + f" See `{prefix + '_' + m.group(0)}`.\n"

    field: tuple[str, Any, None | dict]
    fields: list[tuple[str, Any, None | dict]] = list()
    for option_name, dtype in option_to_dtype.items():
        if numpy.issubdtype(dtype, numpy.integer):
            field = option_name, int | None, dataclasses.field(default=None)
        else:
            field = option_name, Any | None, dataclasses.field(default=None)
        fields.append(field)

    # Add class attributes.

    field = "option_to_enum", ClassVar[dict], dataclasses.field(default=option_to_enum)
    fields.append(field)

    field = "option_to_dtype", ClassVar[dict], dataclasses.field(default=option_to_dtype)
    fields.append(field)

    # Create the options class.
    options_class = dataclasses.make_dataclass(options_class_name, fields)
    options_class.__doc__ = doc

    return options_class


def snake_to_camel(names):
    name = ""
    for i, sub_name in enumerate(names):
        if i == 0:
            name += sub_name.lower()
        else:
            name += sub_name[0].upper() + sub_name[1:]
    name += "_t"
    return name


def camel_to_snake(name, upper=True):
    """
    Convert string from camel case to snake style.
    """

    def transformer(m):
        prefix = iter(("", "_", ""))
        for i in 1, 3, 5:
            first, second = i, i + 1
            s = next(prefix)
            if m.group(second):
                if m.group(first):  # If the second group doesn't exist, the first won't either by the design of the RE.
                    s += m.group(first).lower() + "_"
                s += m.group(second).lower()
                break
        return s

    name = re.sub(r"^([A-Z]*)([A-Z])|(?<!_)([A-Z]*)([A-Z])|([A-Z]*)([A-Z])", transformer, name)
    if upper:
        name = name.upper()
    return name


def determine_enum_prefix(enum_class, chomp):
    """
    This function assumes that the convention used to translate C enumerators to Python enum
    names holds.
    """

    prefix = enum_class.__module__.split(".")[-1].upper()
    prefix += "_" + camel_to_snake(enum_class.__name__)
    prefix = re.sub(chomp, "", prefix)
    return prefix


def add_enum_class_doc(enum_class, chomp):
    """
    Add docstring to enum classes.
    """
    for e in enum_class:
        e.__doc__ = f"See `{determine_enum_prefix(enum_class, chomp) + '_' + e.name.upper()}`."
