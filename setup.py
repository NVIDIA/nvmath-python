# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from Cython.Build import cythonize
from setuptools import setup, Extension


def calculate_ext(module_: str, prefix: str = "", pre_module: str = "", source_suffix: str = "") -> Extension:
    """Create a C++ Extension object with .pyx sources for a given module.

    Args:
        module_: The name of the module in dot notation e.g. "package.subpackage.module".

        prefix: A prefix to prepend to the final module name. e.g.
        "prefixpackage.subpackage.module"

        pre_module: A submodule to insert before the module name. e.g.
        "pre_module.package.subpackage.module".

        source_suffix: A suffix to append to the source filename such as "_linux",
            "_windows". e.g. the source file would be
            package.subpackage.modulesource_suffix.pyx instead of
            package.subpackage.module.pyx

    Returns:
        A Cython Extension object configured with the provided parameters.

    """
    module = module_.split(".")
    if pre_module != "":
        module.insert(-1, pre_module)
    module[-1] = f"{prefix}{module[-1]}"
    pyx = os.path.join(*module[:-1], f"{module[-1]}{source_suffix}.pyx")
    module_ = ".".join(module)

    return Extension(
        module_,
        sources=[pyx],
        language="c++",
    )


def get_ext_modules() -> list[Extension]:
    """Return a list of instantiated C++ Extensions with .pyx sources.

    Modules names are gathered from [tool.nvmath-bindings.modules] and
    [tool.nvmath-bindings.linux_modules] in pyproject.toml from lists of full module names.
    e.g. "nvmath.bindings.cublas"

    """
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    # Access specific sections, e.g., project metadata
    pyproject_data = data.get("tool", {}).get("nvmath-bindings", {})

    # Extension modules in nvmath.bindings for the math libraries.
    modules = pyproject_data["modules"]
    if sys.platform == "linux":
        modules += pyproject_data["linux_modules"]

    ext_modules: list[Extension] = []
    for m in modules:
        ext_modules += [
            calculate_ext(m),
            calculate_ext(m, prefix="cy"),
            calculate_ext(m, pre_module="_internal", source_suffix="_linux" if sys.platform == "linux" else "_windows"),
        ]

    # Extension modules in nvmath.internal for ndbuffer (temporary home).
    nvmath_internal_modules = pyproject_data["internal_modules"]
    ext_nvmath_internal_modules = [calculate_ext(m) for m in nvmath_internal_modules]

    return ext_modules + ext_nvmath_internal_modules


nthreads = os.cpu_count()
setup(
    ext_modules=cythonize(
        get_ext_modules(),
        verbose=True,
        language_level=3,
        compiler_directives={"embedsignature": True},
        nthreads=nthreads,
    ),
)
