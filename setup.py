# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import glob
import os
import shutil
import sys
import tempfile

from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

# this is tricky: sys.path gets overwritten at different stages of the build
# flow, so we need to hack sys.path ourselves...
source_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(source_root, 'builder'))
import utils  # this is builder.utils


# List the main modules, and infer the auxiliary modules automatically
ext_modules = [
    "nvmath.bindings.cublas",
    "nvmath.bindings.cublasLt",
    "nvmath.bindings.cusolver",
    "nvmath.bindings.cusolverDn",
    "nvmath.bindings.cufft",
    "nvmath.bindings.cusparse",
    "nvmath.bindings.curand",
]


# WAR: Check if this is still valid
# TODO: can this support cross-compilation?
if sys.platform == 'linux':
    src_files = glob.glob('**/**/_internal/*_linux.pyx')
elif sys.platform == 'win32':
    src_files = glob.glob('**/**/_internal/*_windows.pyx')
else:
    raise RuntimeError(f'platform is unrecognized: {sys.platform}')
dst_files = []
for src in src_files:
    # Set up a temporary file; it must be under the cache directory so
    # that atomic moves within the same filesystem can be guaranteed
    with tempfile.NamedTemporaryFile(delete=False, dir='.') as f:
        shutil.copy2(src, f.name)
        f_name = f.name
    dst = src.replace('_linux', '').replace('_windows', '')
    # atomic move with the destination guaranteed to be overwritten
    os.replace(f_name, f"./{dst}")
    dst_files.append(dst)


@atexit.register
def cleanup_dst_files():
    for dst in dst_files:
        try:
            os.remove(dst)
        except FileNotFoundError:
            pass


def calculate_modules(module):
    module = module.split('.')

    lowpp_mod = module.copy()
    lowpp_mod_pyx = os.path.join(*module[:-1], f"{module[-1]}.pyx")
    lowpp_mod = '.'.join(lowpp_mod)
    lowpp_ext = Extension(
        lowpp_mod,
        sources=[lowpp_mod_pyx],
        language="c++",
    )

    cy_mod = module.copy()
    cy_mod[-1] = f"cy{cy_mod[-1]}"
    cy_mod_pyx = os.path.join(*cy_mod[:-1], f"{cy_mod[-1]}.pyx")
    cy_mod = '.'.join(cy_mod)
    cy_ext = Extension(
        cy_mod,
        sources=[cy_mod_pyx],
        language="c++",
    )

    inter_mod = module.copy()
    inter_mod.insert(-1, '_internal')
    inter_mod_pyx = os.path.join(*inter_mod[:-1], f"{inter_mod[-1]}.pyx")
    inter_mod = '.'.join(inter_mod)
    inter_ext = Extension(
        inter_mod,
        sources=[inter_mod_pyx],
        language="c++",
    )

    return lowpp_ext, cy_ext, inter_ext


# Note: the extension attributes are overwritten in build_extension()
ext_modules = [
    e for ext in ext_modules for e in calculate_modules(ext)
] + [
    Extension(
        "nvmath.bindings._internal.utils",
        sources=["nvmath/bindings/_internal/utils.pyx"],
        language="c++",
    ),
]


cmdclass = {
    'build_ext': utils.build_ext,
    'bdist_wheel': utils.bdist_wheel,
}


setup(
    ext_modules=cythonize(ext_modules,
        verbose=True, language_level=3,
        compiler_directives={'embedsignature': True}),
    packages=find_packages(include=['nvmath', 'nvmath.*']),
    package_data=dict.fromkeys(
        find_packages(include=["nvmath.*"]),
        ["*.pxd", "*.pyx", "*.py"],
    ),
    zip_safe=False,
    cmdclass=cmdclass,
)
