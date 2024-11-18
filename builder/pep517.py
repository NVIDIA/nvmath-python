# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# This module implements basic PEP 517 backend support, see e.g.
# - https://peps.python.org/pep-0517/
# - https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks
# Specifically, there are 5 APIs required to create a proper build backend, see below.
# For now it's mostly a pass-through to setuptools, except that we need to determine
# some dependencies at build time.
#
# Note that we purposely do not implement the PEP-660 API hooks so that "pip install ...
# --no-build-isolation -e ." behaves as expected (in-place build/installation without
# creating a wheel). This may require pip>21.3.0.

from setuptools import build_meta as _build_meta

import utils  # this is builder.utils (the build system has sys.path set up)


prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_wheel = _build_meta.build_wheel
build_sdist = _build_meta.build_sdist


# Note: this function returns a list of *build-time* dependencies, so it's not affected
# by "--no-deps" based on the PEP-517 design.
def get_requires_for_build_wheel(config_settings=None):
    return _build_meta.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _build_meta.get_requires_for_build_sdist(config_settings)
