# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# How does the build system for cuquantum-python work?
#
# - When building a wheel ("pip wheel", "pip install .", or "python setup.py
#   bdist_wheel" (discouraged!)), we want to build against the cutensor &
#   cuquantum wheels that would be installed to site-packages, so we need
#   two things:
#     1. make them the *build-time* dependencies
#     2. set up linker flags to modify rpaths
#
# - For 1. we opt in to use PEP-517, as setup_requires is known to not work
#   automatically for users. This is the "price" we pay (by design of
#   PEP-517), as it creates a new, "isolated" environment (referred to as
#   build isolation) to which all build-time dependencies that live on PyPI
#   are installed. Another "price" (also by design) is in the non-editable
#   mode (without the "-e" flag) it always builds a wheel for installation.
#
# - For 2. the solution is to create our own bdist_wheel (called first) and
#   build_ext (called later) commands. The former would inform the latter
#   whether we are building a wheel.
#
# - There is an escape hatch for 1. which is to set "--no-build-isolation".
#   Then, users are expected to set CUQUANTUM_ROOT (or CUSTATEVEC_ROOT &
#   CUTENSORNET_ROOT) and manage all build-time dependencies themselves.
#   This, together with "-e", would not produce any wheel, which is the old
#   behavior offered by the environment variable CUQUANTUM_IGNORE_SOLVER=1
#   that we removed and no longer works.
#
# - In any case, the custom build_ext command is in use, which would compute
#   the needed compiler flags (depending on it's building a wheel or not)
#   and overwrite the incoming Extension instances.
#
# - In any case, the dependencies (on PyPI wheels) are set up by default,
#   and "--no-deps" can be passed as usual to tell pip to ignore the
#   *run-time* dependencies.
