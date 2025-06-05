# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Create map from package names to package interface objects.
"""

__all__ = ["PACKAGE", "AnyStream", "StreamHolder"]

from .package_ifc import Package, AnyStream, StreamHolder
from .package_ifc_cuda import CUDAPackage

PACKAGE: dict[str, type[Package]] = {"cuda": CUDAPackage}
