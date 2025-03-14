# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Create map from package names to package interface objects.
"""

__all__ = ["PACKAGE"]

from .package_ifc import Package
from .package_ifc_cupy import CupyPackage

PACKAGE: dict[str, type[Package]] = {"cupy": CupyPackage}
try:
    import torch  # noqa: F401
    from .package_ifc_torch import TorchPackage

    PACKAGE["torch"] = TorchPackage
except ImportError:
    pass
