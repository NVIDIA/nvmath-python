# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Create map from package names to package interface objects.
"""

__all__ = ['PACKAGE']

from .package_ifc_cupy import CupyPackage

PACKAGE = {'cupy': CupyPackage}
try:
    import torch
    from .package_ifc_torch import TorchPackage
    PACKAGE['torch'] = TorchPackage
except ImportError as e:
    pass

