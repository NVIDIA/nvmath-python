# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This set of tests verifies the lower level interfaces
"""

import numpy as np

from nvmath.linalg.advanced import Matmul
from nvmath.linalg._internal.matmul_desc_ifc import MatmulDescInterface

import pytest

try:
    import cupy
except ModuleNotFoundError:
    pytest.skip("cupy required for matmul tests", allow_module_level=True)


def test_matmul_desc_ifc():
    """
    Test MatmulDescInterface.__getattr__ (not used anywhere yet)
    """
    mm = Matmul(np.zeros((1, 1)), np.zeros((1, 1)))
    desc = MatmulDescInterface(mm.mm_desc)
    desc.epilog = 123
    assert desc.epilog == 123
