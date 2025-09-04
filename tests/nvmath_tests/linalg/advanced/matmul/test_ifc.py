# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This set of tests verifies the lower level interfaces
"""

import numpy as np

from nvmath.linalg.advanced import Matmul
from nvmath.linalg._internal.matmul_desc_ifc import MatmulDescInterface
from nvmath.linalg._internal.matrix_layout_ifc import MatrixLayoutInterface
from nvmath.linalg._internal.matmul_pref_ifc import MatmulPreferenceInterface

from nvmath.internal import typemaps


def test_matmul_desc_ifc():
    """
    Test MatmulDescInterface.__getattr__ (not used anywhere yet)
    """
    a = np.zeros((1, 1))
    with Matmul(a, a) as mm:
        desc = MatmulDescInterface(mm.mm_desc)
        desc.epilogue = 123
        assert desc.epilogue == 123


def test_matrix_layout_ifc():
    """
    Test MatrixLayoutInterface.__getattr__
    """
    a = np.zeros((1, 1), dtype=np.float32)
    with Matmul(a, a) as mm:
        mm.plan()
        layout_a_ifc = MatrixLayoutInterface(mm.a_layout_ptr)
        assert typemaps.DATA_TYPE_TO_NAME[layout_a_ifc.type] == "float32"


def test_matmul_pref_ifc():
    """
    Test MatmulPreferenceInterface.__getattr__
    """
    a = np.zeros((1, 1))
    with Matmul(a, a) as mm:
        mm.plan()
        pref_ifc = MatmulPreferenceInterface(mm.preference_ptr)
        assert pref_ifc.max_workspace_bytes == mm.memory_limit
