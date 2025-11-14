# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
import sys
import pytest
import cuda.core.experimental as ccx

if sys.platform == "win32":
    pytest.skip("Skipping tensor contraction tests because they are not supported on Windows.", allow_module_level=True)


# starting cutensor 2.3.0, support only compute capability > 7.0
def pytest_collection_modifyitems(config, items):
    """Skip all tests in this directory if compute capability <= 7.0"""
    if ccx.Device().compute_capability <= (7, 0):
        skip_marker = pytest.mark.skip(reason="cuTensor 2.3.1+ requires compute capability > 7.0")
        for item in items:
            item.add_marker(skip_marker)
