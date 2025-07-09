# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import pytest

try:
    import cupy  # noqa: F401
except ModuleNotFoundError:
    pytest.skip("cupy required for sparse tests", allow_module_level=True)

from ..test_utils import run_sample

samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "sparse", "advanced", "direct_solver")
sample_files = glob.glob(samples_path + "**/*.py", recursive=True)


@pytest.mark.parametrize("sample", sample_files)
class TestMatmulSamples:
    def test_sample(self, sample):
        run_sample(samples_path, sample)
