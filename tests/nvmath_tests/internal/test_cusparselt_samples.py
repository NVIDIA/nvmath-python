# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import sys

import pytest

# Add example_tests to sys.path to import run_sample
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "example_tests"))
from test_utils import run_sample

from nvmath_tests.helpers import requires_cusparselt

samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "internal", "examples", "_bindings", "cusparseLt")
sample_files = glob.glob(os.path.join(samples_path, "**", "*.py"), recursive=True)


@requires_cusparselt
@pytest.mark.parametrize("sample", sample_files)
class TestCusparseLtSamples:
    def test_sample(self, sample):
        run_sample(samples_path, os.path.basename(sample), {"__name__": "__main__"})
