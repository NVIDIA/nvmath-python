# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import pytest
from nvmath.bindings import mathdx

from ..test_utils import run_sample


samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "device")
sample_files = glob.glob(samples_path + "**/cu*.py", recursive=True)


@pytest.mark.parametrize("sample", sample_files)
class TestDeviceSamples:
    def test_sample(self, sample):
        if os.path.basename(sample) == "cublasdx_device_gemm_performance.py" and mathdx.get_version() < 201:
            # Skip the test if libmathdx version is less than 0.2.1 because we
            # are using global memory alignment in the sample.
            pytest.skip("Skipping test for cublasdx_device_gemm_performance.py, requires libmathdx >= 0.2.1")
        run_sample(samples_path, sample, {"__name__": "__main__"})
