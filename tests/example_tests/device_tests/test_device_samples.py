# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import pytest

from ..test_utils import run_sample


samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "device")
sample_files = glob.glob(samples_path + "**/cu*.py", recursive=True)


@pytest.mark.parametrize("sample", sample_files)
class TestDeviceSamples:
    def test_sample(self, sample):
        if os.path.basename(sample) == "cublasdx_fp64_emulation.py":
            # TODO: Uncomment once issue with LTO IR version resolved
            # spec = importlib.util.find_spec("cuda.cccl")
            # if spec is None:
            pytest.skip("Skipping test for cublasdx_fp64_emulation.py, requires cuda.cccl module")
        if os.path.basename(sample) == "cublasdx_gemm_fft_fp16.py":
            pytest.skip("NVBug 5218000")
        run_sample(samples_path, sample, {"__name__": "__main__"})
