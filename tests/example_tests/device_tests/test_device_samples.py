# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import pytest

from nvmath._utils import get_nvrtc_version
from ..test_utils import run_sample, DEVICE_COUNT, cc


samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "device")
sample_files = glob.glob(samples_path + "**/cu*.py", recursive=True)

if DEVICE_COUNT < 1:
    pytest.skip(allow_module_level=True, reason="Device examples require at least one CUDA device.")


@pytest.mark.parametrize("sample", sample_files)
class TestDeviceSamples:
    def test_sample(self, sample):
        filename = os.path.basename(sample)

        if filename == "cublasdx_fp64_emulation.py":
            # TODO: Uncomment once issue with LTO IR version resolved
            # spec = importlib.util.find_spec("cuda.cccl")
            # if spec is None:
            pytest.skip("Skipping test for cublasdx_fp64_emulation.py, requires cuda.cccl module")
        if filename == "cublasdx_gemm_fft_fp16.py":
            pytest.skip("NVBug 5218000")

        # Skip pipeline tests for CTK < 13.0 and SM 100+ & CTK < 13.1
        if "pipeline" in filename:
            sm_version = cc[0] * 10 + cc[1]
            ctk_version = get_nvrtc_version()
            if sm_version >= 100 and ctk_version < (13, 1, 0) or ctk_version < (13, 0, 0):
                pytest.skip(
                    f"Skipping pipeline test {filename}, requires CTK >= 13.1 (current: {ctk_version[0]}.{ctk_version[1]})"
                )

        run_sample(samples_path, sample, {"__name__": "__main__"})
