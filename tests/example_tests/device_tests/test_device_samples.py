# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import pytest
from nvmath_tests.device.helpers import skip_if_pipeline_unsupported

from nvmath._utils import get_nvrtc_version

from ..test_utils import DEVICE_COUNT, run_sample

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
            skip_if_pipeline_unsupported()

        if "cusolverdx" in filename:
            ctk_version = get_nvrtc_version()
            if ctk_version < (12, 6, 85):
                pytest.skip(
                    f"Skipping cuSolverDx test {filename}, requires "
                    "CTK >= 12.6 Update 3 "
                    f"(current: {ctk_version[0]}.{ctk_version[1]}.{ctk_version[2]})"
                )

        run_sample(samples_path, sample, {"__name__": "__main__"})
