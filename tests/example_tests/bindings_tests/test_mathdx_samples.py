# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import pytest
from nvmath_tests.device.helpers import skip_if_pipeline_unsupported

from nvmath._utils import get_nvrtc_version

from ..test_utils import run_sample

samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "_bindings", "mathdx")
sample_files = glob.glob(samples_path + "**/cu*.py", recursive=True)


@pytest.mark.parametrize("sample", sample_files)
class TestDeviceSamples:
    def test_sample(self, sample):
        filename = os.path.basename(sample)

        # Skip pipeline tests for CTK < 13.0
        if "pipeline" in filename:
            skip_if_pipeline_unsupported()

        # Skip cuSolverDX tests for CTK < 12.6 Update 3
        if "cusolver" in filename:
            ctk_version = get_nvrtc_version()
            if ctk_version < (12, 6, 85):
                pytest.skip(
                    f"Skipping cuSolverDX test {filename}, requires "
                    "CTK >= 12.6 Update 3 "
                    f"(current: {ctk_version[0]}.{ctk_version[1]}.{ctk_version[2]})"
                )

        run_sample(samples_path, sample, {"__name__": "__main__"})
