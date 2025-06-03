# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import pytest

from ..test_utils import run_sample


try:
    from mpi4py import MPI  # noqa: F401

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "distributed/")
sample_files = glob.glob(samples_path + "**/*.py", recursive=True)


@pytest.mark.parametrize("sample", sample_files)
class TestDeviceSamples:
    def test_sample(self, sample):
        if not HAS_MPI:
            pytest.skip(f"Sample ({sample}) is skipped because mpi4py is not installed")

        run_sample(samples_path, sample, {"__name__": "__main__"}, use_mpi=True, use_subprocess=True)
