# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import pytest


from nvmath.bindings import cutensor
from nvmath.bindings._internal.utils import NotSupportedError, FunctionNotFoundError

from ..test_utils import run_sample, cc


try:
    cutensor.get_version()
    HAS_CUTENSOR = True
except (NotSupportedError, FunctionNotFoundError, RuntimeError):
    HAS_CUTENSOR = False

samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "tensor", "contraction")
sample_files = glob.glob(samples_path + "**/*.py", recursive=True)


@pytest.mark.skipif(not HAS_CUTENSOR, reason="cuTensor is not available")
@pytest.mark.skipif(cc <= (7, 0), reason="cuTensor 2.3.1+ requires compute capability > 7.0")
@pytest.mark.parametrize("sample", sample_files)
class TestContractionSamples:
    def test_sample(self, sample):
        run_sample(samples_path, sample)
