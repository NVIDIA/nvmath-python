# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re

import pytest

from nvmath import bindings
from ..test_utils import run_sample


samples_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'examples', 'device')
sample_files = glob.glob(samples_path+'**/cu*.py', recursive=True)


@pytest.mark.parametrize(
    'sample', sample_files
)
class TestDeviceSamples:

    def test_sample(self, sample):
        run_sample(samples_path, sample, {'__name__': '__main__'})
