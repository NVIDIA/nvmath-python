# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import pytest

from ..test_utils import run_sample

samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "sparse", "ust")
sample_files = glob.glob(os.path.join(samples_path, "**", "*.py"), recursive=True)


def _have_numba():
    try:
        import numba  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


HAVE_NUMBA = _have_numba()


@pytest.mark.parametrize("sample", sample_files)
class TestUSTSamples:
    def test_sample(self, sample):
        if not HAVE_NUMBA:
            pytest.skip(f"Sample ({sample}) is skipped due to not having Numba.")

        # use_subprocess=True is required here. Some examples call
        # logging.basicConfig(level=logging.DEBUG), which attaches a handler to
        # the root logger. Subsequent examples that call basicConfig with a
        # higher level (e.g. INFO) get a no-op, because basicConfig does
        # nothing once handlers are already attached. The root logger therefore
        # stays at DEBUG for the remainder of the process, which triggers a
        # numba-cuda bug: formatting a DEBUG message in byteflow.py raises
        #   TypeError: pformat() got an unexpected keyword argument 'lazy_func'
        # Running each sample in its own subprocess isolates the logging state.
        run_sample(samples_path, sample, use_subprocess=True)
