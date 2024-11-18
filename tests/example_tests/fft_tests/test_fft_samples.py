# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re

import pytest

from nvmath import bindings
from nvmath.fft._exec_utils import _check_init_cufft, _check_init_fftw
from ..test_utils import run_sample

HAS_CUFFT = True
HAS_FFTW = True


try:
    _check_init_cufft()
except RuntimeError as e:
    if "FFT CUDA execution" in str(e):
        HAS_CUFFT = False
    else:
        raise


try:
    _check_init_fftw()
except RuntimeError as e:
    if "The FFT CPU execution is not available" in str(e):
        HAS_FFTW = False
    else:
        raise


def _has_numba():
    try:
        import numba

        return True
    except ModuleNotFoundError:
        return False


# The HAS_CUDA filtering may be extended for runs with no CUFFT at all
skip_cufft_jit_callback = (
    not HAS_CUFFT
    or not _has_numba()
    or bindings._internal.cufft._inspect_function_pointer("__cufftXtSetJITCallback") == 0
)

samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "fft")
sample_files = glob.glob(samples_path + "**/*.py", recursive=True)

# Handle MPI tests separately.
mpi_re = r".*_mpi[_]?.*\.py"
sample_files = list(filter(lambda f: not re.search(mpi_re, f), sample_files))


# Cases that raise error due to lack of proper support for 3D batched
# FFTs when run with CTKs older than 11.4U2
_allowed_to_file_3d_cases = re.compile(
    "|".join(
        [
            "example16_cupy_nd_fft_benchmark",
            "example18_5D_trunc",
            "example15_cupy_nd_fft_benchmark",
        ]
    )
)


@pytest.mark.parametrize("sample", sample_files)
class TestFFTSamples:
    def test_sample(self, sample):
        if "cpu_execution" in sample:
            if not HAS_FFTW:
                pytest.skip(f"Sample ({sample}) is skipped because no FFT CPU library was found")
        else:
            if not HAS_CUFFT:
                pytest.skip(f"Sample ({sample}) is skipped due to missing cufft or cupy")

            if skip_cufft_jit_callback and "callback" in sample:
                pytest.skip(f"Sample ({sample}) is skipped due to missing function pointer")

            if bindings.cufft.get_version() < 10502 and _allowed_to_file_3d_cases.search(sample) is not None:
                pytest.skip(f"Sample ({sample}) is skipped due to CTK version not supporting 3D batched FFTs")

        run_sample(samples_path, sample, use_subprocess=True)
