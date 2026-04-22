# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re

import pytest

from nvmath import bindings

from ..test_utils import cc, run_sample

samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "linalg", "generic", "matmul")
sample_files = glob.glob(samples_path + "**/*.py", recursive=True)

# Handle MPI tests separately.
mpi_re = r".*_mpi[_]?.*\.py"
sample_files = list(filter(lambda f: not re.search(mpi_re, f), sample_files))

min_cublas_version = {}

min_cc = {}

test_requires_nvpl = {
    "example01_numpy_cpu_execution.py": True,
    "example04_stateful_torch_cpu_execution.py": True,
}
try:
    cublas_version = bindings.cublasLt.get_version()
except:
    cublas_version = 0

try:
    from nvmath.bindings._internal.utils import FunctionNotFoundError
    from nvmath.bindings.nvpl.blas import get_version

    get_version()
    del get_version
    NVPL_AVAILABLE = True
except FunctionNotFoundError as e:
    if "function nvpl_blas_get_version is not found" not in str(e):
        raise e
    # An NVPL alternative was loaded which doesn't implement nvpl_blas_get_version
    NVPL_AVAILABLE = True
except RuntimeError as e:
    if "Failed to dlopen all of the following libraries" not in str(e):
        raise e
    # Neither NVPL or an alternative was loaded
    NVPL_AVAILABLE = False


@pytest.mark.parametrize("sample", sample_files)
class TestMatmulSamples:
    def test_sample(self, sample):
        filename = os.path.basename(sample)
        required_cublas_version = min_cublas_version.get(filename, 0)
        if cublas_version < required_cublas_version:
            pytest.skip(f"cublas version {cublas_version} lower than required ({required_cublas_version})")
        required_cc = min_cc.get(filename, (0, 0))
        if cc < required_cc:
            pytest.skip(f"compute capability {cc} lower than required {required_cc}")
        nvpl_required = test_requires_nvpl.get(filename, False)
        if nvpl_required and not NVPL_AVAILABLE:
            pytest.skip("NVPL is required, but not available.")
        run_sample(samples_path, sample)
