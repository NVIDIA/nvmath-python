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
    os.path.dirname(__file__), '..', '..', '..', 'examples', 'linalg', 'advanced', 'matmul')
sample_files = glob.glob(samples_path+'**/*.py', recursive=True)

# Handle MPI tests separately.
mpi_re = r".*_mpi[_]?.*\.py"
sample_files = list(filter(lambda f: not re.search(mpi_re, f), sample_files))

min_cublas_version = {
    "example09_epilog_bias.py": 11501,
    "example09_epilog_gelu_bias.py": 11501,
    "example10_epilog_relu_aux.py": 111103,
    "example10_epilog_drelu.py": 111103,
    "example10_epilog_dgelu.py": 111103,
    "example10_epilog_drelu.py": 111103,
    "example11_epilog_drelu_bgrad.py": 111103,
    "example12_epilog_bgrada.py": 111103,
    "example12_epilog_bgradb.py": 111103,
    "example13_epilog_stateful_reset.py":  11501,
    "example14_autotune.py": 11501,
    "example16_reuse_algorithms.py": 11501,
}

cublas_version = bindings.cublasLt.get_version()

@pytest.mark.parametrize(
    'sample', sample_files
)
class TestMatmulSamples:

    def test_sample(self, sample):
        filename = os.path.basename(sample)
        required_cublas_version = min_cublas_version.get(filename, 0)
        if cublas_version < required_cublas_version:
            pytest.skip(f"cublas version {cublas_version} lower than required ({required_cublas_version})")
        run_sample(samples_path, sample)
