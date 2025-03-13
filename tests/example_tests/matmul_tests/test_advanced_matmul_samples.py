# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re

import pytest

try:
    import cupy  # noqa: F401
except ModuleNotFoundError:
    pytest.skip("cupy required for matmul tests", allow_module_level=True)

from nvmath import bindings
from ..test_utils import run_sample


samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "linalg", "advanced", "matmul")
sample_files = glob.glob(samples_path + "**/*.py", recursive=True)

# Handle MPI tests separately.
mpi_re = r".*_mpi[_]?.*\.py"
sample_files = list(filter(lambda f: not re.search(mpi_re, f), sample_files))

min_cublas_version = {
    "example09_epilog_bias.py": 11501,
    "example09_epilog_gelu_bias.py": 11501,
    "example10_epilog_relu_aux.py": 111103,
    "example10_epilog_drelu.py": 111103,
    "example10_epilog_dgelu.py": 111103,
    "example11_epilog_drelu_bgrad.py": 111103,
    "example12_epilog_bgrada.py": 111103,
    "example12_epilog_bgradb.py": 111103,
    "example13_epilog_stateful_reset.py": 11501,
    "example14_autotune.py": 11501,
    "example16_reuse_algorithms.py": 11501,
    "example17_fp8.py": 120800,
    "example18_fp8_types.py": 120800,
    "example19_fp8_reset.py": 120800,
    "example20_fp8_inplace_scale_change.py": 120800,
    "example21_fp8_amax.py": 120800,
    "example22_fp8_delayed_scaling.py": 120800,
    "example23_fp8_epilog.py": 120800,
    "example24_fp8_epilog_aux.py": 120800,
    "example25_mxfp8.py": 120800,
    "example26_mxfp8_d_out.py": 120800,
    "example27_mxfp8_chaining.py": 120800,
    "example28_mxfp8_epilog.py": 120800,
    "example29_mxfp8_layout.py": 120800,
}

min_cc = {
    "example17_fp8.py": (8, 9),
    "example18_fp8_types.py": (8, 9),
    "example19_fp8_reset.py": (8, 9),
    "example20_fp8_inplace_scale_change.py": (8, 9),
    "example21_fp8_amax.py": (8, 9),
    "example22_fp8_delayed_scaling.py": (8, 9),
    "example23_fp8_epilog.py": (8, 9),
    "example24_fp8_epilog_aux.py": (8, 9),
    "example25_mxfp8.py": (10, 0),
    "example26_mxfp8_d_out.py": (10, 0),
    "example27_mxfp8_chaining.py": (10, 0),
    "example28_mxfp8_epilog.py": (10, 0),
    "example29_mxfp8_layout.py": (10, 0),
}

cublas_version = bindings.cublasLt.get_version()
device_properties = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
cc = (device_properties["major"], device_properties["minor"])


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
        run_sample(samples_path, sample)
