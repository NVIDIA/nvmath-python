# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nvmath import bindings
import cupy

try:
    import torch
except ModuleNotFoundError:
    torch = None

notebook_path = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks")
notebook_files = glob.glob(notebook_path + "/**/*.ipynb", recursive=True)

torch_notebooks = ("matmul/03_backpropagation.ipynb",)

min_cublas_version = {
    "matmul/04_fp8.ipynb": 120800,
}

min_cc = {"matmul/04_fp8.ipynb": (10, 0)}

cublas_version = bindings.cublasLt.get_version()
device_properties = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
cc = (device_properties["major"], device_properties["minor"])


@pytest.mark.parametrize("notebook", notebook_files, ids=[os.path.relpath(f, start=notebook_path) for f in notebook_files])
def test_notebook(notebook):
    notebook_name = os.path.relpath(notebook, notebook_path)
    if torch is None and notebook_name in torch_notebooks:
        pytest.skip("PyTorch not present")
    required_cublas_version = min_cublas_version.get(notebook_name, 0)
    if cublas_version < required_cublas_version:
        pytest.skip(f"cublas version {cublas_version} lower than required ({required_cublas_version})")
    required_cc = min_cc.get(notebook_name, (0, 0))
    if cc < required_cc:
        pytest.skip(f"compute capability {cc} lower than required {required_cc}")
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": notebook_path}})
