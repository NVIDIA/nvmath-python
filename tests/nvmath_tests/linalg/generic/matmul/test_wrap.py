# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pytest

from nvmath.internal.typemaps import cudaDataType
from nvmath.linalg.generic import ExecutionCPU, ExecutionCUDA
from nvmath.linalg.generic._configuration import wrap


from . import CUBLAS_AVAILABLE, NVPL_AVAILABLE


@pytest.mark.skipif(not NVPL_AVAILABLE, reason="NVPL BLAS required for this test.")
def test_nvpl_blas_function_not_found():
    logger = logging.getLogger()
    with pytest.raises(NotImplementedError):
        wrap.nvpl_mm_function(
            execution=ExecutionCPU(),
            dtype=cudaDataType.CUDA_R_32F,
            matrix_descr_abbreviation="xx",
            logger=logger,
            batch_type="group",
        )


@pytest.mark.skipif(not CUBLAS_AVAILABLE, reason="cuBLAS required for this test.")
def test_cublas_function_not_found():
    logger = logging.getLogger()
    with pytest.raises(NotImplementedError):
        wrap.cublas_mm_function(
            execution=ExecutionCUDA(),
            dtype=cudaDataType.CUDA_R_32F,
            matrix_descr_abbreviation="xx",
            logger=logger,
            batch_type="group",
        )
