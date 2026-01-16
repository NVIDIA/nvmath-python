# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

try:
    import cupy as cp

    CP_NDARRAY = cp.ndarray
except ImportError:
    cp = CP_NDARRAY = None
try:
    import torch

    TORCH_TENSOR = torch.Tensor
except ImportError:
    torch = TORCH_TENSOR = None

from .common_axes import Framework, DType


numpy_dtype = {
    DType.float16: np.float16,
    DType.float32: np.float32,
    DType.float64: np.float64,
    DType.complex64: np.complex64,
    DType.complex128: np.complex128,
}

if cp is None:
    cupy_dtype = {}
else:
    cupy_dtype = {
        DType.float16: cp.float16,
        DType.float32: cp.float32,
        DType.float64: cp.float64,
        DType.complex64: cp.complex64,
        DType.complex128: cp.complex128,
    }

if torch is not None:
    torch_dtype = {
        DType.float16: torch.float16,
        DType.bfloat16: torch.bfloat16,
        DType.float32: torch.float32,
        DType.float64: torch.float64,
        DType.complex32: torch.complex32,
        DType.complex64: torch.complex64,
        DType.complex128: torch.complex128,
    }
else:
    torch_dtype = {}

framework_dtype = {
    Framework.numpy: numpy_dtype,
    Framework.cupy: cupy_dtype,
    Framework.torch: torch_dtype,
}


def is_complex(dtype: DType):
    assert isinstance(dtype, DType)
    return dtype in [DType.complex32, DType.complex64, DType.complex128]


def get_framework_dtype(framework: Framework, dtype: DType):
    return framework_dtype[framework][dtype]


def get_framework_module(framework: Framework):
    if framework == Framework.numpy:
        return np
    elif framework == Framework.cupy:
        return cp
    elif framework == Framework.torch:
        return torch
    else:
        raise ValueError(f"Unknown framework {framework}")
