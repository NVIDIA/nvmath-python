# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from .common_axes import Backend, Framework, DType, OptFftType

import numpy as np
import cupy as cp
try:
    import torch
    TORCH_TENSOR = torch.Tensor
except:
    torch = TORCH_TENSOR = None

size_of_dtype = {
    DType.int8: 1,
    DType.int8: 1,
    DType.uint16: 2,
    DType.int16: 2,
    DType.uint32: 4,
    DType.int32: 4,
    DType.uint64: 8,
    DType.int64: 8,
    DType.float16: 2,
    DType.bfloat16: 2,
    DType.float32: 4,
    DType.float64: 8,
    DType.complex32: 4,
    DType.complex64: 8,
    DType.complex128: 16,
}

numpy_dtype = {
    DType.uint8: np.uint8,
    DType.int8: np.int8,
    DType.uint16: np.uint16,
    DType.int16: np.int16,
    DType.uint32: np.uint32,
    DType.int32: np.int32,
    DType.uint64: np.uint64,
    DType.int64: np.int64,
    DType.float16: np.float16,
    DType.float32: np.float32,
    DType.float64: np.float64,
    DType.complex64: np.complex64,
    DType.complex128: np.complex128,
}

numpy_dtype_rev = {np.dtype(value): key for key, value in numpy_dtype.items()}

cupy_dtype = {
    DType.uint8: cp.uint8,
    DType.int8: cp.int8,
    DType.uint16: cp.uint16,
    DType.int16: cp.int16,
    DType.uint32: cp.uint32,
    DType.int32: cp.int32,
    DType.uint64: cp.uint64,
    DType.int64: cp.int64,
    DType.float16: cp.float16,
    DType.float32: cp.float32,
    DType.float64: cp.float64,
    DType.complex64: cp.complex64,
    DType.complex128: cp.complex128,
}

cupy_dtype_rev = {cp.dtype(value): key for key, value in cupy_dtype.items()}

if torch is not None:
    torch_dtype = {
        DType.uint8: torch.uint8,
        DType.int8: torch.int8,
        DType.int16: torch.int16,
        DType.int32: torch.int32,
        DType.int64: torch.int64,
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

torch_dtype_rev = {value: key for key, value in torch_dtype.items()}

r2c_dtype = {
    DType.float16: DType.complex32,
    DType.float32: DType.complex64,
    DType.float64: DType.complex128,
    DType.complex32: DType.complex32,
    DType.complex64: DType.complex64,
    DType.complex128: DType.complex128,
}

c2r_dtype = {
    DType.complex32: DType.float16,
    DType.complex64: DType.float32,
    DType.complex128: DType.float64,
}

framework_dtype = {
    Framework.numpy: numpy_dtype,
    Framework.cupy: cupy_dtype,
    Framework.torch: torch_dtype,
}

framework_dtype_rev = {
    Framework.numpy: numpy_dtype_rev,
    Framework.cupy: cupy_dtype_rev,
    Framework.torch: torch_dtype_rev,
}


def enum_val_str(value):
    return str(value).split(".")[-1]


def is_complex(dtype: DType):
    assert isinstance(dtype, DType)
    return dtype in [DType.complex32, DType.complex64, DType.complex128]


def is_half(dtype: DType):
    assert isinstance(dtype, DType)
    return dtype in [DType.float16, DType.bfloat16, DType.complex32]


def size_of(dtype: DType):
    return size_of_dtype[dtype]


def get_framework_dtype(framework: Framework, dtype: DType):
    return framework_dtype[framework][dtype]


def get_framework_from_array(array: Union[np.ndarray, cp.ndarray, TORCH_TENSOR]):
    if isinstance(array, np.ndarray):
        return Framework.numpy
    elif isinstance(array, cp.ndarray):
        return Framework.cupy
    elif isinstance(array, TORCH_TENSOR):
        return Framework.torch
    else:
        raise ValueError(f"Unknown array type {array}")


def get_dtype_from_array(array: Union[np.ndarray, cp.ndarray, TORCH_TENSOR]) -> DType:
    framework = get_framework_from_array(array)
    return framework_dtype_rev[framework][array.dtype]


def get_array_backend(array: Union[np.ndarray, cp.ndarray, TORCH_TENSOR]):
    if isinstance(array, np.ndarray):
        return Backend.cpu
    elif isinstance(array, cp.ndarray):
        return Backend.gpu
    elif isinstance(array, TORCH_TENSOR):
        return Backend.gpu if array.is_cuda else Backend.cpu
    else:
        raise ValueError(f"Unknown array type {array}")


def get_fft_dtype(in_dtype: DType):
    return r2c_dtype[in_dtype]


def get_ifft_dtype(in_dtype: DType, fft_type: OptFftType):
    assert fft_type in [OptFftType.c2c, OptFftType.c2r]
    return in_dtype if fft_type == OptFftType.c2c else c2r_dtype[in_dtype]

def is_array(array):
    if torch is None:
        return isinstance(array, (np.ndarray, cp.ndarray))
    else:
        return isinstance(array, (np.ndarray, cp.ndarray, TORCH_TENSOR))

def get_framework_module(framework: Framework):
    if framework == Framework.numpy:
        return np
    elif framework == Framework.cupy:
        return cp
    elif framework == Framework.torch:
        return torch
    else:
        raise ValueError(f"Unknown framework {framework}")
