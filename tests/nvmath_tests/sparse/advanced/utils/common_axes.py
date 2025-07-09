# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Literal

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


try:
    import torch
except ImportError:
    torch = None


try:
    import cupyx.scipy.sparse as csp
except ImportError:
    csp = None


try:
    import scipy.sparse as sp
except ImportError:
    sp = None


class Framework(Enum):
    numpy = 1
    cupy = 2
    torch = 3
    cupyx = 4
    scipy = 5

    @classmethod
    def enabled(cls):
        yield cls.numpy
        if cp is not None:
            yield cls.cupy
        if torch is not None:
            yield cls.torch
        if csp is not None:
            yield cls.cupyx
        if sp is not None:
            yield cls.scipy


class DType(Enum):
    uint8 = 1  # unsupported
    int8 = 2  # unsupported
    uint16 = 3  # unsupported
    int16 = 4  # unsupported
    uint32 = 5  # unsupported
    int32 = 6  # unsupported
    uint64 = 7  # unsupported
    int64 = 8  # unsupported

    float16 = 100  # unsupported
    bfloat16 = 101  # unsupported
    float32 = 102
    float64 = 103

    complex32 = 200  # unsupported
    complex64 = 201
    complex128 = 202


def is_complex(dtype: DType):
    return dtype in [DType.complex32, DType.complex64, DType.complex128]


def real_dtype(dtype: DType):
    match dtype:
        case DType.complex32:
            return DType.float16
        case DType.complex64:
            return DType.float32
        case DType.complex128:
            return DType.float64
        case _:
            return dtype


class ExecutionSpace(Enum):
    cudss_cuda = 1
    cudss_hybrid = 2

    @property
    def nvname(self):
        return "cuda" if self == ExecutionSpace.cudss_cuda else "hybrid"


class OperandPlacement(Enum):
    host = 1
    device = 2


# TODO(ktokarski) Include HybridMemoryModeOptions
# TODO(ktokarski) Include DirectSolverOptions


class DenseRHS(Enum):
    vector = 1
    matrix = 2
    batch = 3


class RHSMatrix:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.type = DenseRHS.matrix

    def pretty_name(self):
        return f"{self.type}.{self.n}x{self.k}"


class RHSVector:
    def __init__(self, n: int):
        self.n = n
        self.k = None
        self.type = DenseRHS.vector

    def pretty_name(self):
        return f"{self.type}.{self.n}"


class RHSBatch:
    def __init__(self, n: int, k: int, batch_dims: int | tuple[int, ...]):
        self.n = n
        self.k = k
        if isinstance(batch_dims, int):
            batch_dims = (batch_dims,)
        self.batch_dims = batch_dims
        self.type = DenseRHS.batch

    def pretty_name(self):
        batch_dims = "x".join(str(dim) for dim in self.batch_dims)
        return f"{self.type}.{batch_dims}x{self.n}x{self.k}"


class SparseArrayType(Enum):
    dense = 0  # unsupported
    CSR = 1
    CSC = 2  # unsupported


class Threading(Enum):
    single_thread = 1
    multi_thread = 2


class BatchMode(Enum):
    no_batch = 1
    implicit = 2
    explicit = 3


size_of_dtype = {
    DType.uint8: 1,
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

if cp is None:
    cupy_dtype = {}
    cupy_dtype_rev = {}
else:
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
    torch_dtype_rev = {value: key for key, value in torch_dtype.items()}
else:
    torch_dtype = {}
    torch_dtype_rev = {}


if csp is not None:
    assert cp is not None
    cupyx_dtype = {
        DType.float32: cp.float32,
        DType.float64: cp.float64,
        DType.complex64: cp.complex64,
        DType.complex128: cp.complex128,
    }
    cupyx_dtype_rev = {cp.dtype(value): key for key, value in cupyx_dtype.items()}
else:
    cupyx_dtype = {}
    cupyx_dtype_rev = {}

if sp is not None:
    assert np is not None
    scipy_dtype = {
        DType.int8: np.int8,
        DType.uint8: np.uint8,
        DType.int16: np.int16,
        DType.uint16: np.uint16,
        DType.int32: np.int32,
        DType.uint32: np.uint32,
        DType.int64: np.int64,
        DType.uint64: np.uint64,
        DType.float32: np.float32,
        DType.float64: np.float64,
        DType.complex64: np.complex64,
        DType.complex128: np.complex128,
    }
    scipy_dtype_rev = {np.dtype(value): key for key, value in scipy_dtype.items()}
else:
    scipy_dtype = {}
    scipy_dtype_rev = {}


framework2dtype = {
    Framework.numpy: numpy_dtype,
    Framework.cupy: cupy_dtype,
    Framework.torch: torch_dtype,
    Framework.cupyx: cupyx_dtype,
    Framework.scipy: scipy_dtype,
}

framework2dtype_rev = {
    Framework.numpy: numpy_dtype_rev,
    Framework.cupy: cupy_dtype_rev,
    Framework.torch: torch_dtype_rev,
    Framework.cupyx: cupyx_dtype_rev,
    Framework.scipy: scipy_dtype_rev,
}

framework2operand_placement = {
    Framework.numpy: [OperandPlacement.host],
    Framework.cupy: [OperandPlacement.device],
    Framework.torch: [OperandPlacement.host, OperandPlacement.device],
    Framework.cupyx: [OperandPlacement.device],
    Framework.scipy: [OperandPlacement.host],
}

framework2tensor_framework = {
    Framework.torch: Framework.torch,
    Framework.cupyx: Framework.cupy,
    Framework.scipy: Framework.numpy,
}

framework2index_dtype = {
    Framework.torch: [DType.int8, DType.int16, DType.int32, DType.int64],
    Framework.cupyx: [DType.int32],
    Framework.scipy: [DType.int32],
}

sparse_supporting_frameworks = [Framework.cupyx, Framework.scipy, Framework.torch]


def framework_from_array(a):
    if isinstance(a, np.ndarray):
        return Framework.numpy
    elif cp is not None and isinstance(a, cp.ndarray):
        return Framework.cupy
    elif torch is not None and isinstance(a, torch.Tensor):
        return Framework.torch
    elif sp is not None and isinstance(a, sp.csr_matrix):
        return Framework.scipy
    elif csp is not None and isinstance(a, csp.csr_matrix):
        return Framework.cupyx
    raise RuntimeError(f"Uncrecognized array type {type(a)}")


def operand_placement_from_array(a):
    framework = framework_from_array(a)
    match framework:
        case Framework.numpy | Framework.scipy:
            return OperandPlacement.host
        case Framework.cupy | Framework.cupyx:
            return OperandPlacement.device
        case Framework.torch:
            if a.is_cuda:
                return OperandPlacement.device
            else:
                return OperandPlacement.host
        case _:
            raise ValueError(f"Unsupported type {type(a)}")


def get_values_dtype_from_array(a):
    framework = framework_from_array(a)
    return framework2dtype_rev[framework][a.dtype]


def device_id_from_array(a) -> int | Literal["cpu"]:
    framework = framework_from_array(a)
    match framework:
        case Framework.numpy | Framework.scipy:
            return "cpu"
        case Framework.cupy:
            return a.device.id
        case Framework.cupyx:
            return a.data.device.id
        case Framework.torch:
            idx = a.device.index
            if idx is None:
                return "cpu"
            else:
                return idx
        case _:
            raise ValueError(f"Unsupported type {type(a)}")


class Param:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __bool__(self):
        return bool(self.value)

    def pretty_name(self):
        if isinstance(self.value, Enum):
            value_str = self.value.name
        else:
            value_str = str(self.value)
        return f"{self.name}.{value_str}"
