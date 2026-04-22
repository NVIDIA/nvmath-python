# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from packaging.version import Version

try:
    import torch
except ModuleNotFoundError:

    class torch:
        __version__ = "0.0.0"


from .common_axes import (
    ComputeType,
    DType,
    Framework,
    MemBackend,
)

framework_backend_support = {
    Framework.cupy: [MemBackend.cuda],
    Framework.numpy: [MemBackend.cpu],
    Framework.torch: [MemBackend.cpu, MemBackend.cuda],
}

framework_type_support = {
    Framework.cupy: [
        DType.float16,
        DType.float32,
        DType.float64,
        DType.complex64,
        DType.complex128,
    ],
    Framework.numpy: [
        DType.float16,
        DType.float32,
        DType.float64,
        DType.complex64,
        DType.complex128,
    ],
    Framework.torch: [
        *((DType.float16,) if Version(torch.__version__) >= Version("2.2.0") else ()),
        DType.float32,
        DType.float64,
        DType.complex64,
        DType.complex128,
    ],
}

compute_type_support = {
    DType.float16: [
        ComputeType.float32,
    ],
    DType.bfloat16: [
        ComputeType.float32,
    ],
    DType.float32: [
        ComputeType.float32,
        ComputeType.tf32,
        ComputeType.three_xtf32,
        ComputeType.float16,
        ComputeType.bfloat16,
        ComputeType.nine_x16bf,
        ComputeType.four_x16f,
    ],
    DType.float64: [
        ComputeType.float64,
        ComputeType.float32,
        ComputeType.eight_xint8,
    ],
    DType.complex64: [
        ComputeType.float32,
        ComputeType.tf32,
        ComputeType.three_xtf32,
        ComputeType.float16,
        ComputeType.bfloat16,
        ComputeType.nine_x16bf,
        ComputeType.four_x16f,
    ],
    DType.complex128: [
        ComputeType.float64,
        ComputeType.float32,
        ComputeType.eight_xint8,
    ],
}
