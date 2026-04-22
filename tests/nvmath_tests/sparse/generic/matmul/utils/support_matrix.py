# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


from nvmath.sparse.ust import NamedFormats

from ....utils.common_axes import (
    DType,
    SparseArrayType,
)

supported_dtypes = (
    DType.float16,
    DType.bfloat16,
    DType.float32,
    DType.float64,
    DType.complex32,
    DType.complex64,
    DType.complex128,
)
supported_callback_dtypes = (DType.float32, DType.float64, DType.complex64, DType.complex128)
supported_index_dtypes = (DType.int32, DType.int64)
supported_codegen_index_dtypes = (
    DType.int32,
    DType.int64,
)

supported_dispatch_formats = (
    SparseArrayType.CSR,
    SparseArrayType.CSC,
    SparseArrayType.BSR,
    SparseArrayType.BSC,
    SparseArrayType.COO,
)
supported_codegen_formats = (SparseArrayType.DIA,) + supported_dispatch_formats
supported_formats = supported_codegen_formats

# NOTE: NamedFormats that are not covered by `from_package()` tests are listed below.

BSR3_format = NamedFormats.BSR3((2, 2, 2))
direct_named_formats = (
    NamedFormats.COO,
    NamedFormats.CSR,
    NamedFormats.CSC,
    NamedFormats.DCSR,
    NamedFormats.DCSC,
    NamedFormats.CROW,
    NamedFormats.CCOL,
    NamedFormats.DIAI,
    NamedFormats.SkewDIAI,
    NamedFormats.SkewDIAJ,
    NamedFormats.BatchedCSR,
    NamedFormats.BatchedDIAINonUniform,
    NamedFormats.BatchedDIAIUniform,
    NamedFormats.DELTA(8),
    BSR3_format,
)

batched_named_formats = {
    NamedFormats.BatchedCSR,
    NamedFormats.BatchedDIAINonUniform,
    NamedFormats.BatchedDIAIUniform,
    BSR3_format,
}
