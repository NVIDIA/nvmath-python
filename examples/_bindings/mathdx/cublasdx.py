# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath.bindings import mathdx

m = 32
n = 8
k = 16
num_threads = 32
arch = 80

h = mathdx.cublasdx_create_descriptor()

mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.FUNCTION, mathdx.CublasdxFunction.MM)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.EXECUTION, mathdx.CommondxExecution.BLOCK)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.API, mathdx.CublasdxApi.SMEM)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.PRECISION, mathdx.CommondxPrecision.F16)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.SM, arch * 10)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.TYPE, mathdx.CublasdxType.REAL)
mathdx.cublasdx_set_operator_int64s(
    h,
    mathdx.CublasdxOperatorType.BLOCK_DIM,
    3,
    [num_threads, 1, 1],
)
mathdx.cublasdx_set_operator_int64s(
    h,
    mathdx.CublasdxOperatorType.SIZE,
    3,
    [m, n, k],
)
mathdx.cublasdx_set_operator_int64s(
    h,
    mathdx.CublasdxOperatorType.TRANSPOSE_MODE,
    2,
    [
        mathdx.CublasdxTransposeMode.NON_TRANSPOSED,
        mathdx.CublasdxTransposeMode.NON_TRANSPOSED,
    ],
)
mathdx.cublasdx_set_option_str(h, mathdx.CommondxOption.SYMBOL_NAME, "my_gemm")

lto_size = mathdx.cublasdx_get_ltoir_size(h)
print(f"lto size: {lto_size}")

buffer = bytearray(lto_size)
mathdx.cublasdx_get_ltoir(h, lto_size, buffer)

print(f"Successfully generated LTOIR, {lto_size} bytes for Matmul of size {m} x {n} x {k}\n")

mathdx.cublasdx_destroy_descriptor(h)
