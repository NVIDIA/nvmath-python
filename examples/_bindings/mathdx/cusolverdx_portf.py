# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath.bindings import mathdx

size = [64]
block_dim = [256, 1, 1]
arch = 800

h = mathdx.cusolverdx_create_descriptor()

mathdx.cusolverdx_set_operator_int64s(h, mathdx.CusolverdxOperatorType.SIZE, len(size), size)
mathdx.cusolverdx_set_operator_int64s(h, mathdx.CusolverdxOperatorType.BLOCK_DIM, 3, block_dim)
mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.TYPE, mathdx.CusolverdxType.REAL)
mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.API, mathdx.CusolverdxApi.SMEM)
mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.FUNCTION, mathdx.CusolverdxFunction.POTRF)
mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.EXECUTION, mathdx.CommondxExecution.BLOCK)
mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.PRECISION, mathdx.CommondxPrecision.F64)
mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.FILL_MODE, mathdx.CusolverdxFillMode.LOWER)
mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.SM, arch)
mathdx.cusolverdx_set_option_str(h, mathdx.CommondxOption.SYMBOL_NAME, "my_solver")

# Compile the device function to lto_90
code = mathdx.commondx_create_code()
mathdx.commondx_set_code_option_int64(code, mathdx.CommondxOption.TARGET_SM, arch)
mathdx.cusolverdx_finalize_code(code, h)
lto_size = mathdx.commondx_get_code_ltoir_size(code)
lto = bytearray(lto_size)
mathdx.commondx_get_code_ltoir(code, lto_size, lto)
mathdx.commondx_destroy_code(code)

print(f"Generated LTOIR ({lto_size} bytes) for POTRF solver with size: {size}")

fatbin_size = mathdx.cusolverdx_get_universal_fatbin_size(h)
fatbin = bytearray(fatbin_size)
mathdx.cusolverdx_get_universal_fatbin(h, fatbin_size, fatbin)

print(f"Successfully generated LTOIR, {lto_size} Bytes for POTRF of size {size}, with {fatbin_size} bytes of universal fatbin")

shared_memory_size = mathdx.cusolverdx_get_trait_int64(h, mathdx.CusolverdxTraitType.SHARED_MEMORY_SIZE)
print(f"Function requires {shared_memory_size} B of shared memory")
mathdx.cusolverdx_destroy_descriptor(h)
