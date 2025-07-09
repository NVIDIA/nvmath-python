# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath.bindings import mathdx

import numpy

size = 32
ept = 4
bpb = 2

h = mathdx.cufftdx_create_descriptor()

mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.API, mathdx.CufftdxApi.LMEM)
mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.EXECUTION, mathdx.CommondxExecution.BLOCK)
mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.SIZE, size)
mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.TYPE, mathdx.CufftdxType.C2C)
mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.DIRECTION, mathdx.CufftdxDirection.FORWARD)
mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.PRECISION, mathdx.CommondxPrecision.F32)
mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.SM, 800)
mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.ELEMENTS_PER_THREAD, ept)
mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.FFTS_PER_BLOCK, bpb)
# TODO: Uncomment once CUFFT LTO supported
# mathdx.cufftdx_set_operator_int64(
#     h, mathdx.CufftdxOperatorType.CODE_TYPE, mathdx.CufftdxCodeType.LTOIR,
# )
mathdx.cufftdx_set_option_str(h, mathdx.CommondxOption.SYMBOL_NAME, "my_fft")


code = mathdx.commondx_create_code()
mathdx.commondx_set_code_option_int64(
    code,
    mathdx.CommondxOption.TARGET_SM,
    800,
)
mathdx.cufftdx_finalize_code(code, h)

num_ltoirs = mathdx.commondx_get_code_num_ltoirs(code)

sizes = numpy.zeros(num_ltoirs, dtype=numpy.int64)
mathdx.commondx_get_code_ltoir_sizes(code, num_ltoirs, sizes.ctypes.data)

ltos = [numpy.zeros(size, dtype=numpy.int8) for size in sizes]
lto_pointers = [lto.ctypes.data for lto in ltos]
mathdx.commondx_get_code_ltoirs(code, num_ltoirs, lto_pointers)
ltos = [lto.tobytes() for lto in ltos]

print(f"Successfully generated LTOIR for FFT of size {size}, with {ept} elements per thread and {bpb} FFTs per block")

for i, lto in enumerate(ltos):
    print(f"{sizes[i]} bytes with first 20 bytes of content:\n\t{lto[:20]}")

shared_memory_size = mathdx.cufftdx_get_trait_int64(h, mathdx.CufftdxTraitType.SHARED_MEMORY_SIZE)

block_dim = numpy.zeros(3, dtype=numpy.int64)
mathdx.cufftdx_get_trait_int64s(h, mathdx.CufftdxTraitType.BLOCK_DIM, 3, block_dim.ctypes.data)

print(f"Function requires {shared_memory_size} B of shared memory and a block_dim of {block_dim}\n")

mathdx.cufftdx_destroy_descriptor(h)
