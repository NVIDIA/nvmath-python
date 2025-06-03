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
mathdx.cufftdx_set_option_str(h, mathdx.CommondxOption.SYMBOL_NAME, "my_fft")

lto_size = mathdx.cufftdx_get_ltoir_size(h)
print(f"lto size: {lto_size}")

buffer = bytearray(lto_size)
mathdx.cufftdx_get_ltoir(h, lto_size, buffer)

print(
    f"Successfully generated LTOIR, {lto_size} bytes for FFT of size {size}, "
    f"with {ept} elements per thread and {bpb} FFTs per block\n"
)

shared_memory_size = mathdx.cufftdx_get_trait_int64(h, mathdx.CufftdxTraitType.SHARED_MEMORY_SIZE)

block_dim = numpy.zeros(3, dtype=numpy.int64)
mathdx.cufftdx_get_trait_int64s(h, mathdx.CufftdxTraitType.BLOCK_DIM, 3, block_dim.ctypes.data)

print(f"Function requires {shared_memory_size} B of shared memory and a block_dim of {block_dim}\n")

mathdx.cufftdx_destroy_descriptor(h)
