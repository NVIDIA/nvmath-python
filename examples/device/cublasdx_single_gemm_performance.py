# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/single_gemm_performance.cu
#

import numpy as np
from numba import cuda
from nvmath.device import matmul
from common import random_complex, mm_perf_GFlops, fp16x2_to_complex64, complex64_to_fp16x2
from common_numba import time_numba, load_to_shared_1d_float16x2, store_from_shared_1d_float16x2


def main():
    m, n, k = 32, 32, 64
    repeat = 4000
    block_size = 256
    alpha, beta = 1 + 0j, 0 + 0j
    ncycles = 1
    data_type = "complex"
    precision = np.float16

    MM = matmul(
        size=(m, n, k),
        precision=precision,
        data_type=data_type,
        transpose_mode=("non_transposed", "transposed"),
        execution="Block",
        block_size=block_size,
        compiler="numba",
        leading_dimension="suggested",
    )

    a_size = MM.a_size
    b_size = MM.b_size
    c_size = MM.c_size
    a_dim = MM.a_dim
    b_dim = MM.b_dim
    c_dim = MM.c_dim
    ld = MM.leading_dimension
    lda, ldb, ldc = ld.a, ld.b, ld.c
    block_dim = MM.block_dim
    grid_dim = 1
    value_type = MM.value_type

    @cuda.jit(link=MM.files)
    def f(a, b, c, alpha, beta, output, repeat):
        smem_a = cuda.shared.array(shape=(a_size,), dtype=value_type)
        smem_b = cuda.shared.array(shape=(b_size,), dtype=value_type)
        smem_c = cuda.shared.array(shape=(c_size,), dtype=value_type)

        load_to_shared_1d_float16x2(a, smem_a, a_dim, lda)
        load_to_shared_1d_float16x2(b, smem_b, b_dim, ldb)
        load_to_shared_1d_float16x2(c, smem_c, c_dim, ldc)

        cuda.syncthreads()

        for r in range(repeat):
            MM(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        store_from_shared_1d_float16x2(smem_c, output, c_dim, ldc)

    a = random_complex(a_dim, np.float32)
    b = random_complex(b_dim, np.float32)
    c = random_complex(c_dim, np.float32)
    o = np.zeros_like(c)

    a_d = cuda.to_device(complex64_to_fp16x2(a))
    b_d = cuda.to_device(complex64_to_fp16x2(b))
    c_d = cuda.to_device(complex64_to_fp16x2(c))
    o_d = cuda.to_device(complex64_to_fp16x2(o))

    time_ms = time_numba(f, grid_dim, block_dim, 0, ncycles, a_d, b_d, c_d, alpha, beta, o_d, repeat)
    time_2x_ms = time_numba(f, grid_dim, block_dim, 0, ncycles, a_d, b_d, c_d, alpha, beta, o_d, 2 * repeat)
    time_mm_ms = (time_2x_ms - time_ms) / repeat
    perf = mm_perf_GFlops((m, n, k), 1, time_mm_ms)

    print(f"Time {time_mm_ms} ms\nPerf {perf} GFlop/s")

    print(f"m, n, k: {m}, {n}, {k}")
    print(f"Data type: {data_type}")
    print(f"Precision: {precision}")
    print(f"Block size: {block_size}")
    print(f"Leading dimensions: {lda}, {ldb}, {ldc}")
    print(f"Shared memory: {a_size + b_size + c_size} elements")
    print(f"Avg time [ms]: {time_mm_ms}")
    print(f"Time (all) [ms]: {time_mm_ms * repeat}")
    print(f"Performance [GFLOPS]: {perf}")

    data_test = fp16x2_to_complex64(o_d.copy_to_host())
    data_ref = alpha * (a @ b.T) + beta * c
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-2


if __name__ == "__main__":
    main()
