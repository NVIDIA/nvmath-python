# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/blockdim_gemm_fp16.cu
#

import numpy as np
from numba import cuda
from nvmath.device import matmul, Dim3
from common import random_real
from common_numba import load_to_shared, store_from_shared


def main():
    m, n, k = 64, 64, 64
    precision = np.float16

    # See restrictions from https://docs.nvidia.com/cuda/cublasdx/api/operators.html#blockdim-operator

    blas_block_dims = [
        Dim3(128),
        Dim3(128),
        Dim3(128),
        Dim3(32, 4),
    ]

    kernel_block_dims = [Dim3(128), Dim3(255), Dim3(196, 2), Dim3(32, 4, 2)]

    for scenario, (blas_block_dim, kernel_block_dim) in enumerate(zip(blas_block_dims, kernel_block_dims, strict=True)):
        print(f"Scenario with BLAS dim {blas_block_dim} and kernel dim {kernel_block_dim}")

        MM = matmul(
            size=(m, n, k),
            precision=precision,
            data_type="real",
            arrangement=("row_major", "col_major", "col_major"),
            execution="Block",
            compiler="numba",
            block_dim=blas_block_dim,
        )

        @cuda.jit(link=MM.files)
        def f(a, b, c, alpha, beta, output):
            smem_a = cuda.shared.array(shape=MM.a_size, dtype=MM.a_value_type)
            smem_b = cuda.shared.array(shape=MM.b_size, dtype=MM.b_value_type)
            smem_c = cuda.shared.array(shape=MM.c_size, dtype=MM.c_value_type)
            [lda, ldb, ldc] = MM.leading_dimension

            load_to_shared(a, smem_a, MM.a_dim, lda, row_major=True)
            load_to_shared(b, smem_b, MM.b_dim, ldb)
            load_to_shared(c, smem_c, MM.c_dim, ldc)

            cuda.syncthreads()

            match scenario:
                case 0 | 1:
                    MM.execute(alpha, smem_a, smem_b, beta, smem_c)
                case 2:
                    if cuda.threadIdx.y == 0:
                        MM.execute(alpha, smem_a, smem_b, beta, smem_c)
                case 3:
                    if cuda.threadIdx.z == 0:
                        MM.execute(alpha, smem_a, smem_b, beta, smem_c)

            cuda.syncthreads()

            store_from_shared(smem_c, output, MM.c_dim, ldc)

        a = random_real(MM.a_dim, precision)
        b = random_real(MM.b_dim, precision)
        c = random_real(MM.c_dim, precision)
        o = np.zeros_like(c)

        a_d = cuda.to_device(a)
        b_d = cuda.to_device(b)
        c_d = cuda.to_device(c)
        o_d = cuda.to_device(o)

        alpha = 1.0
        beta = 2.0

        f[1, kernel_block_dim](a_d, b_d, c_d, alpha, beta, o_d)
        cuda.synchronize()

        data_test = o_d.copy_to_host()
        data_ref = alpha * (a @ b) + beta * c
        error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
        assert error < 1e-2


if __name__ == "__main__":
    main()
