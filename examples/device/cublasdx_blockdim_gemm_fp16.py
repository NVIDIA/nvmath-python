# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
            transpose_mode=("non_transposed", "transposed"),
            execution="Block",
            compiler="numba",
            block_dim=blas_block_dim,
        )

        value_type = MM.value_type
        a_size = MM.a_size
        b_size = MM.b_size
        a_dim = MM.a_dim
        b_dim = MM.b_dim
        c_dim = MM.c_dim
        shared_memory_size = MM.shared_memory_size
        ld = MM.leading_dimension
        lda, ldb, ldc = (ld.a, ld.b, ld.c)

        @cuda.jit(link=MM.files)
        def f(a, b, c, alpha, beta, output):
            smem = cuda.shared.array(shape=(0,), dtype=value_type)

            smem_a = smem[0:]
            smem_b = smem[a_size:]
            smem_c = smem[a_size + b_size :]

            load_to_shared(a, smem_a, a_dim, lda)
            load_to_shared(b, smem_b, b_dim, ldb)
            load_to_shared(c, smem_c, c_dim, ldc)

            cuda.syncthreads()

            match scenario:
                case 0 | 1:
                    MM(alpha, smem_a, smem_b, beta, smem_c)
                case 2:
                    if cuda.threadIdx.y == 0:
                        MM(alpha, smem_a, smem_b, beta, smem_c)
                case 3:
                    if cuda.threadIdx.z == 0:
                        MM(alpha, smem_a, smem_b, beta, smem_c)

            cuda.syncthreads()

            store_from_shared(smem_c, output, c_dim, ldc)

        a = random_real(a_dim, precision)
        b = random_real(b_dim, precision)
        c = random_real(c_dim, precision)
        o = np.zeros_like(c)

        a_d = cuda.to_device(a)
        b_d = cuda.to_device(b)
        c_d = cuda.to_device(c)
        o_d = cuda.to_device(o)

        alpha = 1.0
        beta = 2.0

        f[1, kernel_block_dim, 0, shared_memory_size](a_d, b_d, c_d, alpha, beta, o_d)
        cuda.synchronize()

        data_test = o_d.copy_to_host()
        data_ref = alpha * (a @ b.T) + beta * c
        error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
        assert error < 1e-2


if __name__ == "__main__":
    main()
