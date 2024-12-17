# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/simple_gemm_leading_dimensions.cu
#

import numpy as np
from numba import cuda
from nvmath.device import matmul, Dim3
from common import random_real
from common_numba import load_to_shared, store_from_shared


def main():
    m, n, k = 30, 31, 33

    lda, ldb, ldc = 32, 33, 31

    block_dim = Dim3(16, 16, 1)

    kwargs = {
        "size": (m, n, k),
        "precision": np.float64,
        "data_type": "real",
        "transpose_mode": ("non_transposed", "transposed"),
        "execution": "Block",
        "block_dim": block_dim,
        "compiler": "numba",
    }

    MM_static_ld = matmul(**kwargs, leading_dimension=(lda, ldb, ldc))
    MM_runtime_ld = matmul(**kwargs)

    value_type = MM_static_ld.value_type
    a_size = MM_static_ld.a_size
    b_size = MM_static_ld.b_size
    a_dim = MM_static_ld.a_dim
    b_dim = MM_static_ld.b_dim
    c_dim = MM_static_ld.c_dim
    shared_memory_size = MM_static_ld.shared_memory_size

    @cuda.jit(link=MM_static_ld.files)
    def f_static_ld(alpha, a, b, beta, c, output):
        smem = cuda.shared.array(shape=(0,), dtype=value_type)
        smem_a = smem[0:]
        smem_b = smem[a_size:]
        smem_c = smem[a_size + b_size :]

        load_to_shared(a, smem_a, a_dim, lda)
        load_to_shared(b, smem_b, b_dim, ldb)
        load_to_shared(c, smem_c, c_dim, ldc)

        cuda.syncthreads()

        MM_static_ld(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        store_from_shared(smem_c, output, c_dim, ldc)

    @cuda.jit(link=MM_runtime_ld.files)
    def f_runtime_ld(alpha, a, lda, b, ldb, beta, c, ldc, output):
        smem = cuda.shared.array(shape=(0,), dtype=value_type)
        smem_a = smem[0:]
        smem_b = smem[a_size:]
        smem_c = smem[a_size + b_size :]

        load_to_shared(a, smem_a, a_dim, lda)
        load_to_shared(b, smem_b, b_dim, ldb)
        load_to_shared(c, smem_c, c_dim, ldc)

        cuda.syncthreads()

        MM_runtime_ld(alpha, smem_a, lda, smem_b, ldb, beta, smem_c, ldc)

        cuda.syncthreads()

        store_from_shared(smem_c, output, c_dim, ldc)

    a = random_real(a_dim, np.float64)
    b = random_real(b_dim, np.float64)
    c = random_real(c_dim, np.float64)
    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_static_ld_d = cuda.to_device(o)
    o_runtime_ld_d = cuda.to_device(o)

    alpha = 1.0
    beta = 2.0

    f_static_ld[1, block_dim, 0, shared_memory_size](alpha, a_d, b_d, beta, c_d, o_static_ld_d)
    cuda.synchronize()

    f_runtime_ld[1, block_dim, 0, shared_memory_size](alpha, a_d, lda, b_d, ldb, beta, c_d, ldc, o_runtime_ld_d)
    cuda.synchronize()

    data_ref = alpha * (a @ b.T) + beta * c
    for o_d in [o_static_ld_d, o_runtime_ld_d]:
        data_test = o_d.copy_to_host()
        error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
        print(error)
        assert error < 1e-5


if __name__ == "__main__":
    main()
