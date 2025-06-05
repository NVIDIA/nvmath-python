# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/batched_gemm_fp64.cu
#

import numpy as np
from numba import cuda
from nvmath.device import matmul
from common import random_real
from common_numba import load_to_shared_batched, store_from_shared_batched


def main():
    m, n, k = 16, 16, 16
    block_size = 64
    batches = 2

    MM = matmul(
        size=(m, n, k),
        precision=np.float64,
        data_type="real",
        arrangement=("row_major", "col_major", "col_major"),
        execution="Block",
        compiler="numba",
        block_size=block_size,
    )

    assert MM.block_dim == (block_size, 1, 1)
    block_dim = (block_size, batches, 1)
    a_size_batched = batches * MM.a_size
    b_size_batched = batches * MM.b_size
    c_size_batched = batches * MM.c_size

    @cuda.jit(link=MM.files)
    def f(a, b, c, alpha, beta, output):
        bid = cuda.threadIdx.y

        smem_a = cuda.shared.array(shape=a_size_batched, dtype=MM.a_value_type)
        smem_b = cuda.shared.array(shape=b_size_batched, dtype=MM.b_value_type)
        smem_c = cuda.shared.array(shape=c_size_batched, dtype=MM.c_value_type)
        [lda, ldb, ldc] = MM.leading_dimension

        batch_smem_a = smem_a[bid * MM.a_size :]
        batch_smem_b = smem_b[bid * MM.b_size :]
        batch_smem_c = smem_c[bid * MM.c_size :]

        load_to_shared_batched(a, smem_a, bid, MM.a_dim, lda, row_major=True)
        load_to_shared_batched(b, smem_b, bid, MM.b_dim, ldb)
        load_to_shared_batched(c, smem_c, bid, MM.c_dim, ldc)

        cuda.syncthreads()

        MM.execute(alpha, batch_smem_a, batch_smem_b, beta, batch_smem_c)

        cuda.syncthreads()

        store_from_shared_batched(smem_c, output, bid, MM.c_dim, ldc)

    a = random_real((batches, *MM.a_dim), np.float64)
    b = random_real((batches, *MM.b_dim), np.float64)
    c = random_real((batches, *MM.c_dim), np.float64)
    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    alpha = 1.0
    beta = 2.0

    f[1, block_dim, 0, batches * MM.get_shared_storage_size()](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = alpha * np.einsum("bij,bjk->bik", a, b) + beta * c
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-10


if __name__ == "__main__":
    main()
