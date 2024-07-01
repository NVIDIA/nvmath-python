# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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

    MM = matmul(size=(m, n, k), precision=np.float64, data_type='real', transpose_mode=('non_transposed', 'transposed'),
                execution='Block', compiler='numba', block_size=block_size)

    value_type          = MM.value_type
    a_size              = MM.a_size
    b_size              = MM.b_size
    c_size              = MM.c_size
    a_dim               = MM.a_dim
    b_dim               = MM.b_dim
    c_dim               = MM.c_dim
    shared_memory_size  = MM.shared_memory_size
    assert MM.block_dim == (block_size, 1, 1)
    block_dim           = (block_size, batches, 1)

    ld                  = MM.leading_dimension
    lda, ldb, ldc       = (ld.a, ld.b, ld.c)

    @cuda.jit(link=MM.files)
    def f(a, b, c, alpha, beta, output):

        bid = cuda.threadIdx.y

        smem = cuda.shared.array(shape=(0,), dtype=value_type)

        smem_a = smem[0:]
        smem_b = smem[batches*a_size:]
        smem_c = smem[batches*(a_size+b_size):]

        batch_smem_a = smem_a[bid*a_size:]
        batch_smem_b = smem_b[bid*b_size:]
        batch_smem_c = smem_c[bid*c_size:]

        load_to_shared_batched(a, smem_a, bid, a_dim, lda)
        load_to_shared_batched(b, smem_b, bid, b_dim, ldb)
        load_to_shared_batched(c, smem_c, bid, c_dim, ldc)

        cuda.syncthreads()

        MM(alpha, batch_smem_a, batch_smem_b, beta, batch_smem_c)

        cuda.syncthreads()

        store_from_shared_batched(smem_c, output, bid, c_dim, ldc)

    a = random_real((batches, *a_dim), np.float64)
    b = random_real((batches, *b_dim), np.float64)
    c = random_real((batches, *c_dim), np.float64)
    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    alpha = 1.0
    beta = 2.0

    f[1, block_dim, 0, batches * shared_memory_size](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = alpha * np.einsum('bij,bkj->bik', a, b) + beta * c
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-10

if __name__ == "__main__":
    main()