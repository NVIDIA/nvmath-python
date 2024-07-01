# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/simple_gemm_fp32.cu
#

import numpy as np
from numba import cuda
from nvmath.device import matmul
from common import random_real
from common_numba import load_to_shared_2d, store_from_shared_2d

def main():

    m, n, k = 32, 16, 64
    block_size = 256

    MM = matmul(size=(m, n, k), precision=np.float32, data_type='real', transpose_mode=('non_transposed', 'transposed'),
                execution='Block', block_size=block_size, compiler='numba')

    a_size              = MM.a_size
    b_size              = MM.b_size
    c_size              = MM.c_size
    a_dim               = MM.a_dim
    b_dim               = MM.b_dim
    c_dim               = MM.c_dim
    block_dim           = MM.block_dim

    @cuda.jit(link=MM.files)
    def f(a, b, c, alpha, beta, output):

        # cuBLASDx requires column-major arrays but cuda.shared.array creates row-major arrays (only)
        # so we emulate a column-major array by flipping dimensions
        smem_a = cuda.shared.array(shape=(a_dim[1], a_dim[0]), dtype=np.float32)
        smem_b = cuda.shared.array(shape=(b_dim[1], b_dim[0]), dtype=np.float32)
        smem_c = cuda.shared.array(shape=(c_dim[1], c_dim[0]), dtype=np.float32)

        load_to_shared_2d(a, smem_a, a_dim)
        load_to_shared_2d(b, smem_b, b_dim)
        load_to_shared_2d(c, smem_c, c_dim)

        cuda.syncthreads()

        MM(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        store_from_shared_2d(smem_c, output, c_dim)

    a = random_real(a_dim, np.float32)
    b = random_real(b_dim, np.float32)
    c = random_real(c_dim, np.float32)
    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    alpha = 2.0
    beta = 5.0

    f[1, block_dim](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = alpha * (a @ b.T) + beta * c
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5

if __name__ == "__main__":
    main()