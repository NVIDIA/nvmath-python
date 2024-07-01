# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/simple_gemm_cfp16.cu
#

import numpy as np
from numba import cuda
from nvmath.device import matmul
from common import random_complex
from common_numba import load_to_shared, store_from_shared

def main():

    m, n, k = 64, 64, 64

    MM = matmul(size=(m, n, k), precision=np.float16, data_type='complex', transpose_mode=('non_transposed', 'transposed'),
                execution='Block', compiler='numba')

    value_type          = MM.value_type
    a_size              = MM.a_size
    b_size              = MM.b_size
    c_size              = MM.c_size
    a_dim               = MM.a_dim
    b_dim               = MM.b_dim
    c_dim               = MM.c_dim
    block_dim           = MM.block_dim
    block_size          = block_dim[0]
    ld                  = MM.leading_dimension
    lda, ldb, ldc       = ld.a, ld.b, ld.c
    shared_memory_size  = MM.shared_memory_size

    @cuda.jit(link=MM.files)
    def f(a, b, c, alpha, beta, output):

        smem = cuda.shared.array(shape=(0,), dtype=value_type)
        smem_a = smem[0:]
        smem_b = smem[a_size:]
        smem_c = smem[a_size+b_size:]

        load_to_shared(a, smem_a, a_dim, lda)
        load_to_shared(b, smem_b, b_dim, ldb)
        load_to_shared(c, smem_c, c_dim, ldc)

        cuda.syncthreads()

        MM(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        store_from_shared(smem_c, output, c_dim, ldc)

    # Note: Numpy does not have a complex<half>
    # so those are really arrays of complex64.
    a = random_complex(a_dim, np.float16)
    b = random_complex(b_dim, np.float16)
    c = random_complex(c_dim, np.float16)
    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    alpha = 1+1j
    beta = 2+2j

    f[1, block_dim, 0, shared_memory_size](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = alpha * (a @ b.T) + beta * c
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-2

if __name__ == "__main__":
    main()