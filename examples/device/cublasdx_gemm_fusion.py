# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/gemm_fusion.cu
#

import numpy as np
from numba import cuda
from nvmath.device import matmul
from common import random_real
from common_numba import set_max_dynamic_shared_size_bytes, load_to_shared, store_from_shared


def main():
    m1 = 64
    n1 = 64
    k1 = 64
    m2 = m1
    n2 = 128
    k2 = n1

    block_size = 128

    MM1 = matmul(
        size=(m1, n1, k1),
        precision=np.float16,
        data_type="real",
        transpose_mode=("non_transposed", "non_transposed"),
        execution="Block",
        block_size=block_size,
        compiler="numba",
    )

    MM2 = matmul(
        size=(m2, n2, k2),
        precision=np.float16,
        data_type="real",
        transpose_mode=("non_transposed", "non_transposed"),
        execution="Block",
        block_size=block_size,
        compiler="numba",
    )

    a_size = MM1.a_size
    c_size = MM1.c_size

    d_size = MM2.b_size

    a_dim = MM1.a_dim
    b_dim = MM1.b_dim
    c_dim = MM1.c_dim

    d_dim = MM2.b_dim
    f_dim = MM2.c_dim

    lda = MM1.leading_dimension.a
    ldb = MM1.leading_dimension.b
    ldc = MM1.leading_dimension.c

    ldd = MM1.leading_dimension.b
    ldf = MM1.leading_dimension.c

    block_dim = MM1.block_dim
    shared_memory_size = max(MM1.shared_memory_size, MM2.shared_memory_size)

    assert MM2.a_dim == MM1.c_dim
    assert MM2.block_dim == MM1.block_dim
    assert MM1.c_size == MM2.a_size

    @cuda.jit(link=MM1.files + MM2.files)
    def kernel(alpha1, a, b, beta1, c, alpha2, d, beta2, f, output):
        smem = cuda.shared.array(shape=(0,), dtype=np.float16)

        # MM1 takes (a, b, c) --> c
        # smem = [ c c c c c c c | a a a a a a a | b b b b b b ]
        smem_c = smem[0:]
        smem_a = smem[c_size:]
        smem_b = smem[c_size + a_size :]

        # MM2 takes (c, d, f) --> f
        # smem = [ c c c c c c c | d d d d d | f f f f f f f f f f f]
        smem_d = smem[c_size:]  # MM2.a_size
        smem_f = smem[c_size + d_size :]

        # Load MM1's A (a)
        load_to_shared(a, smem_a, a_dim, lda)

        # Load MM1's B (b)
        load_to_shared(b, smem_b, b_dim, ldb)

        # Load MM1's C (a)
        load_to_shared(c, smem_c, c_dim, ldc)

        cuda.syncthreads()

        MM1(alpha1, smem_a, smem_b, beta1, smem_c)

        cuda.syncthreads()

        # Load MM2's B (d)
        load_to_shared(d, smem_d, d_dim, ldd)

        # Load MM2's C (f)
        load_to_shared(f, smem_f, f_dim, ldf)

        cuda.syncthreads()

        MM2(alpha2, smem_c, smem_d, beta2, smem_f)

        cuda.syncthreads()

        # Store MM2's C (f)
        store_from_shared(smem_f, output, f_dim, ldf)

    a = random_real(a_dim, np.float16)
    b = random_real(b_dim, np.float16)
    c = random_real(c_dim, np.float16)
    d = random_real(d_dim, np.float16)
    f = random_real(f_dim, np.float16)
    o = np.zeros_like(f)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    d_d = cuda.to_device(d)
    f_d = cuda.to_device(f)
    o_d = cuda.to_device(o)

    alpha1 = 1.0
    beta1 = 0.0
    alpha2 = 1.0
    beta2 = 1.0

    set_max_dynamic_shared_size_bytes(kernel, shared_memory_size, alpha1, a_d, b_d, beta1, c_d, alpha2, d_d, beta2, f_d, o_d)

    kernel[1, block_dim, 0, shared_memory_size](alpha1, a_d, b_d, beta1, c_d, alpha2, d_d, beta2, f_d, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = alpha2 * ((alpha1 * (a @ b) + beta1 * c) @ d) + beta2 * f
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
