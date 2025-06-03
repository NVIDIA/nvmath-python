# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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

    MM = matmul(
        size=(m, n, k),
        precision=np.float16,
        data_type="complex",
        arrangement=("row_major", "col_major", "col_major"),
        execution="Block",
        compiler="numba",
    )

    @cuda.jit(link=MM.files)
    def f(a, b, c, alpha, beta, output):
        # all value types are the same
        smem = cuda.shared.array(shape=(0,), dtype=MM.a_value_type)
        smem_a = smem[0:]
        smem_b = smem[MM.a_size :]
        smem_c = smem[MM.a_size + MM.b_size :]
        [lda, ldb, ldc] = MM.leading_dimension

        load_to_shared(a, smem_a, MM.a_dim, lda, row_major=True)
        load_to_shared(b, smem_b, MM.b_dim, ldb)
        load_to_shared(c, smem_c, MM.c_dim, ldc)

        cuda.syncthreads()

        MM.execute(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        store_from_shared(smem_c, output, MM.c_dim, ldc)

    # Note: Numpy does not have a complex<half>
    # so those are really arrays of complex64.
    a = random_complex(MM.a_dim, np.float16)
    b = random_complex(MM.b_dim, np.float16)
    c = random_complex(MM.c_dim, np.float16)
    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    alpha = 1 + 1j
    beta = 2 + 2j

    f[1, MM.block_dim, 0, MM.get_shared_storage_size()](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = alpha * (a @ b) + beta * c
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-2


if __name__ == "__main__":
    main()
