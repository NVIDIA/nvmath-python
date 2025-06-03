# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from nvmath.device import matmul
from common import random_real
from common_numba import load_to_shared_2d, store_from_shared_2d


def main():
    m, n, k = 32, 16, 64
    block_size = 256

    MM = matmul(
        size=(m, n, k),
        precision=np.float32,
        data_type="real",
        # transpose_mode is deprecated since cublasdx 0.2.0 and may be removed
        # in future versions. Please use arrangement instead.
        transpose_mode=("non_transposed", "transposed"),
        execution="Block",
        block_size=block_size,
        compiler="numba",
    )

    @cuda.jit(link=MM.files)
    def f(a, b, c, alpha, beta, output):
        # cuBLASDx requires column-major arrays but cuda.shared.array creates row-major
        # arrays (only) so we emulate a column-major array by flipping dimensions
        smem_a = cuda.shared.array(shape=MM.a_dim[::-1], dtype=MM.a_value_type)
        smem_b = cuda.shared.array(shape=MM.b_dim[::-1], dtype=MM.b_value_type)
        smem_c = cuda.shared.array(shape=MM.c_dim[::-1], dtype=MM.c_value_type)

        load_to_shared_2d(a, smem_a, MM.a_dim)
        load_to_shared_2d(b, smem_b, MM.b_dim)
        load_to_shared_2d(c, smem_c, MM.c_dim)

        cuda.syncthreads()

        # Deprecated, use MM.execute(...) instead
        MM(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        store_from_shared_2d(smem_c, output, MM.c_dim)

    a = random_real(MM.a_dim, np.float32)
    b = random_real(MM.b_dim, np.float32)
    c = random_real(MM.c_dim, np.float32)
    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    alpha = 2.0
    beta = 5.0

    f[1, MM.block_dim](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = alpha * (a @ b.T) + beta * c
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
