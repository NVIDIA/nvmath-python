# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/fused_gemm_performance.cu
#

import numpy as np
from numba import cuda
from nvmath.device import matmul
from common import random_complex
from common_numba import set_max_dynamic_shared_size_bytes, load_to_shared, store_from_shared, time_numba


def main():
    m1 = 32
    n1 = 32
    k1 = 32
    m2 = m1
    n2 = 32
    k2 = n1

    block_size = 512
    precision = np.float32

    kwargs = {
        "precision": precision,
        "data_type": "complex",
        "arrangement": ("col_major", "col_major", "col_major"),
        "execution": "Block",
        "block_size": block_size,
        "compiler": "numba",
    }

    MM1 = matmul(size=(m1, n1, k1), **kwargs)

    MM2 = matmul(size=(m2, n2, k2), **kwargs)

    value_type = MM1.a_value_type  # all value types are the same

    a_size = MM1.a_size
    c_size = MM1.c_size

    d_size = MM2.b_size

    a_dim = MM1.a_dim
    b_dim = MM1.b_dim
    c_dim = MM1.c_dim

    d_dim = MM2.b_dim
    f_dim = MM2.c_dim

    block_dim = MM1.block_dim
    shared_memory_size = max(
        MM1.get_shared_storage_size(),
        MM2.get_shared_storage_size(),
    )

    assert MM2.a_dim == MM1.c_dim
    assert MM2.block_dim == MM1.block_dim
    assert MM1.c_size == MM2.a_size
    assert MM1.leading_dimension.c == MM2.leading_dimension.a

    @cuda.jit(link=MM1.files)
    def kernel(alpha1, a, b, beta1, c, alpha2, d, beta2, f, output):
        smem = cuda.shared.array(shape=(0,), dtype=value_type)

        # MM1 takes (a, b, c) --> c
        # smem = [ c c c c c c c | a a a a a a a | b b b b b b ]
        smem_c = smem[0:]
        smem_a = smem[c_size:]
        smem_b = smem[c_size + a_size :]

        # MM2 takes (c, d, f) --> f
        # smem = [ c c c c c c c | d d d d d | f f f f f f f f f f f]
        smem_d = smem[c_size:]
        smem_f = smem[c_size + d_size :]

        [lda, ldb, ldc] = MM1.leading_dimension
        [ldc, ldd, ldf] = MM2.leading_dimension

        # Load MM1's A (a)
        load_to_shared(a, smem_a, a_dim, lda)

        # Load MM1's B (b)
        load_to_shared(b, smem_b, b_dim, ldb)

        # Load MM1's C (a)
        load_to_shared(c, smem_c, c_dim, ldc)

        cuda.syncthreads()

        MM1.execute(alpha1, smem_a, smem_b, beta1, smem_c)

        cuda.syncthreads()

        # Load MM2's B (d)
        load_to_shared(d, smem_d, d_dim, ldd)

        # Load MM2's C (f)
        load_to_shared(f, smem_f, f_dim, ldf)

        cuda.syncthreads()

        MM2.execute(alpha2, smem_c, smem_d, beta2, smem_f)

        cuda.syncthreads()

        # Store MM2's C (f)
        store_from_shared(smem_f, output, f_dim, ldf)

    a = random_complex(a_dim, np.float32)
    b = random_complex(b_dim, np.float32)
    c = random_complex(c_dim, np.float32)
    d = random_complex(d_dim, np.float32)
    f = random_complex(f_dim, np.float32)
    o = np.zeros_like(f)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    d_d = cuda.to_device(d)
    f_d = cuda.to_device(f)
    o_d = cuda.to_device(o)

    alpha1 = 1 + 2j
    beta1 = 0 + 0j
    alpha2 = 3 + 4j
    beta2 = 0 + 0j

    set_max_dynamic_shared_size_bytes(kernel, shared_memory_size, alpha1, a_d, b_d, beta1, c_d, alpha2, d_d, beta2, f_d, o_d)

    time_ms = time_numba(
        kernel, 1, block_dim, shared_memory_size, 100, alpha1, a_d, b_d, beta1, c_d, alpha2, d_d, beta2, f_d, o_d
    )

    print("m1, n1, k1: ", m1, n1, k1)
    print("m2, n2, k2: ", m2, n2, k2)
    print("Type: ", value_type)
    print("Precision: ", precision)
    print("cuBLASDx average time [ms] = ", time_ms)

    data_test = o_d.copy_to_host()
    data_ref = alpha2 * ((alpha1 * (a @ b) + beta1 * c) @ d) + beta2 * f
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
