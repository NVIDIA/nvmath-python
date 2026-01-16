# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/gemm_fusion.cu
#

import numpy as np
from numba import cuda
from nvmath.device import Matmul, FFT
from common import random_complex
from common_numba import load_to_shared


def main():
    m, n, k = 8, 8, 8

    fft = FFT(
        fft_type="c2c",
        size=m * n,
        precision=np.float32,
        direction="forward",
        elements_per_thread=2,
        ffts_per_block=1,
        execution="Block",
    )

    mm = Matmul(
        size=(m, n, k),
        precision=np.float32,
        data_type="complex",
        arrangement=("col_major", "col_major", "col_major"),
        execution="Block",
        block_dim=fft.block_dim,
    )

    shared_memory_size = max(mm.get_shared_storage_size(), fft.shared_memory_size)

    @cuda.jit
    def kernel(a, b, c, alpha, beta, output):
        thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)

        local_fft_id = cuda.threadIdx.y
        index = cuda.threadIdx.x

        smem_a = shared_mem[0:]
        smem_b = shared_mem[mm.a_size :]
        smem_c = shared_mem[mm.a_size + mm.b_size :]
        [lda, ldb, ldc] = mm.leading_dimension

        load_to_shared(a, smem_a, mm.a_dim, lda)
        load_to_shared(b, smem_b, mm.b_dim, ldb)
        load_to_shared(c, smem_c, mm.c_dim, ldc)

        cuda.syncthreads()

        # smem_a is 8 * 8
        # smem_b is 8 * 8
        # -> smem_c is 8 * 8 = 64 elements
        mm.execute(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        index = local_fft_id * fft.ffts_per_block + cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            thread_data[i] = smem_c[index]
            index += fft.stride

        cuda.syncthreads()

        fft.execute(thread_data, shared_mem)

        index = local_fft_id * fft.ffts_per_block + cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            output[index] = thread_data[i]
            index += fft.stride

    a = random_complex(mm.a_dim, np.float32)
    b = random_complex(mm.b_dim, np.float32)
    c = random_complex(mm.c_dim, np.float32)
    o = np.zeros((mm.c_dim[0] * mm.c_dim[1],), dtype=np.complex64)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    alpha = 2.0 + 0j
    beta = 3.0 + 0j

    kernel[1, fft.block_dim, 0, shared_memory_size](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = np.fft.fft((alpha * (a @ b) + beta * c).T.reshape((-1,)))

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
