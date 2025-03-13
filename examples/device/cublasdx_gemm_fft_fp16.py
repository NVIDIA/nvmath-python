# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/gemm_fft_fp16.cu
#

import numpy as np
from numba import cuda
from nvmath.device import matmul, fft, float16x4, float16x2
from common import random_complex, complex64_to_fp16x2, fp16x2_to_complex64
from common_numba import load_to_shared_1d_float16x2, store_from_shared_1d_float16x2


def main():
    # batch_size must be 2
    batch_size = 2

    m, n, k = 64, batch_size, 64

    FFT = fft(
        fft_type="c2c",
        size=k,
        precision=np.float16,
        direction="forward",
        elements_per_thread=2,
        ffts_per_block=batch_size,
        execution="Block",
        compiler="numba",
    )

    assert FFT.block_dim.y == 1
    assert FFT.ffts_per_block == FFT.implicit_type_batching

    MM = matmul(
        size=(m, n, k),
        precision=np.float16,
        data_type="complex",
        transpose_mode=("non_transposed", "non_transposed"),
        execution="Block",
        block_size=FFT.block_dim.x,
        compiler="numba",
    )

    elements_per_thread = FFT.elements_per_thread
    fft_complex_type = FFT.value_type
    mm_complex_type = MM.value_type
    storage_size = FFT.storage_size
    stride = FFT.stride

    a_size = MM.a_size
    b_size = MM.b_size

    a_dim = MM.a_dim
    b_dim = MM.b_dim
    c_dim = MM.c_dim

    lda = MM.leading_dimension.a
    ldc = MM.leading_dimension.c
    shared_memory_size = max(MM.shared_memory_size, FFT.shared_memory_size)

    # A is m x k
    # B is k x n
    # C is m x n
    #
    # We compute
    # B[:,r] = FFT(B[:,r])
    # C = alpha A * B + beta C
    @cuda.jit(link=MM.files + FFT.files)
    def kernel(a, b, c, alpha, beta, output):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=fft_complex_type)  # dtype = float16x4
        fft_shared_mem = cuda.shared.array(shape=(0,), dtype=fft_complex_type)
        mm_shared_mem = cuda.shared.array(shape=(0,), dtype=mm_complex_type)  # dtype = float16x2

        smem_a = mm_shared_mem[0:]
        smem_b = mm_shared_mem[a_size:]
        smem_c = mm_shared_mem[a_size + b_size :]

        # Load B to thread_data
        # - B
        #     shape (k, 2*n)
        #     format (r i)
        #     dtype np.float16
        # - thread_data
        #     shape (k,),
        #     format (r0 r1 i0 i1)
        #     dtype float16x4

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            r0, i0 = b[index, 0], b[index, 1]
            r1, i1 = b[index, 2], b[index, 3]
            thread_data[i] = float16x4(r0, r1, i0, i1)
            index += stride

        FFT(thread_data, fft_shared_mem)

        cuda.syncthreads()

        # Copy B to smem_b
        # We must convert from thread_data (float16x4) to smem_b (float16x2)
        # - thread_data
        #     shape (k,),
        #     format (r0 r1 i0 i1)
        #     dtype float16x4
        # - smem_b
        #     shape (k,n)
        #     format (r i)
        #     dtype float16x2

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            v = thread_data[i]
            r0, r1, i0, i1 = v.x, v.y, v.z, v.w
            smem_b[index + ldc * 0] = float16x2(r0, i0)
            smem_b[index + ldc * 1] = float16x2(r1, i1)
            index += stride

        # Load A to smem_a, C to smem_c
        load_to_shared_1d_float16x2(a, smem_a, a_dim, lda)
        load_to_shared_1d_float16x2(c, smem_c, c_dim, ldc)

        cuda.syncthreads()

        # MM
        MM(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        # Store C

        store_from_shared_1d_float16x2(smem_c, output, c_dim, ldc)

    a = random_complex(a_dim, np.float32)
    b = random_complex(b_dim, np.float32)
    c = random_complex(c_dim, np.float32)
    o = np.zeros_like(c)

    a_d = cuda.to_device(complex64_to_fp16x2(a))
    b_d = cuda.to_device(complex64_to_fp16x2(b))
    c_d = cuda.to_device(complex64_to_fp16x2(c))
    o_d = cuda.to_device(complex64_to_fp16x2(o))

    alpha = 2.0 + 0j
    beta = 3.0 + 0j

    kernel[1, FFT.block_dim, 0, shared_memory_size](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = fp16x2_to_complex64(o_d.copy_to_host())
    data_ref = alpha * (a @ np.fft.fft(b, axis=0)) + beta * c

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-2


if __name__ == "__main__":
    main()
