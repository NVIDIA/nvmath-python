# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/gemm_fusion.cu
#

import numpy as np
from numba import cuda
from nvmath.device import matmul, fft
from common import random_complex
from common_numba import load_to_shared


def main():
    m, n, k = 8, 8, 8

    FFT = fft(
        fft_type="c2c",
        size=m * n,
        precision=np.float32,
        direction="forward",
        elements_per_thread=2,
        ffts_per_block=1,
        execution="Block",
        compiler="numba",
    )

    MM = matmul(
        size=(m, n, k),
        precision=np.float32,
        data_type="complex",
        transpose_mode=("non_transposed", "non_transposed"),
        execution="Block",
        block_dim=FFT.block_dim,
        compiler="numba",
    )

    elements_per_thread = FFT.elements_per_thread
    ffts_per_block = FFT.ffts_per_block
    complex_type = FFT.value_type
    storage_size = FFT.storage_size
    stride = FFT.stride

    a_size = MM.a_size
    b_size = MM.b_size

    a_dim = MM.a_dim
    b_dim = MM.b_dim
    c_dim = MM.c_dim

    lda = MM.leading_dimension.a
    ldb = MM.leading_dimension.b
    ldc = MM.leading_dimension.c
    shared_memory_size = max(MM.shared_memory_size, FFT.shared_memory_size)

    @cuda.jit(link=MM.files + FFT.files)
    def kernel(a, b, c, alpha, beta, output):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=complex_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=complex_type)

        local_fft_id = cuda.threadIdx.y
        index = cuda.threadIdx.x

        smem_a = shared_mem[0:]
        smem_b = shared_mem[a_size:]
        smem_c = shared_mem[a_size + b_size :]

        load_to_shared(a, smem_a, a_dim, lda)
        load_to_shared(b, smem_b, b_dim, ldb)
        load_to_shared(c, smem_c, c_dim, ldc)

        cuda.syncthreads()

        # smem_a is 8 * 8
        # smem_b is 8 * 8
        # -> smem_c is 8 * 8 = 64 elements
        MM(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        index = local_fft_id * ffts_per_block + cuda.threadIdx.x
        for i in range(elements_per_thread):
            thread_data[i] = smem_c[index]
            index += stride

        cuda.syncthreads()

        FFT(thread_data, shared_mem)

        index = local_fft_id * ffts_per_block + cuda.threadIdx.x
        for i in range(elements_per_thread):
            output[index] = thread_data[i]
            index += stride

    a = random_complex(a_dim, np.float32)
    b = random_complex(b_dim, np.float32)
    c = random_complex(c_dim, np.float32)
    o = np.zeros((c_dim[0] * c_dim[1],), dtype=np.complex64)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    alpha = 2.0 + 0j
    beta = 3.0 + 0j

    kernel[1, FFT.block_dim, 0, shared_memory_size](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = np.fft.fft((alpha * (a @ b) + beta * c).T.reshape((-1,)))

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
