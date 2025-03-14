# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/gemm_fft_performance.cu
#

import numpy as np
import cupy as cp
from numba import cuda
from nvmath.device import matmul, fft
from common import random_complex
from common_numba import load_to_shared, time_numba
from common_cupy import time_cupy


@cuda.jit(inline="always")
def nb_transform(x):
    return x * x


def main():
    ncycles = 10
    batch_size = 128 * 1024
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

        batch = cuda.blockIdx.x
        local_fft_id = cuda.threadIdx.y
        index = cuda.threadIdx.x

        smem_a = shared_mem[0:]
        smem_b = shared_mem[a_size:]
        smem_c = shared_mem[a_size + b_size :]

        # Load data
        load_to_shared(a[batch, :, :], smem_a, a_dim, lda)
        load_to_shared(b[batch, :, :], smem_b, b_dim, ldb)
        load_to_shared(c[batch, :, :], smem_c, c_dim, ldc)

        cuda.syncthreads()

        # Transform A
        for i in range(cuda.threadIdx.x, a_size, cuda.blockDim.x):
            smem_a[i] = nb_transform(smem_a[i])

        # Execute GEMM
        MM(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        # Load data into local array
        index = local_fft_id * ffts_per_block + cuda.threadIdx.x
        for i in range(elements_per_thread):
            thread_data[i] = smem_c[index]
            index += stride

        cuda.syncthreads()

        # Execute FFT
        FFT(thread_data, shared_mem)

        # Transform and store data
        index = local_fft_id * ffts_per_block + cuda.threadIdx.x
        for i in range(elements_per_thread):
            output[batch, index] = nb_transform(thread_data[i])
            index += stride

    a = cp.array(random_complex((batch_size, *a_dim), np.float32))
    b = cp.array(random_complex((batch_size, *b_dim), np.float32))
    c = cp.array(random_complex((batch_size, *c_dim), np.float32))
    data_test = cp.zeros((batch_size, c_dim[0] * c_dim[1]), dtype=np.complex64)

    alpha = 2.0 + 0j
    beta = 3.0 + 0j
    grid_dim = batch_size // ffts_per_block
    block_dim = FFT.block_dim

    kernel[grid_dim, block_dim, 0, shared_memory_size](a, b, c, alpha, beta, data_test)
    cuda.synchronize()

    def cp_kernel(a, b, c):
        cp_transform = lambda x: cp.multiply(x, x)
        abc = cp.swapaxes(alpha * cp.einsum("bik,bkj->bij", cp_transform(a), b) + beta * c, 1, 2).reshape((batch_size, -1))
        return cp_transform(cp.fft.fft(abc, axis=-1))

    data_ref = cp_kernel(a, b, c)

    error = cp.linalg.norm(data_test - data_ref) / cp.linalg.norm(data_ref)
    assert error < 1e-5

    time_cp = time_cupy(cp_kernel, ncycles, a, b, c)
    time_nb = time_numba(kernel, grid_dim, block_dim, shared_memory_size, ncycles, a, b, c, alpha, beta, data_test)

    print(f"cupy average time: {time_cp} [ms]")
    print(f"numba average time: {time_nb} [ms]")


if __name__ == "__main__":
    main()
