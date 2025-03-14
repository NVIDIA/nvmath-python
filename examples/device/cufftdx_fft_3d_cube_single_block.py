# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/fft_3d/fft_3d_cube_single_block.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft, Dim3
from common import random_complex


def main():
    fft_size = 16

    FFT = fft(fft_type="c2c", size=fft_size, direction="forward", precision=np.float32, execution="Thread", compiler="numba")

    block_dim = Dim3(fft_size, fft_size, 1)
    grid_dim = Dim3(1, 1, 1)
    storage_size = FFT.storage_size
    value_type = FFT.value_type
    shared_memory_size = fft_size * fft_size * fft_size * np.complex64(1).itemsize
    elements_per_thread = FFT.elements_per_thread

    stride_x = fft_size * fft_size
    stride_y = fft_size

    @cuda.jit(link=FFT.files)
    def f(input, output):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)

        ## Load
        j, k = cuda.threadIdx.y, cuda.threadIdx.x
        for i in range(elements_per_thread):
            thread_data[i] = input[i, j, k]

        ## FFT along X
        FFT(thread_data)

        # Exchange/transpose via shared memory
        index = cuda.threadIdx.x + cuda.threadIdx.y * fft_size
        for i in range(elements_per_thread):
            shared_mem[index] = thread_data[i]
            index += stride_x
        cuda.syncthreads()
        index = cuda.threadIdx.x + cuda.threadIdx.y * fft_size * fft_size
        for i in range(elements_per_thread):
            thread_data[i] = shared_mem[index]
            index += stride_y

        # FFT along Y
        FFT(thread_data)

        # Exchange/transpose via shared memory
        index = cuda.threadIdx.x + cuda.threadIdx.y * fft_size * fft_size
        for i in range(elements_per_thread):
            shared_mem[index] = thread_data[i]
            index += stride_y
        cuda.syncthreads()
        index = (cuda.threadIdx.x + cuda.threadIdx.y * fft_size) * fft_size
        for i in range(elements_per_thread):
            thread_data[i] = shared_mem[index]
            index += 1
        # for i in range(0, elements_per_thread, 2): # Manually vectorized
        #     fast_copy_2x(shared_mem, index, thread_data, i)
        #     index += 2

        # FFT along Z
        FFT(thread_data)

        # Shared memory IO - exchange data to store with coalesced stores
        index = (cuda.threadIdx.x + cuda.threadIdx.y * fft_size) * fft_size
        for i in range(elements_per_thread):
            shared_mem[index] = thread_data[i]
            index += 1
        # for i in range(0, elements_per_thread, 2): # Manually vectorized
        #     fast_copy_2x(thread_data, i, shared_mem, index)
        #     index += 2
        cuda.syncthreads()
        index = cuda.threadIdx.x + cuda.threadIdx.y * fft_size
        for i in range(elements_per_thread):
            thread_data[i] = shared_mem[index]
            index += stride_x

        # Store
        j, k = cuda.threadIdx.y, cuda.threadIdx.x
        for i in range(elements_per_thread):
            output[i, j, k] = thread_data[i]

    input = random_complex((fft_size, fft_size, fft_size), real_dtype=np.float32)
    output = np.zeros((fft_size, fft_size, fft_size), dtype=np.complex64)
    input_d = cuda.to_device(input)
    output_d = cuda.to_device(output)

    print("input [:2,:2,:2]:", input[:2, :2, :2])

    f[grid_dim, block_dim, 0, shared_memory_size](input_d, output_d)
    cuda.synchronize()

    output_test = output_d.copy_to_host()

    print("output [:10,:10]:", output_test[:2, :2, :2])

    output_ref = np.fft.fftn(input)
    error = np.linalg.norm(output_ref - output_test) / np.linalg.norm(output_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
