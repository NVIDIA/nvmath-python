# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/fft_3d/fft_3d_cube_single_block.cu
#

import numpy as np
from numba import cuda
from nvmath.device import FFT, Dim3
from common import random_complex


def main():
    fft = FFT(fft_type="c2c", size=16, direction="forward", precision=np.float32, execution="Thread")

    block_dim = Dim3(fft.size, fft.size, 1)
    grid_dim = Dim3(1, 1, 1)
    shared_memory_size = fft.size * fft.size * fft.size * np.complex64(1).itemsize

    stride_x = fft.size * fft.size
    stride_y = fft.size

    @cuda.jit
    def f(input, output):
        thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)

        ## Load
        j, k = cuda.threadIdx.y, cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            thread_data[i] = input[i, j, k]

        ## FFT along X
        fft.execute(thread_data)

        # Exchange/transpose via shared memory
        index = cuda.threadIdx.x + cuda.threadIdx.y * fft.size
        for i in range(fft.elements_per_thread):
            shared_mem[index] = thread_data[i]
            index += stride_x
        cuda.syncthreads()
        index = cuda.threadIdx.x + cuda.threadIdx.y * fft.size * fft.size
        for i in range(fft.elements_per_thread):
            thread_data[i] = shared_mem[index]
            index += stride_y

        # FFT along Y
        fft.execute(thread_data)

        # Exchange/transpose via shared memory
        index = cuda.threadIdx.x + cuda.threadIdx.y * fft.size * fft.size
        for i in range(fft.elements_per_thread):
            shared_mem[index] = thread_data[i]
            index += stride_y
        cuda.syncthreads()
        index = (cuda.threadIdx.x + cuda.threadIdx.y * fft.size) * fft.size
        for i in range(fft.elements_per_thread):
            thread_data[i] = shared_mem[index]
            index += 1
        # for i in range(0, elements_per_thread, 2): # Manually vectorized
        #     fast_copy_2x(shared_mem, index, thread_data, i)
        #     index += 2

        # FFT along Z
        fft.execute(thread_data)

        # Shared memory IO - exchange data to store with coalesced stores
        index = (cuda.threadIdx.x + cuda.threadIdx.y * fft.size) * fft.size
        for i in range(fft.elements_per_thread):
            shared_mem[index] = thread_data[i]
            index += 1
        # for i in range(0, elements_per_thread, 2): # Manually vectorized
        #     fast_copy_2x(thread_data, i, shared_mem, index)
        #     index += 2
        cuda.syncthreads()
        index = cuda.threadIdx.x + cuda.threadIdx.y * fft.size
        for i in range(fft.elements_per_thread):
            thread_data[i] = shared_mem[index]
            index += stride_x

        # Store
        j, k = cuda.threadIdx.y, cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            output[i, j, k] = thread_data[i]

    input = random_complex((fft.size, fft.size, fft.size), real_dtype=np.float32)
    output = np.zeros((fft.size, fft.size, fft.size), dtype=np.complex64)
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
