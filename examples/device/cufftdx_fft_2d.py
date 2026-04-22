# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/fft_2d/fft_2d.cu
#

import functools

import numpy as np
from common import random_complex
from numba import cuda

from nvmath.device import FFT, Dim3


def main():
    fft_size_y = 1024
    fft_size_x = 1024

    ept_y = 16
    fpb_y = 1
    ept_x = 16
    fpb_x = 8

    FFT_base = functools.partial(FFT, fft_type="c2c", direction="forward", precision=np.float32, execution="Block")
    fft_y = FFT_base(size=fft_size_y, elements_per_thread=ept_y, ffts_per_block=fpb_y)
    fft_x = FFT_base(size=fft_size_x, elements_per_thread=ept_x, ffts_per_block=fpb_x)

    grid_dim_y = Dim3(fft_size_x // fpb_y, 1, 1)
    grid_dim_x = Dim3(fft_size_y // fpb_x, 1, 1)

    @cuda.jit
    def f_y(input, output):
        thread_data = cuda.local.array(shape=(fft_y.storage_size,), dtype=fft_y.value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft_y.value_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * fpb_y + local_fft_id

        index = cuda.threadIdx.x
        for i in range(ept_y):
            thread_data[i] = input[fft_id, index]
            index += fft_y.stride

        fft_y.execute(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(ept_y):
            output[fft_id, index] = thread_data[i]
            index += fft_y.stride

    @cuda.jit
    def f_x(input, output):
        thread_data = cuda.local.array(shape=(fft_x.storage_size,), dtype=fft_x.value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft_x.value_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * fpb_x + local_fft_id

        index = cuda.threadIdx.x
        for i in range(ept_x):
            thread_data[i] = input[index, fft_id]
            index += fft_x.stride

        fft_x.execute(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(ept_x):
            output[index, fft_id] = thread_data[i]
            index += fft_x.stride

    input = random_complex((fft_size_x, fft_size_y), real_dtype=np.float32)
    output = np.zeros((fft_size_x, fft_size_y), dtype=np.complex64)
    input_d = cuda.to_device(input)
    output_d = cuda.to_device(output)

    print("input [:10,:10]:", input[:10, :10])

    f_y[grid_dim_y, fft_y.block_dim, 0, fft_y.shared_memory_size](input_d, output_d)
    f_x[grid_dim_x, fft_x.block_dim, 0, fft_x.shared_memory_size](output_d, output_d)
    cuda.synchronize()

    output_test = output_d.copy_to_host()

    print("output [:10,:10]:", output_test[:10, :10])

    output_ref = np.fft.fftn(input)
    error = np.linalg.norm(output_test - output_ref) / np.linalg.norm(output_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
