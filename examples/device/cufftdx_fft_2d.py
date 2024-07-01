# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/fft_2d/fft_2d.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft, Dim3
from common import random_complex
import functools

def main():

    fft_size_y = 1024
    fft_size_x = 1024

    ept_y = 16
    fpb_y = 1
    ept_x = 16
    fpb_x = 8

    FFT_base = functools.partial(fft, fft_type='c2c', direction='forward', precision=np.float32, execution='Block', compiler='numba')
    FFT_y = FFT_base(size=fft_size_y, elements_per_thread=ept_y, ffts_per_block=fpb_y)
    FFT_x = FFT_base(size=fft_size_x, elements_per_thread=ept_x, ffts_per_block=fpb_x)

    value_type           = FFT_x.value_type
    storage_size_x       = FFT_x.storage_size
    storage_size_y       = FFT_y.storage_size
    stride_x             = FFT_x.stride
    stride_y             = FFT_y.stride

    grid_dim_y = Dim3(fft_size_x // fpb_y, 1, 1)
    grid_dim_x = Dim3(fft_size_y // fpb_x, 1, 1)

    @cuda.jit(link=FFT_y.files)
    def f_y(input, output):

        thread_data = cuda.local.array(shape=(storage_size_y,), dtype=value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * fpb_y + local_fft_id

        index = cuda.threadIdx.x
        for i in range(ept_y):
            thread_data[i] = input[fft_id, index]
            index += stride_y

        FFT_y(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(ept_y):
            output[fft_id, index] = thread_data[i]
            index += stride_y

    @cuda.jit(link=FFT_x.files)
    def f_x(input, output):

        thread_data = cuda.local.array(shape=(storage_size_x,), dtype=value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * fpb_x + local_fft_id

        index = cuda.threadIdx.x
        for i in range(ept_x):
            thread_data[i] = input[index, fft_id]
            index += stride_x

        FFT_x(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(ept_x):
            output[index, fft_id] = thread_data[i]
            index += stride_x

    input = random_complex((fft_size_x, fft_size_y), real_dtype=np.float32)
    output = np.zeros((fft_size_x, fft_size_y), dtype=np.complex64)
    input_d = cuda.to_device(input)
    output_d = cuda.to_device(output)

    print("input [:10,:10]:", input[:10,:10])

    f_y[grid_dim_y, FFT_y.block_dim, 0, FFT_y.shared_memory_size](input_d, output_d)
    f_x[grid_dim_x, FFT_x.block_dim, 0, FFT_x.shared_memory_size](output_d, output_d)
    cuda.synchronize()

    output_test = output_d.copy_to_host()

    print("output [:10,:10]:", output_test[:10,:10])

    output_ref = np.fft.fftn(input)
    error = np.linalg.norm(output_test - output_ref) / np.linalg.norm(output_ref)
    assert error < 1e-5

if __name__ == "__main__":
    main()
