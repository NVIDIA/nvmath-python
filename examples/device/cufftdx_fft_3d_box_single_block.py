# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/fft_3d/fft_3d_box_single_block.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft, Dim3
from common import random_complex
import functools

def main():

    fft_size_x = 16
    fft_size_y = 15
    fft_size_z = 14

    FFT_base = functools.partial(fft, fft_type='c2c', direction='forward', precision=np.float32, execution='Thread', compiler='numba')
    FFT_x = FFT_base(size=fft_size_x)
    FFT_y = FFT_base(size=fft_size_y)
    FFT_z = FFT_base(size=fft_size_z)

    value_type            = FFT_x.value_type
    max_dim               = max(fft_size_x, fft_size_y, fft_size_z)
    block_dim             = Dim3(max_dim, max_dim, 1)
    shared_memory_size    = (fft_size_x * fft_size_y * fft_size_z) * np.complex64(1.0).itemsize
    storage_size          = max(FFT_x.storage_size, FFT_y.storage_size, FFT_z.storage_size)
    grid_dim              = Dim3(1, 1, 1)

    eptx = FFT_x.elements_per_thread
    epty = FFT_y.elements_per_thread
    eptz = FFT_z.elements_per_thread

    stride_x = fft_size_y * fft_size_z
    stride_y = fft_size_z
    stride_z = 1

    @cuda.jit(link=FFT_x.files + FFT_y.files + FFT_z.files)
    def f(input, output):

        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)

        tidx = cuda.threadIdx.x
        tidy = cuda.threadIdx.y

        # given thread --> X
        # threadIdx.y  --> Y
        # threadIdx.x  --> Z
        if tidy < fft_size_y and tidx < fft_size_z:

            for i in range(eptx):
                # fast_copy(input, i * stride_x + tidy * stride_y + tidx * stride_z, thread_data, i)
                thread_data[i] = input[i, tidy, tidx]

            FFT_x(thread_data)

            index = tidy * stride_y + tidx * stride_z
            for i in range(eptx):
                shared_mem[index] = thread_data[i]
                index += stride_x

        cuda.syncthreads()

        # threadIdx.y  --> X
        # given thread --> Y
        # threadIdx.x  --> Z
        if tidy < fft_size_x and tidx < fft_size_z:

            index = tidy * stride_x + tidx * stride_z
            for i in range(epty):
                thread_data[i] = shared_mem[index]
                index += stride_y

            FFT_y(thread_data)

            index = tidy * stride_x + tidx
            for i in range(epty):
                shared_mem[index] = thread_data[i]
                index += stride_y

        cuda.syncthreads()

        # threadIdx.y  --> X
        # threadIdx.x  --> Y
        # given thread --> Z
        if tidy < fft_size_x and tidx < fft_size_y:

            index = tidy * stride_x + tidx * stride_y
            # for i in range(0, eptz, 2): # eptz is even
            #     fast_copy_2x(shared_mem, index, thread_data, i)
            #     index += 2
            for i in range(eptz):
                thread_data[i] = shared_mem[index]
                index += stride_z

            FFT_z(thread_data)

            # Reshuffle in shared
            index = tidy * stride_x + tidx * stride_y
            # for i in range(0, eptz, 2): # eptz is even
            #     fast_copy_2x(thread_data, i, shared_mem, index)
            #     index += 2
            for i in range(eptz):
                shared_mem[index] = thread_data[i]
                index += stride_z

        cuda.syncthreads()

        # given thread --> X
        # threadIdx.y  --> Y
        # threadIdx.x  --> Z
        if tidy < fft_size_y and tidx < fft_size_z:

            index = tidy * stride_y + tidx * stride_z
            for i in range(eptx):
                thread_data[i] = shared_mem[index]
                index += stride_x

            for i in range(eptx):
                # fast_copy(thread_data, i, output, i * stride_x + tidy * stride_y + tidx * stride_z)
                output[i, tidy, tidx] = thread_data[i]

    input = random_complex((fft_size_x, fft_size_y, fft_size_z), real_dtype=np.float32)
    output = np.zeros((fft_size_x, fft_size_y, fft_size_z), dtype=np.complex64)
    input_d = cuda.to_device(input)
    output_d = cuda.to_device(output)

    print("input [:2,:2,:2]:", input[:2,:2,:2])

    f[grid_dim, block_dim, 0, shared_memory_size](input_d, output_d)
    cuda.synchronize()

    output_test = output_d.copy_to_host()

    print("output [:10,:10]:", output_test[:2,:2,:2])

    output_ref = np.fft.fftn(input)
    error = np.linalg.norm(output_ref - output_test) / np.linalg.norm(output_ref)
    assert error < 1e-5

if __name__ == "__main__":
    main()
