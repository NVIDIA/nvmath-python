# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/fft_2d/fft_2d_r2c_c2r.cu
#

import numpy as np
from numba import cuda
from nvmath.device import fft
from common import random_real
import functools


def main():
    fft_size_y = 1024
    fft_size_x = 1024

    ept_y = 16
    fpb_y = 1
    ept_x = 16
    fpb_x = 8

    FFT_base = functools.partial(fft, precision=np.float32, execution="Block", compiler="numba")
    # R2C along Y (fft_size_x batches, logical FFT size is fft_size_y, complex size is fft_size_y//2+1)  # noqa: W505
    FFT_y_r2c = FFT_base(fft_type="r2c", size=fft_size_y, elements_per_thread=ept_y, ffts_per_block=fpb_y)
    # C2Cf along X (fft_size_y//2+1 batches, logical FFT size is fft_size_x)
    FFT_x_c2c_f = FFT_base(
        fft_type="c2c", direction="inverse", size=fft_size_x, elements_per_thread=ept_x, ffts_per_block=fpb_x
    )
    # C2Ci along X (fft_size_y//2+1 batches, logical FFT size is fft_size_x)
    FFT_x_c2c_i = FFT_base(
        fft_type="c2c", direction="forward", size=fft_size_x, elements_per_thread=ept_x, ffts_per_block=fpb_x
    )
    # C2R along Y (fft_size_x batches, logical FFT size is fft_size_y, complex size is fft_size_y//2+1)  # noqa: W505
    FFT_y_c2r = FFT_base(fft_type="c2r", size=fft_size_y, elements_per_thread=ept_y, ffts_per_block=fpb_y)

    complex_type = FFT_y_r2c.value_type
    real_type = np.float32
    storage_size_r2c = FFT_y_r2c.storage_size
    storage_size_c2c = max(FFT_x_c2c_f.storage_size, FFT_x_c2c_i.storage_size)
    storage_size_c2r = FFT_y_c2r.storage_size
    stride_r2c = FFT_y_r2c.stride
    stride_c2c = FFT_x_c2c_f.stride
    stride_c2r = FFT_y_c2r.stride

    assert FFT_x_c2c_f.stride == FFT_x_c2c_i.stride
    assert FFT_x_c2c_f.block_dim == FFT_x_c2c_i.block_dim
    assert FFT_x_c2c_f.shared_memory_size == FFT_x_c2c_i.shared_memory_size

    grid_dim_y = (fft_size_x + fpb_y - 1) // fpb_y
    grid_dim_x = ((fft_size_y // 2 + 1) + fpb_x - 1) // fpb_x

    # storage_size_x       = FFT_x.storage_size
    # storage_size_y       = FFT_y.storage_size
    # stride_x             = FFT_x.stride
    # stride_y             = FFT_y.stride

    @cuda.jit(link=FFT_y_r2c.files)
    def f_y_r2c(input, output):
        thread_data = cuda.local.array(shape=(storage_size_r2c,), dtype=complex_type)
        thread_data_real = thread_data.view(real_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=complex_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * fpb_y + local_fft_id
        if fft_id >= fft_size_x:
            return

        index = cuda.threadIdx.x
        for i in range(ept_y):
            if index < fft_size_y:
                thread_data_real[i] = input[fft_id, index]
                index += stride_r2c

        FFT_y_r2c(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(ept_y):
            if index < (fft_size_y // 2 + 1):
                output[fft_id, index] = thread_data[i]
                index += stride_r2c

    @cuda.jit(link=FFT_x_c2c_f.files + FFT_x_c2c_i.files)
    def f_x(input, output):
        thread_data = cuda.local.array(shape=(storage_size_c2c,), dtype=complex_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=complex_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * fpb_x + local_fft_id
        if fft_id >= (fft_size_y // 2 + 1):
            return

        index = cuda.threadIdx.x
        for i in range(ept_x):
            if index < fft_size_x:
                thread_data[i] = input[index, fft_id]
                index += stride_c2c

        FFT_x_c2c_f(thread_data, shared_mem)

        # Can do some elementwise operation here
        # index = cuda.threadIdx.x
        # for i in range(ept_x):
        #     if index < fft_size_x:
        #         thread_data[i] = thread_data[i] / (fft_size_x * fft_size_y)
        #         index += stride_c2c

        FFT_x_c2c_i(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(ept_x):
            if index < fft_size_x:
                output[index, fft_id] = thread_data[i]
                index += stride_c2c

    @cuda.jit(link=FFT_y_c2r.files)
    def f_y_c2r(input, output):
        thread_data = cuda.local.array(shape=(storage_size_c2r,), dtype=complex_type)
        thread_data_real = thread_data.view(real_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=complex_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * fpb_y + local_fft_id
        if fft_id >= fft_size_x:
            return

        index = cuda.threadIdx.x
        for i in range(ept_y):
            if index < (fft_size_y // 2 + 1):
                thread_data[i] = input[fft_id, index]
                index += stride_c2r

        FFT_y_c2r(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(ept_y):
            if index < fft_size_y:
                output[fft_id, index] = thread_data_real[i]
                index += stride_c2r

    real = random_real((fft_size_x, fft_size_y), real_dtype=np.float32)
    complex = np.zeros((fft_size_x, fft_size_y // 2 + 1), dtype=np.complex64)

    real_d = cuda.to_device(real)
    complex_d = cuda.to_device(complex)

    print("real (input) [:10,:10]:", real[:10, :10])

    f_y_r2c[grid_dim_y, FFT_y_r2c.block_dim, 0, FFT_y_r2c.shared_memory_size](real_d, complex_d)
    f_x[grid_dim_x, FFT_x_c2c_f.block_dim, 0, FFT_x_c2c_f.shared_memory_size](complex_d, complex_d)
    f_y_c2r[grid_dim_y, FFT_y_c2r.block_dim, 0, FFT_y_c2r.shared_memory_size](complex_d, real_d)
    cuda.synchronize()

    real_test = real_d.copy_to_host()
    real_ref = real * fft_size_x * fft_size_y

    print("real (output) [:10,:10]:", real_test[:10, :10])

    error = np.linalg.norm(real_test - real_ref) / np.linalg.norm(real_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
