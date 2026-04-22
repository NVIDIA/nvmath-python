# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/fft_2d/fft_2d_single_kernel.cu
#

import functools

import numpy as np
from common import random_complex
from numba import cuda

from nvmath.device import FFT


def main():
    fft_size_y = 256
    fft_size_x = 128

    ept_y = 16
    fpb_y = 4
    ept_x = 8
    fpb_x = fpb_y

    FFT_base = functools.partial(FFT, fft_type="c2c", direction="forward", precision=np.float32, execution="Block")
    fft_y = FFT_base(size=fft_size_y, elements_per_thread=ept_y, ffts_per_block=fpb_y)
    fft_x = FFT_base(size=fft_size_x, elements_per_thread=ept_x, ffts_per_block=fpb_x)

    value_type = fft_y.value_type
    storage_size = max(fft_x.storage_size, fft_y.storage_size)
    shared_memory_size = max(fft_x.shared_memory_size, fft_y.shared_memory_size)

    assert fft_x.block_dim == fft_y.block_dim
    block_dim = fft_x.block_dim
    grid_dim = (max(fft_size_y // fpb_y, fft_size_x // fpb_x), 1, 1)

    @cuda.jit
    def f(input, output):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)

        ## FFTs along Y (contiguous dimension)

        fft_id = cuda.blockIdx.x * fpb_x + cuda.threadIdx.y

        if fft_id < fft_size_x:
            index = cuda.threadIdx.x
            for i in range(ept_y):
                thread_data[i] = input[fft_id, index]
                # fast_copy(input, fft_id * fft_size_y + index, thread_data, i)
                index += fft_y.stride

            fft_y.execute(thread_data, shared_mem)

            index = cuda.threadIdx.x
            for i in range(ept_y):
                output[fft_id, index] = thread_data[i]
                # fast_copy(thread_data, i, output, fft_id * fft_size_y + index)
                index += fft_y.stride

        ## Grid sync
        g = cuda.cg.this_grid()
        g.sync()

        ## FFTs along X (strided dimension)

        fft_id = cuda.blockIdx.x * fpb_y + cuda.threadIdx.y

        if fft_id < fft_size_y:
            index = cuda.threadIdx.x
            for i in range(ept_x):
                thread_data[i] = output[index, fft_id]
                # fast_copy(output, index * fft_size_y + fft_id, thread_data, i)
                index += fft_x.stride

            # Compute
            fft_x.execute(thread_data, shared_mem)

            # Store
            index = cuda.threadIdx.x
            for i in range(ept_x):
                output[index, fft_id] = thread_data[i]
                # fast_copy(thread_data, i, output, index * fft_size_y + fft_id)
                index += fft_x.stride

    input = random_complex((fft_size_x, fft_size_y), real_dtype=np.float32)
    output = np.zeros((fft_size_x, fft_size_y), dtype=np.complex64)
    input_d = cuda.to_device(input)
    output_d = cuda.to_device(output)

    try:
        f[grid_dim, block_dim, 0, shared_memory_size](input_d, output_d)
    except cuda.cudadrv.driver.LinkerError as e:
        if str(e) == "libcudadevrt.a not found":
            print(
                f"\n=== Numba linker error: {e}. Please use the System CTK option (see Installation in the documentation) "
                "to run this example. ===\n"
            )
        raise e
    cuda.synchronize()

    output_test = output_d.copy_to_host()

    print("input [:10,:10]:", input[:10, :10])

    print("output [:10,:10]:", output_test[:10, :10])

    output_ref = np.fft.fftn(input)
    error = np.linalg.norm(output_test - output_ref) / np.linalg.norm(output_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
