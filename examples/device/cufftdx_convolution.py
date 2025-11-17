# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/convolution/convolution.cu
#

import numpy as np
from numba import cuda
from nvmath.device import FFT
from common import random_complex
import functools


def main():
    FFT_base = functools.partial(
        FFT,
        fft_type="c2c",
        size=64,
        precision=np.float32,
        ffts_per_block=2,
        elements_per_thread=8,
        execution="Block",
    )
    fft = FFT_base(direction="forward")
    ifft = FFT_base(direction="inverse")

    @cuda.jit
    def f(data):
        thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * fft.ffts_per_block + local_fft_id

        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            thread_data[i] = data[fft_id, index]
            index += fft.stride

        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)
        fft.execute(thread_data, shared_mem)

        for i in range(fft.elements_per_thread):
            thread_data[i] = thread_data[i] / fft.size

        ifft.execute(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            data[fft_id, index] = thread_data[i]
            index += fft.stride

    data = random_complex((fft.ffts_per_block, fft.size), real_dtype=np.float32)
    data_d = cuda.to_device(data)

    print("input [1st FFT]:", data[0, :])

    f[1, fft.block_dim, 0, fft.shared_memory_size](data_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()

    print("output [1st FFT]:", data_test[0, :])

    error = np.linalg.norm(data_test - data) / np.linalg.norm(data)
    assert error < 1e-5


if __name__ == "__main__":
    main()
