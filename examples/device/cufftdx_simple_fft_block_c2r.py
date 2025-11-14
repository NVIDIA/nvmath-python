# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/simple_fft_block_c2r.cu
#

import numpy as np
from numba import cuda
from nvmath.device import FFT


def main():
    fft = FFT(
        fft_type="c2r",
        size=128,
        precision=np.float32,
        elements_per_thread=8,
        ffts_per_block=2,
        execution="Block",
    )

    @cuda.jit
    def f(input, output):
        thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)
        thread_data_real = thread_data.view(np.float32)
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)

        local_fft_id = cuda.threadIdx.y

        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            if index < fft.size // 2 + 1:
                thread_data[i] = input[local_fft_id, index]
            index += fft.stride

        fft.execute(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(fft.elements_per_thread):
            if index < fft.size:
                output[local_fft_id, index] = thread_data_real[i]
            index += fft.stride

    input = np.ones((fft.ffts_per_block, fft.size // 2 + 1), dtype=np.complex64)
    output = np.zeros((fft.ffts_per_block, fft.size), dtype=np.float32)
    input_d = cuda.to_device(input)
    output_d = cuda.to_device(output)

    print("input [1st FFT]:", input[0, :])

    f[1, fft.block_dim, 0, fft.shared_memory_size](input_d, output_d)
    cuda.synchronize()

    output_test = output_d.copy_to_host()

    print("output [1st FFT]:", output_test[0, :])

    data_ref = np.fft.irfft(input, axis=-1, n=fft.size, norm="forward")
    error = np.linalg.norm(output_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
