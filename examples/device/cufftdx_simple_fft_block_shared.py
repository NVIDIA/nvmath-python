# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# Mirrors https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuFFTDx/simple_fft_block_shared.cu
#

import numpy as np
from numba import cuda
from nvmath.device import FFT


def main():
    fft = FFT(
        fft_type="c2c",
        size=128,
        precision=np.float32,
        direction="forward",
        elements_per_thread=8,
        ffts_per_block=2,
        execution="Block",
    )

    @cuda.jit
    def f(data):
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)
        local_fft_id = cuda.threadIdx.y

        index = cuda.threadIdx.x
        for _ in range(fft.elements_per_thread):
            shared_mem[local_fft_id * fft.size + index] = data[local_fft_id, index]
            index += fft.stride

        cuda.syncthreads()

        fft(shared_mem)

        cuda.syncthreads()

        index = cuda.threadIdx.x
        for _ in range(fft.elements_per_thread):
            data[local_fft_id, index] = shared_mem[local_fft_id * fft.size + index]
            index += fft.stride

    data = np.ones((fft.ffts_per_block, fft.size), dtype=np.complex64)
    data_d = cuda.to_device(data)

    print("input [1st FFT]:", data[0, :])

    shared_memory_size = max(fft.shared_memory_size, np.complex64(1.0).itemsize * fft.size)
    f[1, fft.block_dim, 0, shared_memory_size](data_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()

    print("output [1st FFT]:", data_test[0, :])

    data_ref = np.fft.fft(data, axis=-1)
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5


if __name__ == "__main__":
    main()
