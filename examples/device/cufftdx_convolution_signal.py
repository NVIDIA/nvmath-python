# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import functools

import numpy as np
from common import random_complex
from numba import cuda

from nvmath.device import FFT


def main():
    size = 128
    ffts_per_block = 1
    batch_size = 1

    FFT_base = functools.partial(
        FFT,
        fft_type="c2c",
        size=size,
        precision=np.float32,
        ffts_per_block=ffts_per_block,
        elements_per_thread=2,
        execution="Block",
    )
    fft = FFT_base(direction="forward")
    ifft = FFT_base(direction="inverse")

    @cuda.jit
    def f(signal, filter):
        thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)

        fft_id = (cuda.blockIdx.x * ffts_per_block) + cuda.threadIdx.y
        if fft_id >= batch_size:
            return
        offset = cuda.threadIdx.x

        for i in range(fft.elements_per_thread):
            thread_data[i] = signal[fft_id, offset + i * fft.stride]

        fft.execute(thread_data, shared_mem)

        for i in range(fft.elements_per_thread):
            thread_data[i] = thread_data[i] * filter[fft_id, offset + i * fft.stride]

        ifft.execute(thread_data, shared_mem)

        for i in range(fft.elements_per_thread):
            signal[fft_id, offset + i * fft.stride] = thread_data[i]

    data = random_complex((ffts_per_block, size), np.float32)
    filter = random_complex((ffts_per_block, size), np.float32)

    data_d = cuda.to_device(data)
    filter_d = cuda.to_device(filter)

    f[1, fft.block_dim, 0, fft.shared_memory_size](data_d, filter_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()
    data_ref = np.fft.ifft(np.fft.fft(data, axis=-1) * filter, axis=-1) * size

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    print(f"L2 error {error}")

    assert error < 1e-5


if __name__ == "__main__":
    main()
