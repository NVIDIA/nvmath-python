# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from nvmath.device import fft
from common import random_complex


def main():
    size = 128
    ffts_per_block = 1
    batch_size = 1

    FFT_fwd = fft(
        fft_type="c2c",
        size=size,
        precision=np.float32,
        direction="forward",
        ffts_per_block=ffts_per_block,
        elements_per_thread=2,
        execution="Block",
        compiler="numba",
    )
    FFT_inv = fft(
        fft_type="c2c",
        size=size,
        precision=np.float32,
        direction="inverse",
        ffts_per_block=ffts_per_block,
        elements_per_thread=2,
        execution="Block",
        compiler="numba",
    )

    value_type = FFT_fwd.value_type
    storage_size = FFT_fwd.storage_size
    shared_memory_size = FFT_fwd.shared_memory_size
    fft_stride = FFT_fwd.stride
    ept = FFT_fwd.elements_per_thread
    block_dim = FFT_fwd.block_dim

    @cuda.jit(link=FFT_fwd.files + FFT_inv.files)
    def f(signal, filter):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)

        fft_id = (cuda.blockIdx.x * ffts_per_block) + cuda.threadIdx.y
        if fft_id >= batch_size:
            return
        offset = cuda.threadIdx.x

        for i in range(ept):
            thread_data[i] = signal[fft_id, offset + i * fft_stride]

        FFT_fwd(thread_data, shared_mem)

        for i in range(ept):
            thread_data[i] = thread_data[i] * filter[fft_id, offset + i * fft_stride]

        FFT_inv(thread_data, shared_mem)

        for i in range(ept):
            signal[fft_id, offset + i * fft_stride] = thread_data[i]

    data = random_complex((ffts_per_block, size), np.float32)
    filter = random_complex((ffts_per_block, size), np.float32)

    data_d = cuda.to_device(data)
    filter_d = cuda.to_device(filter)

    f[1, block_dim, 0, shared_memory_size](data_d, filter_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()
    data_ref = np.fft.ifft(np.fft.fft(data, axis=-1) * filter, axis=-1) * size

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    print(f"L2 error {error}")

    assert error < 1e-5


if __name__ == "__main__":
    main()
