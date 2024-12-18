# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from nvmath.device import current_device_lto, FFTOptions
from common_numba import time_numba


def main():
    batch = 1024 * 32
    ncycles = 10

    base_FFT = FFTOptions(
        fft_type="c2c",
        size=256,
        precision=np.float32,
        direction="forward",
        execution="Block",
        code_type=current_device_lto(),
    )

    data = np.ones((batch, base_FFT.size), dtype=np.complex64)
    data_ref = np.fft.fft(data, axis=-1)

    for ept, fpb in base_FFT.valid("elements_per_thread", "ffts_per_block"):
        FFT = base_FFT.create(elements_per_thread=ept, ffts_per_block=fpb, compiler="numba")

        value_type = FFT.value_type
        storage_size = FFT.storage_size
        shared_memory_size = FFT.shared_memory_size
        stride = FFT.stride
        block_dim = FFT.block_dim
        ffts_per_block = FFT.ffts_per_block
        elements_per_thread = FFT.elements_per_thread
        grid_dim = (batch + ffts_per_block - 1) // ffts_per_block

        assert ept == elements_per_thread
        assert fpb == ffts_per_block

        @cuda.jit(link=FFT.files)
        def f(input, output):
            thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)

            local_fft_id = cuda.threadIdx.y
            fft_id = cuda.blockIdx.x * ffts_per_block + local_fft_id
            if fft_id >= batch:
                return

            index = cuda.threadIdx.x
            for i in range(elements_per_thread):
                thread_data[i] = input[fft_id, index]
                index += stride

            shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)
            FFT(thread_data, shared_mem)

            index = cuda.threadIdx.x
            for i in range(elements_per_thread):
                output[fft_id, index] = thread_data[i]
                index += stride

        input_d = cuda.to_device(data)
        output_d = cuda.to_device(data)
        cuda.synchronize()
        time_ms = time_numba(f, grid_dim, block_dim, shared_memory_size, ncycles, input_d, output_d)
        cuda.synchronize()
        data_test = output_d.copy_to_host()
        error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
        assert error < 1e-5
        print(f"Performance (elements_per_thread={elements_per_thread}, ffts_per_block={ffts_per_block}): {time_ms} [ms.]")


if __name__ == "__main__":
    main()
