# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import numpy as np
from numba import cuda
from nvmath.device import current_device_sm, FFT
from common_numba import time_numba


def main():
    batch = 1024 * 32
    ncycles = 10

    BaseFFT = functools.partial(
        FFT,
        fft_type="c2c",
        size=256,
        precision=np.float32,
        direction="forward",
        execution="Block",
        sm=current_device_sm(),
    )
    fft = BaseFFT()

    data = np.ones((batch, fft.size), dtype=np.complex64)
    data_ref = np.fft.fft(data, axis=-1)

    valid_ept_fpb = fft.valid("elements_per_thread", "ffts_per_block")

    for ept, fpb in valid_ept_fpb:
        fft = BaseFFT(elements_per_thread=ept, ffts_per_block=fpb)

        grid_dim = (batch + fft.ffts_per_block - 1) // fft.ffts_per_block

        assert ept == fft.elements_per_thread
        assert fpb == fft.ffts_per_block

        @cuda.jit
        def f(input, output):
            thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)

            local_fft_id = cuda.threadIdx.y
            fft_id = cuda.blockIdx.x * fft.ffts_per_block + local_fft_id
            if fft_id >= batch:
                return

            index = cuda.threadIdx.x
            for i in range(fft.elements_per_thread):
                thread_data[i] = input[fft_id, index]
                index += fft.stride

            shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)
            fft.execute(thread_data, shared_mem)

            index = cuda.threadIdx.x
            for i in range(fft.elements_per_thread):
                output[fft_id, index] = thread_data[i]
                index += fft.stride

        input_d = cuda.to_device(data)
        output_d = cuda.to_device(data)
        cuda.synchronize()
        time_ms = time_numba(f, grid_dim, fft.block_dim, fft.shared_memory_size, ncycles, input_d, output_d)
        cuda.synchronize()
        data_test = output_d.copy_to_host()
        error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
        assert error < 1e-5
        print(
            f"Performance (elements_per_thread={fft.elements_per_thread}, ffts_per_block={fft.ffts_per_block}): {time_ms} [ms.]"
        )


if __name__ == "__main__":
    main()
