# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from nvmath.device import fft, float32x2_type
from common import random_real


def main():
    FFT_r2c = fft(
        fft_type="r2c",
        size=64,
        precision=np.float32,
        ffts_per_block=2,
        elements_per_thread=4,
        real_fft_options={"complex_layout": "packed", "real_mode": "folded"},
        execution="Block",
        compiler="numba",
    )

    FFT_c2r = fft(
        fft_type="c2r",
        size=64,
        precision=np.float32,
        ffts_per_block=2,
        elements_per_thread=4,
        real_fft_options={"complex_layout": "packed", "real_mode": "folded"},
        execution="Block",
        compiler="numba",
    )

    complex_type = FFT_r2c.value_type
    storage_size = FFT_r2c.storage_size
    shared_memory_size = FFT_r2c.shared_memory_size
    ffts_per_block = FFT_r2c.ffts_per_block
    stride = FFT_r2c.stride
    size = FFT_r2c.size
    elements_per_thread = FFT_r2c.elements_per_thread
    block_dim = FFT_r2c.block_dim

    assert complex_type == float32x2_type
    assert storage_size == 2
    assert ffts_per_block == 2
    assert all([file.endswith(".ltoir") for file in FFT_r2c.files])
    assert stride == 16
    assert size == 64
    assert elements_per_thread == 4
    assert block_dim == (16, 2, 1)

    assert FFT_r2c.value_type == FFT_c2r.value_type
    assert FFT_r2c.precision == FFT_c2r.precision
    assert FFT_r2c.storage_size == FFT_c2r.storage_size
    assert FFT_r2c.shared_memory_size == FFT_c2r.shared_memory_size
    assert FFT_r2c.ffts_per_block == FFT_c2r.ffts_per_block
    assert FFT_r2c.stride == FFT_c2r.stride
    assert FFT_r2c.size == FFT_c2r.size
    assert FFT_r2c.elements_per_thread == FFT_c2r.elements_per_thread
    assert FFT_r2c.block_dim == FFT_c2r.block_dim

    @cuda.jit(link=FFT_r2c.files + FFT_c2r.files)
    def f(inout):
        # Registers
        complex_thread_data = cuda.local.array(shape=(storage_size,), dtype=complex_type)
        real_thread_data = complex_thread_data.view(np.float32)

        # Figure out fft / batch IDs
        local_fft_id = cuda.threadIdx.y
        global_fft_id = (cuda.blockIdx.x * ffts_per_block) + local_fft_id

        for i in range(elements_per_thread):
            idx = i * stride + cuda.threadIdx.x
            if idx < size // 2:
                # Fold optimized, so we load complex (ie 2 consecutive reals) instead of
                # reals
                real_thread_data[2 * i + 0] = inout[global_fft_id, 2 * idx + 0]
                real_thread_data[2 * i + 1] = inout[global_fft_id, 2 * idx + 1]

        # Allocate shared
        shared_mem = cuda.shared.array(shape=(0,), dtype=complex_type)

        # R2C
        FFT_r2c(complex_thread_data, shared_mem)

        # Normalize
        # `complex_thread_data` has a packed (not natural) layout
        for i in range(elements_per_thread):
            if i * stride + cuda.threadIdx.x < size // 2:
                complex_thread_data[i] = complex_thread_data[i] / size

        # C2R
        FFT_c2r(complex_thread_data, shared_mem)

        # Save results
        for i in range(elements_per_thread):
            idx = i * stride + cuda.threadIdx.x
            if idx < size // 2:
                # Fold optimized, so we load complex (ie 2 consecutive reals) instead of
                # reals
                inout[global_fft_id, 2 * idx + 0] = real_thread_data[2 * i + 0]
                inout[global_fft_id, 2 * idx + 1] = real_thread_data[2 * i + 1]

    input = random_real((ffts_per_block, size), real_dtype=np.float64)
    inout_d = cuda.to_device(input)

    f[1, block_dim, 0, shared_memory_size](inout_d)
    cuda.synchronize()

    output_test = inout_d.copy_to_host()
    output_ref = input

    error = np.linalg.norm(output_test - output_ref) / np.linalg.norm(output_ref)
    print(f"L2 error {error}")
    assert error < 1e-5


if __name__ == "__main__":
    main()
