# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import functools

import numpy as np
from common import random_real
from numba import cuda

from nvmath.device import FFT
from nvmath.device.types import complex64


def main():
    FFT_base = functools.partial(
        FFT,
        size=64,
        precision=np.float32,
        ffts_per_block=2,
        elements_per_thread=4,
        real_fft_options={"complex_layout": "packed", "real_mode": "folded"},
        execution="Block",
    )
    fft = FFT_base(fft_type="r2c")
    ifft = FFT_base(fft_type="c2r")

    assert fft.value_type == complex64
    assert fft.storage_size == 2
    assert fft.ffts_per_block == 2
    assert fft.stride == 16
    assert fft.size == 64
    assert fft.elements_per_thread == 4
    assert fft.block_dim == (16, 2, 1)

    assert fft.value_type == ifft.value_type
    assert fft.precision == ifft.precision
    assert fft.storage_size == ifft.storage_size
    assert fft.shared_memory_size == ifft.shared_memory_size
    assert fft.ffts_per_block == ifft.ffts_per_block
    assert fft.stride == ifft.stride
    assert fft.size == ifft.size
    assert fft.elements_per_thread == ifft.elements_per_thread
    assert fft.block_dim == ifft.block_dim

    @cuda.jit
    def f(inout):
        # Registers
        complex_thread_data = cuda.local.array(shape=(fft.storage_size,), dtype=fft.value_type)
        real_thread_data = complex_thread_data.view(np.float32)

        # Figure out fft / batch IDs
        local_fft_id = cuda.threadIdx.y
        global_fft_id = (cuda.blockIdx.x * fft.ffts_per_block) + local_fft_id

        for i in range(fft.elements_per_thread):
            idx = i * fft.stride + cuda.threadIdx.x
            if idx < fft.size // 2:
                # Fold optimized, so we load complex (ie 2 consecutive reals) instead of
                # reals
                real_thread_data[2 * i + 0] = inout[global_fft_id, 2 * idx + 0]
                real_thread_data[2 * i + 1] = inout[global_fft_id, 2 * idx + 1]

        # Allocate shared
        shared_mem = cuda.shared.array(shape=(0,), dtype=fft.value_type)

        # R2C
        fft.execute(complex_thread_data, shared_mem)

        # Normalize
        # `complex_thread_data` has a packed (not natural) layout
        for i in range(fft.elements_per_thread):
            if i * fft.stride + cuda.threadIdx.x < fft.size // 2:
                complex_thread_data[i] = complex_thread_data[i] / fft.size

        # C2R
        ifft.execute(complex_thread_data, shared_mem)

        # Save results
        for i in range(fft.elements_per_thread):
            idx = i * fft.stride + cuda.threadIdx.x
            if idx < fft.size // 2:
                # Fold optimized, so we load complex (ie 2 consecutive reals) instead of
                # reals
                inout[global_fft_id, 2 * idx + 0] = real_thread_data[2 * i + 0]
                inout[global_fft_id, 2 * idx + 1] = real_thread_data[2 * i + 1]

    input = random_real((fft.ffts_per_block, fft.size), real_dtype=np.float64)
    inout_d = cuda.to_device(input)

    f[1, fft.block_dim, 0, fft.shared_memory_size](inout_d)
    cuda.synchronize()

    output_test = inout_d.copy_to_host()
    output_ref = input

    error = np.linalg.norm(output_test - output_ref) / np.linalg.norm(output_ref)
    print(f"L2 error {error}")
    assert error < 1e-5


if __name__ == "__main__":
    main()
