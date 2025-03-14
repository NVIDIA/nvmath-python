# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example is a slight modification of the block FFT example `cufftdx_block_fft.py`, and
shows how to use RNG device APIs (nvmath.device.random) within a kernel in conjunction with
an FFT.
"""

import numpy as np
from numba import cuda
from nvmath.device import fft, random, float32x2

# Compile the random device APIs for the current device.
compiled_random_apis = random.Compile(cc=None)


def main():
    size = 64

    FFT = fft(fft_type="c2c", size=size, precision=np.float32, direction="forward", execution="Block", compiler="numba")

    size = FFT.size
    value_type = FFT.value_type
    storage_size = FFT.storage_size
    shared_memory_size = FFT.shared_memory_size
    stride = FFT.stride
    block_dim = FFT.block_dim
    ffts_per_block = FFT.ffts_per_block
    elements_per_thread = FFT.elements_per_thread

    # Kernel for initializing the RNG state.
    @cuda.jit(link=compiled_random_apis.files, extensions=compiled_random_apis.extension)
    def setup_random(states):
        index_in_block = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y
        random.init(1234, index_in_block, 0, states[index_in_block])

    # Kernel with threads generating local data and performing an FFT.
    @cuda.jit(link=FFT.files + compiled_random_apis.files, extensions=compiled_random_apis.extension)
    def f(data, result, states):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)

        local_fft_id = cuda.threadIdx.y
        fft_id = cuda.blockIdx.x * ffts_per_block + local_fft_id

        index = cuda.threadIdx.x
        index_in_block = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y
        for i in range(elements_per_thread):
            v1 = random.normal(states[index_in_block])
            v2 = random.normal(states[index_in_block])
            data[fft_id, index] = thread_data[i] = float32x2(v1, v2)
            index += stride

        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)
        FFT(thread_data, shared_mem)

        index = cuda.threadIdx.x
        for i in range(elements_per_thread):
            result[fft_id, index] = thread_data[i]
            index += stride

    # Create host and device buffers to hold the input data and result, respectively.
    data = np.empty((ffts_per_block, size), dtype=np.complex64)
    data_d = cuda.to_device(data)
    result = np.empty((ffts_per_block, size), dtype=np.complex64)
    result_d = cuda.to_device(result)

    states = random.StatesXORWOW(block_dim.x * block_dim.y)
    setup_random[1, block_dim](states)

    f[1, block_dim, 0, shared_memory_size](data_d, result_d, states)
    cuda.synchronize()

    data = data_d.copy_to_host()
    result_test = result_d.copy_to_host()

    result_ref = np.fft.fft(data, axis=-1)
    error = np.linalg.norm(result_test - result_ref) / np.linalg.norm(result_ref)
    assert error < 1e-5
    print(f"The results match against the reference, with relative error = {error:.3e}.")


if __name__ == "__main__":
    main()
