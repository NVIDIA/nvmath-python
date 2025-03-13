# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to use the Philox4_32_10 bit generator to sample four
single-precision values from a uniform distribution in a single call. The values are wrapped
into an object of type `uint32x4` (Numba vector type).

Following recommended practice, the implementation is split into a state initialization
kernel and a sample generation kernel.
"""

import numpy as np

from numba import cuda

from nvmath.device import random

# Various parameters

threads = 64
blocks = 64
nthreads = blocks * threads

sample_count = 10000
repetitions = 50

# Compile the random device APIs for the current device.
compiled_apis = random.Compile(cc=None)


def test_random_uniform4():
    # State initialization kernel
    @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
    def setup(states):
        i = cuda.grid(1)
        random.init(1234, i, 0, states[i])

    # Generation kernel
    @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
    def count_upper_half(states, n, result):
        i = cuda.grid(1)
        count = 0

        # Count the number of samples that falls greater than 0.5, getting 4 values at a
        # time.
        for sample in range(n // 4):
            v = random.uniform4(states[i])
            a = v.x, v.y, v.z, v.w
            for j in range(4):
                if a[j] > 0.5:
                    count += 1

        result[i] += count

    states = random.StatesPhilox4_32_10(nthreads)
    setup[blocks, threads](states)

    results = cuda.to_device(np.zeros(nthreads, dtype=np.int32))

    for i in range(repetitions):
        count_upper_half[blocks, threads](states, sample_count, results)

    host_results = results.copy_to_host()

    total = 0
    for i in range(nthreads):
        total += host_results[i]

    fraction = np.float32(total) / np.float32(sample_count * nthreads * repetitions)

    print(fraction)
    print(np.isclose(fraction, 0.5, atol=1e-4))
    assert np.isclose(fraction, 0.5, atol=1e-4)


test_random_uniform4()
