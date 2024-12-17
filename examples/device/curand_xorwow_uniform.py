# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to use the XORWOW bit generator to sample double-precision values
from a uniform distribution.

Following recommended practice, the implementation is split into a state initialization
kernel and a sample generation kernel.
"""

import numpy as np
from numpy.testing import assert_allclose

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


def test_random_uniform():
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

        # Count the number of samples that falls greater than 0.5
        for sample in range(n):
            x = random.uniform_double(states[i])
            if x > 0.5:
                count += 1

        result[i] += count

    states = random.StatesXORWOW(nthreads)
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


test_random_uniform()
