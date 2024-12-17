# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This program uses the device CURAND API to calculate what proportion of quasi-random 3D
points fall within a sphere of radius 1, and to derive the volume of the sphere.

In particular it uses 64 bit scrambled Sobol direction vectors returned by the host helper
API `get_direction_vectors64` to generate double-precision uniform samples. The host helper
APIs can be accessed from the `nvmath.device.random.random_helpers` module.

See https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example for the
corresponding C example.
"""

import cffi

import numpy as np
from numpy.testing import assert_allclose

from numba import cuda
from nvmath.device import random

# Various parameters

ndim = 3  # Each thread uses three random streams.

threads = 64
blocks = 64
nthreads = blocks * threads

sample_count = 10000
repetitions = 50

# Compile the random device APIs for the current device.
compiled_apis = random.Compile(cc=None)


def test_sobol_scramble():
    ffi = cffi.FFI()

    @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
    def setup(sobolDirectionVectors, sobolScrambleConstants, states):
        id = cuda.grid(1)
        offset = ndim * id

        for z in range(ndim):
            dirptr = ffi.from_buffer(sobolDirectionVectors[(offset + z) :])
            random.init(
                dirptr,
                sobolScrambleConstants[offset + z],
                1234,
                states[offset + z],
            )

    @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
    def count_within_unit_sphere(states, n, result):
        id = cuda.grid(1)
        offset = ndim * id
        count = 0

        for sample in range(n):
            x = random.uniform_double(states[offset])
            y = random.uniform_double(states[offset + 1])
            z = random.uniform_double(states[offset + 2])

            if x * x + y * y + z * z < 1.0:
                count += 1

        result[id] += count

    # The direction vectors and scramble constants are initialized on the host, using the
    # helper functions in the `nvmath.device.random.random_helpers` module.
    hostVectors = random.random_helpers.get_direction_vectors64(
        random.random_helpers.DirectionVectorSet.SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
        nthreads * ndim,
    )
    sobolDirectionVectors = cuda.to_device(hostVectors)

    hostScrambleConstants = random.random_helpers.get_scramble_constants64(nthreads * ndim)
    sobolScrambleConstants = cuda.to_device(hostScrambleConstants)

    states = random.StatesScrambledSobol64(ndim * nthreads)

    devResult = cuda.to_device(np.zeros(nthreads, dtype=np.int32))

    setup[blocks, threads](sobolDirectionVectors, sobolScrambleConstants, states)

    for i in range(repetitions):
        count_within_unit_sphere[blocks, threads](states, sample_count, devResult)

    result = devResult.copy_to_host()

    total = 0
    for i in range(nthreads):
        total += result[i]

    fraction = np.float64(total) / np.float64(sample_count * nthreads * repetitions)

    assert_allclose(fraction * 8.0, 4.0 / 3 * np.pi, atol=1e-4)


test_sobol_scramble()
