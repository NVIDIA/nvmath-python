# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import nvmath.device.random as R
from functools import cache
from .generators import *
from .compiled_apis import compiled_apis

NP_DTYPES = {
    "float": np.float32,
    "double": np.float64,
    "uint32": np.uint32,
}


@cache
def _make_random_kernel(curand_distribution, group_size):
    """
    Returns cuda kernel generating random numbers from the specified distribution.
    """

    @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
    def random_kernel(states, nsamples, result, curand_distribution_args):
        """
        CUDA kernel generating random numbers from the specified distribution.
        """

        i = cuda.grid(1)

        for sample_idx in range(nsamples // group_size):
            gen = curand_distribution(states[i], *curand_distribution_args)
            if group_size == 1:
                result[i][sample_idx] = gen
            elif group_size == 2:
                result[i][2 * sample_idx] = gen.x
                result[i][2 * sample_idx + 1] = gen.y
            elif group_size == 4:
                result[i][4 * sample_idx] = gen.x
                result[i][4 * sample_idx + 1] = gen.y
                result[i][4 * sample_idx + 2] = gen.z
                result[i][4 * sample_idx + 3] = gen.w

    return random_kernel


def prepare_states(
    *,
    generator: Generator,
    seed,
    threads,
    blocks,
    offset=0,
):
    """
    Initializes states an returns them.
    """
    nthreads = blocks * threads
    states = generator.curand_states(nthreads)

    generator.curand_setup(blocks=blocks, threads=threads, seed=seed, states=states, offset=offset)
    return states


def generate_random_numbers(
    *,
    states,
    distribution,
    dtype_name,
    nsamples,
    threads,
    blocks,
    group_size=1,
):
    """
    Runs numba kernel generating random numbers from the specified distribution and states.
    Each thread generates `nsample` values. The result is a numpy array of shape
    (threads*blocks, nsamples).
    """
    assert nsamples % group_size == 0
    nthreads = blocks * threads
    curand_distribution, curand_distribution_args = distribution.curand(dtype_name, group_size)
    results = cuda.to_device(np.zeros((nthreads, nsamples), dtype=NP_DTYPES[dtype_name]))
    kernel = _make_random_kernel(curand_distribution, group_size)
    kernel[blocks, threads](states, nsamples, results, curand_distribution_args)
    return results.copy_to_host()


@cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
def skipahead_kernel(states, ns):
    """
    Skips ns[i] values in i-th thread.
    """
    i = cuda.grid(1)
    R.skipahead(ns[i], states[i])


def per_thread_skipahead(
    *,
    blocks,
    threads,
    states,
    ns,
):
    """
    Runs numba kernel skipping ns[i] values in i-th thread. ns should be a numpy array.
    """
    skipahead_kernel[blocks, threads](states, cuda.to_device(ns))


@cache
def _make_skipahead_sequence_kernel(skipahead_function):
    @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
    def skipahead_sequence_kernel(states, ns):
        """
        Skips ns[i] sequences in i-th thread.
        """
        i = cuda.grid(1)
        skipahead_function(ns[i], states[i])

    return skipahead_sequence_kernel


def per_thread_skipahead_sequence(
    *,
    generator,
    blocks,
    threads,
    states,
    ns,
):
    """
    Runs numba kernel skipping ns[i] sequences in i-th thread. ns should be a numpy array.
    """
    kernel = _make_skipahead_sequence_kernel(generator.get_skipahead_subsequence_function())
    kernel[blocks, threads](states, cuda.to_device(ns))


def prepare_states_and_generate(
    *,
    generator: Generator,
    seed,
    threads,
    blocks,
    offset=0,
    distribution,
    dtype_name,
    nsamples,
    group_size=1,
):
    """
    A wrapper for prepare_states and generate_random_numbers.

    Creates states and runs numba kernel generating random numbers from the specified
    distribution. Each thread generates `nsample` values. The result is a numpy array of
    shape (threads*blocks, nsamples).
    """
    states = prepare_states(generator=generator, seed=seed, threads=threads, blocks=blocks, offset=offset)
    return generate_random_numbers(
        states=states,
        distribution=distribution,
        dtype_name=dtype_name,
        nsamples=nsamples,
        threads=threads,
        blocks=blocks,
        group_size=group_size,
    )


def all_supported_configs(*distributions):
    """
    Returns pytest params representing all configs supported by the distributions.
    Each config is (distribution, dtype, group size, generator).
    """
    configs = []
    for d in distributions:
        for dtype_name, group_size, generator in d.curand_variants():
            configs.append(
                pytest.param(
                    d,
                    dtype_name,
                    group_size,
                    generator,
                    id=f"{d.pretty()}-{dtype_name}-{group_size}-{generator.name()}",
                )
            )
    return configs
