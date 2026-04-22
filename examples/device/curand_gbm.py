# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to generate the Geometric Brownian Motion (GBM) for stock prices
simulation. The normal distribution sample generator is used in combination with the
Philox4_32_10 bit generator, which is converted into GBM in the numba.cuda kernel code.

Following recommended practice, the implementation is split into a state initialization
kernel and a path generation kernel. The generation kernel leverages that the Philox4_32_10
generator returns 4 variates at a time allowing to produce 4 time steps in a single loop
iteration.

To learn more about the GBM, see: https://en.wikipedia.org/wiki/Geometric_Brownian_motion.
"""

import math

import cupy as cp
import numpy as np
from numba import cuda
from scipy import stats

from nvmath.device import random

# Pre-compile the random number generator into IR to use alongside other device code
compiled_rng = random.Compile(cc=None)

# GBM parameters
rng_seed = 7777  # Random seed
n_time_steps = 252  # Trading days in a year
n_paths = 8_000  # Number of simulated paths
mu = 0.003
sigma = 0.027
s0 = 100.0  # Initial stock price

# Set up CUDA kernel launch configuration
threads_per_block = 32
blocks = n_paths // threads_per_block + bool(n_paths % threads_per_block)
nthreads = threads_per_block * blocks


# RNG initialization kernel
@cuda.jit(link=compiled_rng.files, extensions=compiled_rng.extension)
def init_rng_gpu(states, seed):
    idx = cuda.grid(1)
    random.init(seed, idx, 0, states[idx])


# GBM path generation kernel. Note that the random numbers are generated
# as they are needed, unlike for the CPU implementation where they are
# generated upfront and stored.
@cuda.jit(link=compiled_rng.files, extensions=compiled_rng.extension)
def generate_gbm_paths_gpu(states, paths, nsteps, mu, sigma, s0):
    idx = cuda.grid(1)
    if idx >= paths.shape[0]:
        return

    # Each thread generates one path in the time domain
    paths[idx, 0] = s0

    # Consume 4 normal variates at a time for better throughput
    for i in range(1, nsteps, 4):
        v = random.normal4(states[idx])  # Returned as float32x4 type
        vals = v.x, v.y, v.z, v.w  # Decompose into a tuple of float32
        for j in range(i, min(i + 4, nsteps)):  # Process a chunk of 4 time steps
            paths[idx, j] = paths[idx, j - 1] * math.exp(mu + sigma * vals[j - i])


# Reference GBM path generation on CPU for validation
def generate_gbm_paths_cpu(npaths, nsteps, mu, sigma, s0, seed):
    """Generate GBM paths on the CPU and return paths_host.

    This function internally generates Brownian increments and accumulates them to
    form the GBM paths (so there's no separate generate_brownian_paths function).
    """
    np.random.seed(seed)
    dBt = np.random.randn(npaths, nsteps - 1) * sigma + mu
    dBt = np.insert(dBt, 0, 0.0, axis=1)  # The process starts at 0
    Bt = np.cumsum(dBt, axis=1)
    paths = s0 * np.exp(Bt)
    return paths


def main():
    # Allocate space for paths
    paths_device = cp.empty((n_paths, n_time_steps), dtype=cp.float32, order="F")

    # Allocate space for random states
    states = random.StatesPhilox4_32_10(nthreads)

    # Initialize RNG states for GPU and CPU
    init_rng_gpu[blocks, threads_per_block](states, rng_seed)

    # Generate GBM paths on GPU
    generate_gbm_paths_gpu[blocks, threads_per_block](states, paths_device, n_time_steps, mu, sigma, s0)

    mean_device = cp.mean(paths_device[:, -1])
    stdev_device = cp.std(paths_device[:, -1])
    print(f"Mean stock price at maturity (GPU): {mean_device:.2f}, std.dev.: {stdev_device:.2f}")

    # Generate reference GBM paths on CPU
    paths_host = generate_gbm_paths_cpu(n_paths, n_time_steps, mu, sigma, s0, rng_seed)
    mean_host = np.mean(paths_host[:, -1])
    stdev_host = np.std(paths_host[:, -1])
    print(f"Mean stock price at maturity (CPU): {mean_host:.2f}, std.dev.: {stdev_host:.2f}")

    # Validate results
    _, p_value = stats.levene(paths_host[:, -1], paths_device.get()[:, -1])  # Leven's test for equal variances
    assert p_value > 0.05, "The variances are not equal (reject H0) - the test FAILED"

    _, p_value = stats.ttest_ind(paths_host[:, -1], paths_device.get()[:, -1], equal_var=False)  # T-test for equal means
    assert p_value > 0.05, "The means are not equal (reject H0) - the test FAILED"


if __name__ == "__main__":
    main()
