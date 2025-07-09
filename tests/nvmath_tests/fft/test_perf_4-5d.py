# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import itertools
import functools
import sys
import os

from .utils.common_axes import ExecBackend
from .utils.support_matrix import supported_backends

if ExecBackend.cufft in supported_backends.exec:
    import cupy
    from nvmath_tests.helpers import time_cupy, print_aligned_table, fft_perf_GFlops

    def test_fft():
        try:
            SYS_PATH_BACKUP = sys.path.copy()
            samples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")
            sys.path.append(samples_path)
            run_test()
        finally:
            sys.path = SYS_PATH_BACKUP


def run_test():
    from fft.caching import fft as cachedfft, FFTCache
    from fft.fftn2 import fftn

    cols = [
        "axes",
        "size",
        "batch",
        "dataset_size [MiB]",
        "cupy [ms]",
        "nvmath-python [ms]",
        "nvmath-python-caching [ms]",
        "cupy [GFlop/s]",
        "nvmath-python-noncaching [GFlop/s]",
        "nvmath-python [GFlop/s]",
        "speedup nvmath-python over cupy",
    ]
    data = []

    shape = 32, 8, 128, 64, 16
    axes = 0, 1, 2, 3, 4
    ncycles = 10
    precision = np.float32

    a = cupy.random.rand(*shape, dtype=precision) + 1j * cupy.random.rand(*shape, dtype=precision)

    rank = len(shape)
    start = 4

    for num_axes in range(start, rank + 1):
        for axes in [p for c in itertools.combinations(range(rank), num_axes) for p in itertools.permutations(c)]:
            fft_size = int(np.prod([shape[a] for a in axes]))
            batch_size = int(np.prod([shape[a] for a in range(rank) if a not in axes]))

            time_nvmath_noncaching = time_cupy(lambda: fftn(a, axes=axes), ncycles)

            with FFTCache() as cache:
                cached_fft = functools.partial(cachedfft, cache=cache)
                time_nvmath_caching = time_cupy(lambda: fftn(a, axes=axes, engine=cached_fft), ncycles)

            time_cp = time_cupy(lambda: cupy.fft.fftn(a, axes=axes), ncycles)

            data.append(
                {
                    "axes": str(axes),
                    "size": str([shape[a] for a in axes]),
                    "batch": batch_size,
                    "dataset_size [MiB]": np.prod(shape) * precision(1).itemsize / (2**20),
                    "cupy [ms]": time_cp["time_ms"],
                    "nvmath-python [ms]": time_nvmath_noncaching["time_ms"],
                    "nvmath-python-caching [ms]": time_nvmath_caching["time_ms"],
                    "cupy [GFlop/s]": fft_perf_GFlops(fft_size, batch_size, time_cp["time_ms"]),
                    "nvmath-python-noncaching [GFlop/s]": fft_perf_GFlops(
                        fft_size, batch_size, time_nvmath_noncaching["time_ms"]
                    ),
                    "nvmath-python [GFlop/s]": fft_perf_GFlops(fft_size, batch_size, time_nvmath_caching["time_ms"]),
                    "speedup nvmath-python over cupy": time_cp["time_ms"] / time_nvmath_caching["time_ms"],
                }
            )

    print("\n")
    print_aligned_table(cols, data)
