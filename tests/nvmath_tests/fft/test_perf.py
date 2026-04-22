# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import nvmath

try:
    import cupy
except ImportError:
    cupy = None

if cupy is not None:
    from ..helpers import fft_perf_GFlops, print_aligned_table, random_complex, time_cupy

    def test_fft():
        return _test_fft()


def _test_fft():
    cols = [
        "axes",
        "size",
        "batch",
        "dataset_size [MiB]",
        "cupy [ms]",
        "nvmath-python [ms]",
        "cupy [GFlop/s]",
        "nvmath-python [GFlop/s]",
        "speedup nvmath-python over cupy",
    ]

    data = []

    count = 2**24  # 16 2^20 ~= 8M elements
    ncycles = 10
    precision = np.float32

    for ax in [0, -1]:
        for size in [2**i for i in range(28)]:
            batch = max(count // size, 1)
            data_in = random_complex((batch, size), precision, module=cupy)

            with nvmath.fft.FFT(data_in, axes=[ax]) as f:
                f.plan()

                time_nvmath = time_cupy(lambda: f.execute(), ncycles)

            time_cp = time_cupy(lambda x, ax=ax: cupy.fft.fftn(x, axes=[ax]), ncycles, data_in)

            data.append(
                {
                    "axes": ax,
                    "size": size,
                    "batch": batch,
                    "nvmath-python [ms]": time_nvmath["time_ms"],
                    "cupy [ms]": time_cp["time_ms"],
                    "dataset_size [MiB]": size * batch * precision(1).itemsize / (2**20),
                    "cupy [GFlop/s]": fft_perf_GFlops(size, batch, time_cp["time_ms"]),
                    "nvmath-python [GFlop/s]": fft_perf_GFlops(size, batch, time_nvmath["time_ms"]),
                    "speedup nvmath-python over cupy": time_cp["time_ms"] / time_nvmath["time_ms"],
                }
            )

    print("\n")
    print_aligned_table(cols, data)
