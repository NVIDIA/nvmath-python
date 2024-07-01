# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import cupy
import nvmath
import numpy as np

from ..helpers import time_cupy, random_complex, print_aligned_table, fft_perf_GFlops

def test_fft():

    cols = ['axes', 'size', 'batch', 'dataset_size [MiB]', 'cupy [ms]', 'nvmath-python [ms]', 'cupy [GFlop/s]', 'nvmath-python [GFlop/s]', 'speedup nvmath-python over cupy']

    data = []

    count = 2**24
    ncycles = 10
    precision = np.float32

    for axes in [(0,1), (1,2)]:

        for size1 in [2 ** i for i in range(15)]:

            size2 = max(1, size1 // 2)
            size3 = max(count // (size1 * size2), 1)
            sizes = [size3, size2, size1]

            fft_size = int(np.prod([sizes[a] for a in axes]))
            batch_size = int(np.prod([sizes[a] for a in range(3) if a not in axes]))

            data_in = random_complex(sizes, precision, module=cupy)

            with nvmath.fft.FFT(data_in, axes=axes) as f:

                f.plan()

                time_nvmath = time_cupy(lambda: f.execute(), ncycles)

            time_cp = time_cupy(lambda x: cupy.fft.fftn(x, axes=axes), ncycles, data_in)

            data.append({
                'axes': str(axes),
                'size': str([sizes[a] for a in axes]),
                'batch': batch_size,
                'nvmath-python [ms]': time_nvmath['time_ms'],
                'cupy [ms]': time_cp['time_ms'],
                'dataset_size [MiB]': fft_size * batch_size * precision(1).itemsize / (2**20),
                'cupy [GFlop/s]': fft_perf_GFlops(fft_size, batch_size, time_cp['time_ms']),
                'nvmath-python [GFlop/s]': fft_perf_GFlops(fft_size, batch_size, time_nvmath['time_ms']),
                'speedup nvmath-python over cupy': time_cp['time_ms'] / time_nvmath['time_ms']
            })

    print("\n")
    print_aligned_table(cols, data)