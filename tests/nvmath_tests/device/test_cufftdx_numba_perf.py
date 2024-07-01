# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from nvmath.device import CodeType, FFTOptions
from .helpers import smallest_multiple, time_check_cupy, set_device, random_complex
from ..helpers import fft_conv_perf_GFlops, print_aligned_table
import cupy
from .cpp_conv import FFTConvCpp
from .numba_conv import FFTConvNumba
import argparse

TEST_CASES = [ \

    (4,    np.float32),
    (8,    np.float32),
    (16,   np.float32),
    (32,   np.float32),
    (64,   np.float32),
    (128,  np.float32),
    (256,  np.float32),
    (512,  np.float32),
    (1024, np.float32),
    (2048, np.float32),
    (4096, np.float32),

    (4,    np.float64),
    (8,    np.float64),
    (16,   np.float64),
    (32,   np.float64),
    (64,   np.float64),
    (128,  np.float64),
    (256,  np.float64),
    (512,  np.float64),
    (1024, np.float64),
    (2048, np.float64),

]

def run_conv_perf(test_cases):

    cols = ['size', 'batch', 'precision', 'cupy [ms]', 'Numba [ms]', 'Nb+vec [ms]', 'C++ [ms]', 'cupy [GFlop/s]', 'Numba [GFlop/s]', 'Nb+vec [GFlop/s]', 'C++ [GFlop/s]']

    data = []

    SM = set_device()
    ncycles = 10

    for (size, precision) in test_cases:

        # Figure out EPT/BPB
        BASE = FFTOptions(fft_type='c2c', size=size, precision=precision, direction='forward', \
                        elements_per_thread='suggested', ffts_per_block='suggested', \
                        execution='Block', code_type=CodeType('lto', (SM[0], SM[1])))

        ffts_per_block = BASE.ffts_per_block
        elements_per_thread = BASE.elements_per_thread
        min_batch = (1024 * 1024 * 64) // size
        batch = smallest_multiple(min_batch, ffts_per_block)
        assert batch % ffts_per_block == 0
        assert batch >= min_batch

        print(f"Numba vs cupy host APIs vs CUDA C++ (convolution), batch = {batch}, size = {size}, precision = {precision}, bpb = {ffts_per_block}, ept = {elements_per_thread}")

        #
        # cupy
        #
        input = random_complex((batch, size), precision, cupy)
        filter = random_complex((batch, size), precision, cupy)
        fun = lambda input, filter: cupy.fft.ifft(cupy.fft.fft(input, axis=-1) * filter, norm='forward', axis=-1)
        reference = fun(input, filter)

        t_cupy_ms = time_check_cupy(fun, reference, ncycles, input, filter)

        #
        # Numba
        #
        numba_test = FFTConvNumba(size=size, precision=precision, fft_type='c2c', ffts_per_block=ffts_per_block, elements_per_thread=elements_per_thread, use_vectorized_load_store=False)
        t_numba = numba_test.run(input=input, filter=filter, reference=reference, ncycles=ncycles)

        #
        # Numba vectorized
        #
        numba_vectorized_test = FFTConvNumba(size=size, precision=precision, fft_type='c2c', ffts_per_block=ffts_per_block, elements_per_thread=elements_per_thread, use_vectorized_load_store=True)
        t_numba_vectorized = numba_vectorized_test.run(input=input, filter=filter, reference=reference, ncycles=ncycles)

        #
        # CUDA C++
        #
        test = FFTConvCpp(size=size, precision=precision, fft_type='c2c', sm=SM, ffts_per_block=ffts_per_block, elements_per_thread=elements_per_thread)
        t_cpp = test.run(input=input, filter=filter, reference=reference, ncycles=ncycles)

        data.append({
            'size': size,
            'batch': batch,
            'precision': precision,
            'cupy [ms]': t_cupy_ms['time_ms'],
            'cupy [GFlop/s]': fft_conv_perf_GFlops(size, batch, t_cupy_ms['time_ms']),
            'Numba [ms]': t_numba['time_ms'],
            'Numba [GFlop/s]': fft_conv_perf_GFlops(size, batch, t_numba['time_ms']),
            'Nb+vec [ms]': t_numba_vectorized['time_ms'],
            'Nb+vec [GFlop/s]': fft_conv_perf_GFlops(size, batch, t_numba_vectorized['time_ms']),
            'C++ [ms]': t_cpp['time_ms'],
            'C++ [GFlop/s]': fft_conv_perf_GFlops(size, batch, t_cpp['time_ms'])
        })

    print_aligned_table(cols, data)

def test():

    run_conv_perf(TEST_CASES)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--precision", type=str, default='float32')
    args = parser.parse_args()

    size = args.size
    PREC_MAP = {
        'float32': np.float32,
        'float64': np.float64
    }
    precision = PREC_MAP[args.precision]

    run_conv_perf([(size, precision)])