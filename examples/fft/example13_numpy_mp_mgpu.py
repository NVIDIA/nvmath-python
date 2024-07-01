# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example using a process pool to launch multiple independent FFT operations in parallel on multiple GPUs.
"""
import multiprocessing

import numpy as np

import nvmath

def runner(arg):
    operand, axes, device_id = arg
    return nvmath.fft.fft(operand, axes=axes, options={'device_id': device_id})

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    shape = 256, 512, 512
    axes  = 0, 1

    # Creating two input operands
    a = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    b = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    args = (a, axes, 0), (b, axes, 1) # two args with different device id
    # Launching two independent FFT operations using thread pool
    with multiprocessing.Pool(processes=min(len(args), 4)) as pool:
        r = pool.map(runner, args, chunksize=1)

        for i, (fft_input, fft_output) in enumerate(zip(args, r)):
            print(f"Input {i} type = {type(fft_input)}, FFT output type = {type(fft_output)}")
