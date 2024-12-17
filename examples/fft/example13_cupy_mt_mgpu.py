# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example using a thread pool to launch multiple independent FFT operations in parallel on
multiple GPUs.
"""

from functools import partial
import multiprocessing.dummy

import cupy as cp

import nvmath

if __name__ == "__main__":
    shape = 256, 512, 512
    axes = 0, 1

    # Creating two input operands on two different devices
    with cp.cuda.Device(0):
        a = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)
    with cp.cuda.Device(1):
        b = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)

    fft = partial(nvmath.fft.fft, axes=axes)
    args = a, b
    # Launching two independent FFT operations using thread pool
    with multiprocessing.dummy.Pool(processes=min(len(args), 4)) as pool:
        r = pool.map(fft, args, chunksize=1)

        for i, (fft_input, fft_output) in enumerate(zip(args, r, strict=True)):
            # Synchronize the default stream
            cp.cuda.get_current_stream(i).synchronize()
            print(f"Input {i} type = {type(fft_input)}, device = {fft_input.device}")
            print(f"FFT output {i} type = {type(fft_output)}, device = {fft_output.device}")
