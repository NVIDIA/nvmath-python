# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example using a process pool to launch multiple independent FFT operations in parallel on multiple GPUs.

The NumPy ndarrays reside in CPU memory. There are two ways to process CPU tensors with
nvmath: either use host library to process the tensor directly or copy the tensor to GPU
memory and process it with cuFFT.

The default behaviour has changed in Beta2, the cpu arrays will default to processing with
host library: NVPL (Nvidia Performance Libraries), MKL or any other FFTW3-compatible
library. In this example, we explicitly set ``execution="cuda"``, to copy the data on
GPU for processing with cuFFT.
"""

import multiprocessing

import numpy as np

import nvmath


def runner(arg):
    operand, axes, device_id = arg
    return nvmath.fft.fft(
        operand,
        axes=axes,
        execution={"name": "cuda", "device_id": device_id},
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    shape = 256, 512, 512
    axes = 0, 1

    # Creating two input operands
    a = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    b = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    args = (a, axes, 0), (b, axes, 1)  # two args with different device id
    # Launching two independent FFT operations using thread pool
    with multiprocessing.Pool(processes=min(len(args), 4)) as pool:
        r = pool.map(runner, args, chunksize=1)

        for i, (fft_input, fft_output) in enumerate(zip(args, r, strict=True)):
            print(f"Input {i} type = {type(fft_input)}, FFT output type = {type(fft_output)}")
