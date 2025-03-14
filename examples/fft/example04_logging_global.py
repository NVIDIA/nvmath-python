# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of the global Python logger.
"""

import logging

import cupy as cp

import nvmath

shape = 512, 512, 256
axes = 0, 1

# Turn on logging. Here we use the global logger, set the level to "debug", and use a custom
# format for the log. Any of the features provided by the logging module can be used.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

# Forward FFT along the specified axes, batched along the complement.
b = nvmath.fft.fft(a, axes=axes)

# Synchronize the default stream
cp.cuda.get_current_stream().synchronize()
print(f"Input type = {type(a)}, device = {a.device}")
print(f"FFT output type = {type(b)}, device = {b.device}")
