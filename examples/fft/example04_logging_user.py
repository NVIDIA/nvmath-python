# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of the a user-provided logger.
"""

import logging

import cupy as cp

import nvmath

shape = 512, 512, 256
axes = 0, 1

# Create and configure a user logger.
# Any of the features provided by the logging module can be used.
logger = logging.getLogger("userlogger")
logging.getLogger().setLevel(logging.NOTSET)

# Create a console handler for the logger and set level.
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Create a formatter and associate with handler.
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")
handler.setFormatter(formatter)

# Associate handler with logger, resulting in a logger with the desired level, format, and
# console output.
logger.addHandler(handler)


a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

# Specify the custom logger in the FFT options.
o = nvmath.fft.FFTOptions(logger=logger)

# Specify the options to the FFT operation.
b = nvmath.fft.fft(a, axes=axes, options=o)

print("---")

# Recall that the options can also be provided as a dict, so the following is an
# alternative, entirely equivalent way to specify options.
b = nvmath.fft.fft(a, axes=axes, options={"logger": logger})

# Synchronize the default stream
cp.cuda.get_current_stream().synchronize()
print(f"Input type = {type(a)}, device = {a.device}")
print(f"FFT output type = {type(b)}, device = {b.device}")
