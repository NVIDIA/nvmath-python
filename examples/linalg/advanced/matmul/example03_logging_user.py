# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates the use of a user-provided logger.
"""
import logging

import cupy as cp

import nvmath

# Create and configure a user logger.
# Any of the features provided by the logging module can be used.
logger = logging.getLogger('userlogger')
logging.getLogger().setLevel(logging.NOTSET)

# Create a console handler for the logger and set level.
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create a formatter and associate with handler.
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
handler.setFormatter(formatter)

# Associate handler with logger, resulting in a logger with the desired level, format, and console output.
logger.addHandler(handler)

# Prepare sample input data.
m, n, k = 64, 128, 256
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)
alpha = 0.45

# Specify the custom logger in the matrix multiplication options.
o = nvmath.linalg.advanced.MatmulOptions(logger=logger)
# Specify the options to the matrix multiplication operation.
result = nvmath.linalg.advanced.matmul(a, b, alpha=alpha, options=o)

print("---")

# Recall that the options can also be provided as a dict, so the following is an alternative, entirely
#   equivalent way to specify options.
result = nvmath.linalg.advanced.matmul(a, b, alpha=alpha, options={'logger': logger})

# Synchronize the default stream, since by default the execution is non-blocking for GPU operands.
cp.cuda.get_current_stream().synchronize()
