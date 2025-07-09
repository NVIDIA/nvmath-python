# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates the use of a user-provided logger.
"""

import logging

import cupy as cp
import cupyx.scipy.sparse as sp

import nvmath

# Create and configure a user logger.
# Any of the features provided by the logging module can be used.
logger = logging.getLogger("userlogger")
logging.getLogger().setLevel(logging.NOTSET)

# Create a console handler for the logger and set level.
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create a formatter and associate with handler.
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")
handler.setFormatter(formatter)

# Associate handler with logger, resulting in a logger with the desired level, format, and
# console output.
logger.addHandler(handler)

# The number of equations.
n = 8

# Prepare sample input data.
# Create a diagonally-dominant random CSR matrix.
a = sp.random(n, n, density=0.5, format="csr", dtype="float64")
a += sp.diags([2.0] * n, format="csr", dtype="float64")

# Create the RHS, which can be a matrix or vector in column-major layout.
b = cp.ones((n, 2), order="F")

# Specify the custom logger in the sparse direct solver options.
o = nvmath.sparse.advanced.DirectSolverOptions(logger=logger)
# Specify the options to the sparse direct solver operation.
x = nvmath.sparse.advanced.direct_solver(a, b, options=o)

print("---")

# Recall that the options can also be provided as a dict, so the following is an
#   alternative, entirely equivalent way to specify options.
x = nvmath.sparse.advanced.direct_solver(a, b, options={"logger": logger})

# Synchronize the default stream, since by default the execution is non-blocking for GPU
# operands.
cp.cuda.get_current_stream().synchronize()

print(x)

# Check if the result is cupy array as well.
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(x)}.")
assert isinstance(x, cp.ndarray)
