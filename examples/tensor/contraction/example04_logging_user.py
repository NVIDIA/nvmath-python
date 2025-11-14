# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of a user-provided Python logger to observe the
computational details of a binary tensor contraction operation.
"""

import logging

import cupy as cp

import nvmath

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


a = cp.random.rand(4, 4, 12, 12)
b = cp.random.rand(12, 12, 8, 8)

c = cp.random.rand(4, 4, 8, 8)

# result[i,j,m,n] = \sum_{k,l} a[i,j,k,l] * b[k,l,m,n] + c[i,j,m,n]
result = nvmath.tensor.binary_contraction("ijkl,klmn->ijmn", a, b, c=c, beta=1, options={"logger": logger})

assert cp.allclose(result, cp.einsum("ijkl,klmn->ijmn", a, b) + c)

print(f"Input type = {type(a), type(b), type(c)}, contraction result type = {type(result)}")
