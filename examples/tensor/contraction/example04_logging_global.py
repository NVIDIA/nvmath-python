# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of the global Python logger to observe the
computational details of a binary tensor contraction operation.
"""

import logging

import cupy as cp

import nvmath

# Turn on logging. Here we use the global logger, set the level to "debug", and use a custom
# format for the log. Any of the features provided by the logging module can be used.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

a = cp.random.rand(4, 4, 12, 12)
b = cp.random.rand(12, 12, 8, 8)

c = cp.random.rand(4, 4, 8, 8)

# result[i,j,m,n] = \sum_{k,l} a[i,j,k,l] * b[k,l,m,n] + c[i,j,m,n]
result = nvmath.tensor.binary_contraction("ijkl,klmn->ijmn", a, b, c=c, beta=1)

assert cp.allclose(result, cp.einsum("ijkl,klmn->ijmn", a, b) + c)

print(f"Input type = {type(a), type(b), type(c)}, contraction result type = {type(result)}")
