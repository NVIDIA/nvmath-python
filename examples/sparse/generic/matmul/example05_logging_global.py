# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates the use of the global Python logger with basic SpMM
(sparse matrix multiplication with a dense matrix) of the form
`c := alpha a @ b + beta c`, where the operands are UST backed by CuPy ndarrays.

nvmath-python supports multiple sparse formats, frameworks, memory spaces, and
execution spaces. The sparse operand can be provided from SciPy, CuPy, PyTorch
in a variety of supported formats such as BSR, BSC, COO, CSR, CSC, DIA or
as a universal sparse tensor (UST), which supports custom formats in addition to
the standard named formats.
"""

# Turn on logging. Here we use the global logger, set the level to "debug", and use a custom
# format for the log.
import logging

import cupy as cp
import cupyx.scipy.sparse as sp

import nvmath

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# The index (int32, int64, ...) and data (float32, complex128, ...) types.
index_type, dtype = cp.int64, cp.float64

crow_indices = cp.array([0, 2, 4, 6, 8], dtype=index_type)
col_indices = cp.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=index_type)
values = cp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=dtype)

shape = 4, 4

# Create a CuPy CSR matrix.
a = sp.csr_matrix((values, col_indices, crow_indices), shape=shape)

# View the operands as UST. As we will see in later examples, this allows
# for the code generation path in addition to library dispatch.
a = nvmath.sparse.ust.Tensor.from_package(a)
print(f"a = \n{a}")

# Dense 'b' and 'c', also viewed as UST objects.
b = cp.ones(shape, dtype=dtype)
b = nvmath.sparse.ust.Tensor.from_package(b)
print(f"b = \n{b}")
c = cp.zeros(shape, dtype=dtype)
c = nvmath.sparse.ust.Tensor.from_package(c)

# c := 2.0 * a @ b + c
r = nvmath.sparse.matmul(a, b, c, alpha=2.0, beta=1.0)
print(f"c := 2.0 * a @ b = \n{r}")

# The result can also be viewed as a CuPy ndarray.
r = nvmath.sparse.ust.Tensor.to_package(r)
print(f"c (CuPy) = \n{r}")
