# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic SpMM (sparse matrix multiplication with a
dense matrix) of the form `c := alpha a @ b + beta c`.

nvmath-python supports multiple sparse formats, frameworks, memory spaces, and
execution spaces. The sparse operand can be provided from SciPy, CuPy, PyTorch
in a variety of supported formats such as BSR, BSC, COO, CSR, CSC, DIA or
as a universal sparse tensor (UST), which supports custom formats in addition to
the standard named formats.
"""

import numpy as np
import scipy.sparse as sp

import nvmath

# The index (int32, int64, ...) and data (float32, complex128, ...) types.
index_type, dtype = np.int32, np.float32

crow_indices = np.array([0, 1, 2], dtype=index_type)
col_indices = np.array([0, 1], dtype=index_type)
values = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=dtype)

shape = 4, 4

# Create a SciPy BSR array.
a = sp.bsr_array((values, col_indices, crow_indices), shape=shape)
print(f"a = \n{a}")

# Dense 'b' and 'c', from NumPy.
b = np.ones(shape, dtype=dtype)
print(f"b = \n{b}")
c = np.zeros(shape, dtype=dtype)

# c := a @ b + c
r = nvmath.sparse.matmul(a, b, c, beta=1.0)
assert r is c, "Error: the operation is not in-place."
print(f"c := a @ b + c = \n{r}")
