# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to create a universal sparse tensor (UST) from
an existing sparse SciPy BSR matrix.

The nvmath-python UST currently supports multiple sparse matrix and tensor
libraries (SciPy, CuPy, PyTorch), multiple named sparse formats, and multiple
memory spaces.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

import numpy as np
import scipy.sparse as sp

from nvmath.sparse import ust

# Create a SciPy BSR matrix with 2x3 blocks on the CPU.
crow_indices = np.array([0, 1, 2], dtype=np.int32)
col_indices = np.array([0, 1], dtype=np.int32)
values = np.array([[[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]], [[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]]], dtype=np.float64)
shape = 4, 6
a = sp.bsr_matrix((values, col_indices, crow_indices), shape=shape)

# Most operations require the sparse matrix to be coalesced (data structures sorted,
# with no duplicates). SciPy and CuPy use the sum_duplicates() method for this.
a.sum_duplicates()
print(f"The SciPy BSR matrix is:\n {a}.")

# Create an UST from the SciPy BSR matrix. The `from_package` method is zero-copy (it
# shares the data with the original SciPy matrix).
u = ust.Tensor.from_package(a)
print(f"\nThe UST is:\n {u}.")
message = "We'll discuss the information printed above in a later example, after \
we look at the UST DSL."
print(message)

# The UST object is a view into the provided sparse tensor's data structure. A
# modification of either will be reflected in the other.
a.data[0] = 3.14
print(f"\nThe modified SciPy BSR matrix is:\n {a}.")
print(f"\nThe changes above are reflected in the UST `u`:\n {u}.")
