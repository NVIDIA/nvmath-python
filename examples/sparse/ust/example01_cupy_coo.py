# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to create a universal sparse tensor (UST) from
an existing sparse CuPy COO array on the GPU.

The nvmath-python UST currently supports multiple sparse matrix and tensor
libraries (SciPy, CuPy, PyTorch), multiple named sparse formats, and multiple
memory spaces.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

import cupy as cp
import cupyx.scipy.sparse as sp

from nvmath.sparse import ust

# Create a CuPy COO matrix on the GPU.
indices = cp.array([[0, 1], [0, 1]], dtype=cp.int32)
values = cp.array([2.0, 4.0], dtype=cp.float64)
shape = 2, 2
a = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)

# Most operations require the sparse matrix to be coalesced (data structures sorted,
# with no duplicates). SciPy and CuPy use the sum_duplicates() method for this.
a.sum_duplicates()
print(f"The CuPy COO matrix is:\n {a}.")

# Create an UST from the CuPy COO matrix. The `from_package` method is zero-copy (it
# shares the data with the original CuPy matrix).
u = ust.Tensor.from_package(a)
print(f"\nThe UST is:\n {u}.")
message = "We'll discuss the information printed above in a later example, after \
we look at the UST DSL."
print(message)

# The UST object is a view into the provided sparse tensor's data structure. A
# modification of either will be reflected in the other.
a.data[0] = 3.14
print(f"\nThe modified CuPy COO matrix is:\n {a}.")
print(f"\nThe changes above are reflected in the UST `u`:\n {u}.")
