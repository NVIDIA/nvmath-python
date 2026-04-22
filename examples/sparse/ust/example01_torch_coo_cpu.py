# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to create a universal sparse tensor (UST) from
an existing sparse PyTorch COO array on the CPU.

The nvmath-python UST currently supports multiple sparse matrix and tensor
libraries (SciPy, CuPy, PyTorch), multiple named sparse formats, and multiple
memory spaces.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

import torch

from nvmath.sparse import ust

# Set the device id.
device_id = "cpu"

# Create a PyTorch COO matrix on the CPU.
indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device=device_id)
values = torch.tensor([2.0, 4.0], dtype=torch.float64, device=device_id)
shape = 2, 2
a = torch.sparse_coo_tensor(indices, values, size=shape)

# Most operations require the sparse matrix to be coalesced (data structures sorted,
# with no duplicates). PyTorch provides the coalesce() method for this.
a = a.coalesce()
print(f"The PyTorch COO matrix is:\n {a}.")

# Create an UST from the PyTorch COO matrix. The `from_package` method is zero-copy (it
# shares the data with the original PyTorch matrix).
u = ust.Tensor.from_package(a)
print(f"\nThe UST is:\n {u}.")
message = "We'll discuss the information printed above in a later example, after \
we look at the UST DSL."
print(message)

# The UST object is a view into the provided sparse tensor's data structure. A
# modification of either will be reflected in the other.
a.values()[0] = 3.14
print(f"\nThe modified PyTorch COO matrix is:\n {a}.")
print(f"\nThe changes above are reflected in the UST `u`:\n {u}.")
