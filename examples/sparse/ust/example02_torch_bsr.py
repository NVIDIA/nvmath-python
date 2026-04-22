# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to create a universal sparse tensor (UST) from
an existing sparse PyTorch BSR tensor on the GPU.

The nvmath-python UST currently supports multiple sparse matrix and tensor
libraries (SciPy, CuPy, PyTorch), multiple named sparse formats, and multiple
memory spaces.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

import torch

from nvmath.sparse import ust

# Set the device id.
device_id = 0

# Create a PyTorch BSR matrix on the GPU.
crow_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=device_id)
col_indices = torch.tensor([0, 1], dtype=torch.int32, device=device_id)
values = torch.tensor(
    [[[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]], [[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]]], dtype=torch.float64, device=device_id
)
shape = 4, 6
a = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=shape)
print(f"The PyTorch BSR matrix is:\n {a}.")

# Create an UST from the PyTorch BSR matrix. The `from_package` method is zero-copy (it
# shares the data with the original PyTorch matrix).
u = ust.Tensor.from_package(a)
print(f"\nThe UST is:\n {u}.")
message = "We'll discuss the information printed above in a later example, after \
we look at the UST DSL."
print(message)

# The UST object is a view into the provided sparse tensor's data structure. A
# modification of either will be reflected in the other.
a.values()[0] = 3.14
print(f"\nThe modified PyTorch BSR matrix is:\n {a}.")
print(f"\nThe changes above are reflected in the UST `u`:\n {u}.")
