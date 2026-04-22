# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to transform elements of a sparse tensor
using a user-defined operation written in Python.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

import math

import torch

from nvmath.sparse import ust

# Set the device id.
device_id = 0

# Create a PyTorch CSR matrix on the GPU.
crow_indices = torch.tensor([0, 2, 3], dtype=torch.int32, device=device_id)
col_indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device_id)
values = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64, device=device_id)
shape = 2, 2
a = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=shape)

print(f"The PyTorch CSR matrix is:\n {a}.")

# Create a UST from the PyTorch CSR matrix. The `from_package` method is zero-copy (it
# shares the data with the original PyTorch matrix).
u = ust.Tensor.from_package(a)
print(f"\nThe UST is:\n {u}.")
print("We'll discuss the information printed above in a later example, after we look at the UST DSL.")


def transform_all(v):
    return math.pi * math.sin(v)


u.set_kernel(transform_all, with_indices=False)
u.run_kernel()

# The UST object is a view into the provided sparse tensor's data structure. A
# modification in one will be reflected in the other.
print(f"\nThe UST after index-independent modification is:\n {u}.")
print(f"\nThe changes above are reflected in the torch tensor `a`:\n {a}.")


def transform_some(v, i, j):
    """
    Apply a transformation based on the coordinates. In this simple example,
    we choose one transformation for the first row and another for the rest.
    """
    if i == 0:
        return math.asin(v / math.pi)

    return -v


u.set_kernel(transform_some, with_indices=True)
u.run_kernel()

# The UST object is a view into the provided sparse tensor's data structure. A
# modification in one will be reflected in the other.
print(f"\nThe UST after index-dependent modification is:\n {u}.")
print(f"\nThe changes above are reflected in the torch tensor `a`:\n {a}.")
