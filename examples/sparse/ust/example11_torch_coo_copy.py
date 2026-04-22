# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to copy a universal sparse tensor (UST) across
memory spaces. We'll look at both in-place `copy_()` and out-of-place `to()`
copy methods.

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
u_cpu = ust.Tensor.from_package(a)
print(f"\nThe UST on the CPU is:\n {u_cpu}.")

u_gpu = u_cpu.to(device_id=0)
print(f"\nThe UST on the GPU is:\n {u_gpu}.")

# Create a new compatible torch COO tensor, with different sparsity but with
# the same shape, data and index types, and number of non-zeros.
device_id = 0
indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32, device=device_id)
values = torch.tensor([12.0, 14.0], dtype=torch.float64, device=device_id)
shape = 2, 2
b_gpu = torch.sparse_coo_tensor(indices, values, size=shape).coalesce()

# Create a new UST.
v_gpu = ust.Tensor.from_package(b_gpu)

# Copy the contents of the new UST v_gpu to u_gpu in-place.
u_gpu.copy_(v_gpu)
print(f"\nThe UST on the GPU after in-place copy is:\n {u_gpu}.")
