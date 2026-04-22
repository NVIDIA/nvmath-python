# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to convert an universal sparse tensor (UST) format
into another.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

import torch

from nvmath.sparse import ust

# Set the device_id to the GPU ordinal or to "cpu".
device_id = "cpu"

# Create a PyTorch COO matrix on the specified device.
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
u_coo = ust.Tensor.from_package(a)
print(f"\nThe UST in COO format:\n {u_coo}.")

# Convert the UST in COO format to one in CSR format.
u_csr = u_coo.convert(tensor_format=ust.NamedFormats.CSR)
print(f"\nThe UST in CSR format:\n {u_csr}.")

# Create a novel format, like delta-compression with 3 bits.
i, j = ust.Dimension(dimension_name="i"), ust.Dimension(dimension_name="j")
delta_format = ust.TensorFormat([i, j], {i: ust.LevelFormat.DENSE, j: (ust.LevelFormat.DELTA, 3)})

# Convert from CSR (or COO) to delta3.
u_delta = u_csr.convert(tensor_format=delta_format)
print(f"\nThe UST in delta-compressed format:\n {u_delta}.")
