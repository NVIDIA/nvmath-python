# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example describes how to interpret the UST tensor representation, now
that we are familiar with the UST DSL.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

import torch

from nvmath.sparse import ust

# Set the device id.
device_id = 0

# Create a PyTorch CSR matrix on the GPU.
crow_indices = torch.tensor([0, 2, 3], dtype=torch.int32, device=device_id)
col_indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device_id)
values = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64, device=device_id)
shape = 2, 2
a = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=shape)

print(f"The PyTorch CSR matrix is:\n {a}.")

# Create an UST from the PyTorch CSR matrix. The `from_package` method is zero-copy (it
# shares the data with the original PyTorch matrix).
u = ust.Tensor.from_package(a)
print(f"\nThe UST is:\n {u}.")

print("""The first line describes key information about the UST: the datatype (VAL),
the index types (POS, CRD), and the number of dimensions and levels. As a reminder,
dimensions are logical while levels correspond to the physical representation (the
sparse format). For example, blocked sparse formats like BSR have more levels than
dimensions. This is followed by the sparse format in the UST DSL.

The next line shows the memory space (CPU or CUDA), following which the logical extents
(the shape) is shown. This is followed by the extents of each level (the physical
representation). The number of stored elements (NSE) is printed next - this is also
known as the number of non-zeros (NNZ) in other contexts. The position and coordinates
array are printed only for compressed levels, while the flat values array is always
present. Finally the memory in bytes consumed by the data structure along with the
sparsity fraction is printed.
""")
