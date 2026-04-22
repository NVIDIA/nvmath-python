# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to create a universal sparse tensor (UST) from
an existing PyTorch  strided tensor.  Currently, the UST requires that the strided
tensor is dense (no holes) -- the UST is not a universal tensor (yet).

The nvmath-python UST currently supports multiple sparse matrix and tensor
libraries (SciPy, CuPy, PyTorch), multiple named sparse formats, and multiple
memory spaces.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

import torch

from nvmath.sparse import ust

device_id = 0

# Create a 3-D PyTorch tensor on the GPU with device id = 0.
shape = 2, 3, 4
a = torch.rand(*shape, dtype=torch.float64, device=device_id)
print(f"The PyTorch tensor is:\n {a}.")

# Create an UST from the PyTorch tensor. The `from_package` method is zero-copy (it
# shares the data with the original PyTorch tensor).
u = ust.Tensor.from_package(a)
print(f"\nThe UST is:\n {u}.")
message = "We'll discuss the information printed above in a later example, after \
we look at the UST DSL."
print(message)

# The UST object is a view into the provided sparse tensor's data structure. A
# modification of either will be reflected in the other.
a[0, :, 0] = 3.14
print(f"\nThe modified PyTorch tensor is:\n {a}.")
print(f"\nThe changes above are reflected in the UST `u`:\n {u}.")
