# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to create a novel UST format and use it in an SpMM operation.

The operands are UST backed by torch.
"""

import logging

import torch

import nvmath
from nvmath.sparse.ust import Dimension, LevelFormat, Tensor, TensorFormat

# The problem details.
device_id = 0
dtype = torch.float64
m, n, k = 3, 2, 8

# Create a dense torch tensor and view it as UST.
a = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 2], [0, 0, 3, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 5]], dtype=dtype)
a = Tensor.from_package(a)
print(f"The initial dense UST = \n{a}")

# Create a delta-compression format in the UST DSL.
i, j = Dimension(dimension_name="i"), Dimension(dimension_name="j")
delta_format = TensorFormat([i, j], {i: LevelFormat.DENSE, j: (LevelFormat.DELTA, 2)})

# Convert dense UST to the delta-compressed UST. Note that this is NOT a view
# (zero-copy) operation since the representation changes during the conversion.
a = a.convert(tensor_format=delta_format)
print(f"The UST after delta-compression = \n{a}")

# Move `a` to device
a = a.to(device_id=device_id)

# Create dense operands `b` and `c` for the SpMM.
b = torch.ones(k, n, dtype=dtype, device=device_id)
b = nvmath.sparse.ust.Tensor.from_package(b)
print(f"b = \n{b}")

c = torch.zeros(m, n, dtype=dtype, device=device_id)
c = nvmath.sparse.ust.Tensor.from_package(c)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Create, plan, and execute the SpMM operation.
with nvmath.sparse.Matmul(a, b, c=c) as mm:
    # Plan the SpMM.
    mm.plan()

    # Execute it.
    r = mm.execute()

# View the UST result as a torch tensor.
r = r.to_package()
print(r)
