# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic SpMM (sparse matrix multiplication with a
dense matrix) of the form `c := alpha a @ b + beta c`.

nvmath-python supports multiple sparse formats, frameworks, memory spaces, and
execution spaces. The sparse operand can be provided from SciPy, CuPy, PyTorch
in a variety of supported formats such as BSR, BSC, COO, CSR, CSC, DIA or
as a universal sparse tensor (UST), which supports custom formats in addition to
the standard named formats.
"""

import torch

import nvmath

# The memory space (GPU ordinal or "cpu").
device_id = 0

# The index (int32, int64, ...) and data (float32, complex128, ...) types.
index_type, dtype = torch.int32, torch.float32

indices = torch.tensor([[0, 1], [0, 1]], dtype=index_type)
values = torch.tensor([2.0, 4.0], dtype=dtype)
size = 2, 2

# Create a torch sparse tensor on the specified device. Note that the sparse operand
# provided to the matrix multiplication must be coalesced.
a = torch.sparse_coo_tensor(indices, values, size, device=device_id)
a = a.coalesce()
print(f"a = \n{a}")

# Dense 'b' and 'c', from the same package in the same memory space.
b = torch.ones(2, 2, dtype=dtype, device=device_id)
print(f"b = \n{b}")
c = torch.zeros(2, 2, dtype=dtype, device=device_id)

# c := a @ b + c
r = nvmath.sparse.matmul(a, b, c, beta=1.0)
assert r is c, "Error: the operation is not in-place."
print(f"c := a @ b + c = \n{r}")
