# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic SpMM (sparse matrix multiplication with a
dense matrix) of the form `c := alpha a @ b + beta c`, where the operands
are UST backed by torch tensors (on the CPU or GPU).

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
index_type, dtype = torch.int32, torch.float16

indices = torch.tensor([[0, 1], [0, 1]], dtype=index_type)
values = torch.tensor([2.0, 4.0], dtype=dtype)
size = 2, 2

# Create a torch sparse tensor on the specified device. Note that the sparse operand
# must be coalesced before it can be viewed as a UST object.
a = torch.sparse_coo_tensor(indices, values, size, device=device_id)
a = a.coalesce()
# View the operands as UST. As we will see in later examples, this allows
# for the code generation path in addition to library dispatch.
a = nvmath.sparse.ust.Tensor.from_package(a)
print(f"a = \n{a}")


# Dense 'b' and 'c', also viewed as UST objects.
b = torch.ones(2, 2, dtype=dtype, device=device_id)
b = nvmath.sparse.ust.Tensor.from_package(b)
print(f"b = \n{b}")
c = torch.zeros(2, 2, dtype=dtype, device=device_id)
c = nvmath.sparse.ust.Tensor.from_package(c)

# c := a @ b + c
r = nvmath.sparse.matmul(a, b, c, beta=1.0)
print(f"c := a @ b = \n{r}")

# The result can also be viewed as a torch tensor.
r = nvmath.sparse.ust.Tensor.to_package(r)
print(f"c (torch) = \n{r}")
