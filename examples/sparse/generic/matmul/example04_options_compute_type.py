# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic SpMM (sparse matrix multiplication with a
dense matrix) of the form `c := alpha a @ b + beta c`, with the computation
performed in the specified type.

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

crow_indices = torch.tensor([0, 1, 2], dtype=index_type, device=device_id)
col_indices = torch.tensor([0, 1], dtype=index_type, device=device_id)
values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=dtype, device=device_id)

shape = 4, 4

# Create a torch BSR tensor.
a = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=shape)
a = nvmath.sparse.ust.Tensor.from_package(a)
print(f"a = \n{a}")

# Dense 'b' and 'c'.
b = torch.ones(*shape, dtype=dtype, device=device_id)
b = nvmath.sparse.ust.Tensor.from_package(b)
print(f"b = \n{b}")
c = torch.zeros(shape, dtype=dtype, device=device_id)
c = nvmath.sparse.ust.Tensor.from_package(c)

# Specify the compute type. We'll see in a later example what the codegen option does.
compute_type = nvmath.sparse.ComputeType.CUDA_R_64F
options = nvmath.sparse.MatmulOptions(compute_type=compute_type, codegen=True)

# c := a @ b + c, with computation in float64.
r = nvmath.sparse.matmul(a, b, c, beta=1.0, options=options)

# The options can also be provided as a dict instead of a dataclass.
r = nvmath.sparse.matmul(a, b, c, beta=1.0, options={"compute_type": compute_type, "codegen": True})

print(f"c := a @ b + c = \n{r}")
