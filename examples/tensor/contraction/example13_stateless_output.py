# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to write to a pre-allocated output array with stateless
tensor contraction APIs.
"""

import cupy as cp

import nvmath

# If CPU arrays are provided, execute on nvpl tensor by default
# user also gets to use cutensor by providing `execution='cuda'`
# If GPU arrays are provided, execute on cutensor by default
# user also gets to use nvpl tensor by providing `execution='cpu'`

a = cp.random.rand(4, 4, 4)
b = cp.random.rand(4, 4, 4)

# Case I: writing to a pre-allocated output array
c = cp.empty((4, 4))
out = nvmath.tensor.binary_contraction("ijk,jkl->il", a, b, out=c)
assert out is c and cp.allclose(out, cp.einsum("ijk,jkl->il", a, b))


# Case II: writing to a pre-allocated output array with in-place update
c = cp.random.rand(4, 4)
reference = cp.einsum("ijk,jkl->il", a, b) + c
out = nvmath.tensor.binary_contraction("ijk,jkl->il", a, b, c=c, out=c, beta=1)

assert out is c and cp.allclose(out, reference)


# Case III: writing to a slice of a pre-allocated output array with in-place update
full_matrix = cp.random.rand(8, 8)
matrix_slice = full_matrix[2:6, 2:6]

reference = cp.einsum("ijk,jkl->il", a, b) + matrix_slice
out = nvmath.tensor.binary_contraction("ijk,jkl->il", a, b, c=matrix_slice, out=matrix_slice, beta=1)
assert out is matrix_slice and cp.allclose(out, reference)
