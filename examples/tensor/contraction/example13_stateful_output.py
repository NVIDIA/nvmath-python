# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to write to a pre-allocated output array with stateful
tensor contraction APIs.
"""

import cupy as cp

import nvmath

a = cp.random.rand(4, 4, 4)
b = cp.random.rand(4, 4, 4)

# Case I: writing to a pre-allocated output array
c = cp.empty((4, 4))
with nvmath.tensor.BinaryContraction("ijk,jkl->il", a, b, out=c) as contraction:
    contraction.plan()
    contraction.execute()
    assert cp.allclose(c, cp.einsum("ijk,jkl->il", a, b))

# Case II: with in-place update, where the output tensor is the same as the addend operand c
c = cp.random.rand(4, 4)
reference = cp.einsum("ijk,jkl->il", a, b) + c
with nvmath.tensor.BinaryContraction("ijk,jkl->il", a, b, c=c, out=c) as contraction:
    contraction.plan()
    contraction.execute(beta=1)
    assert cp.allclose(c, reference)

# Case III: writing to a slice of a pre-allocated output array with in-place update
full_matrix = cp.random.rand(8, 8)
matrix_slice = full_matrix[2:6, 2:6]

reference = cp.einsum("ijk,jkl->il", a, b) + matrix_slice
with nvmath.tensor.BinaryContraction("ijk,jkl->il", a, b, c=matrix_slice, out=matrix_slice) as contraction:
    contraction.plan()
    contraction.execute(beta=1)
    assert cp.allclose(matrix_slice, reference)

# Case IV: resetting the target output operand to different buffer
# Note that the updated tensor must be compatible with the original tensor

c = cp.random.rand(4, 4)

out1 = cp.empty((4, 4))
out2 = cp.empty((4, 4))

with nvmath.tensor.BinaryContraction("ijk,jkl->il", a, b, c=c, out=out1) as contraction:
    contraction.plan()

    alpha = 1.4
    beta = 0.7
    contraction.execute(alpha=alpha, beta=beta)
    assert cp.allclose(out1, alpha * cp.einsum("ijk,jkl->il", a, b) + beta * c)

    alpha = 0.5
    beta = 0.3
    contraction.reset_operands(a=a, b=b, c=c, out=out2)
    contraction.execute(alpha=alpha, beta=beta)
    assert cp.allclose(out2, alpha * cp.einsum("ijk,jkl->il", a, b) + beta * c)
