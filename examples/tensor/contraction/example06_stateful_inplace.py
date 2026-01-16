# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of inplace update of input operands in stateful tensor
contraction APIs.

The inputs as well as the result are CuPy ndarrays.

NOTE: The operands should be updated inplace only when they are in a memory space that is
accessible from the execution space. In this case, the operands reside on the GPU while the
execution also happens on the GPU.
"""

import cupy as cp

import nvmath

a = cp.random.rand(4, 4, 12, 12)
b = cp.random.rand(12, 12, 8, 8)
c = cp.random.rand(4, 4, 8, 8)

A = cp.random.rand(4, 4, 12, 12)
B = cp.random.rand(12, 12, 8, 8)
C = cp.random.rand(4, 4, 8, 8)

# Create a stateful BinaryContraction object 'contraction'.
with nvmath.tensor.BinaryContraction("ijkl,klmn->ijmn", a, b, c=c) as contraction:
    # Plan the Contraction.
    contraction.plan()

    # Execute the Contraction.
    result = contraction.execute(beta=1)
    assert cp.allclose(result, cp.einsum("ijkl,klmn->ijmn", a, b) + c)

    # Inplace update the input operand 'a,b,c' with the result.
    a[:] = A
    b[:] = B
    c[:] = C

    # Re-execute the Contraction with the updated input operand.
    result = contraction.execute(beta=1)
    assert cp.allclose(result, cp.einsum("ijkl,klmn->ijmn", A, B) + C)
