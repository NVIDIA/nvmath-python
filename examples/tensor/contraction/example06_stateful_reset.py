# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to reset operands in stateful tensor contraction APIs, and
reuse the object for multiple executions. This is needed when the memory space of the
operands is not accessible from the execution space, or if it's desired to bind new
(compatible) operands to the stateful object.

The inputs as well as the result are NumPy ndarrays.
"""

import numpy as np

import nvmath

a = np.random.rand(4, 4, 12, 12)
b = np.random.rand(12, 12, 8, 8)
c = np.random.rand(4, 4, 8, 8)

A = np.random.rand(4, 4, 12, 12)
B = np.random.rand(12, 12, 8, 8)
C = np.random.rand(4, 4, 8, 8)

# Create a stateful BinaryContraction object 'contraction'.
with nvmath.tensor.BinaryContraction("ijkl,klmn->ijmn", a, b, c=c) as contraction:
    # Plan the Contraction.
    contraction.plan()

    # Execute the Contraction.
    result = contraction.execute(beta=1)
    assert np.allclose(result, np.einsum("ijkl,klmn->ijmn", a, b) + c)

    # Reset the input operands with new values.
    contraction.reset_operands(a=A, b=B, c=C)

    # Re-execute the Contraction with the updated input operand.
    result = contraction.execute(beta=1)
    assert np.allclose(result, np.einsum("ijkl,klmn->ijmn", A, B) + C)
