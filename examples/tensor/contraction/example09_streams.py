# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of multiple CUDA streams with the tensor contraction APIs.
"""

import cupy as cp

import nvmath

a = cp.random.rand(4, 4, 12, 12)
b = cp.random.rand(12, 12, 8, 8)

c = cp.random.rand(4, 4, 8, 8)

# Create a CUDA stream to use for instantiating, planning, and first execution of a stateful
# BinaryContraction object 'contraction'.
s1 = cp.cuda.Stream()

# Create a stateful BinaryContraction object 'contraction' on stream s1.
with nvmath.tensor.BinaryContraction("ijkl,klmn->ijmn", a, b, c=c, options={"blocking": "auto"}, stream=s1) as contraction:
    # Plan the BinaryContraction on stream s1.
    contraction.plan(stream=s1)

    # Execute the BinaryContraction on stream s1.
    d = contraction.execute(beta=1, stream=s1)

    assert cp.allclose(d, cp.einsum("ijkl,klmn->ijmn", a, b) + c)

    # Record an event on s1 for use later.
    e1 = s1.record()

    # Create a new stream to on which the new operand c for the second execution will be
    # filled.
    s2 = cp.cuda.Stream()

    # Fill c on s2.
    with s2:
        a1 = cp.random.rand(*a.shape)
        b1 = cp.random.rand(*b.shape)
        c1 = cp.random.rand(*c.shape)

    # In the following blocks, we will use stream s2 to perform subsequent operations. Note
    # that it's our responsibility as a user to ensure proper ordering, and we want to order
    # `reset_operand` after event e1 corresponding to the execute() call above.
    s2.wait_event(e1)

    # Alternatively, if we want to use stream s1 for subsequent operations (s2 only for
    # operand creation), we need to order `reset_operands` after the event for
    # cupy.random.rand on s2, e.g.: e2 = s2.record(); s1.wait_event(e2)

    # Set a new operand c on stream s2.
    contraction.reset_operands(a=a1, b=b1, c=c1, stream=s2)

    # Execute the new BinaryContraction on stream s2.
    d = contraction.execute(beta=1, stream=s2)

    # Synchronize s2 at the end
    s2.synchronize()

    assert cp.allclose(d, cp.einsum("ijkl,klmn->ijmn", a1, b1) + c1)
