# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to manage memory resources used by stateful objects. This is useful
when the tensor contraction operation needs a lot of memory and calls to execution method
on a stateful object are interleaved with calls to other operations
(including another tensor contraction) also requiring a lot of memory.

In this example, two tensor contraction operations are performed in a loop in an
interleaved manner. We assume that the available device memory is large enough for only
one tensor contraction at a time.
"""

import logging

import cupy as cp

import nvmath

# Turn on logging and set the level to DEBUG to print memory management messages.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

a = cp.random.rand(16, 16, 32, 32)
b = cp.random.rand(32, 32, 32, 32)

A = cp.random.rand(16, 16, 32, 32)
B = cp.random.rand(32, 32, 32, 32)


# Create and prepare two BinaryContraction objects.
contraction1 = nvmath.tensor.BinaryContraction("ijkl,klmn->ijmn", a, b)
contraction1.plan()

contraction2 = nvmath.tensor.BinaryContraction("ijkl,klmn->ijmn", A, B)
contraction2.plan()

num_iter = 3
# Use the BinaryContraction objects as context managers so that internal library resources
#   are properly cleaned up.
with contraction1, contraction2:
    for i in range(num_iter):
        print(f"Iteration {i}")
        # Perform the first contraction, and request that the workspace be released at the
        #   end of the operation so that there is enough memory for the second one.
        r = contraction1.execute(release_workspace=True)

        assert cp.allclose(r, cp.einsum("ijkl,klmn->ijmn", a, b))

        # Update contraction1's operands for the next iteration.
        if i < num_iter - 1:
            a[:] = cp.random.rand(*a.shape)
            b[:] = cp.random.rand(*b.shape)

        # Perform the second contraction, and request that the workspace be released
        #   at the end of the operation so that there is enough memory for the first
        #   contraction in the next iteration.
        r = contraction2.execute(release_workspace=True)

        assert cp.allclose(r, cp.einsum("ijkl,klmn->ijmn", A, B))

        # Update contraction2's operands for the next iteration.
        if i < num_iter - 1:
            A[:] = cp.random.rand(*A.shape)
            B[:] = cp.random.rand(*B.shape)

        # Synchronize the default stream
        cp.cuda.get_current_stream().synchronize()
