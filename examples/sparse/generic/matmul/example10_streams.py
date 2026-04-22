# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of multiple CUDA streams.

The operands are UST on the GPU backed by CuPy ndarrays.
"""

import cupy as cp
import cupyx.scipy.sparse as sp

import nvmath

dtype = cp.float64
m, n, k = 4, 2, 6

# Create a CuPy CSR array.
a = sp.random(m, k, density=0.2, format="csr")

# View it as UST.
a = nvmath.sparse.ust.Tensor.from_package(a)
print(f"a = \n{a}")

# Dense 'b' and 'c'.
b = cp.ones((k, n), dtype=dtype)
b = nvmath.sparse.ust.Tensor.from_package(b)
print(f"b = \n{b}")

c = cp.ones((m, n), dtype=dtype)
c = nvmath.sparse.ust.Tensor.from_package(c)

# Create a CUDA stream to use for instantiating, planning, and first execution of a stateful
# SpMM object `mm`.
s1 = cp.cuda.Stream()

# Create a stateful object that specifies the operation c := alpha a @ b + beta c on
# stream s1.
alpha, beta = 1.2, 2.4
with nvmath.sparse.Matmul(a, b, c, alpha=alpha, beta=beta, stream=s1) as mm:
    # Plan the SpMM operation on stream s1.
    mm.plan(stream=s1)

    # Execute the operation on stream s1.
    r1 = mm.execute(stream=s1)

    # Record an event on s1 for use later.
    e1 = s1.record()

    # Create a new stream on which the new operand `c` for the second execution will be
    # filled.
    s2 = cp.cuda.Stream()

    # Fill new c on s2.
    with s2:
        c1 = 3.14 * cp.ones((m, n), dtype=dtype)
        c1 = nvmath.sparse.ust.Tensor.from_package(c1, stream=s2)

    # We will use stream s2 to perform subsequent operations. Note that it's our
    # responsibility as a user to ensure proper ordering, and we want to order
    # `reset_operand` after event e1 corresponding to the execute() call above.
    s2.wait_event(e1)

    # Alternatively, if we want to use stream s1 for subsequent operations (s2 only for
    # operand creation), we need to order `reset_operand` after the event for
    # cupy.ones() on s2, e.g: e2 = s2.record(); s1.wait_event(e2).

    # Set the new operand c=c1 on stream s2.
    mm.reset_operands(c=c1, stream=s2)

    # Execute the new SpMM on stream s2.
    r2 = mm.execute(stream=s2)

# Synchronize before printing since the data is being generated on a custom stream. This
# is important when a stream is non-blocking or a per-thread default stream is used. Since
# this is not the case in this example, the synchronization is redundant (assuming that the
# user has not set `CUPY_PER_THREAD_DEFAULT_STREAM=1`). It is used here to remind the user
# that explicit synchronization (or ordering using events) is typically essential to avoid
# a race condition when data is produced on one stream and consumed in a different stream
# unless it is the legacy default CUDA stream.
s1.synchronize()
print(f"First execution on stream s1, c = \n{r1}.")
s2.synchronize()
print(f"Second execution on stream s2, c = \n{r2}.")
