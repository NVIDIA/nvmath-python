# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of resetting the operands in stateful sparse matrix
multiplication objects. Stateful objects amortize the cost of preparation across multiple
executions.

The operands are UST on the GPU backed by torch tensors.
"""

import logging

import torch

import nvmath

# The memory space.
device_id = 0

# The problem parameters.
dtype = torch.float64
m, n, k = 4, 2, 8

# Create a torch CSR tensor.
a = torch.ones(m, k, dtype=dtype, device=device_id)
a = a.to_sparse_csr()
# View it as UST.
a_u = nvmath.sparse.ust.Tensor.from_package(a)
print(f"a = \n{a}")

# Dense 'b' and 'c'.
b = torch.ones(k, n, dtype=dtype, device=device_id)
b_u = nvmath.sparse.ust.Tensor.from_package(b)
print(f"b = \n{b}")
c = torch.ones(m, n, dtype=dtype, device=device_id)
c_u = nvmath.sparse.ust.Tensor.from_package(c)

# Turn on logging to see what's happening under the hood.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Create a stateful object that specifies the operation c := alpha a @ b + beta c
alpha, beta = 1.2, 2.4
with nvmath.sparse.Matmul(a_u, b_u, c_u, alpha=alpha, beta=beta) as mm:
    # Plan the SpMM operation. As we will see in later examples, planning
    # can be customized by providing prologs, epilog, and semiring operations.
    # The planning needs to be done only once for each problem specification.
    mm.plan()

    # Execute the operation.
    r = mm.execute()
    print(f"First execution, with b filled with 1.0: c =\n{r}")

    # The operands can be reset in-place using the array library. This change is
    # reflected in the corresponding UST view. Alternatively, a Python function
    # can be provided to set or transform USTs as illustrated in
    # ../../ust/example10_torch_csr_apply.py.
    b *= 3.14

    # Note that since `c` has been updated in-place, it also needs to
    # be reset unless accumulating into it is desired.
    c[:] = 1.0

    # Execute the operation with updated `b`.
    r = mm.execute()
    print(f"Second execution, with b updated inplace to 3.14: c =\n{r}")

    # The operands can also be reset using the reset_operand[_unchecked] methods.
    # This is needed when the memory space doesn't match the execution space or
    # if the operands have not been updated in-place. Any operands that are not
    # reset retain their prior values.

    # The sparse matrix needs to be compatible (have the same sparse format, shape,
    # dtype, NNZ etc.).
    a = 2.718 * torch.ones(m, k, dtype=dtype, device=device_id)
    a = a.to_sparse_csr()
    a_u = nvmath.sparse.ust.Tensor.from_package(a)

    c = 6.28 * torch.ones(m, n, dtype=dtype, device=device_id)
    c_u = nvmath.sparse.ust.Tensor.from_package(c)

    mm.reset_operands(a=a_u, c=c_u)

    # Execute the operation with the reset `a` and `c`, `b` retains its previous value.
    r = mm.execute()
    print(f"Third execution, with `a` and `c` reset: c =\n{r}")

    # The `reset_operands` method can add overhead due to the check for compatibility,
    # which may be redundant with the checks performed by the user. The checks can be
    # avoided by using the unchecked version `reset_operands_unchecked`.

    b = torch.ones(k, n, dtype=dtype, device=device_id)
    b_u = nvmath.sparse.ust.Tensor.from_package(b)

    c = torch.ones(m, n, dtype=dtype, device=device_id)
    c_u = nvmath.sparse.ust.Tensor.from_package(c)

    mm.reset_operands_unchecked(b=b_u, c=c_u)

    # Execute the operation with the reset `b` and `c`, `a` retains its previous value.
    r = mm.execute()
    print(f"Fourth execution, with `b` and `c` reset: c =\n{r}")
