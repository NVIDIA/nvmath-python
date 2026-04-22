# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of matrix qualifiers to specify optional transpose or
conjugate operations: `c := alpha a.T @ b.H + beta c`.

The operands are SciPy and NumPy on the CPU.
"""

import numpy as np
import scipy.sparse as sp

import nvmath

dtype = np.complex64
m, n, k = 4, 2, 6

# Create a SciPy CSR array.
a = sp.random(k, m, density=0.2, format="csr", dtype=dtype)
print(f"a = \n{a}")

# Dense 'b' and 'c'.
b = np.ones((n, k), dtype=dtype)
print(f"b = \n{b}")

c = np.ones((m, n), dtype=dtype)

# Create the qualifiers for the operands. It must be a NumPy array of size 3 of
# dtype `nvmath.sparse.matmul_matrix_qualifiers_dtype`
qualifiers = np.zeros((3,), dtype=nvmath.sparse.matmul_matrix_qualifiers_dtype)

# Set transpose for `a` and hermitian (conjugate-transpose) for `b`.
qualifiers[0]["is_transpose"] = 1
qualifiers[1]["is_transpose"] = qualifiers[1]["is_conjugate"] = 1

# Create a stateful object that specifies the operation c := alpha a @ b + beta c
alpha, beta = 1.2, 2.4
with nvmath.sparse.Matmul(a, b, c, alpha=alpha, beta=beta, qualifiers=qualifiers) as mm:
    # Plan the SpMM operation. As we will see in later examples, planning
    # can be customized by providing prologs, epilog, and semiring operations.
    # The planning needs to be done only once for each problem specification.
    mm.plan()

    # Execute the operation.
    r = mm.execute()

    # The result `r` is `c` since the operation is in-place.
    assert r is c, "Error: the operation is not in-place."

print(f"c := {alpha} a @ b + {beta} c =\n{r}")
