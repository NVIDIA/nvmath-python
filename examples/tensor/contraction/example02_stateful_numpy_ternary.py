# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful ternary tensor contraction objects.
Stateful objects amortize the cost of preparation across multiple executions.

The inputs as well as the result are NumPy ndarrays.
"""

import numpy as np

import nvmath

a = np.random.rand(4, 6, 8)
b = np.random.rand(6, 8, 3)
c = np.random.rand(3, 9)


# result[i,j,m,n] = \sum_{k,l} a[i,j,k] * b[j,k,l] * c[l,n]

# Create a stateful TernaryContraction object 'contraction'.
with nvmath.tensor.TernaryContraction("ijk,jkl,ln->in", a, b, c) as contraction:
    # Plan the Contraction.
    contraction.plan()

    # Execute the Contraction.
    result = contraction.execute()

    assert np.allclose(result, np.einsum("ijk,jkl,ln->in", a, b, c))
    print(f"Input type = {type(a)}, device = 'cpu'")
    print(f"Contraction output type = {type(result)}, device = 'cpu'")
