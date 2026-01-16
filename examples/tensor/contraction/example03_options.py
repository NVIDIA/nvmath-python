# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to specify options to a binary tensor contraction operation.

In this example, we will use NumPy ndarrays as input, and we will look at two equivalent
ways to specify the compute type.
"""

import numpy as np

import nvmath

np.random.seed(0)
a = np.random.rand(8, 8, 12, 12)
b = np.random.rand(12, 12, 8, 8)

c = np.random.rand(8, 8, 8, 8)

# Alternative #1 for specifying options, using ContractionOptions class.
options = nvmath.tensor.ContractionOptions(compute_type=nvmath.tensor.ComputeDesc.COMPUTE_32F(), memory_limit="1GB")
result = nvmath.tensor.binary_contraction("ijkl,klmn->ijmn", a, b, c=c, beta=1, options=options)

assert np.allclose(result, np.einsum("ijkl,klmn->ijmn", a, b) + c)

# Alternative #2 for specifying options, using dict. The two alternatives are entirely
# equivalent.
options = {"compute_type": nvmath.tensor.ComputeDesc.COMPUTE_32F(), "memory_limit": "1GB"}
result = nvmath.tensor.binary_contraction("ijkl,klmn->ijmn", a, b, c=c, beta=1, options=options)

assert np.allclose(result, np.einsum("ijkl,klmn->ijmn", a, b) + c)
