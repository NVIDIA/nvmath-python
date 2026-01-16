# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of the plan preference object to configure the
planning phase of a binary tensor contraction operation.

Different contraction algorithms are profiled to demonstrate how contraction algorithms
can potentially impact the performance of a tensor contraction operation.

For a detailed explanation of the contraction algorithms supported by cuTensor, please
refer to the cuTensor documentation:
https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensoralgo-t
"""

import cupy as cp
from cupyx.profiler import benchmark

import nvmath

a = cp.random.rand(64, 8, 8, 6, 6)
b = cp.random.rand(64, 8, 8, 6, 6)

# Create a stateful BinaryContraction object 'contraction'.
with nvmath.tensor.BinaryContraction("pijkl,pjiab->lakbp", a, b) as contraction:
    # Get the handle to the plan preference object
    plan_preference = contraction.plan_preference
    # update the kernel rank to the third best for the underlying algorithm
    plan_preference.kernel_rank = 2

    for algo in (
        nvmath.tensor.ContractionAlgo.DEFAULT_PATIENT,
        nvmath.tensor.ContractionAlgo.GETT,
        nvmath.tensor.ContractionAlgo.TGETT,
        nvmath.tensor.ContractionAlgo.TTGT,
        nvmath.tensor.ContractionAlgo.DEFAULT,
    ):
        print(f"Algorithm: {algo.name}")
        plan_preference.algo = algo
        # Plan the Contraction to activate the updated plan preference
        contraction.plan()
        print(benchmark(contraction.execute, n_repeat=20))
