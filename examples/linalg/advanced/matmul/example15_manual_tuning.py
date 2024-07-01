# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates the possibility to tweak algorithm's configuration manually.

You are free to modify algorithm configuration as long as it's consistent with its capabilities.
As an alternative to manual fine-tuning, you might want to try autotuning - see `autotune` example.
"""

import nvmath
import cupy as cp

# Prepare sample input data
m = n = k = 1024
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)

# Use the stateful object as a context manager to automatically release resources.
with nvmath.linalg.advanced.Matmul(a, b) as mm:

    # Plan.
    mm.plan()

    # Inspect the algorithms proposed.
    print(f"Planning returned {len(mm.algorithms)} algorithms. The capabilities of the best one are:",)
    best = mm.algorithms[0]
    print(best.capabilities)

    # Modify the tiling configuration of the algorithm. Note that the valid tile configuration depends on
    # the hardware, and not all combinations of the configuration are supported, so we leave it as an exercise.
    best.tile = best.tile
    print(f"Modified the tile to be {best.tile}")
    # Execute the multiplication.
    result = mm.execute()

    # Synchronize the default stream, since by default the execution is non-blocking for GPU operands.
    cp.cuda.get_current_stream().synchronize()
    print(f"Input types = {type(a), type(b)}, device = {a.device, b.device}")
    print(f"Result type = {type(result)}, device = {result.device}")
