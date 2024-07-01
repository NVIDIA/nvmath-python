# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example show how to save algorithms from a planned and possibly autotuned matrix multiplication object.

The saved algorithms can be provided later for another compatible matrix multiplication operation, thereby avoiding
the cost of planning and autotuning.
"""
import os
import pickle

import cupy as cp

import nvmath

# Tip: turn logging on to get information on performance improvement from autotuning.
#import logging
#logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")

# Prepare sample input data
m, n, k = 2048, 4096, 1024
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)
bias = cp.random.rand(m, 1)

pickle_file = f"algorithms_{m}_{n}_{k}_f64_relu_bias.pickle"
# In the first pass, we will plan and autotune the matrix multiplication. Autotuning reorders the
# algorithms based on measured performance from fastest to slowest, and we will pickle the ordered
# algorithms.
print("= Phase 1: Plan, autotune, and save the optimal algorithm sequence. =")
with nvmath.linalg.advanced.Matmul(a, b) as mm:
    epilog = nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS
    mm.plan(epilog=epilog, epilog_inputs={"bias": bias})

    mm.autotune(iterations=5)

    # Save the algorithms as ordered by autotuning.
    with open(pickle_file, "wb") as f:
        pickle.dump(mm.algorithms, f)
    print(f"Saved optimized algorithms to '{pickle_file}' for later use.")

    # Execute the multiplication
    result = mm.execute()


print()
print("= Phase 2: Reuse the optimized algorithm sequence later in another compatible matrix multiplication. =")
# Load the algorithms saved earlier for use in a compatible matrix multiplication.
with open(pickle_file, "rb") as f:
    algorithms = pickle.load(f)
print(f"Loaded optimized algorithms from '{pickle_file}'.")

# In the second pass, we will provide the loaded algorithms to plan() to bypass
# planning and autotuning costs, since we already know the optimal algorithm(s) for this case.
with nvmath.linalg.advanced.Matmul(a, b) as mm:
    epilog = nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS

    # Provide the optimized algorithms directly to plan.
    mm.plan(algorithms=algorithms, epilog=epilog, epilog_inputs={"bias": bias})
    print(f"Provided optimized algorithms to plan(), bypassing planning cost.")
    print(f"No autotuning is needed, since the loaded algorithms sequence is in optimal order.")

    # Execute the multiplication
    result = mm.execute()
    print(f"Executed the matrix multiplication using the provided algorithms.")

    # Synchronize the default stream, since by default the execution is non-blocking for GPU operands.
    cp.cuda.get_current_stream().synchronize()

# Remove the pickle file.
os.remove(pickle_file)
