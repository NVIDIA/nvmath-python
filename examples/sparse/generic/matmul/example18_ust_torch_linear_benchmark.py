# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates injecting UST into a torch NN.
"""

import torch
from cupyx.profiler import benchmark

from nvmath.sparse.ust.interfaces import torch_interface

device_id = 0

# The NN details.
m = 1024 * 8
n = 1024 * 4
batch = 64

# Set up a m x n linear layer with a 99.9% sparse weight matrix. Note that this kind
# of sparsity is unrealistic for a weight matrix but is used for illustration purposes
# only.

linear = torch.nn.Linear(m, n, bias=False)
with torch.no_grad():
    torch.manual_seed(0)
    w = torch.rand(n, m)
    w[w < 0.999] = 0.0
    linear.weight.copy_(w)
linear.to(device=device_id)


# Use two almost identical inference methods for benchmarking (because torch sometimes takes
# slightly different paths through the code for no_grad vs. inference_mode).
def infer1(x):
    with torch.no_grad():
        return linear(x)


def infer2(x):
    with torch.inference_mode():
        return linear(x)


# Create sample data for inference.
x = torch.arange(1.0, batch * m + 1.0, device=device_id).reshape(batch, m)
y = infer1(x)
z = infer2(x)
# Sanity check with dense MM.
assert torch.allclose(y, z), "Error: the results from the two inference methods don't match."

# Benchmark the dense linear layer with MM=GEMM.
p = benchmark(infer1, (x,), n_repeat=100)
print(f"torch (no grad mode)   >>> {p}")
p = benchmark(infer2, (x,), n_repeat=100)
print(f"torch (inference mode) >>> {p}")


# Reformat the "model", consisting of a single linear layer. This will replace
# the weight matrix with a UST in COO format.
def reformat(weight):
    nel = weight.numel()
    nnz = torch.count_nonzero(weight)
    sparsity = (1.0 - float(nnz) / float(nel)) * 100.0
    print(f" The shape of the weight matrix is {tuple(weight.shape)}, with sparsity {sparsity:0.3f} ({nnz} out of {nel}).")
    if sparsity > 0.9:
        return torch_interface.TorchUST.from_torch(weight.to_sparse())


torch_interface.reformat_model(linear, func=reformat)

# Sanity check on output. The GEMM is SpMM now.
y_s = infer1(x)
z_s = infer2(x)
assert torch.allclose(y, y_s), "Error: SpMM result != MM result."
assert torch.allclose(z, z_s), "Error: SpMM result != MM result."

# Benchmark the sparse linear layer with MM=SpMM.
p = benchmark(infer1, (x,), n_repeat=100)
print(f"UST (no grad mode)     >>> {p}")
p = benchmark(infer2, (x,), n_repeat=100)
print(f"UST (inference mode)   >>> {p}")
