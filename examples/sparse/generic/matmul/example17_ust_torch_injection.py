# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates how to transparently "inject" UST into a torch neural
network model. This enables exploring the performance of "sparsified" NNs using
a range of sparse formats (including novel ones written in the UST DSL).
"""

import torch

from nvmath.sparse.ust.interfaces import torch_interface

device_id = 0

# The problem details.
dtype = torch.float32
n1, n2, n3, n4 = 128, 64, 32, 16


# Create a simple neural network, consisting of three linear (fully-connected) layers.
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(n1, n2, bias=True)
        self.fc1 = torch.nn.Linear(n2, n3, bias=True)
        self.fc2 = torch.nn.Linear(n3, n4, bias=True)

    def forward(self, x):
        x0 = x.view(-1, n1)  # Flatten all except batch dimensions.
        a0 = torch.nn.functional.relu(self.fc0(x0))
        a1 = torch.nn.functional.relu(self.fc1(a0))
        a2 = torch.nn.functional.relu(self.fc2(a1))
        return a2


# Set the seed, and create the model.
torch.manual_seed(0)
model = SimpleNet().to(device_id)

# Set all weight matrices to 50% sparse. This step mimics training a model,
# pruning the weights, fine-tuning the model, and then finally loading the
# trained parameters prior to using the model for inference.
with torch.no_grad():
    model.fc0.weight.data[model.fc0.weight.data < 0] = 0
    model.fc1.weight.data[model.fc1.weight.data < 0] = 0
    model.fc2.weight.data[model.fc2.weight.data < 0] = 0

# Set up sample inputs for unbatched inference.
x1 = torch.ones((n1,), dtype=dtype, device=device_id)
x2 = torch.rand((n1,), dtype=dtype, device=device_id)

# Perform the inference with dense GEMM.
with torch.no_grad():
    y1 = model(x1)
    y2 = model(x2)


# User-defined UST injection/reformatting function, conditionally sparsifying to CSR in
# this case.
def reformat_func(weight):
    density = torch.count_nonzero(weight) * 100.0 / weight.numel()
    # This is where any analysis can be done.
    ...
    # This is where any pruning can be done.
    ...
    print(f"The density of the weight matrix with shape {tuple(weight.shape)} is {density:.2f}%")
    # Reformat to CSR (or any other sparse format) based on a threshold.
    if density < 60.0:
        return torch_interface.TorchUST.from_torch(weight.to_sparse_csr())


# Inject the UST into the torch NN model.
torch_interface.reformat_model(model, func=reformat_func)

# Perform the inference with UST SpMMs. Note that the second and subsequent forward passes
# in the model use the already planned and cached MMs from the first invocation.
with torch.no_grad():
    z1 = model(x1)
    z2 = model(x2)

# Check that the results are the same.
assert torch.allclose(y1, z1), "Error: the results don't match y1 != z1."
assert torch.allclose(y2, z2), "Error: the results don't match y2 != z2."
