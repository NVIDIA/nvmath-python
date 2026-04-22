# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to specify a novel format (batched CSR) using the
universal sparse tensor (UST) DSL. Each sample in the batch can potentially
have a different number of non-zero elements.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

from nvmath.sparse.ust import Dimension, LevelFormat, NamedFormats, TensorFormat

# Let's specify a novel sparse format (batched CSR) using the UST DSL.

# Create three dimension objects representing the batch axis as well as the two
# axes in the matrix.
batch, i, j = Dimension(dimension_name="batch"), Dimension(dimension_name="i"), Dimension(dimension_name="j")

# If the batch dimension is DENSE, it leads to a batch variant that supports varying NNZ
# across the samples in the batch. The first dimension in the CSR format is dense,
# while the second dimension is compressed.
batched_csr = TensorFormat(
    [batch, i, j], {batch: LevelFormat.DENSE, i: LevelFormat.DENSE, j: LevelFormat.COMPRESSED}, name="BatchedCSR"
)
print(f"A batched CSR format in the UST DSL supporting varying NNZ across the samples:\n {batched_csr}.")

# This format is also available through the NamedFormats interface.
batched_csr = NamedFormats.BatchedCSR
print(f"The predefined batched CSR named format, also in the UST DSL:\n {batched_csr}.")

# If the batch dimension is BATCH, it leads to an alternate batched CSR format where
# each sample in the batch has the same number of non-zeros. This corresponds to the
# torch version of batched CSR. As before, the first dimension in the CSR format
# is dense, while the second dimension is compressed.

# The BATCH level format can only be used in the outermost (those that appear on the
# left) levels.

torch_batched_csr = TensorFormat(
    [batch, i, j], {batch: LevelFormat.BATCH, i: LevelFormat.DENSE, j: LevelFormat.COMPRESSED}, name="TorchBatchedCSR"
)
print(f"A batched CSR format in the UST DSL requiring the same NNZ across the samples:\n {torch_batched_csr}.")
