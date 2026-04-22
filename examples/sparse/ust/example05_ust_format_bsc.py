# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to specify the BSC format using the universal
sparse tensor (UST) DSL. While the existing formats are already defined and are
available via the `nvmath.sparse.ust.NamedFormats` interface, an understanding
of the DSL is needed for creating novel sparse formats.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

from nvmath.sparse.ust import Dimension, LevelFormat, NamedFormats, TensorFormat

# Let's specify a well-known sparse format BSC using the UST DSL.

# Create two dimension objects representing the two axes in the matrix.
first, second = Dimension(dimension_name="first"), Dimension(dimension_name="second")


# The BSC format is essentially the CSC format with each scalar element replaced with
# a block of dimensions `m` x `n`. This example is the first to introduce the notion
# of level, and each of the two matrix axes maps to two levels. The first level for
# each matrix dimension locates the corresponding block coordinate, while the second
# level locates the coordinate within the block.

# The block shape.
m, n = 2, 3

# The example also introduces the notion of level expression, an algebraic expression
# of the dimension objects that map to the level format. This mapping is called
# the level specification.

# The first two entries in the level specification for the BSC are almost the same as
# for CSC, with the difference that the level expression (the key) now locates the
# *block*. The last two entries specify how to locate a value in the (dense) block.
bsc2x3 = TensorFormat(
    [first, second],
    {
        second // n: LevelFormat.DENSE,
        first // m: LevelFormat.COMPRESSED,
        first % m: LevelFormat.DENSE,
        second % n: LevelFormat.DENSE,
    },
    name="BSC",
)

print(f"BSC format in the UST DSL:\n {bsc2x3}")

# Named formats such as BSC are already available through the NamedFormats interface.
# We use `BSCRight` since the layout *within* each block is in `C` (right) order.
bsc2x3 = NamedFormats.BSCRight((2, 3))
print(f"The predefined BSC named format, also in the UST DSL:\n {bsc2x3}")
