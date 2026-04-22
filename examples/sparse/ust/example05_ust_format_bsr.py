# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to specify the BSR format using the universal
sparse tensor (UST) DSL. While the existing formats are already defined and are
available via the `nvmath.sparse.ust.NamedFormats` interface, an understanding
of the DSL is needed for creating novel sparse formats.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

from nvmath.sparse.ust import Dimension, LevelFormat, NamedFormats, TensorFormat

# Let's specify a well-known sparse format BSR using the UST DSL.

# Create two dimension objects representing the two axes in the matrix.
first, second = Dimension(dimension_name="first"), Dimension(dimension_name="second")


# The BSR format is essentially the CSR format with each scalar element replaced with
# a block of dimensions `m` x `n`. This example is the first to introduce the notion
# of level, and each of the two matrix axes maps to two levels. The first level for
# each matrix dimension locates the corresponding block coordinate, while the second
# level locates the coordinate within the block.

# The block shape.
m, n = 2, 3

# The example also introduces the notion of level expression, an algebraic expression
# of the dimension objects that map to the level format. This mapping is called
# the level specification.

# The first two entries in the level specification for the BSR are almost the same as
# for CSR, with the difference that the level expression (the key) now locates the
# *block*. The last two entries specify how to locate a value in the (dense) block.
bsr2x3 = TensorFormat(
    [first, second],
    {
        first // m: LevelFormat.DENSE,
        second // n: LevelFormat.COMPRESSED,
        first % m: LevelFormat.DENSE,
        second % n: LevelFormat.DENSE,
    },
    name="BSR",
)

print(f"BSR format in the UST DSL:\n {bsr2x3}")

# Named formats such as BSR are already available through the NamedFormats interface.
# We use `BSRRight` since the layout *within* each block is in `C` (right) order.
bsr2x3 = NamedFormats.BSRRight((2, 3))
print(f"The predefined BSR named format, also in the UST DSL:\n {bsr2x3}")
