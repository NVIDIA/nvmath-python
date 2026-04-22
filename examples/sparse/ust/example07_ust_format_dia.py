# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to specify the DIA format using the universal
sparse tensor (UST) DSL. While the existing formats are already defined and are
available via the `nvmath.sparse.ust.NamedFormats` interface, an understanding
of the DSL is needed for creating novel sparse formats.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

from nvmath.sparse.ust import Dimension, LevelFormat, NamedFormats, TensorFormat

# Let's specify the sparse format DIA using the UST DSL. We will see that there
# are two variants, which we denote as DIAI and DIAJ.

# Create two dimension objects representing the two axes in the matrix.
i, j = Dimension(dimension_name="i"), Dimension(dimension_name="j")

# In the prior examples, the "storage axes" followed the tensor axes. For
# formats like CSR and CSC the storage axes are the same as the tensor axes,
# and for formats like BSR and BSC each dimension was "fold-decomposed" into
# two levels. In this example, we will see that the UST DSL is more powerful
# and allows for specifying a storage axes to be along the diagonal (in general,
# a linear combination of the tensor axes).

# The level expression `j - i` specifies a diagonal, which is the first storage
# axis. To locate a value along the storage axis, we need to specify one more
# index, which can be either `i` or `j` resulting in the DIAI and DIAJ formats.
# Note that the level format for the second axis is RANGE, which is a elegant
# way of specifying a compression over a contiguous set of indices.
diai = TensorFormat([i, j], {j - i: LevelFormat.COMPRESSED, i: LevelFormat.RANGE}, name="DIAI")
print(f"DIAI format in the UST DSL:\n {diai}")

# Named formats such as DIAI are already available through the NamedFormats interface.
diai = NamedFormats.DIAI
print(f"The predefined DIAI named format, also in the UST DSL:\n {diai}")

diaj = TensorFormat([i, j], {j - i: LevelFormat.COMPRESSED, j: LevelFormat.RANGE}, name="DIAJ")
print(f"DIAJ format in the UST DSL:\n {diaj}")

# Named formats such as DIAJ are already available through the NamedFormats interface.
diaj = NamedFormats.DIAJ
print(f"The predefined DIAJ named format, also in the UST DSL:\n {diaj}")
