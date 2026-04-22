# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to specify the COO format using the universal
sparse tensor (UST) DSL. While the existing formats are already defined and are
available via the `nvmath.sparse.ust.NamedFormats` interface, an understanding
of the DSL is needed for creating novel sparse formats.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

from nvmath.sparse.ust import Dimension, LevelFormat, LevelProperty, NamedFormats, TensorFormat

# Let's specify a common sparse format COO using the UST DSL.

# Create two dimension objects representing the two axes in the matrix.
i, j = Dimension(dimension_name="i"), Dimension(dimension_name="j")

# The first COO format axis is compressed, similar to how the second axis
# is for CSR or CSC. However, an index along the compressed axis can appear
# more than once unlike for CSR or CSC. This is specified by using a level
# property along with the level format. Together the level format and the
# level property form the level type. With this, there is only one entry for
# the second axis, which is specified using the singleton format.
coo = TensorFormat([i, j], {i: (LevelFormat.COMPRESSED, LevelProperty.NONUNIQUE), j: LevelFormat.SINGLETON}, name="COO")
print(f"COO format in the UST DSL:\n {coo}")

# In summary, the UST tensor format is a sequence of dimensions followed by the level
# specification.
# The level specification is a mapping between each level expression and a level type.
# The level type is a level format or the pair (level format, level property).

# Named formats such as COO are already available through the NamedFormats interface.
coo = NamedFormats.COO
print(f"The predefined COO named format, also in the UST DSL:\n {coo}")
