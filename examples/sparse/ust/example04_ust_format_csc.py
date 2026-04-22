# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to specify the CSC format using the universal
sparse tensor (UST) DSL. While the existing formats are already defined and are
available via the `nvmath.sparse.ust.NamedFormats` interface, an understanding
of the DSL is needed for creating novel sparse formats.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

from nvmath.sparse.ust import Dimension, LevelFormat, NamedFormats, TensorFormat

# Let's specify a common sparse format CSC using the UST DSL.

# Create two dimension objects representing the two axes in the matrix.
first, second = Dimension(dimension_name="first"), Dimension(dimension_name="second")

# The SECOND dimension in the the CSC format is dense, while the FIRST dimension
# is compressed.

# The first argument to the tensor format is a sequence of dimensions, while the
# second is the level specification. For the purposes of this example, we can
# treat each dimension as one level. In general, a dimension can be composed of
# more than one level, as we shall see in later examples.
csc = TensorFormat([first, second], {second: LevelFormat.DENSE, first: LevelFormat.COMPRESSED}, name="CSC")
print(f"CSC format in the UST DSL:\n {csc}")

# Named formats such as CSC are already available through the NamedFormats interface.
csc = NamedFormats.CSC
print(f"The predefined CSC named format, also in the UST DSL:\n {csc}")
