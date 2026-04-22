# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to specify the CSR format using the universal
sparse tensor (UST) DSL. While the existing formats are already defined and are
available via the `nvmath.sparse.ust.NamedFormats` interface, an understanding
of the DSL is needed for creating novel sparse formats.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

from nvmath.sparse.ust import Dimension, LevelFormat, NamedFormats, TensorFormat

# Let's specify a popular sparse format CSR using the UST DSL.

# Create two dimension objects representing the two axes in the matrix.
first, second = Dimension(dimension_name="first"), Dimension(dimension_name="second")

# The first dimension in the the CSR format is dense, while the second dimension
# is compressed.

# The first argument to the tensor format is a sequence of dimensions, while the
# second is the level specification. For the purposes of this example, we can
# treat each dimension as one level. In general, a dimension can be composed of
# more than one level, as we shall see in later examples.
csr = TensorFormat([first, second], {first: LevelFormat.DENSE, second: LevelFormat.COMPRESSED}, name="CSR")
print(f"CSR format in the UST DSL:\n {csr}")

# Named formats such as CSR are already available through the NamedFormats interface.
csr = NamedFormats.CSR
print(f"The predefined CSR named format, also in the UST DSL:\n {csr}")
