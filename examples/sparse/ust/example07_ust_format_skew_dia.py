# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to specify the SKEW DIA format using the universal
sparse tensor (UST) DSL. While the existing formats are already defined and are
available via the `nvmath.sparse.ust.NamedFormats` interface, an understanding
of the DSL is needed for creating novel sparse formats.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

from nvmath.sparse.ust import Dimension, LevelFormat, NamedFormats, TensorFormat

# Let's specify the sparse format SKEW DIA using the UST DSL. We will see that there
# are two variants, which we denote as SkewDIAI and SkewDIAJ.

# Create two dimension objects representing the two axes in the matrix.
i, j = Dimension(dimension_name="i"), Dimension(dimension_name="j")

# In most of the prior examples, the "storage axes" followed the tensor axes. For
# formats like CSR and CSC the storage axes are the same as the tensor axes,
# and for formats like BSR and BSC each dimension was "fold-decomposed" into
# two levels. As we saw in the DIA example, we will see here that the UST DSL is more
# powerful and allows for specifying a storage axes to be along the diagonal (in
# general, a linear combination of the tensor axes).

# The level expression `i + j` specifies a skew-diagonal, which is the first storage
# axis. To locate a value along the storage axis, we need to specify one more
# index, which can be either `i` or `j` resulting in the DIAI and DIAJ formats.
# Note that the level format for the second axis is RANGE, which is a elegant
# way of specifying a compression over a contiguous set of indices.
skew_diai = TensorFormat([i, j], {i + j: LevelFormat.COMPRESSED, i: LevelFormat.RANGE}, name="SkewDIAI")
print(f"SkewDIAI format in the UST DSL:\n {skew_diai}")

# Named formats such as SkewDIAI are already available through the NamedFormats interface.
skew_diai = NamedFormats.SkewDIAI
print(f"The predefined SkewDIAI named format, also in the UST DSL:\n {skew_diai}")

skew_diaj = TensorFormat([i, j], {i + j: LevelFormat.COMPRESSED, j: LevelFormat.RANGE}, name="SkewDIAJ")
print(f"SkewDIAJ format in the UST DSL:\n {skew_diaj}")

# Named formats such as SkewDIAJ are already available through the NamedFormats interface.
skew_diaj = NamedFormats.SkewDIAJ
print(f"The predefined SkewDIAJ named format, also in the UST DSL:\n {skew_diaj}")
