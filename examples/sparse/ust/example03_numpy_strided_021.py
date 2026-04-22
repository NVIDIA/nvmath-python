# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to create a universal sparse tensor (UST) from
an existing NumPy strided ndarray. The layout is neither `C` (right) nor `F`
(left). Currently, the UST requires that the ndarray is dense (no holes) --
the UST is not a universal tensor (yet).

The nvmath-python UST currently supports multiple sparse matrix and tensor
libraries (SciPy, CuPy, PyTorch), multiple named sparse formats, and multiple
memory spaces.

The examples in this directory describe key features of the UST, and it is
recommended to read them in order.
"""

import numpy as np

from nvmath.sparse import ust

# Create a 3-D NumPy ndarray on the CPU.
shape = 2, 4, 3
a = np.random.rand(*shape).transpose(0, 2, 1)
print(f"The NumPy ndarray is:\n {a}.")
print(f"The shape is {a.shape}, with strides {tuple(s // 8 for s in a.strides)}.")

# Create an UST from the NumPy ndarray. The `from_package` method is zero-copy (it
# shares the data with the original NumPy ndarray).
u = ust.Tensor.from_package(a)
print(f"\nThe UST is:\n {u}.")
message = "Note the order of the 'k' and 'j' axes in `format`. We'll discuss the information printed \
above in a later example, after we look at the UST DSL."
print(message)

# The UST object is a view into the provided sparse tensor's data structure. A
# modification of either will be reflected in the other.
a[0, :, 0] = 3.14
print(f"\nThe modified NumPy ndarray is:\n {a}.")
print(f"\nThe changes above are reflected in the UST `u`:\n {u}.")
