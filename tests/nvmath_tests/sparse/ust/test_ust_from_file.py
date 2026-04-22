# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile

import pytest

from nvmath.sparse.ust import Tensor


def test_from_matrix_market():
    data = """%%MatrixMarket matrix coordinate real general
4 5 6
1 1 1.0
2 2 2.0
2 3 3.0
3 4 4.0
4 1 5.0
4 2 6.0
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".mtx") as file:
        file.write(data)
        file.flush()
        ust = Tensor.from_file(file.name)
    assert ust.extents == [4, 5]
    assert ust.nse == 6
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([0, 1]) is None
    assert ust.get_value([1, 1]) == 2.0
    assert ust.get_value([1, 2]) == 3.0
    assert ust.get_value([2, 3]) == 4.0
    assert ust.get_value([3, 0]) == 5.0
    assert ust.get_value([3, 1]) == 6.0


def test_from_frostt():
    _ = pytest.importorskip("torch")  # reading tns uses torch

    data = """# FROSTT format
1 1 1 1.5
9 19 29 9.1
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".tns") as file:
        file.write(data)
        file.flush()
        ust = Tensor.from_file(file.name)
    assert ust.extents == [9, 19, 29]
    assert ust.nse == 2
    assert ust.get_value([0, 0, 0]) == 1.5
    assert ust.get_value([1, 1, 1]) is None
    assert ust.get_value([8, 18, 28]) == 9.1


def test_from_frostt_empty():
    _ = pytest.importorskip("torch")  # reading tns uses torch

    data = "# FROSTT format\n"
    with pytest.raises(ValueError) as e, tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".tns") as file:
        file.write(data)
        file.flush()
        _ = Tensor.from_file(file.name)
    message = str(e.value)
    assert re.match("cannot parse .* with FROSTT structure", message)
