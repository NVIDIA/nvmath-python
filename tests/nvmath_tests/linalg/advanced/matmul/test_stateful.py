# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import nvmath
import sys

from ...utils import assert_tensors_equal, to_numpy

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.mark.skipif(HAS_CUPY is False, reason="CuPy is not available")
@pytest.mark.parametrize("use_plan_execute", [False, True], ids=["no_plan", "with_plan"])
@pytest.mark.parametrize("use_c", [False, True], ids=["no_c", "with_c"])
def test_reference_count(use_plan_execute, use_c):
    """
    Test reference counts consistency before/after context manager.
    Only need a single scenario with CuPy to test scenario when tensors reside on GPU.

    Note: sys.getrefcount() adds 1 for the temp reference in getrefcount,
    so we need to subtract 1 to have the actual value.
    """
    m, n, k = 123, 456, 789
    a = cp.random.rand(m, k)
    b = cp.random.rand(k, n)

    assert sys.getrefcount(a) - 1 == 1, f"pre op: {sys.getrefcount(a) - 1}"
    assert sys.getrefcount(b) - 1 == 1, f"pre op: {sys.getrefcount(b) - 1}"

    # Create Matmul with or without c operand
    if use_c:
        c = cp.random.rand(m, n)
        assert sys.getrefcount(c) - 1 == 1, f"pre op: {sys.getrefcount(c) - 1}"
        mm = nvmath.linalg.advanced.Matmul(a, b, c, beta=1.0)
    else:
        mm = nvmath.linalg.advanced.Matmul(a, b)
    with mm:
        if use_plan_execute:
            mm.plan()
            result = mm.execute()
            cp.cuda.get_current_stream().synchronize()
        else:
            pass

    # After exiting context manager, reference counts should return to 1
    assert sys.getrefcount(a) - 1 == 1, f"post op: {sys.getrefcount(a) - 1}"
    assert sys.getrefcount(b) - 1 == 1, f"post op: {sys.getrefcount(b) - 1}"
    if use_c:
        assert sys.getrefcount(c) - 1 == 1, f"post op: {sys.getrefcount(c) - 1}"
    if use_plan_execute:
        assert sys.getrefcount(result) - 1 == 1, f"post op: {sys.getrefcount(result) - 1}"


@pytest.mark.skipif(HAS_CUPY is False, reason="CuPy is not available")
def test_inplace_autotune():
    """
    Test autotuning along with the inplace option.
    """
    m, n, k = 123, 456, 789
    a = cp.random.rand(m, k)
    b = cp.random.rand(k, n)
    c = cp.random.rand(m, n)
    c_orig = c.copy()

    options = {"inplace": True}
    beta = 1.0
    mm = nvmath.linalg.advanced.Matmul(a, b, c, beta=beta, options=options)
    mm.plan()

    # Autotune, and check that it doesn't clobber `c`.
    mm.autotune()
    assert_tensors_equal(c, c_orig)

    result = mm.execute()
    assert result is c, "Error: the operation is not in-place."

    reference = to_numpy(a) @ to_numpy(b) + beta * to_numpy(c_orig)
    assert_tensors_equal(result, reference)
