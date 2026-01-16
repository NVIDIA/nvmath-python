# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import sys
import pytest

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from nvmath.linalg.generic import (
    ExecutionCUDA,
    GeneralMatrixQualifier,
    Matmul,
    matrix_qualifiers_dtype,
)


def test_unplanned():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    q = np.empty(2, dtype=matrix_qualifiers_dtype)
    q[:] = GeneralMatrixQualifier.create()

    with (
        pytest.raises(
            RuntimeError,
            match=r"Execution cannot be performed before plan\(\) has been called",
        ),
        Matmul(
            a,
            b,
            qualifiers=q,
            execution=ExecutionCUDA(),
        ) as mm,
    ):
        mm.execute()


def test_reset_operands():
    a = np.ones(shape=(4, 4))
    b = np.ones(shape=(4, 4))
    a1 = np.ones(shape=(4, 4)) * 2
    b1 = np.ones(shape=(4, 4)) * 3
    q = np.empty(2, dtype=matrix_qualifiers_dtype)
    q[:] = GeneralMatrixQualifier.create()

    with Matmul(
        a,
        b,
        qualifiers=q,
        execution=ExecutionCUDA(),
    ) as mm:
        mm.plan()
        r = mm.execute()
        mm.reset_operands(a1, b1)
        r1 = mm.execute()
    np.testing.assert_equal(6 * r, r1)


def test_reset_operands_new_shape():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    a1 = np.random.rand(4, 4)
    b1 = np.random.rand(4, 5)
    q = np.empty(2, dtype=matrix_qualifiers_dtype)
    q[:] = GeneralMatrixQualifier.create()

    with (
        pytest.raises(
            ValueError,
            match=r"The extents of the new operand must match the extents of the original operand.",
        ),
        Matmul(
            a,
            b,
            qualifiers=q,
            execution=ExecutionCUDA(),
        ) as mm,
    ):
        mm.plan()
        mm.execute()
        mm.reset_operands(a1, b1)
        mm.execute()


def test_reset_operands_new_dtype():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4).astype(np.double)
    a1 = np.random.rand(4, 4)
    b1 = np.random.rand(4, 4).astype(np.single)
    q = np.empty(2, dtype=matrix_qualifiers_dtype)
    q[:] = GeneralMatrixQualifier.create()

    with (
        pytest.raises(
            ValueError,
            match=r"The data type of the new operand must match the data type of the original operand.",
        ),
        Matmul(
            a,
            b,
            qualifiers=q,
            execution=ExecutionCUDA(),
        ) as mm,
    ):
        mm.plan()
        mm.execute()
        mm.reset_operands(a1, b1)
        mm.execute()


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
    # Prepare sample input data.
    m, n, k = 123, 456, 789
    a = cp.random.rand(m, k)
    b = cp.random.rand(k, n)

    assert sys.getrefcount(a) - 1 == 1, f"pre op: {sys.getrefcount(a) - 1}"
    assert sys.getrefcount(b) - 1 == 1, f"pre op: {sys.getrefcount(b) - 1}"

    # Optionally prepare c operand and create Matmul with or without it
    if use_c:
        c = cp.random.rand(m, n)
        assert sys.getrefcount(c) - 1 == 1, f"pre op: {sys.getrefcount(c) - 1}"
        mm = Matmul(a, b, c=c, beta=1.0)
    else:
        mm = Matmul(a, b)
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
