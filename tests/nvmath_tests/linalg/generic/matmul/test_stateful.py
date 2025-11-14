import numpy as np
import pytest

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
