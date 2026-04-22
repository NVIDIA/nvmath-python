# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import sys

import numpy as np
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

from ....helpers import check_freed_after


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
        mm.reset_operands(a=a1, b=b1)
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
        mm.reset_operands(a=a1, b=b1)
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
        mm.reset_operands(a=a1, b=b1)
        mm.execute()


@pytest.mark.skipif(HAS_CUPY is False, reason="CuPy is not available")
@pytest.mark.parametrize("use_plan_execute", [False, True], ids=["no_plan", "with_plan"])
@pytest.mark.parametrize("use_c", [False, True], ids=["no_c", "with_c"])
def test_reference_count(use_plan_execute, use_c):
    """
    Test reference counts consistency before/after context manager.
    Only need a single scenario with CuPy to test scenario when tensors reside on GPU.
    """
    # Prepare sample input data.
    m, n, k = 123, 456, 789
    a = cp.random.rand(m, k)
    b = cp.random.rand(k, n)
    # Starting from Python 3.14, sys.getrefcount() behavior changes due to
    # LOAD_FAST_BORROW, a bytecode optimization that skips the refcount
    # increment when the compiler's lifetime analysis can prove the local
    # variable outlives the stack reference.  Whether the optimization
    # applies can vary between load sites depending on code structure.
    # Assigning c here (rather than inside the `if use_c` branch) ensures
    # consistent behavior across all sys.getrefcount(c) calls.
    # See https://github.com/python/cpython/issues/130704 for example.
    c = cp.random.rand(m, n) if use_c else None

    initial_refcount_a = sys.getrefcount(a)
    initial_refcount_b = sys.getrefcount(b)
    if c is not None:
        initial_refcount_c = sys.getrefcount(c)

    mm = Matmul(a, b, c=c, beta=1.0 if use_c else None)
    with mm:
        if use_plan_execute:
            mm.plan()
            result = mm.execute()
            cp.cuda.get_current_stream().synchronize()
        else:
            pass

    # After exiting context manager, reference counts should return to initial values
    assert sys.getrefcount(a) == initial_refcount_a, "post op: a refcount changed"
    assert sys.getrefcount(b) == initial_refcount_b, "post op: b refcount changed"
    if c is not None:
        assert sys.getrefcount(c) == initial_refcount_c, "post op: c refcount changed"
    if use_plan_execute:
        with check_freed_after(result, "post op: result should have sole ownership"):
            del result


@pytest.mark.parametrize("memory_space", ["cuda", "cpu"])
@pytest.mark.parametrize("with_c", [False, True], ids=["no_c", "with_c"])
@pytest.mark.parametrize("use_plan_execute", [False, True], ids=["no_plan", "with_plan"])
def test_release_operands_ref_count(memory_space, with_c, use_plan_execute):
    if memory_space == "cuda":
        cp = pytest.importorskip("cupy")

    m, n, k = 64, 32, 23
    execution = ExecutionCUDA()
    if memory_space == "cuda":
        a = cp.random.rand(m, k).astype("float32")
        b = cp.random.rand(k, n).astype("float32")
        c = cp.random.rand(m, n).astype("float32") if with_c else None
    else:
        a = np.random.rand(m, k).astype("float32")
        b = np.random.rand(k, n).astype("float32")
        c = np.random.rand(m, n).astype("float32") if with_c else None

    # Record initial refcounts
    initial_refcounts = {"a": sys.getrefcount(a), "b": sys.getrefcount(b)}
    if c is not None:
        initial_refcounts["c"] = sys.getrefcount(c)

    mm = Matmul(a, b, c, beta=1.0 if with_c else None, execution=execution)
    if use_plan_execute:
        mm.plan()
        _ = mm.execute()

    if memory_space == "cuda":
        cp.cuda.get_current_stream().synchronize()

    mm.release_operands()

    # Verify refcounts return to initial values
    assert sys.getrefcount(a) == initial_refcounts["a"]
    assert sys.getrefcount(b) == initial_refcounts["b"]
    if c is not None:
        assert sys.getrefcount(c) == initial_refcounts["c"]

    mm.free()


@pytest.mark.parametrize("memory_space", ["cuda", "cpu"])
def test_release_operands_then_execute_fails(memory_space):
    """
    Test that execute() raises an error after release_operands().
    """
    if memory_space == "cuda":
        cp = pytest.importorskip("cupy")

    m, n, k = 64, 32, 23
    execution = ExecutionCUDA()
    if memory_space == "cuda":
        a = cp.random.rand(m, k).astype("float32")
        b = cp.random.rand(k, n).astype("float32")
    else:
        a = np.random.rand(m, k).astype("float32")
        b = np.random.rand(k, n).astype("float32")

    mm = Matmul(a, b, execution=execution)
    mm.plan()
    _ = mm.execute()
    if memory_space == "cuda":
        cp.cuda.get_current_stream().synchronize()

    mm.release_operands()
    with pytest.raises(ValueError, match="Operands have been released"):
        mm.execute()

    mm.free()


@pytest.mark.parametrize("memory_space", ["cuda", "cpu"])
def test_release_operands_then_reset_works(memory_space):
    if memory_space == "cuda":
        cp = pytest.importorskip("cupy")

    m, n, k = 64, 32, 23
    execution = ExecutionCUDA()
    if memory_space == "cuda":
        a = cp.random.rand(m, k).astype("float32")
        b = cp.random.rand(k, n).astype("float32")
        a_new = cp.random.rand(m, k).astype("float32")
        b_new = cp.random.rand(k, n).astype("float32")
    else:
        a = np.random.rand(m, k).astype("float32")
        b = np.random.rand(k, n).astype("float32")
        a_new = np.random.rand(m, k).astype("float32")
        b_new = np.random.rand(k, n).astype("float32")

    mm = Matmul(a, b, execution=execution)
    mm.plan()
    _ = mm.execute()
    if memory_space == "cuda":
        cp.cuda.get_current_stream().synchronize()

    mm.release_operands()
    mm.reset_operands(a=a_new, b=b_new)
    result2 = mm.execute()

    if memory_space == "cuda":
        reference = cp.asnumpy(a_new) @ cp.asnumpy(b_new)
        np.testing.assert_allclose(cp.asnumpy(result2), reference, rtol=2e-5)
    else:
        reference = a_new @ b_new
        np.testing.assert_allclose(result2, reference, rtol=2e-5)

    mm.free()


@pytest.mark.parametrize("memory_space", ["cuda", "cpu"])
@pytest.mark.parametrize("with_c", [False, True], ids=["no_c", "with_c"])
def test_release_operands_reset_requires_all(memory_space, with_c):
    if memory_space == "cuda":
        cp = pytest.importorskip("cupy")

    m, n, k = 64, 32, 23
    execution = ExecutionCUDA()
    if memory_space == "cuda":
        a = cp.random.rand(m, k).astype("float32")
        b = cp.random.rand(k, n).astype("float32")
        c = cp.random.rand(m, n).astype("float32") if with_c else None
        a_new = cp.random.rand(m, k).astype("float32")
        b_new = cp.random.rand(k, n).astype("float32")
        c_new = cp.random.rand(m, n).astype("float32") if with_c else None
    else:
        a = np.random.rand(m, k).astype("float32")
        b = np.random.rand(k, n).astype("float32")
        c = np.random.rand(m, n).astype("float32") if with_c else None
        a_new = np.random.rand(m, k).astype("float32")
        b_new = np.random.rand(k, n).astype("float32")
        c_new = np.random.rand(m, n).astype("float32") if with_c else None

    mm = Matmul(a, b, c, beta=1.0 if with_c else None, execution=execution)
    mm.plan()
    _ = mm.execute()

    if memory_space == "cuda":
        cp.cuda.get_current_stream().synchronize()

    mm.release_operands()

    # Test various incomplete reset scenarios
    with pytest.raises(ValueError, match="After release_operands.*all required operands must be provided"):
        mm.reset_operands(a=a_new)
    with pytest.raises(ValueError, match="After release_operands.*all required operands must be provided"):
        mm.reset_operands(b=b_new)

    if with_c:
        with pytest.raises(ValueError, match="After release_operands.*all required operands must be provided"):
            mm.reset_operands(a=a_new, b=b_new)
        with pytest.raises(ValueError, match="After release_operands.*all required operands must be provided"):
            mm.reset_operands(a=a_new, c=c_new)
        with pytest.raises(ValueError, match="After release_operands.*all required operands must be provided"):
            mm.reset_operands(b=b_new, c=c_new)

    # Now provide all operands - should work
    if with_c:
        mm.reset_operands(a=a_new, b=b_new, c=c_new)
    else:
        mm.reset_operands(a=a_new, b=b_new)

    # Verify execute works after proper reset and result is correct
    result = mm.execute()

    if memory_space == "cuda":
        cp.cuda.get_current_stream().synchronize()
        if with_c:
            reference = cp.asnumpy(a_new) @ cp.asnumpy(b_new) + cp.asnumpy(c_new)
        else:
            reference = cp.asnumpy(a_new) @ cp.asnumpy(b_new)
        np.testing.assert_allclose(cp.asnumpy(result), reference, rtol=2e-5)
    else:
        if with_c:
            reference = a_new @ b_new + c_new
        else:
            reference = a_new @ b_new
        np.testing.assert_allclose(result, reference, rtol=2e-5)

    mm.free()
