# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

import nvmath

from ....helpers import check_freed_after
from ...utils import assert_tensors_equal, to_numpy

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np


@pytest.mark.skipif(HAS_CUPY is False, reason="CuPy is not available")
@pytest.mark.parametrize("use_plan_execute", [False, True], ids=["no_plan", "with_plan"])
@pytest.mark.parametrize("use_c", [False, True], ids=["no_c", "with_c"])
def test_reference_count(use_plan_execute, use_c):
    """
    Test reference counts consistency before/after context manager.
    Only need a single scenario with CuPy to test scenario when tensors reside on GPU.
    """
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

    mm = nvmath.linalg.advanced.Matmul(a, b, c, beta=1.0 if use_c else None)
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
    with nvmath.linalg.advanced.Matmul(a, b, c, beta=beta, options=options) as mm:
        mm.plan()

        # Autotune, and check that it doesn't clobber `c`.
        mm.autotune()
        assert_tensors_equal(c, c_orig)

        result = mm.execute()
    assert result is c, "Error: the operation is not in-place."

    reference = to_numpy(a) @ to_numpy(b) + beta * to_numpy(c_orig)
    assert_tensors_equal(result, reference)


@pytest.mark.skipif(HAS_CUPY is False, reason="CuPy is not available")
@pytest.mark.parametrize("memory_space", ["cuda", "cpu"])
@pytest.mark.parametrize("with_c", [False, True], ids=["no_c", "with_c"])
@pytest.mark.parametrize("use_plan_execute", [False, True], ids=["no_plan", "with_plan"])
def test_release_operands_main(memory_space, with_c, use_plan_execute):
    """
    Test that after release_operands(), the refcounts of user-provided
    main operands (a, b, c) return to their initial values.
    """
    m, n, k = 128, 256, 128

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

    mm = nvmath.linalg.advanced.Matmul(a, b, c, beta=1.0 if with_c else None)
    if use_plan_execute:
        mm.plan()
        _ = mm.execute()
    else:
        pass

    if memory_space == "cuda":
        cp.cuda.get_current_stream().synchronize()

    mm.release_operands()

    # Verify
    assert sys.getrefcount(a) == initial_refcounts["a"]
    assert sys.getrefcount(b) == initial_refcounts["b"]
    if c is not None:
        assert sys.getrefcount(c) == initial_refcounts["c"]

    mm.free()


def test_release_operands_cpu_inplace():
    """
    Test that release_operands() releases cpu_c_ref for CPU inplace case.
    """
    m, n, k = 128, 256, 128

    a = np.random.rand(m, k).astype("float32")
    b = np.random.rand(k, n).astype("float32")
    c = np.random.rand(m, n).astype("float32")
    initial_refcount_a = sys.getrefcount(a)
    initial_refcount_b = sys.getrefcount(b)
    initial_refcount_c = sys.getrefcount(c)

    # Create Matmul with inplace=True
    mm = nvmath.linalg.advanced.Matmul(a, b, c, beta=1.0, options={"inplace": True})
    mm.plan()
    result = mm.execute()

    # result should hold a reference to c inplace case
    assert sys.getrefcount(c) > initial_refcount_c, (
        f"c refcount after execute (before del result): {sys.getrefcount(c)}, "
        f"expected: > {initial_refcount_c}. Result should be holding a reference to c for inplace."
    )

    # this must be here or we will leak a extra reference corrupting the test
    del result

    mm.release_operands()

    # Verify
    assert sys.getrefcount(c) == initial_refcount_c
    assert sys.getrefcount(a) == initial_refcount_a
    assert sys.getrefcount(b) == initial_refcount_b

    mm.free()


@pytest.mark.skipif(HAS_CUPY is False, reason="CuPy is not available")
@pytest.mark.parametrize("memory_space", ["cuda", "cpu"])
@pytest.mark.parametrize("epilog_type", ["bias", "relu_aux", "gelu_aux"], ids=["bias_epilog", "drelu_epilog", "dgelu_epilog"])
def test_release_operands_epilog(memory_space, epilog_type):
    """
    Test that after release_operands(), the refcounts of user-provided
    epilog inputs return to their initial values.
    """
    m, n, k = 128, 256, 128

    # Create operands on CPU or GPU based on memory_space
    if memory_space == "cuda":
        a = cp.random.rand(m, k).astype("float32")
        b = cp.random.rand(k, n).astype("float32")
    else:
        a = np.random.rand(m, k).astype("float32")
        b = np.random.rand(k, n).astype("float32")

    # Record initial refcounts
    initial_refcounts = {"a": sys.getrefcount(a), "b": sys.getrefcount(b)}

    # Setup epilog and epilog inputs based on type
    if epilog_type == "bias":
        epilog = nvmath.linalg.advanced.MatmulEpilog.BIAS
        if memory_space == "cuda":
            epilog_input = cp.random.rand(m, 1).astype("float32")
        else:
            epilog_input = np.random.rand(m, 1).astype("float32")
        epilog_inputs = {"bias": epilog_input}
        initial_refcounts["epilog_input"] = sys.getrefcount(epilog_input)

    elif epilog_type == "relu_aux":
        # For DRELU, we need relu_aux from a forward pass
        epilog = nvmath.linalg.advanced.MatmulEpilog.DRELU
        relu_aux_m = ((m + 7) // 8 + 15) // 16 * 16
        if memory_space == "cuda":
            epilog_input = cp.asfortranarray(cp.random.randint(0, 2, size=(relu_aux_m, n), dtype="uint8"))
        else:
            epilog_input = np.asfortranarray(np.random.randint(0, 2, size=(relu_aux_m, n), dtype="uint8"))
        epilog_inputs = {"relu_aux": epilog_input}
        initial_refcounts["epilog_input"] = sys.getrefcount(epilog_input)

    else:  # gelu_aux
        epilog = nvmath.linalg.advanced.MatmulEpilog.DGELU
        padded_m = ((m + 7) // 8) * 8
        if memory_space == "cuda":
            epilog_input = cp.asfortranarray(cp.random.rand(padded_m, n).astype("float32"))
        else:
            epilog_input = np.asfortranarray(np.random.rand(padded_m, n).astype("float32"))
        epilog_inputs = {"gelu_aux": epilog_input}
        initial_refcounts["epilog_input"] = sys.getrefcount(epilog_input)

    mm = nvmath.linalg.advanced.Matmul(a, b)
    mm.plan(epilog=epilog, epilog_inputs=epilog_inputs)
    _ = mm.execute()
    if memory_space == "cuda":
        cp.cuda.get_current_stream().synchronize()

    mm.release_operands()
    # Verify
    assert sys.getrefcount(a) == initial_refcounts["a"]
    assert sys.getrefcount(b) == initial_refcounts["b"]
    assert sys.getrefcount(epilog_input) == initial_refcounts["epilog_input"]

    mm.free()


@pytest.mark.skipif(HAS_CUPY is False, reason="CuPy is not available")
def test_release_operands_then_execute_fails():
    m, n, k = 128, 256, 128
    a = cp.random.rand(m, k).astype("float32")
    b = cp.random.rand(k, n).astype("float32")

    mm = nvmath.linalg.advanced.Matmul(a, b)
    mm.plan()
    _ = mm.execute()
    mm.release_operands()
    with pytest.raises(RuntimeError, match="cannot be performed after the operands have been released"):
        mm.execute()

    mm.free()


@pytest.mark.skipif(HAS_CUPY is False, reason="CuPy is not available")
def test_release_operands_then_reset_works():
    m, n, k = 128, 256, 128
    a = cp.random.rand(m, k).astype("float32")
    b = cp.random.rand(k, n).astype("float32")

    mm = nvmath.linalg.advanced.Matmul(a, b)
    mm.plan()
    _ = mm.execute()
    mm.release_operands()

    a_new = cp.random.rand(m, k).astype("float32")
    b_new = cp.random.rand(k, n).astype("float32")
    mm.reset_operands(a=a_new, b=b_new)
    result2 = mm.execute()
    reference = to_numpy(a_new) @ to_numpy(b_new)
    assert_tensors_equal(result2, reference)

    mm.free()
