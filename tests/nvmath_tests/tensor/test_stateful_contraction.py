# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import sys

import numpy as np
import pytest

from nvmath.tensor import BinaryContraction, ContractionAutotuneMode, ContractionCacheMode, TernaryContraction

from ..helpers import check_freed_after
from .utils.base_testers import run_coefficients_test_impl
from .utils.check_helpers import assert_all_close, get_contraction_ref, get_contraction_tolerance
from .utils.common_axes import (
    AlgoOption,
    AutotuneModeOption,
    CacheModeOption,
    DType,
    Framework,
    IncrementalCountOption,
    JitOption,
    KernelRankOption,
    MemBackend,
)
from .utils.data import binary_contraction_test_cases, contraction_test_cases, ternary_contraction_test_cases
from .utils.input_fixtures import get_random_input_data
from .utils.support_matrix import framework_backend_support, framework_type_support

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.mark.parametrize(
    (
        "test_case",
        "framework",
        "mem_backend",
        "dtype",
    ),
    [
        (
            test_case,
            framework,
            mem_backend,
            dtype,
        )
        for test_case in contraction_test_cases
        for framework in Framework.enabled()
        for mem_backend in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
    ],
)
class TestStatefulContraction:
    @pytest.mark.parametrize("alpha", [False, 0, 0.3, 0.2 + 0.3j])
    @pytest.mark.parametrize("beta", [False, 0, 0.5, 0.4 + 0.5j])
    @pytest.mark.parametrize("use_offset", [False, True])
    def test_coefficients(self, seeder, alpha, beta, use_offset, test_case, framework, mem_backend, dtype):
        run_coefficients_test_impl(test_case, framework, mem_backend, dtype, "stateful", alpha, beta, use_offset)

    def test_autotune(self, seeder, test_case, framework, mem_backend, dtype):
        alpha, beta = 0.3, 0.4
        if test_case.num_inputs == 2:
            a, b = test_case.gen_input_operands(framework, dtype, mem_backend)
            c = test_case.gen_random_output(framework, dtype, mem_backend)
            contraction = BinaryContraction(test_case.equation, a, b, c=c)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, alpha=alpha, beta=beta)
        elif test_case.num_inputs == 3:
            a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend)
            d = test_case.gen_random_output(framework, dtype, mem_backend)
            contraction = TernaryContraction(test_case.equation, a, b, c, d=d)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, d=d, alpha=alpha, beta=beta)
        else:
            raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")

        tolerance = get_contraction_tolerance(dtype.name, None)

        with contraction:
            plan_preference = contraction.plan_preference
            plan_preference.autotune_mode = ContractionAutotuneMode.INCREMENTAL
            plan_preference.incremental_count = 3
            contraction.plan()
            for _ in range(5):
                result = contraction.execute(alpha=alpha, beta=beta)
                assert_all_close(result, reference, **tolerance)

    def test_non_caching(self, seeder, test_case, framework, mem_backend, dtype):
        alpha, beta = 0.3, 0.4
        if test_case.num_inputs == 2:
            a, b = test_case.gen_input_operands(framework, dtype, mem_backend)
            c = test_case.gen_random_output(framework, dtype, mem_backend)
            contraction = BinaryContraction(test_case.equation, a, b, c=c)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, alpha=alpha, beta=beta)
        elif test_case.num_inputs == 3:
            a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend)
            d = test_case.gen_random_output(framework, dtype, mem_backend)
            contraction = TernaryContraction(test_case.equation, a, b, c, d=d)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, d=d, alpha=alpha, beta=beta)
        else:
            raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")

        tolerance = get_contraction_tolerance(dtype.name, None)

        with contraction:
            plan_preference = contraction.plan_preference
            plan_preference.cache_mode = ContractionCacheMode.NONE
            contraction.plan()
            result = contraction.execute(alpha=alpha, beta=beta)
            assert_all_close(result, reference, **tolerance)

    @pytest.mark.parametrize("algo", AlgoOption)
    @pytest.mark.parametrize("kernel_rank", KernelRankOption)
    def test_algorithm_kernal_rank(self, seeder, algo, kernel_rank, test_case, framework, mem_backend, dtype):
        alpha, beta = 0.3, 0.4
        if test_case.num_inputs == 2:
            a, b = test_case.gen_input_operands(framework, dtype, mem_backend)
            c = test_case.gen_random_output(framework, dtype, mem_backend)
            contraction = BinaryContraction(test_case.equation, a, b, c=c)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, alpha=alpha, beta=beta)
        elif test_case.num_inputs == 3:
            a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend)
            d = test_case.gen_random_output(framework, dtype, mem_backend)
            contraction = TernaryContraction(test_case.equation, a, b, c, d=d)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, d=d, alpha=alpha, beta=beta)
        else:
            raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")

        tolerance = get_contraction_tolerance(dtype.name, None)

        with contraction:
            plan_preference = contraction.plan_preference
            plan_preference.algo = algo.value
            plan_preference.kernel_rank = kernel_rank.value
            contraction.plan()
            result = contraction.execute(alpha=alpha, beta=beta)
            assert_all_close(result, reference, **tolerance)

    @pytest.mark.parametrize("jit", JitOption.enabled())
    def test_jit(self, seeder, jit, test_case, framework, mem_backend, dtype):
        alpha, beta = 0.3, 0.4
        if test_case.num_inputs == 2:
            a, b = test_case.gen_input_operands(framework, dtype, mem_backend)
            c = test_case.gen_random_output(framework, dtype, mem_backend)
            contraction = BinaryContraction(test_case.equation, a, b, c=c)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, alpha=alpha, beta=beta)
        elif test_case.num_inputs == 3:
            a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend)
            d = test_case.gen_random_output(framework, dtype, mem_backend)
            contraction = TernaryContraction(test_case.equation, a, b, c, d=d)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, d=d, alpha=alpha, beta=beta)
        else:
            raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")

        tolerance = get_contraction_tolerance(dtype.name, None)

        with contraction:
            plan_preference = contraction.plan_preference
            plan_preference.jit = jit.value
            contraction.plan()
            result = contraction.execute(alpha=alpha, beta=beta)
            assert_all_close(result, reference, **tolerance)


class TestCoefficientValidationAndExceptions:
    """Test that invalid coefficient combinations raise appropriate errors."""

    # We use numpy and cpu because these tests are verifying API validation,
    # not numerical correctness, so we can use a single combination to save time.
    _framework = Framework.numpy
    _mem_backend = MemBackend.cpu

    def test_beta_without_offset_binary(self):
        """Test that beta without offset raises ValueError for binary contraction."""

        test_dtype = DType.float64  # does not matter which dtype we use
        test_case = binary_contraction_test_cases[0]  # any binary contraction test case will do
        a, b = test_case.gen_input_operands(self._framework, test_dtype, self._mem_backend)

        contraction = BinaryContraction(test_case.equation, a, b)
        with contraction:
            contraction.plan()
            with pytest.raises(ValueError, match="beta can only be set if c is specified"):
                contraction.execute(alpha=0.3, beta=0.5)

    def test_offset_without_beta_binary(self):
        """Test that offset without beta raises ValueError for binary contraction."""

        test_dtype = DType.float64  # does not matter which dtype we use
        test_case = binary_contraction_test_cases[0]  # any binary contraction test case will do
        a, b = test_case.gen_input_operands(self._framework, test_dtype, self._mem_backend)
        c = test_case.gen_random_output(self._framework, test_dtype, self._mem_backend)

        contraction = BinaryContraction(test_case.equation, a, b, c=c)
        with contraction:
            contraction.plan()
            with pytest.raises(ValueError, match="beta must be set when c is specified"):
                contraction.execute(alpha=0.3)

    def test_beta_without_offset_ternary(self):
        """Test that beta without offset raises ValueError for ternary contraction."""

        test_dtype = DType.float64  # does not matter which dtype we use
        test_case = ternary_contraction_test_cases[0]  # any ternary contraction test case will do
        a, b, c = test_case.gen_input_operands(self._framework, test_dtype, self._mem_backend)
        contraction = TernaryContraction(test_case.equation, a, b, c)

        with contraction:
            contraction.plan()
            with pytest.raises(ValueError, match="beta can only be set if d is specified"):
                contraction.execute(alpha=0.3, beta=0.5)

    def test_offset_without_beta_ternary(self):
        """Test that offset without beta raises ValueError for ternary contraction."""
        test_dtype = DType.float64  # does not matter which dtype we use
        test_case = ternary_contraction_test_cases[0]  # any ternary contraction test case will do
        a, b, c = test_case.gen_input_operands(self._framework, test_dtype, self._mem_backend)
        d = test_case.gen_random_output(self._framework, test_dtype, self._mem_backend)

        contraction = TernaryContraction(test_case.equation, a, b, c, d=d)
        with contraction:
            contraction.plan()
            with pytest.raises(ValueError, match="beta must be set when d is specified"):
                contraction.execute(alpha=0.3)

    def test_complex_alpha_with_real_dtype(self):
        """Test that complex alpha with non-complex dtype raises TypeError."""

        test_dtype = DType.float64  # must be a non-complex dtype here as stated in the test name

        def impl(contraction):
            with contraction:
                contraction.plan()
                with pytest.raises(TypeError):
                    contraction.execute(alpha=0.2 + 0.3j)

        binary_test_case = binary_contraction_test_cases[0]  # any binary contraction test case will do
        a, b = binary_test_case.gen_input_operands(self._framework, test_dtype, self._mem_backend)
        binary_contraction = BinaryContraction(binary_test_case.equation, a, b)
        impl(binary_contraction)

        ternary_test_case = ternary_contraction_test_cases[0]  # any ternary contraction test case will do
        a, b, c = ternary_test_case.gen_input_operands(self._framework, test_dtype, self._mem_backend)
        ternary_contraction = TernaryContraction(ternary_test_case.equation, a, b, c)
        impl(ternary_contraction)

    def test_complex_beta_with_real_dtype(self):
        """Test that complex beta with non-complex dtype raises TypeError."""

        test_dtype = DType.float64  # must be a non-complex dtype here as stated in the test name

        def impl(contraction):
            with contraction:
                contraction.plan()
                with pytest.raises(TypeError):
                    contraction.execute(alpha=0.3, beta=0.4 + 0.5j)

        binary_test_case = binary_contraction_test_cases[0]  # any binary contraction test case will do
        a, b = binary_test_case.gen_input_operands(self._framework, test_dtype, self._mem_backend)
        c = binary_test_case.gen_random_output(self._framework, test_dtype, self._mem_backend)
        binary_contraction = BinaryContraction(binary_test_case.equation, a, b, c=c)
        impl(binary_contraction)

        ternary_test_case = ternary_contraction_test_cases[0]
        a, b, c = ternary_test_case.gen_input_operands(self._framework, test_dtype, self._mem_backend)
        d = ternary_test_case.gen_random_output(self._framework, test_dtype, self._mem_backend)
        ternary_contraction = TernaryContraction(ternary_test_case.equation, a, b, c, d=d)
        impl(ternary_contraction)


def test_check_einsum_expression():
    """
    Test that {Binary,Ternary}Contraction syntactically
    validate einsum expression strings.
    """
    # Create simple test arrays
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(3, 4, 5)
    c = np.random.rand(4, 5, 6)

    # with None expr
    with pytest.raises(ValueError):
        BinaryContraction(None, a, b)
    # wrong number of operands
    with pytest.raises(ValueError):
        BinaryContraction("ijk->ijk", a, b)
    with pytest.raises(ValueError):
        BinaryContraction("ijk,jkl,lmn->imn", a, b)

    # with None expr
    with pytest.raises(ValueError):
        TernaryContraction(None, a, b, c)
    # with wrong number of operands
    with pytest.raises(ValueError):
        TernaryContraction("ijk,jkl->il", a, b, c)
    with pytest.raises(ValueError):
        TernaryContraction("ijk,jkl,lmn,nop->iop", a, b, c)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not available")
def test_reset_operands_with_different_pointer_alignment():
    a = cp.random.rand(4, 3)
    b = cp.random.rand(3, 4)
    c = cp.random.rand(4, 4)

    a_lower = a[:2]  # 256 alignment
    a_upper = a[2:]  # 16 alignment

    # reset operands to a tensor with incompatible pointer alignment
    with BinaryContraction("ij,jk->ik", a_lower, b) as bc:
        bc.plan()
        out = bc.execute()
        assert cp.allclose(out, a_lower @ b)
        with pytest.raises(ValueError):
            bc.reset_operands(a=a_upper, b=b)

    # reset operands to a tensor with compatible pointer alignment
    with BinaryContraction("ij,jk->ik", a_upper, b) as bc:
        bc.plan()
        out = bc.execute()
        assert cp.allclose(out, a_upper @ b)

        bc.reset_operands(a=a_lower, b=b)
        out = bc.execute()
        assert cp.allclose(out, a_lower @ b)

    # reset operands to a tensor with incompatible pointer alignment
    with TernaryContraction("ij,jk,kl->il", a_lower, b, c) as tc:
        tc.plan()
        out = tc.execute()
        assert cp.allclose(out, a_lower @ b @ c)
        with pytest.raises(ValueError):
            tc.reset_operands(a=a_upper, b=b, c=c)

    # reset operands to a tensor with compatible pointer alignment
    with TernaryContraction("ij,jk,kl->il", a_upper, b, c) as tc:
        tc.plan()
        out = tc.execute()
        assert cp.allclose(out, a_upper @ b @ c)
        tc.reset_operands(a=a_lower, b=b, c=c)
        out = tc.execute()
        assert cp.allclose(out, a_lower @ b @ c)


@pytest.mark.parametrize(
    ("framework", "mem_backend", "output_provided"),
    [
        (
            framework,
            mem_backend,
            output_provided,
        )
        for framework in Framework.enabled()
        for mem_backend in framework_backend_support[framework]
        for output_provided in [True, False]
    ],
)
class TestReferenceCount:
    """
    Test reference counts consistency before/after contraction context manager.
    Only need a single scenario with CuPy to test scenario when tensors reside on GPU.
    """

    def test_binary_contraction(self, framework, mem_backend, output_provided):
        a = get_random_input_data(framework, (4, 4, 4, 4), DType.float64, mem_backend)
        b = get_random_input_data(framework, (4, 4, 4, 4), DType.float64, mem_backend)
        c = get_random_input_data(framework, (4, 4, 4, 4), DType.float64, mem_backend)

        initial_refcount_a = sys.getrefcount(a)
        initial_refcount_b = sys.getrefcount(b)
        initial_refcount_c = sys.getrefcount(c)

        if output_provided:
            out = get_random_input_data(framework, (4, 4, 4, 4), DType.float64, mem_backend)
            initial_refcount_out = sys.getrefcount(out)
        else:
            out = None

        contraction = BinaryContraction("ijkl,klmn->ijmn", a, b, c=c, out=out)
        with contraction:
            contraction.plan()
            result = contraction.execute(beta=2.2)

            if output_provided:
                assert out is result

        assert sys.getrefcount(a) == initial_refcount_a, "a refcount changed after context exit"
        assert sys.getrefcount(b) == initial_refcount_b, "b refcount changed after context exit"
        assert sys.getrefcount(c) == initial_refcount_c, "c refcount changed after context exit"
        if output_provided:
            del result
            assert sys.getrefcount(out) == initial_refcount_out, "out refcount changed after context exit"
        else:
            with check_freed_after(result, "post op: result should have sole ownership"):
                del result

    def test_ternary_contraction(self, framework, mem_backend, output_provided):
        a = get_random_input_data(framework, (4, 4, 4, 4), DType.float64, mem_backend)
        b = get_random_input_data(framework, (4, 4, 4, 4), DType.float64, mem_backend)
        c = get_random_input_data(framework, (4, 4, 4, 4), DType.float64, mem_backend)
        d = get_random_input_data(framework, (4, 4, 4, 4), DType.float64, mem_backend)

        initial_refcount_a = sys.getrefcount(a)
        initial_refcount_b = sys.getrefcount(b)
        initial_refcount_c = sys.getrefcount(c)
        initial_refcount_d = sys.getrefcount(d)

        if output_provided:
            out = get_random_input_data(framework, (4, 4, 4, 4), DType.float64, mem_backend)
            initial_refcount_out = sys.getrefcount(out)
        else:
            out = None

        contraction = TernaryContraction("ijkl,klmn,mnpq->ijpq", a, b, c, d=d, out=out)
        with contraction:
            contraction.plan()
            result2 = contraction.execute(beta=2.2)

            if output_provided:
                assert out is result2

        assert sys.getrefcount(a) == initial_refcount_a, "a refcount changed after context exit"
        assert sys.getrefcount(b) == initial_refcount_b, "b refcount changed after context exit"
        assert sys.getrefcount(c) == initial_refcount_c, "c refcount changed after context exit"
        assert sys.getrefcount(d) == initial_refcount_d, "d refcount changed after context exit"
        if output_provided:
            del result2
            assert sys.getrefcount(out) == initial_refcount_out, "out refcount changed after context exit"
        else:
            with check_freed_after(result2, "post op: result should have sole ownership"):
                del result2


@pytest.mark.parametrize(
    ("framework", "mem_backend"),
    [
        (
            framework,
            mem_backend,
        )
        for framework in Framework.enabled()
        for mem_backend in framework_backend_support[framework]
    ],
)
class TestContractionTrivialContextManager:
    """
    Test contraction objects for trivial context manager usage.
    Ensures objects can be used in context managers without calling plan/execute.
    """

    def test_binary_contraction_empty_context(self, framework, mem_backend):
        dtype = DType.float64

        # choose random shapes for the inputs, since one specific shape is enough
        a = get_random_input_data(framework, (4, 4, 12, 12), dtype, mem_backend)
        b = get_random_input_data(framework, (12, 12, 8, 8), dtype, mem_backend)
        c = get_random_input_data(framework, (4, 4, 8, 8), dtype, mem_backend)
        with BinaryContraction("ijkl,klmn->ijmn", a, b, c=c) as contraction:
            pass
        # Upon exiting the context manager, the contraction object should be invalid
        # because the free() method should have been called.
        assert not contraction.valid_state

    def test_ternary_contraction_empty_context(self, framework, mem_backend):
        dtype = DType.float64

        # choose random shapes for the inputs, since one specific shape is enough
        a = get_random_input_data(framework, (4, 6, 8), dtype, mem_backend)
        b = get_random_input_data(framework, (6, 8, 3), dtype, mem_backend)
        c = get_random_input_data(framework, (3, 9), dtype, mem_backend)
        with TernaryContraction("ijk,jkl,ln->in", a, b, c=c) as contraction:
            pass

        # Upon exiting the context manager, the contraction object should be invalid
        # because the free() method should have been called.
        assert not contraction.valid_state


class TestContractionPlanPreferenceGetterSetter:
    def setup_class(self):
        a = np.random.rand(2, 3)
        b = np.random.rand(3, 4)
        c = np.random.rand(4, 5)

        self.contractions = [
            BinaryContraction("ij,jk->ik", a, b),
            TernaryContraction("ij,jk,kl->il", a, b, c),
        ]

    def teardown_class(self):
        for contraction in self.contractions:
            contraction.free()

    @pytest.mark.parametrize(
        "attr",
        [
            "autotune_mode",
            "cache_mode",
            "incremental_count",
            "algo",
            "kernel_rank",
            "jit",
        ],
    )
    def test_plan_preference_getter_setter(self, attr):
        value_iterator = {
            "autotune_mode": AutotuneModeOption,
            "cache_mode": CacheModeOption,
            "incremental_count": IncrementalCountOption,
            "algo": AlgoOption,
            "kernel_rank": KernelRankOption,
            "jit": JitOption,
        }[attr]

        for contraction in self.contractions:
            for value in value_iterator:
                plan_preference = contraction.plan_preference
                setattr(plan_preference, attr, value.value)
                assert getattr(plan_preference, attr) == value.value


@pytest.mark.parametrize(
    ("framework", "mem_backend"),
    [
        (
            framework,
            mem_backend,
        )
        for framework in Framework.enabled()
        for mem_backend in framework_backend_support[framework]
    ],
)
class TestContractionOutput:
    @pytest.mark.parametrize("output_provided", [True, False])
    def test_binary_contraction_output(self, framework, mem_backend, output_provided):
        dtype = DType.float64
        a = get_random_input_data(framework, (4, 6, 8), dtype, mem_backend)
        b = get_random_input_data(framework, (6, 8, 3), dtype, mem_backend)
        if output_provided:
            out = get_random_input_data(framework, (4, 3), dtype, mem_backend)
        else:
            out = None
        with BinaryContraction("ijk,jkl->il", a, b, out=out) as contraction:
            contraction.plan()
            out0 = contraction.execute()
            out1 = contraction.execute()
            if output_provided:
                assert out0 is out
                assert out1 is out
            else:
                assert out1 is not out0

    @pytest.mark.parametrize("output_provided", [True, False])
    def test_ternary_contraction_output(self, framework, mem_backend, output_provided):
        dtype = DType.float64
        a = get_random_input_data(framework, (4, 6, 8), dtype, mem_backend)
        b = get_random_input_data(framework, (6, 8, 3), dtype, mem_backend)
        c = get_random_input_data(framework, (3, 9), dtype, mem_backend)
        if output_provided:
            out = get_random_input_data(framework, (4, 9), dtype, mem_backend)
        else:
            out = None
        with TernaryContraction("ijk,jkl,ln->in", a, b, c, out=out) as contraction:
            contraction.plan()
            out0 = contraction.execute()
            out1 = contraction.execute()
            if output_provided:
                assert out0 is out
                assert out1 is out
            else:
                assert out1 is not out0


@pytest.mark.parametrize(
    ("framework", "mem_backend"),
    [(framework, mem_backend) for framework in Framework.enabled() for mem_backend in framework_backend_support[framework]],
)
@pytest.mark.parametrize(
    "output_mode",
    ["none", "accumulation", "out"],
    ids=["no_output", "with_c", "with_out"],
)
def test_release_operands_binary(framework, mem_backend, output_mode):
    """
    Test that after release_operands(), the refcounts of all user-provided
    operands return to their initial values.
    """
    from .utils.input_fixtures import get_random_input_data

    dtype = DType.float32
    shape = (32, 32)
    out_shape = (32, 32)

    a = get_random_input_data(framework, shape, dtype, mem_backend)
    b = get_random_input_data(framework, shape, dtype, mem_backend)
    c = get_random_input_data(framework, out_shape, dtype, mem_backend) if output_mode == "accumulation" else None
    out = get_random_input_data(framework, out_shape, dtype, mem_backend) if output_mode == "out" else None

    # Record initial refcounts for all user-provided operands
    initial_refcounts = {"a": sys.getrefcount(a), "b": sys.getrefcount(b)}
    if c is not None:
        initial_refcounts["c"] = sys.getrefcount(c)
    if out is not None:
        initial_refcounts["out"] = sys.getrefcount(out)

    contraction = BinaryContraction("ij,jk->ik", a, b, c=c, out=out)
    contraction.plan()
    if output_mode == "accumulation":
        result = contraction.execute(beta=0.5)
    else:
        result = contraction.execute()
    if output_mode == "out":
        del result
    else:
        with check_freed_after(result, "The caller should hold the only reference to the result buffer"):
            del result

    contraction.release_operands()

    assert sys.getrefcount(a) == initial_refcounts["a"], f"a refcount: {sys.getrefcount(a)}, expected: {initial_refcounts['a']}"
    assert sys.getrefcount(b) == initial_refcounts["b"], f"b refcount: {sys.getrefcount(b)}, expected: {initial_refcounts['b']}"
    if c is not None:
        assert sys.getrefcount(c) == initial_refcounts["c"], (
            f"c refcount: {sys.getrefcount(c)}, expected: {initial_refcounts['c']}"
        )
    if out is not None:
        assert sys.getrefcount(out) == initial_refcounts["out"], (
            f"out refcount: {sys.getrefcount(out)}, expected: {initial_refcounts['out']}"
        )

    contraction.free()


@pytest.mark.parametrize(
    ("framework", "mem_backend"),
    [(framework, mem_backend) for framework in Framework.enabled() for mem_backend in framework_backend_support[framework]],
)
@pytest.mark.parametrize(
    "output_mode",
    ["none", "accumulation", "out"],
    ids=["no_output", "with_d", "with_out"],
)
def test_release_operands_ternary(framework, mem_backend, output_mode):
    """
    Test that after release_operands(), the refcounts of all user-provided
    operands return to their initial values.
    """
    from .utils.input_fixtures import get_random_input_data

    dtype = DType.float32
    shape = (32, 32)
    out_shape = (32, 32)

    a = get_random_input_data(framework, shape, dtype, mem_backend)
    b = get_random_input_data(framework, shape, dtype, mem_backend)
    c = get_random_input_data(framework, shape, dtype, mem_backend)
    d = get_random_input_data(framework, out_shape, dtype, mem_backend) if output_mode == "accumulation" else None
    out = get_random_input_data(framework, out_shape, dtype, mem_backend) if output_mode == "out" else None

    # Record initial refcounts for all user-provided operands
    initial_refcounts = {"a": sys.getrefcount(a), "b": sys.getrefcount(b), "c": sys.getrefcount(c)}
    if d is not None:
        initial_refcounts["d"] = sys.getrefcount(d)
    if out is not None:
        initial_refcounts["out"] = sys.getrefcount(out)

    contraction = TernaryContraction("ij,jk,kl->il", a, b, c, d=d, out=out)
    contraction.plan()
    if output_mode == "accumulation":
        result = contraction.execute(beta=0.5)
    else:
        result = contraction.execute()
    if output_mode == "out":
        del result
    else:
        with check_freed_after(result, "The caller should hold the only reference to the result buffer"):
            del result

    contraction.release_operands()

    assert sys.getrefcount(a) == initial_refcounts["a"], f"a refcount: {sys.getrefcount(a)}, expected: {initial_refcounts['a']}"
    assert sys.getrefcount(b) == initial_refcounts["b"], f"b refcount: {sys.getrefcount(b)}, expected: {initial_refcounts['b']}"
    assert sys.getrefcount(c) == initial_refcounts["c"], f"c refcount: {sys.getrefcount(c)}, expected: {initial_refcounts['c']}"
    if d is not None:
        assert sys.getrefcount(d) == initial_refcounts["d"], (
            f"d refcount: {sys.getrefcount(d)}, expected: {initial_refcounts['d']}"
        )
    if out is not None:
        assert sys.getrefcount(out) == initial_refcounts["out"], (
            f"out refcount: {sys.getrefcount(out)}, expected: {initial_refcounts['out']}"
        )

    contraction.free()


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("gpu_operand_gpu_exec", id="gpu_operand_gpu_exec"),
        pytest.param("cpu_operand_gpu_exec", id="cpu_operand_gpu_exec"),
    ],
)
@pytest.mark.parametrize(
    "contraction_type",
    ["binary", "ternary"],
)
def test_release_then_reset_unchecked(case, contraction_type):
    """
    Test that reset_operands_unchecked works after release_operands for
    both same-space (GPU) and cross-space (CPU) operands.
    """
    if case == "gpu_operand_gpu_exec":
        cp = pytest.importorskip("cupy")
        make_tensor = lambda: cp.random.rand(32, 32).astype(cp.float32)
    else:
        make_tensor = lambda: np.random.rand(32, 32).astype(np.float32)

    a, b, a_new, b_new = make_tensor(), make_tensor(), make_tensor(), make_tensor()

    if contraction_type == "binary":
        expr = "ij,jk->ik"
        ref = get_contraction_ref(expr, a_new, b_new)
        contraction = BinaryContraction(expr, a, b)
        contraction.plan()
        contraction.execute()
        contraction.release_operands()
        contraction.reset_operands_unchecked(a=a_new, b=b_new)
    else:
        c, c_new = make_tensor(), make_tensor()
        expr = "ij,jk,kl->il"
        ref = get_contraction_ref(expr, a_new, b_new, c=c_new)
        contraction = TernaryContraction(expr, a, b, c)
        contraction.plan()
        contraction.execute()
        contraction.release_operands()
        contraction.reset_operands_unchecked(a=a_new, b=b_new, c=c_new)

    result = contraction.execute()
    assert_all_close(result, ref, rtol=1e-4, atol=1e-4)

    contraction.free()
