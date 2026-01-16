# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
import numpy as np

from nvmath.tensor import BinaryContraction, TernaryContraction, ContractionCacheMode, ContractionAutotuneMode
from .utils.check_helpers import get_contraction_ref, assert_all_close, get_contraction_tolerance
from .utils.base_testers import run_coefficients_test_impl

from .utils.common_axes import Framework, JitOption, AlgoOption, KernelRankOption
from .utils.data import contraction_test_cases, binary_contraction_test_cases, ternary_contraction_test_cases
from .utils.support_matrix import framework_backend_support, framework_type_support
from .utils.common_axes import MemBackend, DType

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
class TestReferenceCount:
    """
    Test reference counts consistency before/after contraction context manager.
    Only need a single scenario with CuPy to test scenario when tensors reside on GPU.
    """

    @staticmethod
    def refcount_msg(name, arr, stage=""):
        actual = sys.getrefcount(arr) - 1
        stage_info = f" {stage}" if stage else ""
        return f"CuPy array '{name}' should have refcount 1{stage_info} (actual: {actual})"

    def test_binary_contraction(self):
        a = cp.random.rand(4, 4, 4, 4)
        b = cp.random.rand(4, 4, 4, 4)
        c = cp.random.rand(4, 4, 4, 4)

        # Check initial reference counts should be 1
        # getrefcount adds 1 for the temp reference in the function call, so expect 2
        assert sys.getrefcount(a) == 2, self.refcount_msg("a", a, "before contraction")
        assert sys.getrefcount(b) == 2, self.refcount_msg("b", b, "before contraction")
        assert sys.getrefcount(c) == 2, self.refcount_msg("c", c, "before contraction")

        # Create and execute binary contraction
        contraction = BinaryContraction("ijkl,klmn->ijmn", a, b, c=c)
        with contraction:
            contraction.plan()
            result = contraction.execute(beta=2.2)

        # After exiting context manager, reference counts should return to 1
        # getrefcount adds 1 for the temp reference in the function call, so expect 2
        assert sys.getrefcount(a) == 2, self.refcount_msg("a", a, "after contraction context exit")
        assert sys.getrefcount(b) == 2, self.refcount_msg("b", b, "after contraction context exit")
        assert sys.getrefcount(c) == 2, self.refcount_msg("c", c, "after contraction context exit")
        assert sys.getrefcount(result) == 2, self.refcount_msg("result", result)

    def test_ternary_contraction(self):
        a = cp.random.rand(4, 4, 4, 4)
        b = cp.random.rand(4, 4, 4, 4)
        c = cp.random.rand(4, 4, 4, 4)
        d = cp.random.rand(4, 4, 4, 4)

        # Check initial reference counts should be 1
        assert sys.getrefcount(a) == 2, self.refcount_msg("a", a, "before contraction")
        assert sys.getrefcount(b) == 2, self.refcount_msg("b", b, "before contraction")
        assert sys.getrefcount(c) == 2, self.refcount_msg("c", c, "before contraction")
        assert sys.getrefcount(d) == 2, self.refcount_msg("d", d, "before contraction")

        # Create and execute ternary contraction
        contraction = TernaryContraction("ijkl,klmn,mnpq->ijpq", a, b, c, d=d)
        with contraction:
            contraction.plan()
            result2 = contraction.execute(beta=2.2)

        # After exiting context manager, reference counts should return to 1
        # getrefcount adds 1 for the temp reference in the function call, so expect 2
        assert sys.getrefcount(a) == 2, self.refcount_msg("a", a, "after contraction context exit")
        assert sys.getrefcount(b) == 2, self.refcount_msg("b", b, "after contraction context exit")
        assert sys.getrefcount(c) == 2, self.refcount_msg("c", c, "after contraction context exit")
        assert sys.getrefcount(d) == 2, self.refcount_msg("d", d, "after contraction context exit")
        assert sys.getrefcount(result2) == 2, self.refcount_msg("result", result2)


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
        from .utils.input_fixtures import get_random_input_data
        from .utils.common_axes import DType

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
        from .utils.input_fixtures import get_random_input_data
        from .utils.common_axes import DType

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
