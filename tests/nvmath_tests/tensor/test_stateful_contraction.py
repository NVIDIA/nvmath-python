# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from nvmath.tensor import BinaryContraction, TernaryContraction, ContractionCacheMode, ContractionAutotuneMode
from .utils.check_helpers import get_contraction_ref, assert_all_close, get_contraction_tolerance
from .utils.base_testers import BaseStatefulTester

from .utils.common_axes import Framework, JitOption, AlgoOption, KernelRankOption
from .utils.data import contraction_test_cases
from .utils.support_matrix import framework_backend_support, framework_type_support


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
class TestStatefulContraction(BaseStatefulTester):
    @pytest.mark.parametrize("alpha", [False, 0, 0.3, 0.2 + 0.3j])
    @pytest.mark.parametrize("beta", [False, 0, 0.5, 0.4 + 0.5j])
    @pytest.mark.parametrize("use_offset", [False, True])
    def test_coefficients(self, alpha, beta, use_offset, test_case, framework, mem_backend, dtype):
        self._test_coefficients(alpha, beta, use_offset, test_case, framework, mem_backend, dtype)

    def test_autotune(self, test_case, framework, mem_backend, dtype):
        alpha, beta = 0.3, 0.4
        if test_case.num_inputs == 2:
            a, b = test_case.gen_input_operands(framework, dtype, mem_backend, 23)
            c = test_case.gen_random_output(framework, dtype, mem_backend, 24)
            contraction = BinaryContraction(test_case.equation, a, b, c=c)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, alpha=alpha, beta=beta)
        elif test_case.num_inputs == 3:
            a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend, 23)
            d = test_case.gen_random_output(framework, dtype, mem_backend, 24)
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

    def test_non_caching(self, test_case, framework, mem_backend, dtype):
        alpha, beta = 0.3, 0.4
        if test_case.num_inputs == 2:
            a, b = test_case.gen_input_operands(framework, dtype, mem_backend, 23)
            c = test_case.gen_random_output(framework, dtype, mem_backend, 24)
            contraction = BinaryContraction(test_case.equation, a, b, c=c)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, alpha=alpha, beta=beta)
        elif test_case.num_inputs == 3:
            a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend, 23)
            d = test_case.gen_random_output(framework, dtype, mem_backend, 24)
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
    def test_algorithm_kernal_rank(self, algo, kernel_rank, test_case, framework, mem_backend, dtype):
        alpha, beta = 0.3, 0.4
        if test_case.num_inputs == 2:
            a, b = test_case.gen_input_operands(framework, dtype, mem_backend, 23)
            c = test_case.gen_random_output(framework, dtype, mem_backend, 24)
            contraction = BinaryContraction(test_case.equation, a, b, c=c)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, alpha=alpha, beta=beta)
        elif test_case.num_inputs == 3:
            a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend, 23)
            d = test_case.gen_random_output(framework, dtype, mem_backend, 24)
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
    def test_jit(self, jit, test_case, framework, mem_backend, dtype):
        alpha, beta = 0.3, 0.4
        if test_case.num_inputs == 2:
            a, b = test_case.gen_input_operands(framework, dtype, mem_backend, 23)
            c = test_case.gen_random_output(framework, dtype, mem_backend, 24)
            contraction = BinaryContraction(test_case.equation, a, b, c=c)
            reference = get_contraction_ref(test_case.equation, a, b, c=c, alpha=alpha, beta=beta)
        elif test_case.num_inputs == 3:
            a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend, 23)
            d = test_case.gen_random_output(framework, dtype, mem_backend, 24)
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
