# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from nvmath.tensor import BinaryContraction, TernaryContraction

from .utils.check_helpers import assert_all_close, get_contraction_tolerance
from .utils.common_axes import DType, Framework, MemBackend
from .utils.data import binary_contraction_test_cases, ternary_contraction_test_cases
from .utils.support_matrix import framework_backend_support

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None


def synchronize_if_needed(mem_backend):
    if mem_backend == MemBackend.cuda:
        # Need to synchronize GPU regardless of framework
        # Try different synchronization methods in order of preference
        if cp is not None:
            # Use CuPy's synchronization (most reliable when available)
            cp.cuda.runtime.deviceSynchronize()
        elif torch is not None:
            # Use PyTorch's synchronization as fallback
            torch.cuda.synchronize()
        else:
            # If neither CuPy nor PyTorch available but we're using CUDA backend,
            # this shouldn't happen in normal test execution, but just in case
            pass


@pytest.mark.parametrize(
    (
        "test_case",
        "framework",
        "mem_backend",
    ),
    [
        (test_case, framework, mem_backend)
        for test_case in binary_contraction_test_cases
        for framework in Framework.enabled()
        for mem_backend in framework_backend_support[framework]
    ],
)
class TestBinaryContractionResetOperandsUnchecked:
    """Test reset_operands_unchecked for BinaryContraction."""

    @pytest.mark.parametrize(
        "use_offset",
        [True, False],
        ids=["with_offset", "without_offset"],
    )
    def test_without_output(self, seeder, test_case, framework, mem_backend, use_offset):  # noqa: ARG002
        """Test that reset_operands_unchecked produces the same results as reset_operands.

        Note: We only test resetting all operands because reset_operands() does not support
        partial updates (it sets unspecified operands to None).
        """
        dtype = DType.float32
        alpha = 0.7
        beta = 0.4 if use_offset else None

        # Generate initial operands
        a, b = test_case.gen_input_operands(framework, dtype, mem_backend)
        c = test_case.gen_random_output(framework, dtype, mem_backend) if use_offset else None

        # Generate new operands for reset
        a_new, b_new = test_case.gen_input_operands(framework, dtype, mem_backend)
        c_new = test_case.gen_random_output(framework, dtype, mem_backend) if use_offset else None

        tolerance = get_contraction_tolerance(dtype.name, None)

        # Build reset kwargs - always reset all input operands
        reset_kwargs = {"a": a_new, "b": b_new}
        if use_offset:
            reset_kwargs["c"] = c_new

        # Test with reset_operands (reference)
        with BinaryContraction(test_case.equation, a, b, c=c) as contraction_ref:
            contraction_ref.plan()
            if use_offset:
                result_ref_1 = contraction_ref.execute(alpha=alpha, beta=beta)
            else:
                result_ref_1 = contraction_ref.execute(alpha=alpha)

            # Reset operands using standard method
            contraction_ref.reset_operands(**reset_kwargs)
            if use_offset:
                result_ref_2 = contraction_ref.execute(alpha=alpha, beta=beta)
            else:
                result_ref_2 = contraction_ref.execute(alpha=alpha)

        # Test with reset_operands_unchecked
        with BinaryContraction(test_case.equation, a, b, c=c) as contraction_unchecked:
            contraction_unchecked.plan()
            if use_offset:
                result_unchecked_1 = contraction_unchecked.execute(alpha=alpha, beta=beta)
            else:
                result_unchecked_1 = contraction_unchecked.execute(alpha=alpha)

            # Reset operands using unchecked method
            contraction_unchecked.reset_operands_unchecked(**reset_kwargs)
            if use_offset:
                result_unchecked_2 = contraction_unchecked.execute(alpha=alpha, beta=beta)
            else:
                result_unchecked_2 = contraction_unchecked.execute(alpha=alpha)

        # Synchronize once before accessing results for comparison
        synchronize_if_needed(mem_backend)

        # Verify that both methods produce the same results
        assert_all_close(result_unchecked_1, result_ref_1, **tolerance)
        assert_all_close(result_unchecked_2, result_ref_2, **tolerance)

    def test_with_output(self, seeder, test_case, framework, mem_backend):  # noqa: ARG002
        """Test reset_operands_unchecked when resetting also the output buffer.

        Note: We reset all operands (a, b, and out) because reset_operands()
        doesn't support partial updates.
        """
        dtype = DType.float32
        alpha = 0.8

        # Generate initial operands with output buffer
        a, b = test_case.gen_input_operands(framework, dtype, mem_backend)
        out = test_case.gen_random_output(framework, dtype, mem_backend)

        # Generate new operands and output for reset
        a_new, b_new = test_case.gen_input_operands(framework, dtype, mem_backend)
        out_new = test_case.gen_random_output(framework, dtype, mem_backend)

        tolerance = get_contraction_tolerance(dtype.name, None)

        # Build reset kwargs - reset all operands including output
        reset_kwargs = {"a": a_new, "b": b_new, "out": out_new}

        # Test with reset_operands (reference)
        with BinaryContraction(test_case.equation, a, b, out=out) as contraction_ref:
            contraction_ref.plan()
            result_ref_1 = contraction_ref.execute(alpha=alpha)
            assert result_ref_1 is out

            # Reset all operands including output
            contraction_ref.reset_operands(**reset_kwargs)
            result_ref_2 = contraction_ref.execute(alpha=alpha)
            assert result_ref_2 is out_new

        # Test with reset_operands_unchecked
        with BinaryContraction(test_case.equation, a, b, out=out) as contraction_unchecked:
            contraction_unchecked.plan()
            result_unchecked_1 = contraction_unchecked.execute(alpha=alpha)
            assert result_unchecked_1 is out

            # Reset all operands including output using unchecked method
            contraction_unchecked.reset_operands_unchecked(**reset_kwargs)
            result_unchecked_2 = contraction_unchecked.execute(alpha=alpha)
            assert result_unchecked_2 is out_new

        # Synchronize once before accessing results for comparison
        synchronize_if_needed(mem_backend)

        # Verify that both methods produce the same results
        assert_all_close(result_unchecked_1, result_ref_1, **tolerance)
        assert_all_close(result_unchecked_2, result_ref_2, **tolerance)


@pytest.mark.parametrize(
    (
        "test_case",
        "framework",
        "mem_backend",
    ),
    [
        (test_case, framework, mem_backend)
        for test_case in ternary_contraction_test_cases
        for framework in Framework.enabled()
        for mem_backend in framework_backend_support[framework]
    ],
)
class TestTernaryContractionResetOperandsUnchecked:
    """Test reset_operands_unchecked for TernaryContraction."""

    @pytest.mark.parametrize(
        "use_offset",
        [True, False],
        ids=["with_offset", "without_offset"],
    )
    def test_reset_operands_unchecked(self, seeder, test_case, framework, mem_backend, use_offset):  # noqa: ARG002
        """Test that reset_operands_unchecked produces the same results as reset_operands.

        Note: We only test "all" operands reset because reset_operands() does not support
        partial updates (it sets unspecified operands to None).
        """
        dtype = DType.float32
        alpha = 0.7
        beta = 0.4 if use_offset else None

        # Generate initial operands
        a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend)
        d = test_case.gen_random_output(framework, dtype, mem_backend) if use_offset else None

        # Generate new operands for reset
        a_new, b_new, c_new = test_case.gen_input_operands(framework, dtype, mem_backend)
        d_new = test_case.gen_random_output(framework, dtype, mem_backend) if use_offset else None

        tolerance = get_contraction_tolerance(dtype.name, None)

        # Build reset kwargs - always reset all operands
        reset_kwargs = {"a": a_new, "b": b_new, "c": c_new}
        if use_offset:
            reset_kwargs["d"] = d_new

        # Test with reset_operands (reference)
        with TernaryContraction(test_case.equation, a, b, c, d=d) as contraction_ref:
            contraction_ref.plan()
            if use_offset:
                result_ref_1 = contraction_ref.execute(alpha=alpha, beta=beta)
            else:
                result_ref_1 = contraction_ref.execute(alpha=alpha)

            # Reset operands using standard method
            contraction_ref.reset_operands(**reset_kwargs)
            if use_offset:
                result_ref_2 = contraction_ref.execute(alpha=alpha, beta=beta)
            else:
                result_ref_2 = contraction_ref.execute(alpha=alpha)

        # Test with reset_operands_unchecked
        with TernaryContraction(test_case.equation, a, b, c, d=d) as contraction_unchecked:
            contraction_unchecked.plan()
            if use_offset:
                result_unchecked_1 = contraction_unchecked.execute(alpha=alpha, beta=beta)
            else:
                result_unchecked_1 = contraction_unchecked.execute(alpha=alpha)

            # Reset operands using unchecked method
            contraction_unchecked.reset_operands_unchecked(**reset_kwargs)
            if use_offset:
                result_unchecked_2 = contraction_unchecked.execute(alpha=alpha, beta=beta)
            else:
                result_unchecked_2 = contraction_unchecked.execute(alpha=alpha)

        # Synchronize once before accessing results for comparison
        synchronize_if_needed(mem_backend)

        # Verify that both methods produce the same results
        assert_all_close(result_unchecked_1, result_ref_1, **tolerance)
        assert_all_close(result_unchecked_2, result_ref_2, **tolerance)

    def test_with_output(self, seeder, test_case, framework, mem_backend):  # noqa: ARG002
        """Test reset_operands_unchecked when resetting also the output buffer.

        Note: We reset all operands (a, b, c, and out) because reset_operands()
        doesn't support partial updates.
        """
        dtype = DType.float32
        alpha = 0.8

        # Generate initial operands with output buffer
        a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend)
        out = test_case.gen_random_output(framework, dtype, mem_backend)

        # Generate new operands and output for reset
        a_new, b_new, c_new = test_case.gen_input_operands(framework, dtype, mem_backend)
        out_new = test_case.gen_random_output(framework, dtype, mem_backend)

        tolerance = get_contraction_tolerance(dtype.name, None)

        # Build reset kwargs - reset all operands including output
        reset_kwargs = {"a": a_new, "b": b_new, "c": c_new, "out": out_new}

        # Test with reset_operands (reference)
        with TernaryContraction(test_case.equation, a, b, c, out=out) as contraction_ref:
            contraction_ref.plan()
            result_ref_1 = contraction_ref.execute(alpha=alpha)
            assert result_ref_1 is out

            # Reset all operands including output
            contraction_ref.reset_operands(**reset_kwargs)
            result_ref_2 = contraction_ref.execute(alpha=alpha)
            assert result_ref_2 is out_new

        # Test with reset_operands_unchecked
        with TernaryContraction(test_case.equation, a, b, c, out=out) as contraction_unchecked:
            contraction_unchecked.plan()
            result_unchecked_1 = contraction_unchecked.execute(alpha=alpha)
            assert result_unchecked_1 is out

            # Reset all operands including output using unchecked method
            contraction_unchecked.reset_operands_unchecked(**reset_kwargs)
            result_unchecked_2 = contraction_unchecked.execute(alpha=alpha)
            assert result_unchecked_2 is out_new

        # Synchronize once before accessing results for comparison
        synchronize_if_needed(mem_backend)

        # Verify that both methods produce the same results
        assert_all_close(result_unchecked_1, result_ref_1, **tolerance)
        assert_all_close(result_unchecked_2, result_ref_2, **tolerance)
