# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for reset_operands_unchecked method.
"""

import pytest

try:
    import torch
except ImportError:
    torch = None

try:
    import cupy as cp
except ImportError:
    cp = None

from nvmath.bindings import cublasLt as cublaslt
from nvmath.internal.typemaps import NAME_TO_DATA_TYPE
from nvmath.linalg.advanced import Matmul, MatmulEpilog

from ...utils import assert_tensors_equal, compare_tensors, sample_matrix
from .fp8_utils import assert_fp8_equal, choose_scales, generate_inputs


# Determine available frameworks for parametrization
def _get_available_frameworks():
    """Returns list of available frameworks for test parametrization.

    Tests parametrized with AVAILABLE_FRAMEWORKS will only run for installed
    frameworks. If neither torch nor cupy are available, the list will be empty
    and tests will not be collected.
    """
    frameworks = []
    if torch is not None:
        frameworks.append("torch")
    if cp is not None:
        frameworks.append("numpy/cupy")
    return frameworks


AVAILABLE_FRAMEWORKS = _get_available_frameworks()


# Check FP8 support (same checks as in test_fp8.py)
def _check_fp8_support():
    if torch is None:
        return False, "Torch is not available"
    try:
        cc = torch.cuda.get_device_properties(0)
        if (cc.major, cc.minor) < (8, 9):
            msg = f"Detected compute capability {cc.major}.{cc.minor} < 8.9"
            msg += "but CC>=8.9 is required for FP8 tests"
            return False, msg

    except Exception as e:
        return False, f"Could not get CUDA device properties: {e}"

    if cublaslt.get_version() < 120800:
        return False, f"cuBLASLt version {cublaslt.get_version()} < 120800"

    return True, "FP8 supported"


FP8_SUPPORTED, FP8_SKIP_REASON = _check_fp8_support()
skip_fp8_tests = pytest.mark.skipif(not FP8_SUPPORTED, reason=FP8_SKIP_REASON)


class TestResetOperandsUncheckedBasic:
    """Test resetting basic operands (a, b, c, alpha, beta)."""

    @pytest.mark.parametrize("framework", AVAILABLE_FRAMEWORKS)
    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    def test_reset_ab_operands(self, framework, use_cuda, m, n, k):
        """Test resetting A and B operands."""

        dtype_str = "float32"

        # Create initial operands
        a1 = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b1 = sample_matrix(framework, dtype_str, (k, n), use_cuda)

        # Create new operands
        a2 = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b2 = sample_matrix(framework, dtype_str, (k, n), use_cuda)

        # Test with reset_operands
        with Matmul(a1, b1) as mm:
            mm.plan()
            mm.reset_operands(a=a2, b=b2)
            result1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a1, b1) as mm:
            mm.plan()
            mm.reset_operands_unchecked(a=a2, b=b2)
            result2 = mm.execute()

        # no need to block here because inside the assertion,
        # tensors are moved to the CPU and compared there
        assert_tensors_equal(result1, result2)

    @pytest.mark.parametrize("framework", AVAILABLE_FRAMEWORKS)
    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    def test_reset_abc_operands(self, framework, use_cuda, m, n, k):
        """Test resetting A, B, and C operands."""

        dtype_str = "float32"
        # Create initial operands
        a1 = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b1 = sample_matrix(framework, dtype_str, (k, n), use_cuda)
        c1 = sample_matrix(framework, dtype_str, (m, n), use_cuda)

        # Create new operands
        a2 = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b2 = sample_matrix(framework, dtype_str, (k, n), use_cuda)
        c2 = sample_matrix(framework, dtype_str, (m, n), use_cuda)

        # Test with reset_operands
        with Matmul(a1, b1, c1, beta=1.0) as mm:
            mm.plan()
            mm.reset_operands(a=a2, b=b2, c=c2)
            result1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a1, b1, c1, beta=1.0) as mm:
            mm.plan()
            mm.reset_operands_unchecked(a=a2, b=b2, c=c2)
            result2 = mm.execute()

        # no need to block here because inside the assertion,
        # tensors are moved to the CPU and compared there
        assert_tensors_equal(result1, result2)

    @pytest.mark.parametrize("framework", AVAILABLE_FRAMEWORKS)
    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    def test_reset_alpha_beta(self, framework, use_cuda, m, n, k):
        """Test resetting alpha and beta scaling factors."""
        dtype_str = "float32"
        # Create operands
        a = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b = sample_matrix(framework, dtype_str, (k, n), use_cuda)
        c = sample_matrix(framework, dtype_str, (m, n), use_cuda)

        alpha1, beta1 = 1.0, 0.5
        alpha2, beta2 = 2.0, 0.25

        # Test with reset_operands
        with Matmul(a, b, c, alpha=alpha1, beta=beta1) as mm:
            mm.plan()
            mm.reset_operands(alpha=alpha2, beta=beta2)
            result1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a, b, c, alpha=alpha1, beta=beta1) as mm:
            mm.plan()
            mm.reset_operands_unchecked(alpha=alpha2, beta=beta2)
            result2 = mm.execute()

        # no need to block here because inside the assertion,
        # tensors are moved to the CPU and compared there
        assert_tensors_equal(result1, result2)

    @pytest.mark.parametrize("framework", AVAILABLE_FRAMEWORKS)
    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    def test_reset_abc_operands_inplace(self, framework, use_cuda, m, n, k):
        """Test resetting A, B, and C operands with inplace=True.

        When use_cuda=False, this test verifies that cpu_c_ref is correctly
        updated during reset_operands_unchecked, ensuring the result is copied
        back to the correct CPU tensor after reset.

        When use_cuda=True, this test verifies basic inplace identity behavior
        (result tensor is the same object as the c operand).
        """
        dtype_str = "float32"

        # Create initial operands
        a1 = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b1 = sample_matrix(framework, dtype_str, (k, n), use_cuda)
        c1 = sample_matrix(framework, dtype_str, (m, n), use_cuda)

        # Create new operands
        a2 = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b2 = sample_matrix(framework, dtype_str, (k, n), use_cuda)
        c2 = sample_matrix(framework, dtype_str, (m, n), use_cuda)

        # Test with reset_operands
        with Matmul(a1, b1, c1, beta=1.0, options={"inplace": True}) as mm:
            mm.plan()
            result1_initial = mm.execute()
            # Verify inplace behavior: result should be c1
            assert result1_initial is c1, "Initial result should be c1 for inplace=True"

            mm.reset_operands(a=a2, b=b2, c=c2)
            result1 = mm.execute()
            # After reset, result should now be c2
            assert result1 is c2, "Result after reset should be c2 for inplace=True"

        # Test with reset_operands_unchecked
        with Matmul(a1, b1, c1, beta=1.0, options={"inplace": True}) as mm:
            mm.plan()
            result2_initial = mm.execute()
            # Verify inplace behavior: result should be c1
            assert result2_initial is c1, "Initial result should be c1 for inplace=True"

            mm.reset_operands_unchecked(a=a2, b=b2, c=c2)
            result2 = mm.execute()
            # After reset, result should now be c2
            assert result2 is c2, "Result after reset should be c2 for inplace=True"

        # we cannot use assert_tensors_equal because result1 and result2 are the same object
        # and that function has a check for that. Also no need to block here because
        # inside the assertion, tensors are moved to the CPU and compared there.
        assert compare_tensors(result1, result2)


class TestResetOperandsUncheckedEpilogBias:
    """Test resetting epilog inputs: bias."""

    @pytest.mark.parametrize("framework", AVAILABLE_FRAMEWORKS)
    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    def test_reset_bias_epilog_input(self, framework, use_cuda, m, n, k):
        """Test resetting bias epilog input."""
        dtype_str = "float32"
        # Create operands
        a = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b = sample_matrix(framework, dtype_str, (k, n), use_cuda)

        # Create bias vectors
        bias1 = sample_matrix(framework, dtype_str, (m, 1), use_cuda)
        bias2 = sample_matrix(framework, dtype_str, (m, 1), use_cuda)

        # Test with reset_operands
        with Matmul(a, b) as mm:
            mm.plan(epilog=MatmulEpilog.BIAS, epilog_inputs={"bias": bias1})
            mm.reset_operands(epilog_inputs={"bias": bias2})
            result1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a, b) as mm:
            mm.plan(epilog=MatmulEpilog.BIAS, epilog_inputs={"bias": bias1})
            mm.reset_operands_unchecked(epilog_inputs={"bias": bias2})
            result2 = mm.execute()

        # no need to block here because inside the assertion,
        # tensors are moved to the CPU and compared there
        assert_tensors_equal(result1, result2)

    @pytest.mark.parametrize("framework", AVAILABLE_FRAMEWORKS)
    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    def test_reset_operands_and_bias(self, framework, use_cuda, m, n, k):
        """Test resetting both operands and bias simultaneously."""
        dtype_str = "float32"
        # Create initial operands
        a1 = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b1 = sample_matrix(framework, dtype_str, (k, n), use_cuda)
        bias1 = sample_matrix(framework, dtype_str, (m, 1), use_cuda)

        # Create new operands
        a2 = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b2 = sample_matrix(framework, dtype_str, (k, n), use_cuda)
        bias2 = sample_matrix(framework, dtype_str, (m, 1), use_cuda)

        # Test with reset_operands
        with Matmul(a1, b1) as mm:
            mm.plan(epilog=MatmulEpilog.BIAS, epilog_inputs={"bias": bias1})
            mm.reset_operands(a=a2, b=b2, epilog_inputs={"bias": bias2})
            result1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a1, b1) as mm:
            mm.plan(epilog=MatmulEpilog.BIAS, epilog_inputs={"bias": bias1})
            mm.reset_operands_unchecked(a=a2, b=b2, epilog_inputs={"bias": bias2})
            result2 = mm.execute()

        # no need to block here because inside the assertion,
        # tensors are moved to the CPU and compared there
        assert_tensors_equal(result1, result2)


class TestResetOperandsUncheckedEpilogRelAndGeluAux:
    """Test resetting epilog inputs: relu_aux and gelu_aux."""

    @pytest.mark.parametrize("framework", AVAILABLE_FRAMEWORKS)
    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    @pytest.mark.parametrize(
        "forward_epilog, backward_epilog, epilog_name",
        [(MatmulEpilog.GELU_AUX, MatmulEpilog.DGELU, "gelu_aux"), (MatmulEpilog.RELU_AUX, MatmulEpilog.DRELU, "relu_aux")],
    )
    def test_reset_gelu_aux_epilog_input(self, framework, use_cuda, m, n, k, forward_epilog, backward_epilog, epilog_name):
        """Test resetting epilog input for DGELU or DRELU."""
        dtype_str = "float32"
        # Create operands
        a = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b = sample_matrix(framework, dtype_str, (k, n), use_cuda)

        # Generate aux from forward pass
        with Matmul(a, b) as mm_forward:
            mm_forward.plan(epilog=forward_epilog)
            _, aux = mm_forward.execute()
            aux1 = aux[epilog_name]

        # Generate another aux
        a_new = sample_matrix(framework, dtype_str, (m, k), use_cuda)
        b_new = sample_matrix(framework, dtype_str, (k, n), use_cuda)
        with Matmul(a_new, b_new) as mm_forward:
            mm_forward.plan(epilog=forward_epilog)
            _, aux = mm_forward.execute()
            aux2 = aux[epilog_name]

        # Test with reset_operands for backward pass
        with Matmul(a, b) as mm:
            mm.plan(epilog=backward_epilog, epilog_inputs={epilog_name: aux1})
            mm.reset_operands(epilog_inputs={epilog_name: aux2})
            result1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a, b) as mm:
            mm.plan(epilog=backward_epilog, epilog_inputs={epilog_name: aux1})
            mm.reset_operands_unchecked(epilog_inputs={epilog_name: aux2})
            result2 = mm.execute()

        # no need to block here because inside the assertion,
        # tensors are moved to the CPU and compared there
        assert_tensors_equal(result1, result2)


class TestResetOperandsUncheckedCombined:
    """Test resetting multiple types of operands simultaneously."""

    @pytest.mark.parametrize("framework", AVAILABLE_FRAMEWORKS)
    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    def test_reset_everything(self, framework, use_cuda, m, n, k):
        """Test resetting operands, alpha, and epilog inputs together."""
        # Create initial state (without C operand to avoid BIAS epilog conflict)
        a1 = sample_matrix(framework, "float32", (m, k), use_cuda)
        b1 = sample_matrix(framework, "float32", (k, n), use_cuda)
        bias1 = sample_matrix(framework, "float32", (m, 1), use_cuda)
        alpha1 = 1.0

        # Create new state
        a2 = sample_matrix(framework, "float32", (m, k), use_cuda)
        b2 = sample_matrix(framework, "float32", (k, n), use_cuda)
        bias2 = sample_matrix(framework, "float32", (m, 1), use_cuda)
        alpha2 = 2.0

        # Test with reset_operands
        with Matmul(a1, b1, alpha=alpha1) as mm:
            mm.plan(epilog=MatmulEpilog.BIAS, epilog_inputs={"bias": bias1})
            mm.reset_operands(a=a2, b=b2, alpha=alpha2, epilog_inputs={"bias": bias2})
            result1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a1, b1, alpha=alpha1) as mm:
            mm.plan(epilog=MatmulEpilog.BIAS, epilog_inputs={"bias": bias1})
            mm.reset_operands_unchecked(a=a2, b=b2, alpha=alpha2, epilog_inputs={"bias": bias2})
            result2 = mm.execute()

        # no need to block here because inside the assertion,
        # tensors are moved to the CPU and compared there
        assert_tensors_equal(result1, result2)


@skip_fp8_tests
class TestResetOperandsUncheckedQuantization:
    """Test resetting quantization scales for FP8 operations."""

    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    def test_reset_scalar_quantization_scales(self, use_cuda, m, n, k):
        """Test resetting scalar quantization scales for FP8."""
        # Use same setup as test_fp8.py
        atype, btype, ctype, dtype = "float8_e4m3fn", "float8_e4m3fn", "float16", None
        a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

        # Initial scales
        scales1 = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)

        # New scales
        new_a, new_b, new_c, new_alpha, new_beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
        scales2 = choose_scales(new_a, new_b, new_c, atype, btype, ctype, dtype, alpha=new_alpha, beta=new_beta)

        options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}

        # Test with reset_operands (reset 'a' along with scales, like test_fp8.py)
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales1, options=options) as mm:
            mm.plan()
            mm.reset_operands(a=new_a, quantization_scales=scales2)
            result1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales1, options=options) as mm:
            mm.plan()
            mm.reset_operands_unchecked(a=new_a, quantization_scales=scales2)
            result2 = mm.execute()

        # Use FP8-aware comparison
        assert_fp8_equal(result1, result2)

    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    def test_reset_partial_quantization_scales(self, use_cuda, m, n, k):
        """Test resetting only some quantization scales."""
        # Use same setup as test_fp8.py
        atype, btype, ctype, dtype = "float8_e4m3fn", "float8_e4m3fn", "float16", None
        a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

        # Initial scales
        scales1 = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)

        # New operand and partial scales (only reset 'a' scale)
        new_a, new_b, new_c, new_alpha, new_beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
        new_scales = choose_scales(new_a, new_b, new_c, atype, btype, ctype, dtype, alpha=new_alpha, beta=new_beta)
        scales2 = {"a": new_scales["a"]}  # Only reset scale for A

        options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}

        # Test with reset_operands
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales1, options=options) as mm:
            mm.plan()
            mm.reset_operands(a=new_a, quantization_scales=scales2)
            result1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales1, options=options) as mm:
            mm.plan()
            mm.reset_operands_unchecked(a=new_a, quantization_scales=scales2)
            result2 = mm.execute()

        # Use FP8-aware comparison
        assert_fp8_equal(result1, result2)

    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    @pytest.mark.parametrize("scale_shape", [(), (1,)])
    def test_reset_tensor_quantization_scales(self, use_cuda, m, n, k, scale_shape):
        """Test resetting single-element tensor quantization scales for FP8.

        According to the documentation, scales can be provided as:
        - Scalars (int or float)
        - Single-element tensors of shape () or (1,)

        This test verifies the single-element tensor format works correctly.
        """

        # Use same setup as test_fp8.py
        atype, btype, ctype, dtype = "float8_e4m3fn", "float8_e4m3fn", "float16", None
        a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

        # Get scalar scales first
        scalar_scales1 = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)

        # Convert to single-element tensors with specified shape (float32 required)
        # Place scales on same device as operands (CPU or GPU)
        device = "cuda" if use_cuda else "cpu"
        scales1 = {
            "a": torch.tensor(scalar_scales1["a"], dtype=torch.float32, device=device).reshape(scale_shape),
            "b": torch.tensor(scalar_scales1["b"], dtype=torch.float32, device=device).reshape(scale_shape),
            "c": torch.tensor(scalar_scales1["c"], dtype=torch.float32, device=device).reshape(scale_shape)
            if scalar_scales1["c"] is not None
            else None,
            "d": torch.tensor(scalar_scales1["d"], dtype=torch.float32, device=device).reshape(scale_shape)
            if scalar_scales1["d"] is not None
            else None,
        }

        # New operand and tensor scales
        new_a, new_b, new_c, new_alpha, new_beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
        scalar_scales2 = choose_scales(new_a, new_b, new_c, atype, btype, ctype, dtype, alpha=new_alpha, beta=new_beta)
        scales2 = {
            "a": torch.tensor(scalar_scales2["a"], dtype=torch.float32, device=device).reshape(scale_shape),
            "b": torch.tensor(scalar_scales2["b"], dtype=torch.float32, device=device).reshape(scale_shape),
            "c": torch.tensor(scalar_scales2["c"], dtype=torch.float32, device=device).reshape(scale_shape)
            if scalar_scales2["c"] is not None
            else None,
            "d": torch.tensor(scalar_scales2["d"], dtype=torch.float32, device=device).reshape(scale_shape)
            if scalar_scales2["d"] is not None
            else None,
        }

        options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}

        # Test with reset_operands
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales1, options=options) as mm:
            mm.plan()
            mm.reset_operands(a=new_a, quantization_scales=scales2)
            result1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales1, options=options) as mm:
            mm.plan()
            mm.reset_operands_unchecked(a=new_a, quantization_scales=scales2)
            result2 = mm.execute()

        # Use FP8-aware comparison
        assert_fp8_equal(result1, result2)


@skip_fp8_tests
class TestResetOperandsUncheckedEpilogAuxQuantizationScale:
    """Test resetting epilog inputs: aux_quantization_scale."""

    @pytest.mark.parametrize("use_cuda", [True, False])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    def test_reset_aux_quantization_scale_scalar(self, use_cuda, m, n, k):
        """Test resetting aux_quantization_scale for FP8 with aux output."""
        # Use same setup as test_fp8_epilogs.py test_epilog_aux_scale_reset
        atype, btype, ctype, dtype = "float8_e4m3fn", "float8_e4m3fn", None, None
        epilog_aux_type = "float8_e4m3fn"
        epilog_chosen = MatmulEpilog.GELU_AUX

        a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

        scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
        options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}
        preferences = {
            "epilog": {
                "aux_type": NAME_TO_DATA_TYPE[epilog_aux_type],
            }
        }

        plan_inputs = {"aux_quantization_scale": 10}
        reset_inputs = {"aux_quantization_scale": 0.5}

        # Test with reset_operands
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales, options=options) as mm:
            mm.plan(epilog=epilog_chosen, epilog_inputs=plan_inputs, preferences=preferences)
            mm.reset_operands(epilog_inputs=reset_inputs)
            result1, aux1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales, options=options) as mm:
            mm.plan(epilog=epilog_chosen, epilog_inputs=plan_inputs, preferences=preferences)
            mm.reset_operands_unchecked(epilog_inputs=reset_inputs)
            result2, aux2 = mm.execute()

        # Compare results - they should match if both methods correctly update the scale
        assert_fp8_equal(result1, result2)
        assert_fp8_equal(aux1["gelu_aux"], aux2["gelu_aux"])

    @pytest.mark.parametrize("use_cuda", [True])
    @pytest.mark.parametrize("m,n,k", [(16, 32, 16)])
    @pytest.mark.parametrize("scale_shape", [(), (1,)])
    def test_reset_aux_quantization_scale_tensor(self, use_cuda, m, n, k, scale_shape):
        """Test resetting aux_quantization_scale as tensor for FP8 with aux output.

        Tests both scalar-shaped tensors () and single-element tensors (1,) as per
        the documentation which states scales can be provided as single-element tensors.
        """

        # Use setup similar to test_fp8_epilogs.py test_epilog_aux_scale_reset
        atype, btype, ctype, dtype = "float8_e4m3fn", "float8_e4m3fn", None, None
        epilog_aux_type = "float8_e4m3fn"
        epilog_chosen = MatmulEpilog.GELU_AUX

        a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

        scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
        options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}
        preferences = {
            "epilog": {
                "aux_type": NAME_TO_DATA_TYPE[epilog_aux_type],
            }
        }

        # Create tensor scales on the same device as operands
        device = "cuda" if use_cuda else "cpu"
        aux_scale1 = torch.tensor(1.0, dtype=torch.float32, device=device).reshape(scale_shape)
        aux_scale2 = torch.tensor(0.5, dtype=torch.float32, device=device).reshape(scale_shape)

        # Test with reset_operands - only reset scale, not operands, to isolate scale effect
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales, options=options) as mm:
            mm.plan(epilog=epilog_chosen, epilog_inputs={"aux_quantization_scale": aux_scale1}, preferences=preferences)
            mm.reset_operands(epilog_inputs={"aux_quantization_scale": aux_scale2})
            result1, aux1 = mm.execute()

        # Test with reset_operands_unchecked
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales, options=options) as mm:
            mm.plan(epilog=epilog_chosen, epilog_inputs={"aux_quantization_scale": aux_scale1}, preferences=preferences)
            mm.reset_operands_unchecked(epilog_inputs={"aux_quantization_scale": aux_scale2})
            result2, aux2 = mm.execute()

        # Compare results - they should match if both methods correctly update the scale
        assert_fp8_equal(result1, result2)
        assert_fp8_equal(aux1["gelu_aux"], aux2["gelu_aux"])
