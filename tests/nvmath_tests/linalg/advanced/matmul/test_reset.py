# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This set of tests checks reset_operands
"""
import nvmath
import pytest
from .utils import *


@pytest.mark.parametrize("framework", ("numpy/cupy", "torch"))
@pytest.mark.parametrize("dtype", ("float32",))
@pytest.mark.parametrize(
    "reset_a",
    (
        True,
        False,
    ),
)
@pytest.mark.parametrize(
    "reset_b",
    (
        True,
        False,
    ),
)
@pytest.mark.parametrize(
    "with_alpha, reset_alpha", ((False, False), (True, False), (True, True))
)
@pytest.mark.parametrize(
    "with_c, reset_c, reset_beta, with_epilog, reset_epilog",
    (
        # No c, no epilog
        (False, False, False, False, False),
        # With c, no epilog
        (True, False, False, False, False),
        (True, False, True, False, False),
        (True, True, False, False, False),
        (True, True, True, False, False),
        # No c, with epilog
        (False, False, False, True, False),
        (False, False, False, True, True),
    ),
)
@pytest.mark.parametrize("reset_to_none", (True, False))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_reset(
    framework,
    dtype,
    reset_a,
    reset_b,
    with_alpha,
    reset_alpha,
    with_c,
    reset_c,
    reset_beta,
    with_epilog,
    reset_epilog,
    reset_to_none,
    use_cuda,
):
    """
    Tests resetting particular operands
    """
    if not any((reset_a, reset_b, reset_c, reset_alpha, reset_beta, reset_epilog)):
        pytest.skip("No operand will be reset in this test")

    m, n, k = 12, 34, 56
    a = sample_matrix(framework, dtype, (m, k), use_cuda)
    b = sample_matrix(framework, dtype, (k, n), use_cuda)
    c = sample_matrix(framework, dtype, (m, n), use_cuda)
    alpha = 0.12
    beta = 0.34

    if with_epilog:
        skip_if_cublas_before(11501)  # Epilog inputs not fully supported

    epilog = nvmath.linalg.advanced.MatmulEpilog.BIAS
    epilog_inputs = {"bias": sample_matrix(framework, dtype, (m, 1), use_cuda)}

    matmul_kwargs = {}

    if with_alpha:
        matmul_kwargs["alpha"] = alpha

    if with_c:
        matmul_kwargs["c"] = c
        matmul_kwargs["beta"] = beta

    with nvmath.linalg.advanced.Matmul(a, b, **matmul_kwargs) as mm:
        if with_epilog:
            mm.plan(epilog=epilog, epilog_inputs=epilog_inputs)
        else:
            mm.plan()

        reference1 = a @ b * (alpha if with_alpha else 1)
        if with_c:
            reference1 += c * beta
        if with_epilog:
            reference1 += epilog_inputs["bias"]

        result1 = mm.execute()
        assert_tensors_equal(result1, reference1)

        if reset_to_none:
            mm.reset_operands(None)

        new_a = sample_matrix(framework, dtype, (m, k), use_cuda)
        new_b = sample_matrix(framework, dtype, (k, n), use_cuda)
        new_c = sample_matrix(framework, dtype, (m, n), use_cuda)
        new_alpha = 0.56
        new_beta = 0.78
        new_epilog_inputs = {"bias": sample_matrix(framework, dtype, (m, 1), use_cuda)}

        reset_kwargs = {}
        if reset_a:
            reset_kwargs["a"] = new_a
        if reset_b:
            reset_kwargs["b"] = new_b
        if reset_c:
            reset_kwargs["c"] = new_c
        if reset_alpha:
            reset_kwargs["alpha"] = new_alpha
        if reset_beta:
            reset_kwargs["beta"] = new_beta
        if reset_epilog:
            reset_kwargs["epilog_inputs"] = new_epilog_inputs

        all_operands_reset = (
            reset_a
            and reset_b
            and (reset_c or not with_c)
            and (reset_epilog or not with_epilog)
        )
        if reset_to_none and not all_operands_reset:
            with pytest.raises(ValueError):
                mm.reset_operands(**reset_kwargs)
        else:
            mm.reset_operands(**reset_kwargs)

            reference2 = (new_a if reset_a else a) @ (new_b if reset_b else b)
            reference2 *= new_alpha if reset_alpha else alpha if with_alpha else 1
            if with_c:
                reference2 += (new_c if reset_c else c) * (
                    new_beta if reset_beta else beta
                )
            if with_epilog:
                reference2 += (
                    new_epilog_inputs["bias"] if reset_epilog else epilog_inputs["bias"]
                )

            result2 = mm.execute()
            assert_tensors_equal(result2, reference2)


@pytest.mark.parametrize("framework", ("numpy/cupy",))
@pytest.mark.parametrize("dtype", ("float64",))
@pytest.mark.parametrize("a_mismatch", (True, False))
@pytest.mark.parametrize("b_mismatch", (True, False))
@pytest.mark.parametrize(
    "with_c, c_mismatch, with_epilog, bias_mismatch",
    (
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (False, False, False, False),
        (False, False, True, False),
        (False, False, True, True),
    ),
)
@pytest.mark.parametrize("use_cuda", (True, False))
def test_shape_mismatch(
    framework,
    dtype,
    a_mismatch,
    b_mismatch,
    with_c,
    c_mismatch,
    with_epilog,
    bias_mismatch,
    use_cuda,
):
    """
    Checks if resetting operands to ones of different shapes results in appropriate error message
    """
    m, n, k = 54, 32, 10
    a = sample_matrix(framework, dtype, (m, k), use_cuda)
    b = sample_matrix(framework, dtype, (k, n), use_cuda)
    c = sample_matrix(framework, dtype, (m, n), use_cuda) if with_c else None

    if with_epilog:
        skip_if_cublas_before(11501)  # Epilog inputs not fully supported

    epilog = nvmath.linalg.advanced.MatmulEpilog.BIAS
    epilog_inputs = {"bias": sample_matrix(framework, dtype, (m, 1), use_cuda)}

    with nvmath.linalg.advanced.Matmul(a, b, c=c, beta=2 if with_c else None) as mm:
        if with_epilog:
            mm.plan(epilog=epilog, epilog_inputs=epilog_inputs)
        else:
            mm.plan()
        mm.execute()

        new_a = sample_matrix(
            framework, dtype, (m, k + 1) if a_mismatch else (m, k), use_cuda
        )
        new_b = sample_matrix(
            framework, dtype, (k, n + 3) if b_mismatch else (k, n), use_cuda
        )
        new_c = (
            sample_matrix(
                framework, dtype, (m + 9, n - 3) if c_mismatch else (m, n), use_cuda
            )
            if with_c
            else None
        )
        new_epilog_inputs = (
            {
                "bias": sample_matrix(
                    framework, dtype, (m - 1, 1) if bias_mismatch else (m, 1), use_cuda
                )
            }
            if with_epilog
            else None
        )

        if any((a_mismatch, b_mismatch, c_mismatch, bias_mismatch)):
            with pytest.raises(ValueError, match="The extents .* must match"):
                mm.reset_operands(
                    a=new_a, b=new_b, c=new_c, epilog_inputs=new_epilog_inputs
                )
        else:
            pytest.skip("All shapes match")


@pytest.mark.parametrize("framework", ("numpy/cupy",))
@pytest.mark.parametrize(
    "dtype, bad_dtype", (("float64", "float32"), ("float32", "float64"))
)
@pytest.mark.parametrize("a_mismatch", (True, False))
@pytest.mark.parametrize("b_mismatch", (True, False))
@pytest.mark.parametrize(
    "with_c, c_mismatch, with_epilog, bias_mismatch",
    (
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (False, False, False, False),
        (False, False, True, False),
        (False, False, True, True),
    ),
)
@pytest.mark.parametrize("use_cuda", (True, False))
def test_dtype_mismatch(
    framework,
    dtype,
    bad_dtype,
    a_mismatch,
    b_mismatch,
    with_c,
    c_mismatch,
    with_epilog,
    bias_mismatch,
    use_cuda,
):
    """
    Checks if resetting operands to ones with different dtypes results in appropriate error message
    """

    m, n, k = 19, 28, 37
    a = sample_matrix(framework, dtype, (m, k), use_cuda)
    b = sample_matrix(framework, dtype, (k, n), use_cuda)
    c = sample_matrix(framework, dtype, (m, n), use_cuda) if with_c else None

    if with_epilog:
        skip_if_cublas_before(11501)  # Epilog inputs not fully supported

    epilog = nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS
    epilog_inputs = {"bias": sample_matrix(framework, dtype, (m, 1), use_cuda)}

    with nvmath.linalg.advanced.Matmul(a, b, c=c, beta=2 if with_c else None) as mm:
        if with_epilog:
            mm.plan(epilog=epilog, epilog_inputs=epilog_inputs)
        else:
            mm.plan()
        mm.execute()

        new_a = sample_matrix(
            framework, bad_dtype if a_mismatch else dtype, (m, k), use_cuda
        )
        new_b = sample_matrix(
            framework, bad_dtype if b_mismatch else dtype, (k, n), use_cuda
        )
        new_c = (
            sample_matrix(
                framework, bad_dtype if c_mismatch else dtype, (m, n), use_cuda
            )
            if with_c
            else None
        )
        new_epilog_inputs = (
            {
                "bias": sample_matrix(
                    framework, bad_dtype if bias_mismatch else dtype, (m, 1), use_cuda
                )
            }
            if with_epilog
            else None
        )

        if any((a_mismatch, b_mismatch, c_mismatch, bias_mismatch)):
            with pytest.raises(
                ValueError,
                match="The data type of the new operand must match the data type of the original operand.",
            ):
                mm.reset_operands(
                    a=new_a, b=new_b, c=new_c, epilog_inputs=new_epilog_inputs
                )
        else:
            # All shapes match, just check if nothing explodes here
            mm.reset_operands(
                a=new_a, b=new_b, c=new_c, epilog_inputs=new_epilog_inputs
            )
            mm.execute()


@pytest.mark.parametrize(
    "framework, bad_framework", (("numpy/cupy", "torch"), ("torch", "numpy/cupy"))
)
@pytest.mark.parametrize("dtype", ("float64",))
@pytest.mark.parametrize("a_mismatch", (True, False))
@pytest.mark.parametrize("b_mismatch", (True, False))
@pytest.mark.parametrize(
    "with_c, c_mismatch, with_epilog, bias_mismatch",
    (
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (False, False, False, False),
        (False, False, True, False),
        (False, False, True, True),
    ),
)
@pytest.mark.parametrize("use_cuda", (True, False))
def test_framework_mismatch(
    framework,
    bad_framework,
    dtype,
    a_mismatch,
    b_mismatch,
    with_c,
    c_mismatch,
    with_epilog,
    bias_mismatch,
    use_cuda,
):
    """
    Checks if resetting operands to ones from different framework results in appropriate error message
    """

    m, n, k = 10, 11, 12
    a = sample_matrix(framework, dtype, (m, k), use_cuda)
    b = sample_matrix(framework, dtype, (k, n), use_cuda)
    c = sample_matrix(framework, dtype, (m, n), use_cuda) if with_c else None

    if with_epilog:
        skip_if_cublas_before(11501)  # Epilog inputs not fully supported

    epilog = nvmath.linalg.advanced.MatmulEpilog.BIAS
    epilog_inputs = {"bias": sample_matrix(framework, dtype, (m, 1), use_cuda)}

    with nvmath.linalg.advanced.Matmul(a, b, c=c, beta=2 if with_c else None) as mm:
        if with_epilog:
            mm.plan(epilog=epilog, epilog_inputs=epilog_inputs)
        else:
            mm.plan()
        mm.execute()

        new_a = sample_matrix(
            bad_framework if a_mismatch else framework, dtype, (m, k), use_cuda
        )
        new_b = sample_matrix(
            bad_framework if b_mismatch else framework, dtype, (k, n), use_cuda
        )
        new_c = (
            sample_matrix(
                bad_framework if c_mismatch else framework, dtype, (m, n), use_cuda
            )
            if with_c
            else None
        )
        new_epilog_inputs = (
            {
                "bias": sample_matrix(
                    bad_framework if bias_mismatch else framework,
                    dtype,
                    (m, 1),
                    use_cuda,
                )
            }
            if with_epilog
            else None
        )

        if any((a_mismatch, b_mismatch, c_mismatch, bias_mismatch)):
            with pytest.raises(TypeError, match="Library package mismatch"):
                mm.reset_operands(
                    a=new_a, b=new_b, c=new_c, epilog_inputs=new_epilog_inputs
                )
        else:
            # All dtypes match, just check if nothing explodes here
            mm.reset_operands(
                a=new_a, b=new_b, c=new_c, epilog_inputs=new_epilog_inputs
            )
            mm.execute()


@pytest.mark.parametrize("framework", ("numpy/cupy",))
@pytest.mark.parametrize("dtype", ("float32",))
@pytest.mark.parametrize("ta", (True, False))
@pytest.mark.parametrize("tb", (True, False))
@pytest.mark.parametrize("tc", (True, False))
def test_layout_change(framework, dtype, ta, tb, tc):
    """
    Check if layout change of the input matrix is handled correctly
    """
    m = n = k = 5
    a = sample_matrix(framework, dtype, (m, k), True)
    b = sample_matrix(framework, dtype, (k, n), True)
    c = sample_matrix(framework, dtype, (m, n), True)

    with nvmath.linalg.advanced.Matmul(a, b, c=c, beta=1) as mm:
        mm.plan()
        result1 = mm.execute()
        assert_tensors_equal(result1, a @ b + c)

        a = a.T if ta else a
        b = b.T if tb else b
        c = c.T if tc else c

        if ta or tb or tc:
            with pytest.raises(ValueError, match="The strides .* must match"):
                mm.reset_operands(a=a, b=b, c=c)


@pytest.mark.parametrize("b_conj_init, b_conj_reset", ((True, False), (False, True)))
def test_conjugate_flag(b_conj_init, b_conj_reset):
    """
    Tests if conjugate flag of torch tensors is inferred again on reset.

    Only checks GPU tensors, because conj flag is reset on H2D copy.

    Only checks B, because changing conj flag of A requires transposing it due to cublas requirements,
    which causes stride mismatch.
    """
    m, k, n = 3, 4, 5

    a = random_torch_complex((m, k), True, True)
    b = random_torch_complex((k, n), True, True)
    c = random_torch_complex((m, n), True, False)

    if b_conj_init:
        b = b.conj()

    with nvmath.linalg.advanced.Matmul(a, b, c=c, beta=1) as mm:
        mm.plan()
        result1 = mm.execute()
        assert_tensors_equal(result1, a @ b + c)

        b = random_torch_complex((k, n), True, True)
        if b_conj_reset:
            b = b.conj()

        with pytest.raises(ValueError):
            mm.reset_operands(b=b)
