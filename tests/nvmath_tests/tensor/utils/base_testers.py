# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

from nvmath.tensor import BinaryContraction, ContractionOptions, TernaryContraction, binary_contraction, ternary_contraction

from .axes_utils import is_complex
from .check_helpers import assert_all_close, get_contraction_ref, get_contraction_tolerance
from .common_axes import MemBackend


def _parse_options(options):
    options = {} if options is None else options
    if isinstance(options, ContractionOptions):
        blocking = options.blocking
        compute_type = options.compute_type
    else:
        blocking = "auto"
        compute_type = options.get("compute_type", None)
    return blocking, compute_type


def parse_operands(test_case, framework, mem_backend, dtype, use_offset, c=None, d=None, out=None):
    if test_case.num_inputs == 2:
        a, b = test_case.gen_input_operands(framework, dtype, mem_backend)
        if use_offset:
            assert c is None, "c cannot be provided if use_offset is True"
            c = test_case.gen_random_output(framework, dtype, mem_backend)
    elif test_case.num_inputs == 3:
        assert c is None, "c can not be provided as a keyword argument for ternary contraction"
        a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend)
        if use_offset:
            assert d is None, "d cannot be provided if use_offset is True"
            d = test_case.gen_random_output(framework, dtype, mem_backend)
    else:
        raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")
    return a, b, c, d, out


def run_stateless_impl(
    test_case,
    framework,
    mem_backend,
    dtype,
    a,
    b,
    *,
    c=None,
    d=None,
    out=None,
    context=None,
    stream=None,
    options=None,
    **kwargs,
):
    if context is None:
        context = nullcontext()

    blocking, compute_type = _parse_options(options)
    sync_needed = blocking == "auto" and mem_backend == MemBackend.cuda and stream is not None

    tolerance = get_contraction_tolerance(dtype.name, compute_type)

    with context:
        # reference must be computed first as out may be modified
        #   by the contraction, e.g, when c is the same as out
        ref = get_contraction_ref(test_case.equation, a, b, c=c, d=d, stream=stream, **kwargs)
        if test_case.num_inputs == 2:
            result = binary_contraction(test_case.equation, a, b, c=c, stream=stream, options=options, out=out, **kwargs)
        elif test_case.num_inputs == 3:
            result = ternary_contraction(test_case.equation, a, b, c, d=d, stream=stream, options=options, out=out, **kwargs)
        else:
            raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")
        if sync_needed:
            # stream is guaranteed to be either a
            #   cupy.cuda.Stream or a torch.cuda.Stream object
            stream.synchronize()

        assert_all_close(result, ref, **tolerance)
        if out is not None:
            assert result is out


def run_stateful_impl(
    test_case,
    framework,
    mem_backend,
    dtype,
    a,
    b,
    *,
    c=None,
    d=None,
    context=None,
    stream=None,
    options=None,
    out=None,
    test_reset_operands=None,
    **kwargs,
):
    if context is None:
        context = nullcontext()

    blocking, compute_type = _parse_options(options)
    sync_needed = blocking == "auto" and mem_backend == MemBackend.cuda and stream is not None

    tolerance = get_contraction_tolerance(dtype.name, compute_type)

    if test_case.num_inputs == 2:
        contraction = BinaryContraction(test_case.equation, a, b, c=c, stream=stream, options=options, out=out)
    elif test_case.num_inputs == 3:
        contraction = TernaryContraction(test_case.equation, a, b, c, d=d, stream=stream, options=options, out=out)
    else:
        raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")

    # Order matters here: contraction must be outer context so that
    # context (pytest.raises or nullcontext) can catch any exceptions raised
    # during get_contraction_ref() or contraction.execute()
    with contraction, context:
        ref = get_contraction_ref(test_case.equation, a, b, c=c, d=d, stream=stream, **kwargs)
        contraction.plan(stream=stream)
        result = contraction.execute(**kwargs, stream=stream)
        if sync_needed:
            stream.synchronize()
        assert_all_close(result, ref, **tolerance)
        if out is not None:
            assert result is out

        if test_reset_operands:
            # Use the provided new operands from test_reset_operands dict
            a_new = test_reset_operands.get("a")
            b_new = test_reset_operands.get("b")
            # Only use c_new if c was originally provided
            c_new = test_reset_operands.get("c") if c is not None else None
            # Only use d_new if d was originally provided
            d_new = test_reset_operands.get("d") if d is not None else None

            kwargs["alpha"] = -0.3 * kwargs.get("alpha", 1.0) + 0.2
            if "beta" in kwargs:
                # NOTE: beta can only be updated/specified when offset is specified
                kwargs["beta"] = -0.4 * kwargs["beta"] + 0.1 if kwargs["beta"] is not None else None

            # Compute reference with new operands
            if test_case.num_inputs == 2:
                ref = get_contraction_ref(test_case.equation, a_new, b_new, c=c_new, d=None, **kwargs)
                contraction.reset_operands(a=a_new, b=b_new, c=c_new)
            elif test_case.num_inputs == 3:
                ref = get_contraction_ref(test_case.equation, a_new, b_new, c=c_new, d=d_new, **kwargs)
                contraction.reset_operands(a=a_new, b=b_new, c=c_new, d=d_new)

            result = contraction.execute(**kwargs, stream=stream)
            if sync_needed:
                stream.synchronize()
            assert_all_close(result, ref, **tolerance)
            if out is not None:
                assert result is out


def run_coefficients_test_impl(
    test_case, framework, mem_backend: MemBackend, dtype, impl_type: str, alpha, beta, use_offset: bool
):
    """
    Common implementation for testing different coefficient combinations.
    """
    assert impl_type in ["stateless", "stateful"]

    # Allocate tensors ONCE before the loop
    if test_case.num_inputs == 2:
        a, b = test_case.gen_input_operands(framework, dtype, mem_backend)
        offset_c = test_case.gen_random_output(framework, dtype, mem_backend)
        offset_d = None
    elif test_case.num_inputs == 3:
        a, b, input_c = test_case.gen_input_operands(framework, dtype, mem_backend)
        offset_d = test_case.gen_random_output(framework, dtype, mem_backend)
        offset_c = None
    else:
        raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")

    # Generate fresh operands for reset_operands testing (for stateful impl only)
    reset_operands_dict = {}
    if impl_type == "stateful":
        if test_case.num_inputs == 2:
            a_new, b_new = test_case.gen_input_operands(framework, dtype, mem_backend)
            c_new = test_case.gen_random_output(framework, dtype, mem_backend)
            reset_operands_dict = {"a": a_new, "b": b_new, "c": c_new, "d": None}
        elif test_case.num_inputs == 3:
            a_new, b_new, c_new = test_case.gen_input_operands(framework, dtype, mem_backend)
            d_new = test_case.gen_random_output(framework, dtype, mem_backend)
            reset_operands_dict = {"a": a_new, "b": b_new, "c": c_new, "d": d_new}

    # Skip invalid combinations instead of testing them.
    # These are tested in TestCoefficientValidationAndExceptions.
    # Invalid: offset without beta, or beta without offset
    if (use_offset and beta is False) or (not use_offset and beta is not False):
        return
    # Invalid: complex coefficients with non-complex dtype
    if not is_complex(dtype) and (isinstance(alpha, complex) or isinstance(beta, complex)):
        return

    # Determine which offset to use
    if test_case.num_inputs == 2:
        c = offset_c if use_offset else None
        d = None
    elif test_case.num_inputs == 3:
        c = input_c
        d = offset_d if use_offset else None

    # Build coefficient arguments
    coeffs = {}
    if alpha is not False:
        coeffs["alpha"] = alpha
    if beta is not False:
        coeffs["beta"] = beta

    # Run test with pre-allocated operands (only valid combinations)
    if impl_type == "stateless":
        run_stateless_impl(test_case, framework, mem_backend, dtype, a, b, c=c, d=d, **coeffs)
    elif impl_type == "stateful":
        run_stateful_impl(
            test_case,
            framework,
            mem_backend,
            dtype,
            a,
            b,
            c=c,
            d=d,
            test_reset_operands=reset_operands_dict,
            **coeffs,
        )
    else:
        raise ValueError(f"Invalid implementation type: {impl_type}")
