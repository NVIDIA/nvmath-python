# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

import pytest

from .axes_utils import is_complex
from .common_axes import MemBackend
from .input_fixtures import get_custom_stream

from nvmath.tensor import (
    binary_contraction,
    ternary_contraction,
    ContractionOptions,
    BinaryContraction,
    TernaryContraction,
    Operator,
)
from .check_helpers import assert_all_close, get_contraction_ref, get_contraction_tolerance
from .support_matrix import compute_type_support


class BaseStatelessTester:
    def _test_coefficients(self, alpha, beta, use_offset, test_case, framework, mem_backend, dtype):
        if (use_offset and beta is False) or (not use_offset and beta is not False):
            context = pytest.raises(ValueError)
        elif not is_complex(dtype) and (isinstance(alpha, complex) or isinstance(beta, complex)):
            context = pytest.raises(TypeError)
        else:
            context = nullcontext()

        coeffs = {}
        if alpha is not False:
            coeffs["alpha"] = alpha
        if beta is not False:
            coeffs["beta"] = beta
        self.run_test(test_case, framework, mem_backend, dtype, 23, use_offset=use_offset, context=context, **coeffs)

    def _test_inplace_output(self, offset_format, test_case, framework, mem_backend, dtype):
        kwargs = {
            "alpha": 0.3,
            "beta": 0.5,
        }

        offset_name = "c" if test_case.num_inputs == 2 else "d"

        out = test_case.gen_random_output(framework, dtype, mem_backend, 3)

        if offset_format == "out":
            kwargs[offset_name] = out
        elif offset_format == "new":
            kwargs[offset_name] = test_case.gen_random_output(framework, dtype, mem_backend, 7)
        elif offset_format is False:
            kwargs[offset_name] = None
            kwargs["beta"] = None
        else:
            raise ValueError(f"Invalid offset_format: {offset_format}")

        self.run_test(test_case, framework, mem_backend, dtype, 23, out=out, **kwargs)

    def _test_qualifiers(self, qualifiers, test_case, framework, mem_backend, dtype):
        if not is_complex(dtype) and any(op == Operator.OP_CONJ for op in qualifiers):
            context = pytest.raises(ValueError)
        elif qualifiers[test_case.num_inputs] != Operator.OP_IDENTITY:
            context = pytest.raises(ValueError)  # output operand must be the identity operator
        else:
            context = nullcontext()
        self.run_test(test_case, framework, mem_backend, dtype, 23, context=context, qualifiers=qualifiers)

    def _test_compute_type(self, compute_type, test_case, framework, mem_backend, dtype):
        if compute_type in compute_type_support[dtype]:
            context = nullcontext()
        else:
            context = pytest.raises(ValueError)
        compute_type = compute_type.value
        self.run_test(test_case, framework, mem_backend, dtype, 11, context=context, options={"compute_type": compute_type})

    def _parse_operands(self, test_case, framework, mem_backend, dtype, seed, use_offset, c=None, d=None, out=None, **kwargs):
        if test_case.num_inputs == 2:
            a, b = test_case.gen_input_operands(framework, dtype, mem_backend, seed)
            if use_offset:
                assert c is None, "c cannot be provided if use_offset is True"
                c = test_case.gen_random_output(framework, dtype, mem_backend, seed + 1)
        elif test_case.num_inputs == 3:
            assert c is None, "c can not be provided as a keyword argument for ternary contraction"
            a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend, seed)
            if use_offset:
                assert d is None, "d cannot be provided if use_offset is True"
                d = test_case.gen_random_output(framework, dtype, mem_backend, seed + 1)
        else:
            raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")
        return a, b, c, d, out

    def _parse_options(self, options):
        options = {} if options is None else options
        if isinstance(options, ContractionOptions):
            blocking = options.blocking
            compute_type = options.compute_type
        else:
            blocking = "auto"
            compute_type = options.get("compute_type", None)
        return blocking, compute_type

    def run_test(
        self,
        test_case,
        framework,
        mem_backend,
        dtype,
        seed,
        *,
        use_offset=False,
        context=None,
        stream=None,
        options=None,
        **kwargs,
    ):
        if context is None:
            context = nullcontext()

        if stream is True:
            stream = get_custom_stream(framework)

        a, b, c, d, out = self._parse_operands(test_case, framework, mem_backend, dtype, seed, use_offset, **kwargs)
        for key in ["c", "d", "out"]:
            kwargs.pop(key, None)

        blocking, compute_type = self._parse_options(options)
        sync_needed = blocking == "auto" and mem_backend == MemBackend.cuda and stream is not None

        tolerance = get_contraction_tolerance(dtype.name, compute_type)

        with context:
            # reference must be computed first as out may be modified
            #   by the contraction, e.g, when c is the same as out
            ref = get_contraction_ref(test_case.equation, a, b, c=c, d=d, **kwargs)
            if test_case.num_inputs == 2:
                result = binary_contraction(test_case.equation, a, b, c=c, stream=stream, options=options, out=out, **kwargs)
            elif test_case.num_inputs == 3:
                result = ternary_contraction(
                    test_case.equation, a, b, c, d=d, stream=stream, options=options, out=out, **kwargs
                )
            else:
                raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")
            if sync_needed:
                # stream is guaranteed to be either a
                #   cupy.cuda.Stream or a torch.cuda.Stream object
                stream.synchronize()

            assert_all_close(result, ref, **tolerance)
            if out is not None:
                assert result is out


class BaseStatefulTester:
    _parse_options = BaseStatelessTester._parse_options
    _parse_operands = BaseStatelessTester._parse_operands
    _test_coefficients = BaseStatelessTester._test_coefficients

    def run_test(
        self,
        test_case,
        framework,
        mem_backend,
        dtype,
        seed,
        *,
        use_offset=False,
        context=None,
        stream=None,
        options=None,
        plan_preferences=None,
        **kwargs,
    ):
        if context is None:
            context = nullcontext()

        if stream is True:
            stream = get_custom_stream(framework)

        a, b, c, d, out = self._parse_operands(test_case, framework, mem_backend, dtype, seed, use_offset, **kwargs)
        for key in ["c", "d", "out"]:
            kwargs.pop(key, None)

        blocking, compute_type = self._parse_options(options)
        sync_needed = blocking == "auto" and mem_backend == MemBackend.cuda and stream is not None

        tolerance = get_contraction_tolerance(dtype.name, compute_type)

        if test_case.num_inputs == 2:
            contraction = BinaryContraction(test_case.equation, a, b, c=c, stream=stream, options=options, out=out)
        elif test_case.num_inputs == 3:
            contraction = TernaryContraction(test_case.equation, a, b, c, d=d, stream=stream, options=options, out=out)
        else:
            raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")

        with context:
            ref = get_contraction_ref(test_case.equation, a, b, c=c, d=d, **kwargs)
            with contraction:
                contraction.plan()
                result = contraction.execute(**kwargs, stream=stream)
                if sync_needed:
                    # stream is guaranteed to be either a cupy.cuda.Stream
                    #   or a torch.cuda.Stream object
                    stream.synchronize()
                assert_all_close(result, ref, **tolerance)
                if out is not None:
                    assert result is out

                if plan_preferences is not None:
                    preference = contraction.plan_preference
                    for key, value in plan_preferences.items():
                        setattr(preference, key, value)
                    contraction.plan()
                if test_case.num_inputs == 2:
                    a, b = test_case.gen_input_operands(framework, dtype, mem_backend, seed + 23)
                    if c is not None:
                        c = test_case.gen_random_output(framework, dtype, mem_backend, seed + 24)
                elif test_case.num_inputs == 3:
                    a, b, c = test_case.gen_input_operands(framework, dtype, mem_backend, seed + 23)
                    if d is not None:
                        d = test_case.gen_random_output(framework, dtype, mem_backend, seed + 24)
                else:
                    raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")

                kwargs["alpha"] = -0.3 * kwargs.get("alpha", 1.0) + 0.2
                if "beta" in kwargs:
                    # NOTE: beta can only be updated/specified when offset is specified
                    kwargs["beta"] = -0.4 * kwargs["beta"] + 0.1 if kwargs["beta"] is not None else None
                ref = get_contraction_ref(test_case.equation, a, b, c=c, d=d, **kwargs)
                if test_case.num_inputs == 2:
                    contraction.reset_operands(a=a, b=b, c=c)
                elif test_case.num_inputs == 3:
                    contraction.reset_operands(a=a, b=b, c=c, d=d)
                else:
                    raise ValueError(f"Invalid number of inputs: {test_case.num_inputs}")
                result = contraction.execute(**kwargs, stream=stream)
                if sync_needed:
                    stream.synchronize()
                assert_all_close(result, ref, **tolerance)
                if out is not None:
                    assert result is out
