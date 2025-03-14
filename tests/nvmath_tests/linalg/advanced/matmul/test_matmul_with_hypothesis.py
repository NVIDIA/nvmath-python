# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import collections
import logging
import re
import typing

from hypothesis import given, assume
from hypothesis.extra.numpy import arrays, from_dtype
from hypothesis.strategies import (
    one_of,
    tuples,
    none,
    floats,
    integers,
    sampled_from,
    fixed_dictionaries,
    composite,
)
import pytest

try:
    import cupy as cp
except ModuleNotFoundError:
    pytest.skip("cupy is required for matmul tests", allow_module_level=True)
import numpy as np

import nvmath.linalg
from nvmath import CudaDataType
from nvmath.bindings.cublasLt import cuBLASLtError, ReductionScheme
from nvmath.linalg._internal.typemaps import (
    NAMES_TO_DEFAULT_COMPUTE_TYPE,
    CUBLAS_COMPUTE_TYPE_TO_NAME,
    SCALE_TYPE_TO_DEFAULT_COMPUTE_TYPE,
    COMPUTE_TYPE_TO_DEFAULT_SCALE_TYPE,
)
from nvmath.linalg.advanced import MatmulEpilog, MatmulNumericalImplFlags, MatmulPlanPreferences, matmul
from nvmath.linalg.advanced.matmulmod import EPILOG_INPUT_HANDLERS_MAP, EPILOG_MINIMUM_VERSIONS_MAP
from nvmath.memory import _RawCUDAMemoryManager, BaseCUDAMemoryManager, _CupyCUDAMemoryManager

from nvmath_tests.helpers import nvmath_seed
from .utils import get_absolute_tolerance

MatmulEpilog_BIAS_list = [
    MatmulEpilog.BIAS,
    MatmulEpilog.RELU_BIAS,
    MatmulEpilog.RELU_AUX_BIAS,
    MatmulEpilog.GELU_BIAS,
    MatmulEpilog.GELU_AUX_BIAS,
]
MatmulEpilog_DRELU_list = [MatmulEpilog.DRELU, MatmulEpilog.DRELU_BGRAD]
MatmulEpilog_DGELU_list = [MatmulEpilog.DGELU, MatmulEpilog.DGELU_BGRAD]
MatmulEpilog_RELU_list = [MatmulEpilog.RELU, MatmulEpilog.RELU_AUX, MatmulEpilog.RELU_BIAS, MatmulEpilog.RELU_AUX_BIAS]
MatmulEpilog_GELU_list = [MatmulEpilog.GELU, MatmulEpilog.GELU_AUX, MatmulEpilog.GELU_BIAS, MatmulEpilog.GELU_AUX_BIAS]
MatmulEpilog_valid_pairs_list = [
    (MatmulEpilog.BGRADA, None),
    (MatmulEpilog.BGRADB, None),
    (MatmulEpilog.BIAS, None),
    (MatmulEpilog.GELU_AUX_BIAS, MatmulEpilog.DGELU_BGRAD),
    (MatmulEpilog.GELU_AUX_BIAS, MatmulEpilog.DGELU),
    (MatmulEpilog.GELU_AUX_BIAS, None),
    (MatmulEpilog.GELU_AUX, MatmulEpilog.DGELU_BGRAD),
    (MatmulEpilog.GELU_AUX, MatmulEpilog.DGELU),
    (MatmulEpilog.GELU_AUX, None),
    (MatmulEpilog.GELU_BIAS, None),
    (MatmulEpilog.GELU, None),
    (MatmulEpilog.RELU_AUX_BIAS, MatmulEpilog.DRELU_BGRAD),
    (MatmulEpilog.RELU_AUX_BIAS, MatmulEpilog.DRELU),
    (MatmulEpilog.RELU_AUX_BIAS, None),
    (MatmulEpilog.RELU_AUX, MatmulEpilog.DRELU_BGRAD),
    (MatmulEpilog.RELU_AUX, MatmulEpilog.DRELU),
    (MatmulEpilog.RELU_AUX, None),
    (MatmulEpilog.RELU_BIAS, None),
    (MatmulEpilog.RELU, None),
    (None, None),
]


def round_up(m, base):
    return m + (base - m) % base


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def dgelu(x, dgelu_mask):
    dgelu_mask_ = dgelu_mask[: x.shape[0], :]

    tanh_out = np.tanh(np.sqrt(2 / np.pi) * dgelu_mask_ * (1.0 + 0.044715 * np.power(dgelu_mask_, 2)))
    ff = 0.5 * dgelu_mask_ * (
        (1.0 - np.power(tanh_out, 2)) * (np.sqrt(2 / np.pi) + 0.1070322243 * np.power(dgelu_mask_, 2))
    ) + 0.5 * (1.0 + tanh_out)
    return ff * x


def relu(x):
    return np.maximum(x, 0)


def compare_result(ref, res):
    np.testing.assert_allclose(
        cp.asnumpy(res),
        ref,
        equal_nan=True,
        rtol=(1e-02 if res.dtype == np.float16 else 2e-05),
        atol=2 * get_absolute_tolerance(ref),
    )


def verify_bitmask(x, bitmask):
    n, m = x.shape
    for i in range(n):
        for j in range(m):
            expected = (x[i][j] != 0).item()
            actual = bool(bitmask[i // 8][j].astype(int) & (1 << i % 8))
            if expected != actual:
                return False
    return True


def drelu(x, bitmask):
    result = np.zeros(x.shape, dtype=x.dtype)
    n, m = x.shape
    for i in range(n):
        for j in range(m):
            result[i][j] = bool(bitmask[i // 8][j].astype(int) & (1 << i % 8))
    return x * result


def verify_result(a, b, c, result_c, alpha, beta, epilog, epilog_inputs):
    possible_dtype = CUBLAS_COMPUTE_TYPE_TO_NAME[NAMES_TO_DEFAULT_COMPUTE_TYPE[(str(a.dtype), str(b.dtype))]]
    compute_dtype = possible_dtype[1] if np.iscomplexobj(a) else possible_dtype[0]

    added_singleton_dimensions: list[int] = []
    if a.ndim == 1:
        a = a[None, ...]
        added_singleton_dimensions.append(0)
    if b.ndim == 1:
        b = b[..., None]
        added_singleton_dimensions.append(1)
    if c is not None and c.ndim == 1:
        # nvmath and numpy have different broadcasting for `c`. nvmath assumes that a 1D `c`
        # has length `M`. numpy assumes 1D `c` has length `N` to be consistent with
        # broadcasting behavior where singleton dimensions are always prepended.
        c = c[..., None]

    ab = (
        np.matmul(a, b, dtype=compute_dtype)
        if alpha is None
        else np.matmul(np.multiply(alpha, a, dtype=compute_dtype), b, dtype=compute_dtype)
    )
    ref_c = ab if c is None else np.add(ab, np.multiply(c, beta, dtype=compute_dtype), dtype=compute_dtype)
    abc = ref_c

    if epilog is not None:
        if epilog in MatmulEpilog_BIAS_list:
            ref_c = ref_c + cp.asnumpy(epilog_inputs["bias"].astype(compute_dtype))
        if epilog in MatmulEpilog_RELU_list:
            ref_c = relu(ref_c)
        if epilog in MatmulEpilog_GELU_list:
            ref_c = gelu(ref_c)
        if epilog in MatmulEpilog_DRELU_list:
            ref_c = drelu(ref_c, cp.asnumpy(epilog_inputs["relu_aux"].astype(compute_dtype)))
        if epilog in MatmulEpilog_DGELU_list:
            ref_c = dgelu(ref_c, cp.asnumpy(epilog_inputs["gelu_aux"].astype(compute_dtype)))
        if epilog == MatmulEpilog.BGRADA:
            compare_result(a.sum(axis=1, dtype=compute_dtype).astype(a.dtype), result_c[1]["bgrada"])
        if epilog == MatmulEpilog.BGRADB:
            compare_result(b.sum(axis=0, dtype=compute_dtype).astype(a.dtype), result_c[1]["bgradb"])
        if epilog == MatmulEpilog.DRELU_BGRAD:
            compare_result(ref_c.sum(axis=1, dtype=compute_dtype).astype(a.dtype), result_c[1]["drelu_bgrad"])
        if epilog == MatmulEpilog.DGELU_BGRAD:
            compare_result(ref_c.sum(axis=1, dtype=compute_dtype).astype(a.dtype), result_c[1]["dgelu_bgrad"])
        if epilog == MatmulEpilog.GELU_AUX_BIAS:
            compare_result(
                (abc + cp.asnumpy(epilog_inputs["bias"].astype(compute_dtype))).astype(a.dtype),
                result_c[1]["gelu_aux"][: abc.shape[0], :],
            )
        if epilog == MatmulEpilog.GELU_AUX:
            compare_result(abc.astype(a.dtype), result_c[1]["gelu_aux"][: abc.shape[0], :])
        if epilog == MatmulEpilog.RELU_AUX:
            assert verify_bitmask(abc >= 0, result_c[1]["relu_aux"])
        if epilog == MatmulEpilog.RELU_AUX_BIAS:
            assert verify_bitmask((abc + cp.asnumpy(epilog_inputs["bias"].astype(compute_dtype))) >= 0, result_c[1]["relu_aux"])

    if added_singleton_dimensions:
        ref_c = np.squeeze(ref_c, axis=tuple(added_singleton_dimensions))

    result_c_ = result_c[0] if isinstance(result_c, tuple) else result_c
    compare_result(ref_c.astype(a.dtype), result_c_)


problem_size_mnk = integers(min_value=1, max_value=256)

options_blocking_values = [True, "auto"]
options_allocator_values = [
    _RawCUDAMemoryManager(0, logging.getLogger()),
    _CupyCUDAMemoryManager(0, logging.getLogger()),
]

# FIXME: Add integer types to tests
ab_type_values = [
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]

MatmulInputs = collections.namedtuple(
    "MatmulInputs",
    [
        "a",
        "b",
        "c",
        "m",
        "n",
        "k",
        "ab_type",
        "bias",
        "beta",
        "alpha",
        "epilogs",
    ],
)


def notNone(x):
    return x is not None


@composite
def matrix_multiply_arrays(draw):
    m = draw(one_of(none(), problem_size_mnk))
    n = draw(one_of(none(), problem_size_mnk))
    k = draw(problem_size_mnk)
    ab_type = draw(sampled_from(ab_type_values))
    # Generate data in range [0, 5] to match sample_matrix() from utils
    # Only non-negative reals to avoid catastrophic cancellation
    element_properties: dict[str, typing.Any] = dict(
        allow_infinity=False,
        allow_nan=False,
        allow_subnormal=False,
        max_magnitude=np.sqrt(50),
        min_magnitude=0,
        max_value=5,
        min_value=0,
    )
    # NOTE: It is unfeasible for hypothesis to explore a parameter space where
    # all elements of the input arrays are unique, so most of the time, arrays
    # contain just a few unique values
    a = draw(arrays(dtype=ab_type, shape=(k,) if m is None else (m, k), elements=element_properties))
    b = draw(arrays(dtype=ab_type, shape=(k,) if n is None else (k, n), elements=element_properties))
    m_for_c = 1 if m is None else m
    c = draw(
        one_of(
            none(),
            arrays(
                dtype=ab_type,
                shape=tuple(
                    filter(
                        notNone,
                        (
                            m_for_c,
                            draw(sampled_from([1, n])),
                        ),
                    )
                ),
                elements=element_properties,
            ),
        )
    )
    beta = None if c is None else draw(from_dtype(dtype=np.dtype(ab_type), **element_properties))
    alpha = draw(one_of(none(), from_dtype(dtype=np.dtype(ab_type), **element_properties)))
    epilogs = draw(sampled_from(MatmulEpilog_valid_pairs_list))
    bias = (
        draw(arrays(dtype=ab_type, shape=(m_for_c, 1), elements=element_properties))
        if epilogs[0] in MatmulEpilog_BIAS_list
        else None
    )
    assume(np.all(np.isfinite(a)))
    assume(np.all(np.isfinite(b)))
    assume(c is None or np.all(np.isfinite(c)))
    assume(bias is None or np.all(np.isfinite(bias)))
    # FIXME: We should also test broadcasting of c. i.e. when the shape of c is
    # (m, 1), but currently we are avoiding a bug where broadcasting doesn't
    # work on V100 and double precision
    assume(ab_type != np.float64 or c is None or c.shape != (m_for_c, 1))
    return MatmulInputs(a=a, b=b, c=c, m=m, n=n, k=k, ab_type=ab_type, bias=bias, beta=beta, alpha=alpha, epilogs=epilogs)


@composite
def preference_object_strategy(draw):
    limit = draw(integers(min_value=1, max_value=8))
    reduction_scheme_mask = draw(one_of(sampled_from(ReductionScheme)))
    return MatmulPlanPreferences(reduction_scheme_mask=reduction_scheme_mask, limit=limit)


@nvmath_seed()
@given(
    input_arrays=matrix_multiply_arrays(),
    order=sampled_from(["F", "C"]),
    options=one_of(
        none(),
        fixed_dictionaries(
            {
                "blocking": sampled_from(options_blocking_values),
                "allocator": sampled_from(options_allocator_values),
                "scale_type": one_of(none(), sampled_from(CudaDataType)),
                # "compute_type": one_of(none(), sampled_from(nvmath.linalg.ComputeType)),
            }
        ),
    ),
    preferences=one_of(
        none(),
        fixed_dictionaries(
            {
                "reduction_scheme_mask": one_of(sampled_from(ReductionScheme)),
                "max_waves_count": one_of(floats(min_value=0, max_value=100, width=32)),
            }
        ),
        preference_object_strategy(),
    ),
)
def test_matmul(input_arrays, order, options, preferences):
    """Call nvmath.linalg.advanced.matmul() with valid inputs."""
    a, b, c, m, n, k, ab_type, bias, beta, alpha, epilogs = input_arrays
    epilog, epilog1 = epilogs

    d_a = cp.asarray(a, order=order)
    d_b = cp.asarray(b, order=order)
    c_order = "F" if epilog is not None and epilog in [MatmulEpilog.BGRADB, MatmulEpilog.BGRADA] else order
    d_c = (
        # FIXME: c must be F ordered when using BGRAD[A,B]
        None if c is None else cp.asarray(c, order=c_order)
    )

    epilog_inputs = None if epilog is None else {}

    if epilog is not None and epilog in MatmulEpilog_BIAS_list:
        epilog_inputs["bias"] = cp.asarray(bias, order=order)

    try:
        result_c = matmul(
            d_a,
            d_b,
            c=d_c,
            alpha=alpha,
            beta=beta,
            epilog=epilog,
            epilog_inputs=epilog_inputs,
            preferences=preferences,
            options=options,
        )
        verify_result(a, b, c, result_c, alpha, beta, epilog, epilog_inputs)

        if epilog1 is not None:
            epilog1_inputs = result_c[1]

            result_c = matmul(
                d_a,
                d_b,
                c=d_c,
                alpha=alpha,
                beta=beta,
                epilog=epilog1,
                epilog_inputs=epilog1_inputs,
                preferences=preferences,
                options=options,
            )
            verify_result(a, b, c, result_c, alpha, beta, epilog1, epilog1_inputs)

    # Suppress allowed exceptions
    except cuBLASLtError as e:
        if "NOT_SUPPORTED" in str(e) or "CUBLAS_STATUS_INVALID_VALUE" in str(e):
            # Catch both not_supported and invalid value because some features
            # which are only unsupported on certain devices are raised as invalid
            # value in older libraries
            pass
        else:
            raise e
    except ValueError as e:
        # FIXME: Check for CUDA toolkit version 11
        if (
            re.search("K=1 is not supported for (BGRAD(A|B)|D(R|G)ELU) epilog", str(e))
            or "requires cublaslt >=" in str(e)
            or ("`c` must be at least 2-D." in str(e) and c is not None and len(c.shape) < 2)
            or ("Unsupported scale type." in str(e) and options["scale_type"] not in SCALE_TYPE_TO_DEFAULT_COMPUTE_TYPE)
            or (
                "Unsupported compute type." in str(e)
                and options["compute_type"] not in COMPUTE_TYPE_TO_DEFAULT_SCALE_TYPE["real"]
            )
            or re.search("Selected scale_type=(.*) compute_type=(.*) are not supported for data types", str(e))
        ):
            pass
        else:
            raise e


problem_size = integers(min_value=0, max_value=256)
f32_strategy = arrays(np.float32, shape=tuples(problem_size, problem_size), elements=floats(min_value=1, max_value=2, width=32))
f64_strategy = arrays(np.float64, shape=tuples(problem_size, problem_size), elements=floats(min_value=1, max_value=2, width=32))
c32_strategy = arrays(
    np.complex64, shape=tuples(problem_size, problem_size), elements=floats(min_value=1, max_value=2, width=32)
)
c64_strategy = arrays(
    np.complex128, shape=tuples(problem_size, problem_size), elements=floats(min_value=1, max_value=2, width=32)
)
f64_strategy = arrays(np.float64, shape=tuples(problem_size, problem_size), elements=floats(min_value=1, max_value=2, width=64))
f16_strategy = arrays(np.float16, shape=tuples(problem_size, problem_size), elements=floats(min_value=1, max_value=2, width=16))

options_blocking_values_negative = [True, False, "auto", "none"]

options_allocator_values_negative = [
    _RawCUDAMemoryManager(0, logging.getLogger()),
    _CupyCUDAMemoryManager(0, logging.getLogger()),
    "none",
]


def generate_alpha_beta(value_type, value):
    if value is not None:
        tmp = np.zeros((1,), dtype=value_type)
        tmp[0] = value
        return tmp[0]
    else:
        return None


@nvmath_seed()
@given(
    a=one_of(f16_strategy, f32_strategy, f64_strategy, c32_strategy, c64_strategy),
    b=one_of(f16_strategy, f32_strategy, f64_strategy, c32_strategy, c64_strategy),
    c=one_of(f16_strategy, f32_strategy, f64_strategy, c32_strategy, c64_strategy, none()),
    alpha_value=one_of(floats(min_value=1, max_value=2, width=32), none()),
    beta_value=one_of(floats(min_value=1, max_value=2, width=32), none()),
    epilog=one_of(sampled_from(MatmulEpilog), integers(min_value=0x9999), none()),
    epilog_inputs=one_of(fixed_dictionaries({}), none()),
    options=one_of(
        fixed_dictionaries(
            {
                "blocking": one_of(sampled_from(options_blocking_values_negative), none()),
                "allocator": one_of(sampled_from(options_allocator_values_negative), none()),
                "scale_type": one_of(integers(min_value=0x9999), none()),
            }
        ),
        none(),
    ),
    preferences=one_of(
        fixed_dictionaries(
            {
                "reduction_scheme_mask": one_of(sampled_from(ReductionScheme), none()),
                "max_waves_count": one_of(floats(min_value=0, max_value=99999, width=32), none()),
                "numerical_impl_mask": one_of(
                    sampled_from(MatmulNumericalImplFlags), integers(min_value=0, max_value=999999), none()
                ),
                "limit": one_of(integers(min_value=0, max_value=99999), none()),
            }
        ),
        none(),
    ),
)
def test_matmul_negative(a, b, c, alpha_value, beta_value, epilog, epilog_inputs, options, preferences):
    """Call nvmath.linalg.advanced.matmul() with invalid inputs; catch expected
    exceptions."""
    if c is not None and ((a.dtype != c.dtype) or (a.shape[0] != c.shape[0]) or (c.shape[1] != b.shape[1])):
        return

    d_a = cp.asarray(a, order="F")
    d_b = cp.asarray(b, order="F")
    d_c = cp.asarray(c, order="F") if c is not None else None

    alpha = generate_alpha_beta(a.dtype, alpha_value)
    beta = generate_alpha_beta(a.dtype, beta_value)

    try:
        result_c = matmul(
            d_a,
            d_b,
            c=d_c,
            alpha=alpha,
            beta=beta,
            epilog=epilog,
            epilog_inputs=epilog_inputs,
            preferences=preferences,
            options=options,
        )
        verify_result(a, b, c, result_c, alpha, beta, epilog, epilog_inputs)

    except AssertionError as e:
        if "Unsupported alpha or beta type." in str(e):
            # FIXME: Is this AssertionError error still used?
            assert not (isinstance(alpha, int | float | type(None)) and isinstance(beta, int | float | type(None)))
        elif str(e) == "Not supported.":
            assert epilog is not None and epilog not in EPILOG_INPUT_HANDLERS_MAP
        else:
            raise e
    except ValueError as e:
        if "A value for beta must be provided if operand C is provided." in str(e):
            assert (beta is None) and (c is not None)
        elif f"Unsupported combination of dtypes for operands A {a.dtype} and B {b.dtype}" in str(e):
            assert a.dtype != b.dtype
        elif (
            f"The 'K' extent must match for the operands: K={a.shape[1]} in operand A is not equal to K={b.shape[0]} "
            "in operand B." in str(e)
        ):
            assert a.shape[1] != b.shape[0]
        elif re.search(
            re.compile(r"The (M|N) dimension of the c matrix \(\d+\) must match the (M|N) dimension of (a|b)\."), str(e)
        ):
            assert c.shape[0] != a.shape[0] or c.shape[1] != b.shape[1]
        elif re.search(re.compile(r"The epilog \w+ requires the following input tensors: \{\'\w+\'\}\."), str(e)):
            assert epilog is not None
            assert epilog_inputs is None or epilog_inputs == {}
        elif "The value specified for blocking must be either True or 'auto'." in str(e):
            assert options["blocking"] not in (True, "auto")
        elif "is not a valid CudaDataType" in str(e):
            assert not isinstance(options["scale_type"], CudaDataType)
        elif "Unsupported layout." in str(e):
            assert a.shape[0] == 0 or a.shape[1] == 0 or b.shape[0] == 0 or b.shape[1] == 0
        elif "The extents must be strictly positive" in str(e):
            assert (
                any(e <= 0 for e in a.shape) or any(e <= 0 for e in b.shape) or (c is not None and any(e <= 0 for e in c.shape))
            )
        elif "requires cublaslt >=" in str(e):
            from nvmath.bindings import cublasLt

            assert cublasLt.get_version() < EPILOG_MINIMUM_VERSIONS_MAP[epilog]["cublaslt"] or (
                a.shape[-2] == 1 and c is not None and c.shape[-1] == 1 and epilog & MatmulEpilog.BIAS > 0
            )
        elif re.search("K=1 is not supported for (BGRAD(A|B)|D(R|G)ELU) epilog", str(e)):
            assert a.shape[1] == 1 and b.shape[0] == 1
            assert epilog in [
                MatmulEpilog.BGRADA,
                MatmulEpilog.BGRADB,
                MatmulEpilog.DGELU_BGRAD,
                MatmulEpilog.DGELU,
                MatmulEpilog.DRELU_BGRAD,
                MatmulEpilog.DRELU,
            ]
        else:
            raise e
    except TypeError as e:
        if "an integer is required" in str(e):
            pass  # ignore error in nvmath.bindings.cublasLt.matrix_layout_destroy
        elif (
            "The Matrix multiplication plan preferences must be provided as an object of type MatmulPlanPreferences "
            "or as a dict with valid Matrix multiplication plan preferences." in str(e)
        ):
            assert not isinstance(preferences, MatmulPlanPreferences)
        elif "The allocator must be an object of type that fulfils the BaseCUDAMemoryManager protocol" in str(e):
            assert not isinstance(options["allocator"], BaseCUDAMemoryManager)
        elif "int() argument must be a string, a bytes-like object or a number, not 'NoneType'" in str(e):
            assert preferences["reduction_scheme_mask"] is None or preferences["numerical_impl_mask"] is None
        else:
            raise e
    except cuBLASLtError as e:
        if "NOT_SUPPORTED" in str(e) or "CUBLAS_STATUS_INVALID_VALUE" in str(e):
            # Catch both not_supported and invalid value because some features
            # which are only unsupported on certain devices are raised as invalid
            # value in older libraries
            pass
        else:
            raise e
