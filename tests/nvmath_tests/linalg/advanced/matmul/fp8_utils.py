# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

try:
    import torch
except ImportError:
    torch = None
import pytest
import numpy as np
from .utils import sample_matrix, assert_tensors_equal, to_numpy
from nvmath.internal.utils import check_or_create_options
from nvmath.linalg.advanced import matmul
from nvmath.linalg.advanced import _configuration

if torch is None:
    pytest.skip("Torch is required for FP8 tests", allow_module_level=True)


class Fp8Helper:
    """
    A helper class to compare quantized results.
    """

    def __init__(self, exponent_bits, significand_bits):
        self.exponent_bits, self.significand_bits = exponent_bits, significand_bits

        def is_normal(e, m):
            if (exponent_bits, significand_bits) == (4, 3):
                return e > 0 and (e, m) != (2**exponent_bits - 1, 2**significand_bits - 1)
            elif (exponent_bits, significand_bits) == (5, 2):
                return e > 0 and e != 2**exponent_bits - 1
            else:
                raise RuntimeError(f"is_normal not implemented for E{exponent_bits}M{significand_bits}")

        # Compute all values of FP8 number.
        exponent_bias = 2 ** (exponent_bits - 1) - 1
        normal_values = [
            sign * 2 ** (e - exponent_bias) * (1 + m / 2**significand_bits)
            for e in range(2**exponent_bits)
            for m in range(2**significand_bits)
            for sign in (-1, 1)
            if is_normal(e, m)
        ]
        subnormal_values = [
            sign * 2 ** (1 - exponent_bias) * (0 + m / 2**significand_bits)
            for m in range(1, 2**significand_bits)
            for sign in (-1, 1)
        ]
        values = subnormal_values + normal_values + [0.0]
        assert len(set(values)) == len(values)
        self.values = np.asarray(sorted(values))

        # For each value, calculate the range it covers.
        middles = (self.values[1:] + self.values[:-1]) / 2
        self.lranges = np.append(np.asarray([-np.inf]), middles)
        self.rranges = np.append(middles, np.asarray([np.inf]))

    def range(self, value):
        """
        Finds a representable value closest to `value` and returns its range.
        """

        i = np.abs(self.values - value).argmin()
        return self.lranges[i], self.rranges[i]

    def absdiff(self, quantized, expected):
        """
        Returns absolute difference between the ranges of quantized numbers and the expected
        values.
        """
        l, r = np.vectorize(self.range)(quantized)
        diff = np.minimum(abs(l - expected), abs(r - expected))
        diff[(l <= expected) & (r >= expected)] = 0.0
        return diff

    def allclose(self, quantized, expected, atol=1e-2, rtol=1e-2, return_info=False):
        """
        Checks if quantized values are close enough to the expected ones.
        """
        quantized = to_numpy(quantized.type(torch.float64))
        expected = to_numpy(expected.type(torch.float64))
        ok = np.all(self.absdiff(quantized, expected) <= atol + rtol * np.abs(expected))
        if not return_info:
            return ok
        else:
            aerr = self.absdiff(quantized, expected)
            rerr = aerr / (np.abs(expected) + 0.000001)
            return ok, {
                "aerr": np.max(aerr),
                "atol": atol,
                "rerr": np.max(rerr),
                "rtol": rtol,
            }


fp8helpers = {
    "float8_e4m3fn": Fp8Helper(4, 3),
    "float8_e5m2": Fp8Helper(5, 2),
}


def choose_scales(
    a, b, c, atype, btype, ctype, dtype, alpha=1.0, beta=1.0, allowed_in_range=(0.5, 2), allowed_out_range=(1, 100)
):
    """
    Chooses reasonable scales for each of the operands. Tries to fit (absolute values of)
    a, b and c into `allowed_in_range`, and (absolute values of) d into `allowed_out_range`.
    However, some noise is added at the end, so this is not a hard guarantee.
    """
    a = a.type(torch.float32)
    amax = a.abs().max().item()
    ascale = np.random.uniform(*allowed_in_range) / amax if amax > 0 else 1
    a *= ascale

    b = b.type(torch.float32)
    bmax = b.abs().max().item()
    bscale = np.random.uniform(*allowed_in_range) / bmax if bmax > 0 else 1
    b *= bscale

    if c is not None:
        c = c.type(torch.float32)
        cmax = c.abs().max().item()
        cscale = np.random.uniform(*allowed_in_range) / cmax if cmax > 0 else 1
        c *= cscale
    else:
        cscale = None

    if c is not None:
        d = alpha * a @ b + beta * c
    else:
        d = alpha * a @ b

    dmax = d.abs().max().item()
    dscale = np.random.uniform(*allowed_out_range) / dmax if dmax > 0 else 1

    # Add some noise
    ascale *= np.random.uniform(0.95, 1.05)
    bscale *= np.random.uniform(0.95, 1.05)
    if cscale is not None:
        cscale *= np.random.uniform(0.95, 1.05)
    dscale *= np.random.uniform(0.95, 1.05)

    # Flip the signs randomly
    ascale *= np.random.choice((-1, 1))
    bscale *= np.random.choice((-1, 1))
    if cscale is not None:
        cscale *= np.random.choice((-1, 1))
    dscale *= np.random.choice((-1, 1))

    result_type = dtype or ctype or atype
    if "float8" not in result_type:
        dscale = None  # Not supported, would raise an error

    if ctype and "float8" not in ctype:
        cscale = None  # Not supported, would raise an error

    return {"a": ascale, "b": bscale, "c": cscale, "d": dscale}


def simple_scales(atype, btype, ctype, dtype):
    ascale = 1.2
    bscale = 3.4
    cscale = 0.56
    dscale = 0.78
    result_type = dtype or ctype or atype
    if "float8" not in result_type:
        dscale = None  # Not supported, would raise an error
    if not ctype or "float8" not in ctype:
        cscale = None  # Not supported, would raise an error
    return {"a": ascale, "b": bscale, "c": cscale, "d": dscale}


def fp8_matmul_reference(
    a, b, c=None, *args, epilog_inputs=None, quantization_scales=None, options=None, preferences=None, **kwargs
):
    """
    Computes FP8-like matmul, but with higher precision.
    """
    quantization_scales = check_or_create_options(
        _configuration.MatmulQuantizationScales, quantization_scales, "Matmul quantization_scales"
    )
    options = check_or_create_options(_configuration.MatmulOptions, options, "Matmul options")
    preferences = check_or_create_options(_configuration.MatmulPlanPreferences, preferences, "Matmul preferences")
    options.result_type = None
    options.result_amax = False
    preferences.epilog.aux_type = None
    epilog_aux_amax = preferences.epilog.aux_amax
    preferences.epilog.aux_amax = False

    if epilog_inputs is None:
        epilog_inputs = {}
    epilog_inputs = epilog_inputs.copy()
    aux_scale = epilog_inputs.pop("aux_quantization_scale", None)

    for key in ("bias", "gelu_aux"):
        if epilog_inputs and key in epilog_inputs:
            epilog_inputs[key] = epilog_inputs[key].type(torch.float32)

    a_scale = quantization_scales.a if quantization_scales.a is not None else 1
    b_scale = quantization_scales.b if quantization_scales.b is not None else 1
    c_scale = quantization_scales.c if quantization_scales.c is not None else 1
    d_scale = quantization_scales.d if quantization_scales.d is not None else 1

    ascaled = a.type(torch.float32) * a_scale
    bscaled = b.type(torch.float32) * b_scale
    cscaled = None if c is None else c_scale * c.type(torch.float32)
    result = matmul(ascaled, bscaled, cscaled, *args, options=options, epilog_inputs=epilog_inputs, **kwargs)
    if isinstance(result, tuple):
        d, aux = result
        d *= d_scale
        assert len(aux) == 1
        key = list(aux.keys())[0]
        if epilog_aux_amax:
            aux[f"{key}_amax"] = max(key)
        if aux_scale is not None:
            aux[key] *= aux_scale
    else:
        result *= d_scale

    return result


def assert_fp8_equal(result, reference, atol=1e-2, rtol=1e-2):
    """
    Checks if the result is close enough to the reference. For quantized results, uses
    Fp8Helper.
    """
    result_type = str(result.dtype).split(".")[-1]
    if "float8" in result_type:
        ok, info = fp8helpers[result_type].allclose(result, reference, atol=1e-1, rtol=5e-2, return_info=True)
        if not ok:
            print(f"Absolute error: {info['aerr']} (tolerance {info['atol']})")
            print(f"Relative error: {info['rerr']} (tolerance {info['rtol']})")
            print("Result:")
            print(result)
            print("Reference:")
            print(reference)
        assert ok
    else:
        assert_tensors_equal(result, reference, atol=atol, rtol=rtol)


def generate_inputs(m, n, k, atype, btype, ctype, *, c_transposed=False, min=0, max=5, use_cuda):
    """
    Generates matmul inputs of given shapes and types.
    """

    a = sample_matrix("torch", atype, (m, k), use_cuda=use_cuda, min=min, max=max)
    b = sample_matrix("torch", btype, (n, k), use_cuda=use_cuda, min=min, max=max).T

    if ctype is not None:
        if not c_transposed:
            c = sample_matrix("torch", ctype, (m, n), use_cuda=use_cuda, min=min, max=max)
        else:
            c = sample_matrix("torch", ctype, (n, m), use_cuda=use_cuda, min=min, max=max).T
        beta = np.random.uniform(-2, 2)
    else:
        c = None
        beta = None

    alpha = np.random.uniform(-2, 2)

    return a, b, c, alpha, beta
