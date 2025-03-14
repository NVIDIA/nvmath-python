# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


try:
    import torch
except ImportError:
    torch = None
import pytest
from .fp8_utils import fp8helpers
import numpy as np

if torch is None:
    pytest.skip("Torch is required for FP8 tests", allow_module_level=True)


def fp8_values(dtype, finite=False):
    values = list(torch.frombuffer(bytes(range(256)), dtype=getattr(torch, dtype)))
    values = [x.item() for x in values if not finite or torch.isfinite(x.type(torch.float32))]
    return values


@pytest.mark.parametrize(
    "format,left,value,right",
    (
        ("float8_e4m3fn", 0.00390625, 0.005859375, 0.0078125),
        ("float8_e4m3fn", 0.0, 0.001953125, 0.00390625),
        ("float8_e4m3fn", -0.00390625, -0.001953125, 0.0),
        ("float8_e4m3fn", -288.0, -256.0, -240.0),
        ("float8_e4m3fn", 1.25, 1.375, 1.5),
        ("float8_e4m3fn", 0.6875, 0.75, 0.8125),
        ("float8_e4m3fn", -448.0, -416.0, -384.0),
        ("float8_e4m3fn", -np.inf, -448.0, -416.0),
        ("float8_e4m3fn", 416.0, 448.0, np.inf),
        ("float8_e4m3fn", -0.001953125, 0.0, 0.001953125),
        ("float8_e5m2", 3072.0, 3584.0, 4096.0),
        ("float8_e5m2", -2048.0, -1792.0, -1536.0),
        ("float8_e5m2", 0.875, 1.0, 1.25),
        ("float8_e5m2", -0.625, -0.5, -0.4375),
        ("float8_e5m2", -57344.0, -49152.0, -40960.0),
        ("float8_e5m2", 40960.0, 49152.0, 57344.0),
        ("float8_e5m2", -np.inf, -57344.0, -49152.0),
        ("float8_e5m2", 49152.0, 57344.0, np.inf),
        ("float8_e5m2", 1.52587890625e-05, 3.0517578125e-05, 4.57763671875e-05),
        ("float8_e5m2", -4.57763671875e-05, -3.0517578125e-05, -1.52587890625e-05),
        ("float8_e5m2", -1.52587890625e-05, 0.0, 1.52587890625e-05),
    ),
)
def test_fp8_helper(format, left, value, right):
    helper = fp8helpers[format]
    isclose = lambda x, y: np.allclose(x, y, atol=0, rtol=1e-5)

    # Check ranges
    expected_range_left, expected_range_right = (value + left) / 2, (value + right) / 2
    range_left, range_right = helper.range(value)
    assert isclose(expected_range_left, range_left)
    assert isclose(expected_range_right, range_right)

    # Check if ranges work for non-exact match
    if np.isfinite(left):
        range_left, range_right = helper.range(value * 0.8 + left * 0.2)
        assert isclose(expected_range_left, range_left)
        assert isclose(expected_range_right, range_right)
    if np.isfinite(right):
        range_left, range_right = helper.range(value * 0.7 + right * 0.3)
        assert isclose(expected_range_left, range_left)
        assert isclose(expected_range_right, range_right)

    # Check if absdiff works
    scalar_absdiff = lambda x, y: helper.absdiff(np.asarray([x]), np.asarray([y]))
    assert isclose(scalar_absdiff(value, value), 0)

    if np.isfinite(right):
        assert isclose(scalar_absdiff(value, right), abs(right - range_right))
        assert isclose(scalar_absdiff(value, right + 1), abs(right - range_right) + 1)
        assert isclose(scalar_absdiff(value, value * 0.9 + right * 0.1), 0)
        assert isclose(scalar_absdiff(value, value * 0.5 + right * 0.5), 0)
    if np.isfinite(left):
        assert isclose(scalar_absdiff(value, value * 0.9 + left * 0.1), 0)
        assert isclose(scalar_absdiff(value, value * 0.5 + left * 0.5), 0)
        assert isclose(scalar_absdiff(value, left - 1), abs(left - range_left) + 1)
        assert isclose(scalar_absdiff(value, left), abs(left - range_left))

    if not np.isfinite(right):
        assert isclose(scalar_absdiff(value, value + 1234567), 0)
    if not np.isfinite(left):
        assert isclose(scalar_absdiff(value, value - 7654321), 0)
